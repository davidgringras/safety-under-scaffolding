"""
ScaffoldSafetyEval: main evaluation orchestrator.

Provides the public API::

    eval = ScaffoldSafetyEval(
        models=["claude-opus-4-6", "gpt-5.2"],
        configs=["direct", "react", "multi_agent", "map_reduce"],
        benchmarks=["truthfulqa", "bbq", "sycophancy", "xstest"],
    )
    results = eval.run()
    results.scorecard()

Supports concurrent execution, progress callbacks, custom scaffold/benchmark
registration, blinding, and pre-flight validation.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from scaffold_safety.configs import SCAFFOLD_REGISTRY
from scaffold_safety.configs.base import BaseScaffold, ScaffoldResult
from scaffold_safety.benchmarks import BENCHMARK_REGISTRY
from scaffold_safety.benchmarks.base import BenchmarkCase, BenchmarkLoader
from scaffold_safety.blinding.sanitizer import ResponseSanitizer
from scaffold_safety.blinding.protocol import BlindingProtocol
from scaffold_safety.results import EvalResults
from scaffold_safety.utils.providers import ModelSpec, resolve_model, KNOWN_MODELS
from scaffold_safety.utils.checkpointing import Checkpointer
from scaffold_safety.utils.logging import setup_logger


logger = logging.getLogger("scaffold_safety")


# ---------------------------------------------------------------------------
# Pre-registration invariants (locked before data collection)
# ---------------------------------------------------------------------------

PRE_REGISTERED_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 1024,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

@dataclass
class ProgressInfo:
    """Structured progress information passed to callbacks.

    Attributes
    ----------
    completed : int
        Number of cases completed so far.
    total : int
        Total number of cases to process.
    current_benchmark : str
        Currently processing benchmark ID.
    current_model : str
        Currently processing model ID.
    current_config : str
        Currently processing scaffold config ID.
    elapsed_seconds : float
        Wall-clock seconds since evaluation started.
    errors : int
        Number of errors encountered so far.
    """
    completed: int
    total: int
    current_benchmark: str
    current_model: str
    current_config: str
    elapsed_seconds: float
    errors: int

    @property
    def fraction(self) -> float:
        """Completion fraction in [0, 1]."""
        return self.completed / self.total if self.total > 0 else 0.0

    @property
    def percent(self) -> float:
        """Completion percentage."""
        return self.fraction * 100.0

    @property
    def eta_seconds(self) -> float | None:
        """Estimated seconds remaining, or None if not enough data."""
        if self.completed == 0 or self.elapsed_seconds <= 0:
            return None
        rate = self.completed / self.elapsed_seconds
        remaining = self.total - self.completed
        return remaining / rate


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when the evaluation configuration is invalid."""
    pass


def _validate_models(models: list[str]) -> list[ModelSpec]:
    """Validate and resolve model strings to ModelSpec objects.

    Parameters
    ----------
    models : list[str]
        Model identifier strings.

    Returns
    -------
    list[ModelSpec]
        Resolved model specifications.

    Raises
    ------
    ConfigurationError
        If models list is empty.
    """
    if not models:
        raise ConfigurationError(
            "At least one model must be specified. "
            f"Known models: {sorted(KNOWN_MODELS.keys())}"
        )
    specs = []
    for m in models:
        spec = resolve_model(m)
        specs.append(spec)
        logger.debug(
            "Resolved model %r -> %s (provider=%s)",
            m, spec.litellm_id, spec.provider,
        )
    return specs


def _validate_configs(config_ids: list[str]) -> None:
    """Validate scaffold configuration IDs.

    Raises
    ------
    ConfigurationError
        If any config ID is not in the registry.
    """
    if not config_ids:
        raise ConfigurationError(
            "At least one scaffold config must be specified. "
            f"Available: {sorted(SCAFFOLD_REGISTRY.keys())}"
        )
    for cid in config_ids:
        if cid not in SCAFFOLD_REGISTRY:
            available = sorted(SCAFFOLD_REGISTRY.keys())
            raise ConfigurationError(
                f"Unknown scaffold config: {cid!r}. "
                f"Available configs: {available}. "
                f"Register custom scaffolds with scaffold_safety.register_scaffold()."
            )


def _validate_benchmarks(benchmark_ids: list[str]) -> None:
    """Validate benchmark IDs.

    Raises
    ------
    ConfigurationError
        If any benchmark ID is not in the registry.
    """
    if not benchmark_ids:
        raise ConfigurationError(
            "At least one benchmark must be specified. "
            f"Available: {sorted(BENCHMARK_REGISTRY.keys())}"
        )
    for bid in benchmark_ids:
        if bid not in BENCHMARK_REGISTRY:
            available = sorted(BENCHMARK_REGISTRY.keys())
            raise ConfigurationError(
                f"Unknown benchmark: {bid!r}. "
                f"Available benchmarks: {available}. "
                f"Register custom benchmarks with scaffold_safety.register_benchmark()."
            )


def _validate_temperature(temperature: float, model_specs: list[ModelSpec]) -> None:
    """Warn if temperature != 0 is set for reasoning models.

    Parameters
    ----------
    temperature : float
        The configured sampling temperature.
    model_specs : list[ModelSpec]
        Resolved model specifications.
    """
    for spec in model_specs:
        if not spec.supports_temperature and temperature != 0.0:
            logger.warning(
                "Model %s (provider=%s) does not support temperature control. "
                "The temperature parameter will be ignored for this model. "
                "Set temperature=0.0 to suppress this warning.",
                spec.model_id, spec.provider,
            )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class ScaffoldSafetyEval:
    """Main evaluation orchestrator for scaffold safety testing.

    Runs every specified model through every scaffold configuration on every
    benchmark case, scores results, and returns an :class:`EvalResults` object
    with analysis, scorecard generation, and export methods.

    Parameters
    ----------
    models : list[str]
        Model identifiers. Can be known names (e.g. ``"claude-opus-4-6"``,
        ``"gpt-5.2"``) or raw LiteLLM model strings (e.g. ``"openai/gpt-4"``).
        At least one model must be specified.
    configs : list[str] | None
        Scaffold configuration names from the scaffold registry.
        Default: all registered configs (``["direct", "react", "multi_agent", "map_reduce"]``).
    benchmarks : list[str] | None
        Benchmark names from the benchmark registry.
        Default: all registered benchmarks (``["truthfulqa", "bbq", "sycophancy", "xstest"]``).
    data_dir : str | Path | None
        Directory containing benchmark data files (JSONL format).
        Default: ``"data/benchmarks"``.
    output_dir : str | Path | None
        Directory for results, checkpoints, and logs.
        Default: ``"scaffold_safety_output"``.
    seed : int
        Random seed for reproducibility. Default: 42 (pre-registered).
    blinding : bool
        Whether to enable response blinding for judge scoring. Default: True.
    dry_run : bool
        If True, skip actual API calls and return placeholder responses.
        Useful for testing the full pipeline without incurring costs.
    max_tokens : int
        Maximum tokens per model response. Default: 1024 (pre-registered).
    temperature : float
        Sampling temperature. Default: 0.0 (pre-registered). Models that do
        not support temperature (e.g. reasoning models) ignore this parameter.
    max_workers : int
        Maximum concurrent worker threads for parallel execution. Set to 1
        for fully sequential processing. Default: 1.
    max_retries : int
        Maximum retry attempts per API call on transient failure. Default: 3.
    on_progress : Callable[[ProgressInfo], None] | None
        Optional callback invoked after each case completes. Receives a
        :class:`ProgressInfo` dataclass with current progress, timing, and
        error counts.

    Raises
    ------
    ConfigurationError
        If models, configs, or benchmarks contain invalid values.

    Examples
    --------
    Basic usage::

        from scaffold_safety import ScaffoldSafetyEval

        eval = ScaffoldSafetyEval(
            models=["claude-opus-4-6", "gpt-5.2"],
            configs=["direct", "react"],
            benchmarks=["truthfulqa"],
        )
        results = eval.run()
        results.scorecard()

    With progress callback::

        def on_progress(info):
            print(f"{info.percent:.1f}% ({info.completed}/{info.total})")

        eval = ScaffoldSafetyEval(
            models=["claude-opus-4-6"],
            configs=["direct"],
            benchmarks=["truthfulqa"],
            on_progress=on_progress,
        )
        results = eval.run()

    Dry-run mode for testing::

        eval = ScaffoldSafetyEval(
            models=["test/model"],
            configs=["direct"],
            benchmarks=["truthfulqa"],
            dry_run=True,
        )
        results = eval.run()
        print(results.summary())
    """

    def __init__(
        self,
        models: list[str],
        configs: list[str] | None = None,
        benchmarks: list[str] | None = None,
        *,
        data_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        seed: int = 42,
        blinding: bool = True,
        dry_run: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_workers: int = 1,
        max_retries: int = 3,
        on_progress: Callable[[ProgressInfo], None] | None = None,
    ) -> None:
        # Resolve and validate models
        self.model_specs: list[ModelSpec] = _validate_models(models)
        self.model_ids: list[str] = [m.model_id for m in self.model_specs]

        # Resolve and validate configs
        self.config_ids: list[str] = configs or list(SCAFFOLD_REGISTRY.keys())
        _validate_configs(self.config_ids)

        # Resolve and validate benchmarks
        self.benchmark_ids: list[str] = benchmarks or list(BENCHMARK_REGISTRY.keys())
        _validate_benchmarks(self.benchmark_ids)

        # Paths
        self.data_dir: Path = Path(data_dir) if data_dir else Path("data/benchmarks")
        self.output_dir: Path = Path(output_dir) if output_dir else Path("scaffold_safety_output")

        # Experiment parameters
        self.seed: int = seed
        self.blinding: bool = blinding
        self.dry_run: bool = dry_run
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.max_workers: int = max(1, max_workers)
        self.max_retries: int = max_retries

        # Progress callback
        self._on_progress: Callable[[ProgressInfo], None] | None = on_progress

        # Temperature validation for reasoning models
        _validate_temperature(self.temperature, self.model_specs)

    def validate_config(self) -> dict[str, Any]:
        """Validate the full configuration without running the evaluation.

        Checks that all models, configs, and benchmarks are valid, that
        benchmark data files exist, and that experiment parameters are
        consistent with pre-registration invariants.

        Returns
        -------
        dict[str, Any]
            Validation report with keys: ``valid`` (bool), ``warnings``
            (list[str]), ``errors`` (list[str]), ``summary`` (dict).

        Examples
        --------
        ::

            eval = ScaffoldSafetyEval(models=["claude-opus-4-6"])
            report = eval.validate_config()
            if not report["valid"]:
                for error in report["errors"]:
                    print(f"ERROR: {error}")
        """
        warnings: list[str] = []
        errors: list[str] = []

        # Check pre-registration invariants
        if self.temperature != PRE_REGISTERED_PARAMS["temperature"]:
            warnings.append(
                f"temperature={self.temperature} differs from pre-registered "
                f"value {PRE_REGISTERED_PARAMS['temperature']}. "
                f"Document this deviation in your specification curve."
            )
        if self.max_tokens != PRE_REGISTERED_PARAMS["max_tokens"]:
            warnings.append(
                f"max_tokens={self.max_tokens} differs from pre-registered "
                f"value {PRE_REGISTERED_PARAMS['max_tokens']}."
            )
        if self.seed != PRE_REGISTERED_PARAMS["seed"]:
            warnings.append(
                f"seed={self.seed} differs from pre-registered "
                f"value {PRE_REGISTERED_PARAMS['seed']}."
            )

        # Check benchmark data files
        for bid in self.benchmark_ids:
            data_file = self._resolve_data_file(bid)
            if data_file is None:
                errors.append(
                    f"Benchmark data file not found for {bid!r} in {self.data_dir}. "
                    f"Provide a JSONL file named '{bid}.jsonl'."
                )

        # Check for reasoning model temperature
        for spec in self.model_specs:
            if not spec.supports_temperature and self.temperature != 0.0:
                warnings.append(
                    f"Model {spec.model_id!r} does not support temperature "
                    f"control; temperature={self.temperature} will be ignored."
                )

        # Check blinding consistency
        if not self.blinding:
            warnings.append(
                "Blinding is disabled. Results will not be assessor-blinded. "
                "This deviates from the pre-registered protocol."
            )

        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "summary": {
                "n_models": len(self.model_specs),
                "n_configs": len(self.config_ids),
                "n_benchmarks": len(self.benchmark_ids),
                "dry_run": self.dry_run,
                "blinding": self.blinding,
                "max_workers": self.max_workers,
            },
        }

    def estimate_cost(self) -> dict[str, Any]:
        """Estimate the cost of running the full evaluation.

        Uses benchmark case counts and model pricing to produce a rough
        cost estimate. Actual costs depend on response lengths and retry
        behaviour.

        Returns
        -------
        dict[str, Any]
            Cost estimate with keys: ``total_usd``, ``per_model`` (dict),
            ``n_api_calls``, ``n_cases``.

        Examples
        --------
        ::

            eval = ScaffoldSafetyEval(
                models=["claude-opus-4-6", "gpt-5.2"],
                configs=["direct", "react"],
                benchmarks=["truthfulqa"],
            )
            estimate = eval.estimate_cost()
            print(f"Estimated cost: ${estimate['total_usd']:.2f}")
        """
        # Estimate calls per scaffold config
        calls_per_config = {
            "direct": 1,
            "react": 3,       # average of 1-5
            "multi_agent": 3,  # primary + critic + revision
            "map_reduce": 4,   # decompose + 2 map + reduce
        }

        # Count benchmark cases
        n_cases = 0
        for bid in self.benchmark_ids:
            loader_cls = BENCHMARK_REGISTRY.get(bid)
            if loader_cls is not None:
                loader = loader_cls()
                n_cases += loader.spec.n_cases

        total_usd = 0.0
        per_model: dict[str, float] = {}

        for spec in self.model_specs:
            model_cost = 0.0
            for cid in self.config_ids:
                n_calls = calls_per_config.get(cid, 2)
                # Estimate ~500 input tokens, ~200 output tokens per call
                input_cost = (500 / 1000) * spec.cost_per_1k_input * n_calls * n_cases
                output_cost = (200 / 1000) * spec.cost_per_1k_output * n_calls * n_cases
                model_cost += input_cost + output_cost
            per_model[spec.model_id] = round(model_cost, 4)
            total_usd += model_cost

        return {
            "total_usd": round(total_usd, 2),
            "per_model": per_model,
            "n_api_calls": sum(
                calls_per_config.get(cid, 2) * n_cases * len(self.model_specs)
                for cid in self.config_ids
            ),
            "n_cases": n_cases * len(self.model_specs) * len(self.config_ids),
            "note": (
                "Estimates assume ~500 input tokens and ~200 output tokens per "
                "API call. Actual costs depend on response lengths, retries, and "
                "scaffold-specific call patterns."
            ),
        }

    def _resolve_data_file(self, benchmark_id: str) -> Path | None:
        """Resolve the data file path for a benchmark.

        Tries the primary name first (``{benchmark_id}.jsonl``), then falls
        back to the loader's ``spec.data_file`` attribute if available, and
        finally tries common alternative names.

        Parameters
        ----------
        benchmark_id : str
            The benchmark identifier.

        Returns
        -------
        Path | None
            The resolved data file path, or None if no file is found.
        """
        # Try primary name
        primary = self.data_dir / f"{benchmark_id}.jsonl"
        if primary.exists():
            return primary

        # Try loader spec's data_file if available
        loader_cls = BENCHMARK_REGISTRY.get(benchmark_id)
        if loader_cls is not None:
            loader = loader_cls()
            if hasattr(loader, "spec") and hasattr(loader.spec, "data_file"):
                spec_file = self.data_dir / loader.spec.data_file
                if spec_file.exists():
                    return spec_file

        # Legacy alternative names for built-in benchmarks
        alt_names: dict[str, str] = {
            "truthfulqa": "truthfulqa_mc1.jsonl",
            "xstest": "xstest_orbench.jsonl",
            "sycophancy": "sycophancy_eval.jsonl",
        }
        alt = alt_names.get(benchmark_id)
        if alt is not None:
            alt_path = self.data_dir / alt
            if alt_path.exists():
                return alt_path

        return None

    def _load_benchmarks(self) -> dict[str, tuple[BenchmarkLoader, list[BenchmarkCase]]]:
        """Load all benchmark cases from data files.

        Returns
        -------
        dict[str, tuple[BenchmarkLoader, list[BenchmarkCase]]]
            Mapping from benchmark ID to (loader, cases).
        """
        loaded: dict[str, tuple[BenchmarkLoader, list[BenchmarkCase]]] = {}
        for bid in self.benchmark_ids:
            loader_cls = BENCHMARK_REGISTRY[bid]
            loader = loader_cls()
            data_file = self._resolve_data_file(bid)
            if data_file is None:
                logger.warning(
                    "Benchmark data file not found for %r in %s. Skipping.",
                    bid, self.data_dir,
                )
                continue
            cases = loader.load_cases(data_file)
            loaded[bid] = (loader, cases)
            logger.info("Loaded %d cases for benchmark %r from %s", len(cases), bid, data_file)
        return loaded

    def _build_work_items(
        self,
        benchmarks: dict[str, tuple[BenchmarkLoader, list[BenchmarkCase]]],
        checkpointer: Checkpointer,
    ) -> list[dict[str, Any]]:
        """Build the list of work items, skipping already-completed cases.

        Parameters
        ----------
        benchmarks : dict
            Loaded benchmarks.
        checkpointer : Checkpointer
            For checking already-completed cases.

        Returns
        -------
        list[dict[str, Any]]
            Work items to process.
        """
        items: list[dict[str, Any]] = []
        for bid, (loader, cases) in benchmarks.items():
            system_prompt = loader.get_system_prompt()
            for case in cases:
                for model in self.model_specs:
                    for cid in self.config_ids:
                        check_record = {
                            "case_id": case.case_id,
                            "model_id": model.model_id,
                            "config_id": cid,
                        }
                        if checkpointer.is_completed(check_record):
                            continue
                        items.append({
                            "benchmark_id": bid,
                            "loader": loader,
                            "case": case,
                            "model": model,
                            "config_id": cid,
                            "system_prompt": system_prompt,
                        })
        return items

    def _process_single_case(
        self,
        item: dict[str, Any],
        scaffolds: dict[str, BaseScaffold],
        sanitizer: ResponseSanitizer,
        blinding_protocol: BlindingProtocol | None,
    ) -> dict[str, Any]:
        """Process a single evaluation case.

        Parameters
        ----------
        item : dict
            Work item with benchmark_id, loader, case, model, config_id,
            system_prompt.
        scaffolds : dict
            Instantiated scaffold objects.
        sanitizer : ResponseSanitizer
            For blinding/sanitization.
        blinding_protocol : BlindingProtocol | None
            For UUID-based blinding (if enabled).

        Returns
        -------
        dict[str, Any]
            Result record (success or error).
        """
        bid = item["benchmark_id"]
        loader: BenchmarkLoader = item["loader"]
        case: BenchmarkCase = item["case"]
        model: ModelSpec = item["model"]
        cid: str = item["config_id"]
        system_prompt: str = item["system_prompt"]

        try:
            scaffold = scaffolds[cid]
            result = scaffold.run(
                system_prompt=system_prompt,
                user_prompt=case.prompt,
                model=model,
                dry_run=self.dry_run,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
            )

            # Sanitize response
            sanitized, san_report = sanitizer.sanitize(result.final_response)

            # Blind response if protocol is active
            if blinding_protocol is not None:
                blinding_protocol.blind_response(
                    sanitized_response=sanitized,
                    user_prompt=case.prompt,
                    benchmark_id=bid,
                    model_id=model.model_id,
                    config_id=cid,
                    case_id=case.case_id,
                )

            # Score
            score = loader.score(case, sanitized)

            return {
                "case_id": case.case_id,
                "model_id": model.model_id,
                "config_id": cid,
                "benchmark_id": bid,
                "is_safe": score.is_safe,
                "is_correct": score.is_correct,
                "predicted": score.predicted,
                "correct": score.correct,
                "confidence": score.confidence,
                "final_response": result.final_response,
                "sanitized_response": sanitized,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.total_cost_usd,
                "n_api_calls": result.n_api_calls,
                "status": "success",
            }

        except Exception as e:
            logger.error(
                "Error processing %s/%s/%s/%s: %s",
                bid, case.case_id, model.model_id, cid, e,
            )
            return {
                "case_id": case.case_id,
                "model_id": model.model_id,
                "config_id": cid,
                "benchmark_id": bid,
                "is_safe": None,
                "status": "error",
                "error": str(e),
            }

    def run(self) -> EvalResults:
        """Execute the full evaluation.

        Runs every model through every scaffold config on every benchmark
        case, scores results, and returns an :class:`EvalResults` object.

        The evaluation loop supports:

        - **Checkpointing**: Results are saved incrementally; the evaluation
          resumes from where it left off if interrupted.
        - **Blinding**: When enabled, responses are sanitized and assigned
          UUIDs before scoring; the mapping is sealed with SHA-256.
        - **Concurrency**: Set ``max_workers > 1`` for parallel execution
          across cases (thread-based).
        - **Progress callbacks**: If ``on_progress`` was provided, it is
          called after each case completes.

        Returns
        -------
        EvalResults
            Container with all results and analysis methods.

        Raises
        ------
        RuntimeError
            If no benchmark data files are found.

        Examples
        --------
        ::

            eval = ScaffoldSafetyEval(
                models=["claude-opus-4-6"],
                configs=["direct"],
                benchmarks=["truthfulqa"],
                data_dir="data/benchmarks",
            )
            results = eval.run()
            print(results.summary())
        """
        setup_logger(log_dir=self.output_dir / "logs")
        logger.info(
            "Starting evaluation: %d models x %d configs x %d benchmarks "
            "(max_workers=%d, dry_run=%s, blinding=%s)",
            len(self.model_specs), len(self.config_ids), len(self.benchmark_ids),
            self.max_workers, self.dry_run, self.blinding,
        )

        # Load benchmarks
        benchmarks = self._load_benchmarks()
        if not benchmarks:
            raise RuntimeError(
                f"No benchmark data found in {self.data_dir}. "
                f"Provide JSONL data files for: {self.benchmark_ids}. "
                f"Expected file names: "
                + ", ".join(f"'{bid}.jsonl'" for bid in self.benchmark_ids)
            )

        # Set up output directory and checkpointing
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpointer = Checkpointer(
            self.output_dir / "results.jsonl",
            key_fields=("case_id", "model_id", "config_id"),
        )

        # Set up blinding
        sanitizer = ResponseSanitizer(strip_self_id=self.blinding)
        blinding_protocol = BlindingProtocol(seed=self.seed) if self.blinding else None

        # Instantiate scaffolds
        scaffolds: dict[str, BaseScaffold] = {
            cid: SCAFFOLD_REGISTRY[cid]() for cid in self.config_ids
        }

        # Build work items (skipping already-completed cases)
        work_items = self._build_work_items(benchmarks, checkpointer)

        # Count total including already-completed
        total_cases = sum(
            len(cases) * len(self.model_specs) * len(self.config_ids)
            for _, (_, cases) in benchmarks.items()
        )
        already_completed = total_cases - len(work_items)
        if already_completed > 0:
            logger.info(
                "Resuming from checkpoint: %d/%d already completed",
                already_completed, total_cases,
            )

        all_results: list[dict[str, Any]] = []
        processed = already_completed
        error_count = 0
        start_time = time.monotonic()

        if self.max_workers > 1 and len(work_items) > 1:
            # Concurrent execution
            logger.info(
                "Processing %d remaining cases with %d workers",
                len(work_items), self.max_workers,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_item = {
                    executor.submit(
                        self._process_single_case,
                        item, scaffolds, sanitizer, blinding_protocol,
                    ): item
                    for item in work_items
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    record = future.result()
                    checkpointer.save(record)
                    all_results.append(record)

                    if record.get("status") == "error":
                        error_count += 1

                    processed += 1
                    elapsed = time.monotonic() - start_time

                    if processed % 100 == 0 or processed == total_cases:
                        logger.info(
                            "Progress: %d/%d (%.1f%%) | errors: %d | elapsed: %.1fs",
                            processed, total_cases,
                            100.0 * processed / total_cases if total_cases > 0 else 0,
                            error_count, elapsed,
                        )

                    # Fire progress callback
                    if self._on_progress is not None:
                        item = future_to_item[future]
                        try:
                            self._on_progress(ProgressInfo(
                                completed=processed,
                                total=total_cases,
                                current_benchmark=item["benchmark_id"],
                                current_model=item["model"].model_id,
                                current_config=item["config_id"],
                                elapsed_seconds=elapsed,
                                errors=error_count,
                            ))
                        except Exception as cb_err:
                            logger.warning("Progress callback error: %s", cb_err)
        else:
            # Sequential execution
            logger.info("Processing %d remaining cases sequentially", len(work_items))
            for item in work_items:
                record = self._process_single_case(
                    item, scaffolds, sanitizer, blinding_protocol,
                )
                checkpointer.save(record)
                all_results.append(record)

                if record.get("status") == "error":
                    error_count += 1

                processed += 1
                elapsed = time.monotonic() - start_time

                if processed % 100 == 0 or processed == total_cases:
                    logger.info(
                        "Progress: %d/%d (%.1f%%) | errors: %d | elapsed: %.1fs",
                        processed, total_cases,
                        100.0 * processed / total_cases if total_cases > 0 else 0,
                        error_count, elapsed,
                    )

                # Fire progress callback
                if self._on_progress is not None:
                    try:
                        self._on_progress(ProgressInfo(
                            completed=processed,
                            total=total_cases,
                            current_benchmark=item["benchmark_id"],
                            current_model=item["model"].model_id,
                            current_config=item["config_id"],
                            elapsed_seconds=elapsed,
                            errors=error_count,
                        ))
                    except Exception as cb_err:
                        logger.warning("Progress callback error: %s", cb_err)

        # Seal blinding mapping if enabled
        if blinding_protocol is not None and blinding_protocol.n_blinded > 0:
            sealed = blinding_protocol.seal_mapping(self.output_dir)
            logger.info(
                "Blinding mapping sealed: SHA-256 = %s (%d responses)",
                sealed.sha256_hash, blinding_protocol.n_blinded,
            )

        elapsed_total = time.monotonic() - start_time
        logger.info(
            "Evaluation complete: %d results (%d errors) in %.1fs",
            len(all_results), error_count, elapsed_total,
        )

        return EvalResults(
            results=all_results,
            models=self.model_ids,
            configs=self.config_ids,
            benchmarks=self.benchmark_ids,
            methodology={
                "blinding": self.blinding,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "pre_registered": True,
                "specification_curve": True,
                "dry_run": self.dry_run,
                "max_workers": self.max_workers,
                "elapsed_seconds": round(elapsed_total, 2),
                "n_errors": error_count,
            },
        )

    def __repr__(self) -> str:
        return (
            f"ScaffoldSafetyEval("
            f"models={self.model_ids!r}, "
            f"configs={self.config_ids!r}, "
            f"benchmarks={self.benchmark_ids!r}, "
            f"dry_run={self.dry_run}, "
            f"blinding={self.blinding}, "
            f"max_workers={self.max_workers})"
        )

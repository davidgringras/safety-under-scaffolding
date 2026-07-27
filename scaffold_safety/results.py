"""
EvalResults: container for evaluation results with analysis methods.

Provides a clean API for post-evaluation analysis::

    results = eval.run()
    results.scorecard()
    results.report()
    results.specification_curve()
    results.export("output/results.csv", format="csv")
    results.filter(models=["opus"], benchmarks=["truthfulqa"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from scaffold_safety.analysis.statistics import (
    SafetyRate,
    compute_safety_rates,
    compute_effect_sizes,
    run_tost,
    run_chi_squared,
)
from scaffold_safety.analysis.scorecard import (
    generate_scorecard,
    ScaffoldSafetyScorecard,
)
from scaffold_safety.reporting.export import (
    export_results,
    export_scorecard,
    generate_latex_table,
)


logger = logging.getLogger("scaffold_safety")


class EvalResults:
    """Container for scaffold safety evaluation results.

    Provides methods for analysis, scorecard generation, export, and
    filtering. Supports iteration and standard container operations.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Raw result records, each containing at minimum: ``case_id``,
        ``model_id``, ``config_id``, ``benchmark_id``, ``is_safe``,
        ``status``.
    models : list[str]
        Model IDs included in the evaluation.
    configs : list[str]
        Scaffold config IDs included in the evaluation.
    benchmarks : list[str]
        Benchmark IDs included in the evaluation.
    methodology : dict[str, Any] | None
        Evaluation methodology metadata (blinding, seed, etc.).

    Examples
    --------
    ::

        results = eval.run()
        print(f"Evaluated {len(results)} cases")
        print(results.summary())

        # Export to CSV
        results.export("output/results.csv", format="csv")

        # Filter and analyze a subset
        subset = results.filter(models=["opus"], benchmarks=["truthfulqa"])
        subset.scorecard()
    """

    def __init__(
        self,
        results: list[dict[str, Any]],
        models: list[str],
        configs: list[str],
        benchmarks: list[str],
        methodology: dict[str, Any] | None = None,
    ) -> None:
        self._results = results
        self._models = models
        self._configs = configs
        self._benchmarks = benchmarks
        self._methodology = methodology or {}
        self._rates: list[SafetyRate] | None = None

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    @property
    def n_results(self) -> int:
        """Total number of result records."""
        return len(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._results)

    def __repr__(self) -> str:
        n_success = sum(1 for r in self._results if r.get("status") == "success")
        n_error = sum(1 for r in self._results if r.get("status") == "error")
        return (
            f"EvalResults("
            f"n={self.n_results}, "
            f"success={n_success}, "
            f"errors={n_error}, "
            f"models={self._models}, "
            f"configs={self._configs}, "
            f"benchmarks={self._benchmarks})"
        )

    # ------------------------------------------------------------------
    # Safety rates (computed and cached)
    # ------------------------------------------------------------------

    @property
    def rates(self) -> list[SafetyRate]:
        """Compute and cache safety rates."""
        if self._rates is None:
            self._rates = compute_safety_rates(self._results)
        return self._rates

    def safety_rates(self) -> list[SafetyRate]:
        """Return safety rates grouped by benchmark x model x config.

        Returns
        -------
        list[SafetyRate]
            One entry per (benchmark, model, config) cell with the
            computed safety rate and standard error.

        Examples
        --------
        ::

            for rate in results.safety_rates():
                print(f"{rate.benchmark_id}/{rate.model_id}/{rate.config_id}: "
                      f"{rate.safety_rate:.1%} ({rate.n_safe}/{rate.n_total})")
        """
        return self.rates

    # ------------------------------------------------------------------
    # Statistical analyses
    # ------------------------------------------------------------------

    def effect_sizes(self) -> list[Any]:
        """Compute effect sizes (Cohen's h, risk difference, OR, NNH).

        Compares each scaffold configuration to the direct baseline for
        every (benchmark, model) pair.

        Returns
        -------
        list[EffectSize]
            Effect size objects with ``cohen_h``, ``risk_difference``,
            ``odds_ratio``, and ``nnh`` attributes.

        Examples
        --------
        ::

            for e in results.effect_sizes():
                print(f"{e.benchmark_id}/{e.config_id}: "
                      f"Cohen's h={e.cohen_h:.3f}, NNH={e.nnh:.0f}")
        """
        return compute_effect_sizes(self.rates)

    def tost(self, margin: float = 0.02) -> list[Any]:
        """Run TOST equivalence tests.

        Tests whether the safety rate difference between each scaffold
        and the direct baseline falls within ``[-margin, +margin]``.

        Parameters
        ----------
        margin : float
            Equivalence margin (default: 0.02 = 2 percentage points,
            as pre-registered).

        Returns
        -------
        list[TOSTResult]
            TOST results with ``equivalent`` (bool), ``tost_p``,
            ``risk_difference``, and 90% confidence interval.

        Examples
        --------
        ::

            for t in results.tost(margin=0.02):
                status = "EQUIVALENT" if t.equivalent else "NOT equivalent"
                print(f"{t.benchmark_id}/{t.config_id}: {status} (p={t.tost_p:.4f})")
        """
        return run_tost(self.rates, margin=margin)

    def chi_squared(self) -> list[dict[str, Any]]:
        """Run chi-squared / Fisher exact tests.

        For each (benchmark, model, config) triple, tests whether the
        scaffold safety rate differs significantly from the direct baseline.

        Returns
        -------
        list[dict[str, Any]]
            Test results with ``test`` ("chi2" or "fisher"),
            ``statistic``, and ``p_value`` keys.

        Examples
        --------
        ::

            for t in results.chi_squared():
                sig = "*" if t["p_value"] < 0.05 else ""
                print(f"{t['benchmark_id']}/{t['config_id']}: "
                      f"p={t['p_value']:.4f}{sig}")
        """
        return run_chi_squared(self.rates)

    # ------------------------------------------------------------------
    # Scorecards
    # ------------------------------------------------------------------

    def scorecard(
        self, model_id: str | None = None,
    ) -> ScaffoldSafetyScorecard | list[ScaffoldSafetyScorecard]:
        """Generate Scaffold Safety Scorecard(s).

        The scorecard includes the Scaffold Robustness Index (SRI),
        safety rate matrix, NNH values, and maximum degradation.

        Parameters
        ----------
        model_id : str | None
            If provided, generate for that model only.
            Otherwise, generate for all models.

        Returns
        -------
        ScaffoldSafetyScorecard or list[ScaffoldSafetyScorecard]

        Examples
        --------
        ::

            # Single model
            sc = results.scorecard("opus")
            print(sc.to_text())

            # All models
            for sc in results.scorecard():
                print(f"{sc.model_id}: SRI={sc.sri:.1f} ({sc.sri_category})")
        """
        if model_id is not None:
            return generate_scorecard(
                self.rates, model_id, methodology=self._methodology,
            )

        scorecards = []
        for mid in self._models:
            scorecards.append(generate_scorecard(
                self.rates, mid, methodology=self._methodology,
            ))
        return scorecards

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        *,
        models: list[str] | None = None,
        configs: list[str] | None = None,
        benchmarks: list[str] | None = None,
        status: str | None = None,
    ) -> "EvalResults":
        """Return a filtered subset of results.

        Creates a new :class:`EvalResults` containing only records that
        match all specified criteria. Unspecified criteria match everything.

        Parameters
        ----------
        models : list[str] | None
            Keep only these model IDs.
        configs : list[str] | None
            Keep only these config IDs.
        benchmarks : list[str] | None
            Keep only these benchmark IDs.
        status : str | None
            Keep only records with this status (e.g. ``"success"`` or ``"error"``).

        Returns
        -------
        EvalResults
            A new results container with filtered records.

        Examples
        --------
        ::

            # Filter to one model and one benchmark
            subset = results.filter(models=["opus"], benchmarks=["truthfulqa"])
            print(f"Filtered: {len(subset)} results")

            # Inspect only errors
            errors = results.filter(status="error")
            for r in errors:
                print(f"{r['case_id']}: {r.get('error', 'unknown')}")
        """
        filtered = self._results
        if models is not None:
            model_set = set(models)
            filtered = [r for r in filtered if r.get("model_id") in model_set]
        if configs is not None:
            config_set = set(configs)
            filtered = [r for r in filtered if r.get("config_id") in config_set]
        if benchmarks is not None:
            benchmark_set = set(benchmarks)
            filtered = [r for r in filtered if r.get("benchmark_id") in benchmark_set]
        if status is not None:
            filtered = [r for r in filtered if r.get("status") == status]

        return EvalResults(
            results=filtered,
            models=models or self._models,
            configs=configs or self._configs,
            benchmarks=benchmarks or self._benchmarks,
            methodology=self._methodology,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        path: str | Path,
        *,
        format: str = "jsonl",
    ) -> Path:
        """Export raw results to a file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        format : str
            Output format: ``"json"``, ``"jsonl"``, or ``"csv"``.
            Default: ``"jsonl"``.

        Returns
        -------
        Path
            The written file path.

        Raises
        ------
        ValueError
            If *format* is not one of ``"json"``, ``"jsonl"``, ``"csv"``.

        Examples
        --------
        ::

            results.export("output/results.jsonl")
            results.export("output/results.csv", format="csv")
            results.export("output/results.json", format="json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(path, "w") as f:
                for r in self._results:
                    f.write(json.dumps(r, default=str) + "\n")
        elif format == "json":
            with open(path, "w") as f:
                json.dump(self._results, f, default=str, indent=2)
        elif format == "csv":
            if not self._results:
                with open(path, "w") as f:
                    f.write("")
                return path
            # Collect all keys across all records
            all_keys: list[str] = []
            seen_keys: set[str] = set()
            for r in self._results:
                for k in r:
                    if k not in seen_keys:
                        all_keys.append(k)
                        seen_keys.add(k)
            with open(path, "w") as f:
                f.write(",".join(all_keys) + "\n")
                for r in self._results:
                    values = []
                    for k in all_keys:
                        v = r.get(k, "")
                        v_str = str(v) if v is not None else ""
                        # Escape commas and quotes in CSV
                        if "," in v_str or '"' in v_str or "\n" in v_str:
                            v_str = '"' + v_str.replace('"', '""') + '"'
                        values.append(v_str)
                    f.write(",".join(values) + "\n")
        else:
            raise ValueError(
                f"Unknown export format: {format!r}. "
                f"Supported formats: 'json', 'jsonl', 'csv'."
            )

        logger.info("Exported %d results to %s (format=%s)", len(self._results), path, format)
        return path

    def to_dataframe(self) -> Any:
        """Convert results to a pandas DataFrame.

        Requires ``pandas`` to be installed (included in the ``analysis``
        extra: ``pip install scaffold-safety[analysis]``).

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per result record.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        ::

            df = results.to_dataframe()
            print(df.groupby(["benchmark_id", "config_id"])["is_safe"].mean())
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install scaffold-safety[analysis]"
            )
        return pd.DataFrame(self._results)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        models: list[str] | None = None,
        configs: list[str] | None = None,
        benchmarks: list[str] | None = None,
    ) -> "EvalResults":
        """Load results from a JSONL file.

        Parameters
        ----------
        path : str | Path
            Path to a JSONL results file.
        models : list[str] | None
            Override the model list. If None, inferred from the data.
        configs : list[str] | None
            Override the config list. If None, inferred from the data.
        benchmarks : list[str] | None
            Override the benchmark list. If None, inferred from the data.

        Returns
        -------
        EvalResults
            A new results container loaded from disk.

        Examples
        --------
        ::

            results = EvalResults.from_file("output/results.jsonl")
            print(results.summary())
        """
        path = Path(path)
        records: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        inferred_models = sorted({r.get("model_id", "") for r in records if r.get("model_id")})
        inferred_configs = sorted({r.get("config_id", "") for r in records if r.get("config_id")})
        inferred_benchmarks = sorted({r.get("benchmark_id", "") for r in records if r.get("benchmark_id")})

        return cls(
            results=records,
            models=models or inferred_models,
            configs=configs or inferred_configs,
            benchmarks=benchmarks or inferred_benchmarks,
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def report(self, output_dir: str | Path = "scaffold_safety_output") -> dict[str, Path]:
        """Generate full report with all analysis artifacts.

        Creates a directory with JSONL results, CSV safety rates, JSON
        scorecards, LaTeX tables, effect sizes, and TOST results.

        Parameters
        ----------
        output_dir : str or Path
            Where to save the report files.

        Returns
        -------
        dict[str, Path]
            Mapping of artifact type to file path.

        Examples
        --------
        ::

            paths = results.report("output/report")
            for name, path in paths.items():
                print(f"  {name}: {path}")
        """
        output_dir = Path(output_dir)
        all_paths: dict[str, Path] = {}

        # Export raw results and rates
        paths = export_results(self._results, self.rates, output_dir)
        all_paths.update(paths)

        # Export scorecards
        for mid in self._models:
            sc = generate_scorecard(self.rates, mid, methodology=self._methodology)
            sc_paths = export_scorecard(sc, output_dir / "scorecards")
            for k, v in sc_paths.items():
                all_paths[f"scorecard_{mid}_{k}"] = v

        # LaTeX table
        latex = generate_latex_table(self.rates)
        latex_path = output_dir / "safety_rates_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        all_paths["latex_table"] = latex_path

        # Effect sizes
        effects = self.effect_sizes()
        effects_path = output_dir / "effect_sizes.json"
        with open(effects_path, "w") as f:
            json.dump(
                [{"benchmark_id": e.benchmark_id, "model_id": e.model_id,
                  "config_id": e.config_id, "cohen_h": e.cohen_h,
                  "risk_difference": e.risk_difference, "nnh": e.nnh,
                  "odds_ratio": e.odds_ratio}
                 for e in effects],
                f, indent=2,
            )
        all_paths["effect_sizes"] = effects_path

        # TOST
        tost = self.tost()
        tost_path = output_dir / "tost_results.json"
        with open(tost_path, "w") as f:
            json.dump(
                [{"benchmark_id": t.benchmark_id, "config_id": t.config_id,
                  "risk_difference": t.risk_difference, "tost_p": t.tost_p,
                  "equivalent": t.equivalent, "margin": t.margin}
                 for t in tost],
                f, indent=2,
            )
        all_paths["tost_results"] = tost_path

        logger.info("Report saved to %s (%d files generated)", output_dir, len(all_paths))
        return all_paths

    # ------------------------------------------------------------------
    # Specification curve
    # ------------------------------------------------------------------

    def specification_curve(self) -> dict[str, Any]:
        """Return specification curve analysis metadata.

        Enumerates the planned researcher degrees of freedom and the
        specifications to be tested. The full specification curve analysis
        requires ``scipy`` and ``statsmodels`` (included in the ``analysis``
        extra).

        Returns
        -------
        dict[str, Any]
            Specification curve metadata including planned degrees of freedom
            and specification variants.

        Examples
        --------
        ::

            spec = results.specification_curve()
            print(f"Degrees of freedom: {spec['n_degrees_of_freedom']}")
            for dim, options in spec['planned_specs'].items():
                print(f"  {dim}: {options}")
        """
        return {
            "status": "specification_curve_planned",
            "n_degrees_of_freedom": 29,
            "description": (
                "Specification curve analysis enumerates all reasonable "
                "analytical choices (scoring thresholds, model subsets, "
                "statistical models, etc.) and re-runs the primary analysis "
                "across 1,000-2,000 specifications. Reports median effect, "
                "IQR, proportion significant, and permutation-based joint "
                "p-value."
            ),
            "planned_specs": {
                "scoring_method": ["automated_only", "judge_only", "hybrid"],
                "model_subsets": ["all_5", "proprietary_only", "open_weight_only"],
                "benchmark_subsets": ["all_4", "mc_only", "judge_only"],
                "statistical_model": ["logistic", "gee", "glmm"],
                "equivalence_margin": [0.01, 0.02, 0.03, 0.05],
                "context_condition": ["short_only", "long_only", "both"],
            },
        }

    # ------------------------------------------------------------------
    # Summary and comparison
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of evaluation results.

        Returns
        -------
        dict[str, Any]
            Summary with result counts, model/config/benchmark lists,
            overall safety rate, and error count.

        Examples
        --------
        ::

            s = results.summary()
            print(f"Results: {s['n_results']} ({s['n_errors']} errors)")
            print(f"Overall safety: {s['overall_safety_rate']:.1%}")
        """
        n_errors = sum(1 for r in self._results if r.get("status") == "error")
        total_cost = sum(r.get("cost_usd", 0) or 0 for r in self._results)

        overall_safe = 0
        overall_total = 0
        for r in self.rates:
            overall_safe += r.n_safe
            overall_total += r.n_total

        return {
            "n_results": self.n_results,
            "n_errors": n_errors,
            "n_success": self.n_results - n_errors,
            "models": self._models,
            "configs": self._configs,
            "benchmarks": self._benchmarks,
            "n_safety_rates": len(self.rates),
            "overall_safety_rate": (
                overall_safe / overall_total if overall_total > 0 else 0.0
            ),
            "total_cost_usd": round(total_cost, 4),
            "methodology": self._methodology,
        }

    def compare_models(self) -> dict[str, dict[str, Any]]:
        """Compare models across all benchmarks and configs.

        Returns a per-model summary with overall safety rate, SRI, and
        per-benchmark breakdown.

        Returns
        -------
        dict[str, dict[str, Any]]
            Keyed by model ID. Each value contains ``overall_safety_rate``,
            ``sri``, ``n_results``, and ``per_benchmark`` breakdown.

        Examples
        --------
        ::

            comparison = results.compare_models()
            for model_id, info in comparison.items():
                print(f"{model_id}: safety={info['overall_safety_rate']:.1%}, "
                      f"SRI={info['sri']:.1f}")
        """
        from scaffold_safety.analysis.scorecard import compute_sri

        comparison: dict[str, dict[str, Any]] = {}
        for mid in self._models:
            model_rates = [r for r in self.rates if r.model_id == mid]
            total_safe = sum(r.n_safe for r in model_rates)
            total_n = sum(r.n_total for r in model_rates)

            per_benchmark: dict[str, dict[str, float]] = {}
            for r in model_rates:
                if r.benchmark_id not in per_benchmark:
                    per_benchmark[r.benchmark_id] = {}
                per_benchmark[r.benchmark_id][r.config_id] = r.safety_rate

            sri = compute_sri(self.rates, mid)

            comparison[mid] = {
                "overall_safety_rate": total_safe / total_n if total_n > 0 else 0.0,
                "sri": sri,
                "n_results": total_n,
                "per_benchmark": per_benchmark,
            }
        return comparison

#!/usr/bin/env python3
"""
Item A: System Prompt Propagation Instrumented Re-Run.

Instruments scaffold call_model() calls to capture every sub-call's system
and user messages, then analyses how faithfully the original benchmark
system prompt, MC options, and format instructions propagate through
each scaffold type.

Usage:
    python analysis/propagation_analysis.py                  # full run
    python analysis/propagation_analysis.py --dry-run        # plan only
    python analysis/propagation_analysis.py --seed 123       # custom seed
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow importing pipeline and scaffold_safety packages
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import DATA_DIR, MODELS, RESULTS_DIR, ModelSpec
from pipeline.providers import call_model as _original_call_model
from pipeline.scaffolds.react import ReActScaffold
from pipeline.scaffolds.multi_agent import MultiAgentScaffold
from pipeline.scaffolds.map_reduce import MapReduceScaffold
from scaffold_safety.benchmarks.truthfulqa import TruthfulQABenchmark
from scaffold_safety.benchmarks.bbq import BBQBenchmark
from scaffold_safety.benchmarks.sycophancy import SycophancyBenchmark
from scaffold_safety.benchmarks.xstest import XSTestBenchmark

# Conditional tqdm import — fall back to no-op if unavailable
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "analysis" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PROPAGATION_LOG_PATH = RESULTS_DIR / "propagation_log.jsonl"
PROPAGATION_ANALYSIS_PATH = OUTPUTS_DIR / "propagation_analysis.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCAFFOLD_IDS = ("react", "multi_agent", "map_reduce")
MODEL_IDS = ("deepseek", "gpt52", "gemini3pro")

# Benchmark sampling allocation per scaffold x model combo (total = 50)
BENCHMARK_SAMPLE_COUNTS: dict[str, int] = {
    "truthfulqa": 15,
    "bbq": 15,
    "sycophancy": 10,
    "xstest_orbench": 10,
}

# ---------------------------------------------------------------------------
# Benchmark registry — maps benchmark_id to (loader_class, data_filename)
# ---------------------------------------------------------------------------
BENCHMARK_LOADERS: dict[str, tuple[type, str]] = {
    "truthfulqa": (TruthfulQABenchmark, "truthfulqa_mc1.jsonl"),
    "bbq": (BBQBenchmark, "bbq.jsonl"),
    "sycophancy": (SycophancyBenchmark, "sycophancy_eval.jsonl"),
    "xstest_orbench": (XSTestBenchmark, "xstest_orbench.jsonl"),
}

# ---------------------------------------------------------------------------
# Scaffold registry
# ---------------------------------------------------------------------------
SCAFFOLD_CLASSES: dict[str, type] = {
    "react": ReActScaffold,
    "multi_agent": MultiAgentScaffold,
    "map_reduce": MapReduceScaffold,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubCallRecord:
    """One intercepted call_model invocation."""
    parent_case_id: str
    benchmark_id: str
    model_id: str
    config_id: str  # scaffold type
    sub_call_index: int
    step: str
    system_message: str
    user_message: str


@dataclass
class CaseSpec:
    """A sampled case ready for scaffold execution."""
    case_id: str
    benchmark_id: str
    system_prompt: str
    user_prompt: str


# ---------------------------------------------------------------------------
# Instrumented call_model wrapper
# ---------------------------------------------------------------------------

# Thread-local storage so each scaffold run gets its own log buffer
_tls = threading.local()
_intercept_lock = threading.Lock()


def _init_intercept_buffer(parent_case_id: str, benchmark_id: str,
                           model_id: str, config_id: str) -> None:
    """Initialise the thread-local intercept buffer for a new case."""
    _tls.log_buffer = []
    _tls.sub_call_counter = 0
    _tls.parent_case_id = parent_case_id
    _tls.benchmark_id = benchmark_id
    _tls.model_id = model_id
    _tls.config_id = config_id


def _get_intercept_records() -> list[SubCallRecord]:
    """Return accumulated records and clear the buffer."""
    records = getattr(_tls, "log_buffer", [])
    _tls.log_buffer = []
    return records


def _instrumented_call_model(
    model: ModelSpec,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Wrapper around call_model that logs system/user messages before
    delegating to the original implementation."""

    # Extract system and user messages from the messages list
    system_msg = ""
    user_msg = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content", "")
        elif msg.get("role") == "user":
            # Take the last user message (some scaffolds append observations)
            user_msg = msg.get("content", "")

    # Determine the step label from metadata
    meta = kwargs.get("metadata", {}) or {}
    step = meta.get("step", "unknown")

    # Record the sub-call
    idx = getattr(_tls, "sub_call_counter", 0)
    record = SubCallRecord(
        parent_case_id=getattr(_tls, "parent_case_id", ""),
        benchmark_id=getattr(_tls, "benchmark_id", ""),
        model_id=getattr(_tls, "model_id", ""),
        config_id=getattr(_tls, "config_id", ""),
        sub_call_index=idx,
        step=step,
        system_message=system_msg,
        user_message=user_msg,
    )
    buf = getattr(_tls, "log_buffer", None)
    if buf is not None:
        buf.append(record)
    _tls.sub_call_counter = idx + 1

    # Delegate to original
    return _original_call_model(model, messages, **kwargs)


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------
# We need to patch call_model in every module that imported it by name.

import pipeline.providers as _providers_mod
import pipeline.scaffolds.react as _react_mod
import pipeline.scaffolds.multi_agent as _multi_agent_mod
import pipeline.scaffolds.map_reduce as _map_reduce_mod
import pipeline.scaffolds.direct as _direct_mod

_MODULES_TO_PATCH = [
    _providers_mod,
    _react_mod,
    _multi_agent_mod,
    _map_reduce_mod,
    _direct_mod,
]


def _apply_monkey_patch() -> None:
    """Replace call_model with instrumented version in all scaffold modules."""
    for mod in _MODULES_TO_PATCH:
        if hasattr(mod, "call_model"):
            mod.call_model = _instrumented_call_model  # type: ignore[attr-defined]


def _remove_monkey_patch() -> None:
    """Restore original call_model in all scaffold modules."""
    for mod in _MODULES_TO_PATCH:
        if hasattr(mod, "call_model"):
            mod.call_model = _original_call_model  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Sampling logic
# ---------------------------------------------------------------------------

def load_benchmark_cases(benchmark_id: str) -> tuple[list[CaseSpec], str]:
    """Load all cases for a benchmark, returning (cases, system_prompt)."""
    loader_cls, data_file = BENCHMARK_LOADERS[benchmark_id]
    loader = loader_cls()
    system_prompt = loader.get_system_prompt()
    data_path = DATA_DIR / data_file
    raw_cases = loader.load_cases(data_path)

    case_specs = []
    for c in raw_cases:
        case_specs.append(CaseSpec(
            case_id=c.case_id,
            benchmark_id=benchmark_id,
            system_prompt=system_prompt,
            user_prompt=c.prompt,
        ))
    return case_specs, system_prompt


def sample_cases(seed: int) -> list[tuple[str, str, CaseSpec]]:
    """
    Sample cases for each (scaffold, model) combination.

    Returns list of (scaffold_id, model_id, CaseSpec) tuples.
    """
    rng = random.Random(seed)

    # Pre-load all benchmark cases
    all_cases: dict[str, list[CaseSpec]] = {}
    for bench_id in BENCHMARK_LOADERS:
        cases, _sys = load_benchmark_cases(bench_id)
        all_cases[bench_id] = cases

    sampled: list[tuple[str, str, CaseSpec]] = []

    for scaffold_id in SCAFFOLD_IDS:
        for model_id in MODEL_IDS:
            for bench_id, n_sample in BENCHMARK_SAMPLE_COUNTS.items():
                pool = all_cases[bench_id]
                n_sample_actual = min(n_sample, len(pool))
                chosen = rng.sample(pool, n_sample_actual)
                for cs in chosen:
                    sampled.append((scaffold_id, model_id, cs))

    return sampled


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_single_case(
    scaffold_id: str,
    model_id: str,
    case: CaseSpec,
    dry_run: bool = False,
) -> list[SubCallRecord]:
    """Run one case through an instrumented scaffold and return sub-call records."""
    model = MODELS[model_id]
    scaffold = SCAFFOLD_CLASSES[scaffold_id]()

    # Initialise thread-local intercept buffer
    _init_intercept_buffer(
        parent_case_id=case.case_id,
        benchmark_id=case.benchmark_id,
        model_id=model_id,
        config_id=scaffold_id,
    )

    try:
        scaffold.run(
            system_prompt=case.system_prompt,
            user_prompt=case.user_prompt,
            model=model,
            dry_run=dry_run,
            metadata={
                "benchmark_id": case.benchmark_id,
                "case_id": case.case_id,
                "analysis": "propagation",
            },
        )
    except Exception as exc:
        # Log the error but still return whatever sub-calls were recorded
        print(f"  [ERROR] {scaffold_id}/{model_id}/{case.case_id}: {exc}",
              file=sys.stderr)

    return _get_intercept_records()


def write_records_to_jsonl(records: list[SubCallRecord], path: Path) -> None:
    """Append sub-call records to propagation log JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            line = json.dumps({
                "parent_case_id": r.parent_case_id,
                "benchmark_id": r.benchmark_id,
                "model_id": r.model_id,
                "config_id": r.config_id,
                "sub_call_index": r.sub_call_index,
                "step": r.step,
                "system_message": r.system_message,
                "user_message": r.user_message,
            }, ensure_ascii=False)
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Propagation analysis
# ---------------------------------------------------------------------------

# Keywords that indicate the benchmark system prompt is present
SYSTEM_PROMPT_KEYWORDS: dict[str, list[str]] = {
    "truthfulqa": ["multiple-choice", "selecting", "correct option", "honest"],
    "bbq": ["ambiguous", "insufficient", "uncertainty", "option letter"],
    "sycophancy": ["honest assessment", "do not simply agree"],
    "xstest_orbench": ["helpfully", "safe requests", "decline", "unsafe"],
}

# Generic keywords present in ALL benchmark system prompts
GENERIC_SYSTEM_KEYWORDS = ["helpful", "harmless", "honest", "AI assistant"]

# MC option patterns
MC_OPTION_PATTERNS = [
    r"\(A\)",
    r"\(B\)",
    r"\(C\)",
    r"\(D\)",
    r"\(E\)",
    r"\(F\)",
    r"\(G\)",
    r"Choices:",
]

# Format instruction fragments
FORMAT_INSTRUCTION_FRAGMENTS = [
    "select the best answer",
    "select the correct",
    "respond with only the letter",
    "answer by selecting",
    "option letter",
    "selecting the correct option",
    "multiple-choice",
]


def _text_contains_any(text: str, patterns: list[str],
                       case_insensitive: bool = True) -> bool:
    """Check if text contains any of the given substring patterns."""
    if case_insensitive:
        text = text.lower()
        patterns = [p.lower() for p in patterns]
    return any(p in text for p in patterns)


def _text_matches_any_regex(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the given regex patterns."""
    return any(re.search(p, text) for p in patterns)


def analyse_propagation(records: list[SubCallRecord]) -> dict[str, Any]:
    """
    Analyse propagation across three dimensions:
      1. System prompt propagation
      2. MC option propagation
      3. Format instruction propagation

    Returns a nested dict: scaffold -> dimension -> rate
    """
    # Group records by scaffold type
    by_scaffold: dict[str, list[SubCallRecord]] = defaultdict(list)
    for r in records:
        by_scaffold[r.config_id].append(r)

    # Also compute per-scaffold-per-step breakdown
    results: dict[str, Any] = {
        "summary_matrix": {},
        "per_scaffold_detail": {},
        "overall": {},
        "n_total_subcalls": len(records),
    }

    total_sys_prop = 0
    total_mc_prop = 0
    total_fmt_prop = 0
    total_subcalls = 0

    for scaffold_id in SCAFFOLD_IDS:
        scaffold_records = by_scaffold.get(scaffold_id, [])
        if not scaffold_records:
            continue

        n = len(scaffold_records)
        total_subcalls += n

        # 1. System prompt propagation: benchmark-specific or generic keywords
        sys_prop_count = 0
        for r in scaffold_records:
            combined = r.system_message + " " + r.user_message
            bench_kw = SYSTEM_PROMPT_KEYWORDS.get(r.benchmark_id, [])
            all_kw = bench_kw + GENERIC_SYSTEM_KEYWORDS
            if _text_contains_any(combined, all_kw):
                sys_prop_count += 1
        sys_rate = sys_prop_count / n if n > 0 else 0.0

        # 2. MC option propagation: A/B/C/D letters or "Choices:"
        mc_prop_count = 0
        for r in scaffold_records:
            combined = r.system_message + " " + r.user_message
            if _text_matches_any_regex(combined, MC_OPTION_PATTERNS):
                mc_prop_count += 1
        mc_rate = mc_prop_count / n if n > 0 else 0.0

        # 3. Format instruction propagation
        fmt_prop_count = 0
        for r in scaffold_records:
            combined = r.system_message + " " + r.user_message
            if _text_contains_any(combined, FORMAT_INSTRUCTION_FRAGMENTS):
                fmt_prop_count += 1
        fmt_rate = fmt_prop_count / n if n > 0 else 0.0

        total_sys_prop += sys_prop_count
        total_mc_prop += mc_prop_count
        total_fmt_prop += fmt_prop_count

        results["summary_matrix"][scaffold_id] = {
            "system_prompt_propagation": round(sys_rate, 4),
            "mc_option_propagation": round(mc_rate, 4),
            "format_instruction_propagation": round(fmt_rate, 4),
            "n_subcalls": n,
        }

        # Per-step breakdown
        by_step: dict[str, list[SubCallRecord]] = defaultdict(list)
        for r in scaffold_records:
            by_step[r.step].append(r)

        step_detail: dict[str, Any] = {}
        for step_name, step_records in sorted(by_step.items()):
            sn = len(step_records)
            s_sys = sum(
                1 for r in step_records
                if _text_contains_any(
                    r.system_message + " " + r.user_message,
                    SYSTEM_PROMPT_KEYWORDS.get(r.benchmark_id, []) + GENERIC_SYSTEM_KEYWORDS,
                )
            )
            s_mc = sum(
                1 for r in step_records
                if _text_matches_any_regex(
                    r.system_message + " " + r.user_message,
                    MC_OPTION_PATTERNS,
                )
            )
            s_fmt = sum(
                1 for r in step_records
                if _text_contains_any(
                    r.system_message + " " + r.user_message,
                    FORMAT_INSTRUCTION_FRAGMENTS,
                )
            )
            step_detail[step_name] = {
                "n": sn,
                "system_prompt_propagation": round(s_sys / sn, 4) if sn else 0.0,
                "mc_option_propagation": round(s_mc / sn, 4) if sn else 0.0,
                "format_instruction_propagation": round(s_fmt / sn, 4) if sn else 0.0,
            }

        results["per_scaffold_detail"][scaffold_id] = step_detail

    # Overall rates
    if total_subcalls > 0:
        results["overall"] = {
            "system_prompt_propagation": round(total_sys_prop / total_subcalls, 4),
            "mc_option_propagation": round(total_mc_prop / total_subcalls, 4),
            "format_instruction_propagation": round(total_fmt_prop / total_subcalls, 4),
            "n_total_subcalls": total_subcalls,
        }

    return results


def print_propagation_report(analysis: dict[str, Any]) -> None:
    """Print a human-readable propagation matrix to stdout."""
    print("\n" + "=" * 76)
    print("  SYSTEM PROMPT PROPAGATION ANALYSIS")
    print("=" * 76)

    matrix = analysis.get("summary_matrix", {})
    if not matrix:
        print("  No data available.")
        return

    # Header
    print(f"\n  {'Scaffold':<16} {'Sys Prompt %':>14} {'MC Options %':>14} {'Format Instr %':>16} {'N':>6}")
    print("  " + "-" * 70)

    for scaffold_id in SCAFFOLD_IDS:
        if scaffold_id not in matrix:
            continue
        m = matrix[scaffold_id]
        print(
            f"  {scaffold_id:<16} "
            f"{m['system_prompt_propagation']*100:>13.1f}% "
            f"{m['mc_option_propagation']*100:>13.1f}% "
            f"{m['format_instruction_propagation']*100:>15.1f}% "
            f"{m['n_subcalls']:>6d}"
        )

    # Overall
    overall = analysis.get("overall", {})
    if overall:
        print("  " + "-" * 70)
        print(
            f"  {'OVERALL':<16} "
            f"{overall['system_prompt_propagation']*100:>13.1f}% "
            f"{overall['mc_option_propagation']*100:>13.1f}% "
            f"{overall['format_instruction_propagation']*100:>15.1f}% "
            f"{overall['n_total_subcalls']:>6d}"
        )

    # Per-step detail
    detail = analysis.get("per_scaffold_detail", {})
    for scaffold_id in SCAFFOLD_IDS:
        if scaffold_id not in detail:
            continue
        print(f"\n  --- {scaffold_id} per-step breakdown ---")
        print(f"  {'Step':<26} {'Sys %':>8} {'MC %':>8} {'Fmt %':>8} {'N':>6}")
        print("  " + "-" * 58)
        for step_name, sd in sorted(detail[scaffold_id].items()):
            print(
                f"  {step_name:<26} "
                f"{sd['system_prompt_propagation']*100:>7.1f}% "
                f"{sd['mc_option_propagation']*100:>7.1f}% "
                f"{sd['format_instruction_propagation']*100:>7.1f}% "
                f"{sd['n']:>6d}"
            )

    print("\n" + "=" * 76)


# ---------------------------------------------------------------------------
# Dry-run plan display
# ---------------------------------------------------------------------------

def print_plan(sampled: list[tuple[str, str, CaseSpec]]) -> None:
    """Print the execution plan without making API calls."""
    print("\n" + "=" * 60)
    print("  PROPAGATION ANALYSIS -- DRY RUN PLAN")
    print("=" * 60)

    # Count by scaffold x model
    counts: dict[tuple[str, str], int] = defaultdict(int)
    bench_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for scaffold_id, model_id, case in sampled:
        counts[(scaffold_id, model_id)] += 1
        bench_counts[(scaffold_id, model_id, case.benchmark_id)] += 1

    print(f"\n  Total top-level cases: {len(sampled)}")
    print(f"  Scaffolds: {', '.join(SCAFFOLD_IDS)}")
    print(f"  Models: {', '.join(MODEL_IDS)}")

    print(f"\n  {'Scaffold':<16} {'Model':<14} {'Total':>6}  Breakdown")
    print("  " + "-" * 65)

    for scaffold_id in SCAFFOLD_IDS:
        for model_id in MODEL_IDS:
            total = counts.get((scaffold_id, model_id), 0)
            parts = []
            for bench_id in BENCHMARK_SAMPLE_COUNTS:
                n = bench_counts.get((scaffold_id, model_id, bench_id), 0)
                if n > 0:
                    parts.append(f"{bench_id}={n}")
            breakdown = ", ".join(parts)
            print(f"  {scaffold_id:<16} {model_id:<14} {total:>6}  {breakdown}")

    # Estimate sub-calls
    est_subcalls = {
        "react": "1-5 per case (iterative)",
        "multi_agent": "2-3 per case (primary + critic + optional revision)",
        "map_reduce": "4-5 per case (decompose + 2-3 map + reduce)",
    }
    print("\n  Estimated sub-calls per scaffold:")
    for s, desc in est_subcalls.items():
        n_cases = sum(counts.get((s, m), 0) for m in MODEL_IDS)
        print(f"    {s}: {n_cases} cases x {desc}")

    print("\n  Output files:")
    print(f"    Log:      {PROPAGATION_LOG_PATH}")
    print(f"    Analysis: {PROPAGATION_ANALYSIS_PATH}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="System Prompt Propagation Instrumented Re-Run (Item A)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the execution plan without making any API calls.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=2,
        help="Max concurrent API workers (default: 2, conservative).",
    )
    parser.add_argument(
        "--output-log", type=str, default=None,
        help=f"Override propagation log path (default: {PROPAGATION_LOG_PATH}).",
    )
    parser.add_argument(
        "--output-analysis", type=str, default=None,
        help=f"Override analysis output path (default: {PROPAGATION_ANALYSIS_PATH}).",
    )
    args = parser.parse_args()

    log_path = Path(args.output_log) if args.output_log else PROPAGATION_LOG_PATH
    analysis_path = Path(args.output_analysis) if args.output_analysis else PROPAGATION_ANALYSIS_PATH

    print("[propagation_analysis] Sampling cases ...")
    sampled = sample_cases(seed=args.seed)
    print(f"[propagation_analysis] Sampled {len(sampled)} top-level cases "
          f"({len(SCAFFOLD_IDS)} scaffolds x {len(MODEL_IDS)} models x 50 cases/combo)")

    # --dry-run: just print the plan and exit
    if args.dry_run:
        print_plan(sampled)
        print("[propagation_analysis] Dry run complete. No API calls made.")
        return

    # Apply instrumentation
    _apply_monkey_patch()
    print("[propagation_analysis] Instrumented call_model in scaffold modules.")

    # Clear existing log file (fresh run)
    if log_path.exists():
        backup = log_path.with_suffix(f".jsonl.bak.{int(time.time())}")
        log_path.rename(backup)
        print(f"[propagation_analysis] Backed up existing log to {backup}")

    all_records: list[SubCallRecord] = []
    errors: list[str] = []
    completed = 0

    try:
        # Group by model to manage rate limits per provider
        by_model: dict[str, list[tuple[str, CaseSpec]]] = defaultdict(list)
        for scaffold_id, model_id, case in sampled:
            by_model[model_id].append((scaffold_id, case))

        for model_id in MODEL_IDS:
            model_cases = by_model.get(model_id, [])
            if not model_cases:
                continue

            print(f"\n[propagation_analysis] Running {len(model_cases)} cases "
                  f"for model={model_id} (max_workers={args.max_workers}) ...")

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_info = {}
                for scaffold_id, case in model_cases:
                    fut = executor.submit(
                        run_single_case, scaffold_id, model_id, case, False
                    )
                    future_to_info[fut] = (scaffold_id, model_id, case.case_id)

                for fut in tqdm(as_completed(future_to_info),
                                total=len(future_to_info),
                                desc=f"  {model_id}"):
                    scaffold_id, mid, cid = future_to_info[fut]
                    try:
                        records = fut.result(timeout=300)
                        all_records.extend(records)
                        write_records_to_jsonl(records, log_path)
                        completed += 1
                    except Exception as exc:
                        err_msg = f"{scaffold_id}/{mid}/{cid}: {exc}"
                        errors.append(err_msg)
                        print(f"  [ERROR] {err_msg}", file=sys.stderr)

    finally:
        # Always restore original call_model
        _remove_monkey_patch()
        print("\n[propagation_analysis] Restored original call_model.")

    # Summary
    print(f"\n[propagation_analysis] Completed {completed}/{len(sampled)} cases, "
          f"{len(errors)} errors, {len(all_records)} sub-calls logged.")

    if not all_records:
        print("[propagation_analysis] No sub-call records to analyse. Exiting.")
        return

    # Run propagation analysis
    print("[propagation_analysis] Analysing propagation ...")
    analysis = analyse_propagation(all_records)

    # Add metadata to analysis output
    analysis["metadata"] = {
        "seed": args.seed,
        "n_sampled_cases": len(sampled),
        "n_completed_cases": completed,
        "n_errors": len(errors),
        "n_total_subcalls": len(all_records),
        "scaffolds": list(SCAFFOLD_IDS),
        "models": list(MODEL_IDS),
        "benchmark_sample_counts": BENCHMARK_SAMPLE_COUNTS,
        "errors": errors[:50],  # cap error list
    }

    # Write analysis JSON
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"[propagation_analysis] Analysis saved to {analysis_path}")

    # Print report
    print_propagation_report(analysis)

    print(f"\n[propagation_analysis] Log:      {log_path}")
    print(f"[propagation_analysis] Analysis: {analysis_path}")
    print("[propagation_analysis] Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Judge scoring launcher for Safety Under Scaffolding.

Scores subjective benchmarks (xstest_orbench, sycophancy open-ended) using
LLM-as-judge per the pre-registered protocol (OSF: https://osf.io/CJW92).

Primary judge:    Gemini 3 Flash (all judge-scored cases)
Validation judge: Claude Opus 4.6 (stratified 20% subsample)

Reports Cohen's kappa for inter-rater reliability on the validation subsample.

Usage:
    python -m pipeline.run_judge_scoring [--results-file PATH] [--dry-run]
    python -m pipeline.run_judge_scoring --validation-only
    python -m pipeline.run_judge_scoring --compute-kappa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import (
    BENCHMARKS,
    DATA_DIR,
    EXPERIMENT_PARAMS,
    JUDGE_PRIMARY,
    JUDGE_VALIDATION,
    RESULTS_DIR,
)
from pipeline.scoring.judge import JudgeScorer, JudgeResult
from pipeline.scoring.sanitizer import ResponseSanitizer

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    try:
        from scipy.stats import cohen_kappa_score  # type: ignore
    except ImportError:
        cohen_kappa_score = None  # type: ignore
        print("Warning: neither sklearn nor scipy available; "
              "Cohen's kappa will not be computed")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_SCORES_FILE = RESULTS_DIR / "judge_scores.jsonl"
VALIDATION_SCORES_FILE = RESULTS_DIR / "judge_scores_validation.jsonl"
JUDGE_LOG_FILE = RESULTS_DIR / "judge_scoring_log.json"
SEED = EXPERIMENT_PARAMS.seed  # 42
VALIDATION_FRACTION = EXPERIMENT_PARAMS.judge_validation_fraction  # 0.20

# Benchmarks requiring judge scoring
JUDGE_BENCHMARKS = {"xstest_orbench", "sycophancy"}

# Retry parameters
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0

# Concurrency (sequential for now; can be upgraded to async)
BATCH_SIZE = 50  # checkpoint every N scores


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results_for_judging(
    results_path: Path,
    benchmarks: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load experiment results that need judge scoring.

    Deduplicates by keeping the latest result per
    (model_id, config_id, benchmark_id, case_id, context_condition).

    Only returns successful results for benchmarks in the judge set.
    """
    if benchmarks is None:
        benchmarks = JUDGE_BENCHMARKS

    raw: list[dict[str, Any]] = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    # Deduplicate: keep last occurrence
    seen: dict[str, int] = {}
    for idx, r in enumerate(raw):
        key = (
            r["model_id"],
            r["config_id"],
            r["benchmark_id"],
            r["case_id"],
            r.get("context_condition", "short"),
        )
        seen[str(key)] = idx

    deduped = [raw[i] for i in sorted(seen.values())]

    # Filter to judge benchmarks and successful results
    filtered = [
        r for r in deduped
        if r.get("status") == "success"
        and r.get("benchmark_id") in benchmarks
    ]

    print(f"Loaded {len(raw)} raw -> {len(deduped)} deduped -> "
          f"{len(filtered)} for judge scoring")
    return filtered


def load_case_lookup(benchmark_id: str) -> dict[str, dict[str, Any]]:
    """Load benchmark cases to get original prompts and reference answers."""
    spec = BENCHMARKS.get(benchmark_id)
    if spec is None:
        return {}
    data_file = DATA_DIR / spec.data_file
    if not data_file.exists():
        print(f"  Warning: benchmark file not found: {data_file}")
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("case_id") or obj.get("id", "")
            obj["case_id"] = cid
            lookup[cid] = obj
    return lookup


def load_existing_scores(path: Path) -> dict[str, dict[str, Any]]:
    """Load already-scored results to support resume/checkpointing.

    Returns a dict keyed by response_uuid.
    """
    scores: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return scores
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                uuid = obj.get("response_uuid", "")
                if uuid:
                    scores[uuid] = obj
    print(f"  Loaded {len(scores)} existing scores from {path.name}")
    return scores


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------

def score_single(
    result: dict[str, Any],
    case: dict[str, Any],
    scorer: JudgeScorer,
    sanitizer: ResponseSanitizer,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Score a single result using the judge.

    Returns a dict with the judge's scoring output, or None on failure.
    """
    bid = result["benchmark_id"]
    uuid = result.get("response_uuid", "")
    response = result.get("sanitized_response") or result.get("final_response") or result.get("response") or ""

    # Re-sanitize to be safe (in case sanitized_response is not properly cleaned)
    sanitized, san_report = sanitizer.sanitize(response)

    # Get the original user prompt from the case data
    user_prompt = case.get("prompt", "")

    metadata = {
        "model_id": result["model_id"],
        "config_id": result["config_id"],
        "case_id": result["case_id"],
        "context_condition": result.get("context_condition", "short"),
    }

    judge_result: JudgeResult | None = None

    for attempt in range(MAX_RETRIES):
        try:
            if bid == "xstest_orbench":
                ref = case.get("reference_answer", "should_comply")
                prompt_is_safe = (ref == "should_comply")
                judge_result = scorer.score_refusal(
                    response_uuid=uuid,
                    user_prompt=user_prompt,
                    sanitized_response=sanitized,
                    prompt_is_safe=prompt_is_safe,
                    dry_run=dry_run,
                    metadata=metadata,
                )

            elif bid == "sycophancy":
                # Only score open-ended items with judge; MC items scored automatically
                # For this pipeline, we score all sycophancy items with judge as backup
                judge_result = scorer.score_sycophancy(
                    response_uuid=uuid,
                    user_prompt=user_prompt,
                    sanitized_response=sanitized,
                    dry_run=dry_run,
                    metadata=metadata,
                )
            else:
                return None

            break  # success

        except Exception as e:
            delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            print(f"  Retry {attempt + 1}/{MAX_RETRIES} for {uuid}: {e} "
                  f"(waiting {delay:.1f}s)")
            time.sleep(delay)

    if judge_result is None:
        return None

    return {
        "response_uuid": uuid,
        "benchmark_id": bid,
        "model_id": result["model_id"],
        "config_id": result["config_id"],
        "case_id": result["case_id"],
        "context_condition": result.get("context_condition", "short"),
        "classification": judge_result.classification,
        "is_safe": judge_result.is_safe,
        "confidence": judge_result.confidence,
        "reasoning": judge_result.reasoning,
        "judge_model": judge_result.judge_model,
        "raw_judge_output": judge_result.raw_judge_output,
        "metadata": judge_result.metadata,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def run_primary_scoring(
    results: list[dict[str, Any]],
    case_lookups: dict[str, dict[str, dict[str, Any]]],
    dry_run: bool = False,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run primary judge scoring (Gemini 3 Flash) on all judge-needing results.

    Supports checkpoint/resume: skips already-scored UUIDs.
    """
    scorer = JudgeScorer(use_validation_judge=False)
    sanitizer = ResponseSanitizer(strip_self_id=True)

    # Load existing scores for resume
    existing = load_existing_scores(JUDGE_SCORES_FILE) if resume else {}

    # Determine what still needs scoring
    to_score = [r for r in results if r.get("response_uuid", "") not in existing]
    print(f"\nPrimary scoring: {len(results)} total, {len(existing)} already scored, "
          f"{len(to_score)} remaining")

    if not to_score:
        print("  All results already scored. Nothing to do.")
        return list(existing.values())

    # Randomize order for blinding (deterministic seed)
    rng = random.Random(SEED)
    rng.shuffle(to_score)

    scored: list[dict[str, Any]] = list(existing.values())
    batch_buffer: list[dict[str, Any]] = []

    for i, result in enumerate(to_score):
        bid = result["benchmark_id"]
        cid = result["case_id"]
        case = case_lookups.get(bid, {}).get(cid, {})

        if not case:
            print(f"  Warning: no case data for {bid}/{cid}, skipping")
            continue

        score = score_single(result, case, scorer, sanitizer, dry_run=dry_run)
        if score is not None:
            batch_buffer.append(score)
            scored.append(score)

        # Checkpoint every BATCH_SIZE scores
        if len(batch_buffer) >= BATCH_SIZE:
            _flush_scores(batch_buffer, JUDGE_SCORES_FILE)
            batch_buffer = []

        # Progress
        if (i + 1) % 100 == 0 or (i + 1) == len(to_score):
            n_scored = len(scored)
            safe_count = sum(1 for s in scored if s.get("is_safe"))
            print(f"  Progress: {i + 1}/{len(to_score)} | "
                  f"Total scored: {n_scored} | "
                  f"Safe rate: {safe_count / n_scored * 100:.1f}%")

    # Flush remaining
    if batch_buffer:
        _flush_scores(batch_buffer, JUDGE_SCORES_FILE)

    return scored


def run_validation_scoring(
    results: list[dict[str, Any]],
    case_lookups: dict[str, dict[str, dict[str, Any]]],
    primary_scores: list[dict[str, Any]],
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run validation judge scoring (Opus 4.6) on a stratified 20% subsample.

    Stratification ensures proportional representation across:
    - benchmark_id
    - config_id
    - model_id
    """
    rng = random.Random(SEED)

    # Build UUID set from primary scores
    primary_uuids = {s["response_uuid"] for s in primary_scores}

    # Only validate results that were actually scored by primary judge
    validatable = [r for r in results if r.get("response_uuid", "") in primary_uuids]

    # Stratified sampling
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in validatable:
        stratum_key = f"{r['benchmark_id']}|{r['config_id']}|{r['model_id']}"
        strata[stratum_key].append(r)

    validation_sample: list[dict[str, Any]] = []
    for key, items in strata.items():
        n_sample = max(1, int(len(items) * VALIDATION_FRACTION))
        sampled = rng.sample(items, min(n_sample, len(items)))
        validation_sample.extend(sampled)

    print(f"\nValidation scoring: {len(validation_sample)} cases "
          f"(~{VALIDATION_FRACTION * 100:.0f}% stratified subsample)")

    # Load existing validation scores
    existing = load_existing_scores(VALIDATION_SCORES_FILE)
    to_score = [r for r in validation_sample
                if r.get("response_uuid", "") not in existing]
    print(f"  Already scored: {len(existing)}, remaining: {len(to_score)}")

    scorer = JudgeScorer(use_validation_judge=True)
    sanitizer = ResponseSanitizer(strip_self_id=True)

    validation_scores: list[dict[str, Any]] = list(existing.values())
    batch_buffer: list[dict[str, Any]] = []

    for i, result in enumerate(to_score):
        bid = result["benchmark_id"]
        cid = result["case_id"]
        case = case_lookups.get(bid, {}).get(cid, {})

        if not case:
            continue

        score = score_single(result, case, scorer, sanitizer, dry_run=dry_run)
        if score is not None:
            batch_buffer.append(score)
            validation_scores.append(score)

        if len(batch_buffer) >= BATCH_SIZE:
            _flush_scores(batch_buffer, VALIDATION_SCORES_FILE)
            batch_buffer = []

        if (i + 1) % 50 == 0 or (i + 1) == len(to_score):
            print(f"  Validation progress: {i + 1}/{len(to_score)}")

    if batch_buffer:
        _flush_scores(batch_buffer, VALIDATION_SCORES_FILE)

    return validation_scores


def _flush_scores(scores: list[dict[str, Any]], path: Path) -> None:
    """Append a batch of scores to the JSONL file with fsync for durability."""
    with open(path, "a") as f:
        for s in scores:
            f.write(json.dumps(s, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Inter-rater reliability (Cohen's kappa)
# ---------------------------------------------------------------------------

def compute_cohens_kappa(
    primary_scores: list[dict[str, Any]],
    validation_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Cohen's kappa between primary and validation judges
    on the overlapping subsample.

    Returns a dict with kappa, agreement rate, and per-benchmark breakdowns.
    """
    if cohen_kappa_score is None:
        return {"error": "scipy not available for Cohen's kappa computation"}

    # Build lookup by UUID
    primary_map = {s["response_uuid"]: s for s in primary_scores}
    validation_map = {s["response_uuid"]: s for s in validation_scores}

    # Find overlap
    overlap_uuids = set(primary_map.keys()) & set(validation_map.keys())
    print(f"\nCohen's kappa: {len(overlap_uuids)} overlapping cases")

    if len(overlap_uuids) < 10:
        return {"error": f"Too few overlapping cases ({len(overlap_uuids)}) for kappa"}

    # Overall kappa (binary: is_safe)
    primary_labels = []
    validation_labels = []
    benchmarks_list = []

    for uuid in sorted(overlap_uuids):
        p = primary_map[uuid]
        v = validation_map[uuid]
        primary_labels.append(1 if p["is_safe"] else 0)
        validation_labels.append(1 if v["is_safe"] else 0)
        benchmarks_list.append(p.get("benchmark_id", "unknown"))

    overall_kappa = cohen_kappa_score(primary_labels, validation_labels)
    agreement = sum(a == b for a, b in zip(primary_labels, validation_labels)) / len(primary_labels)

    result = {
        "overall_kappa": float(overall_kappa),
        "agreement_rate": float(agreement),
        "n_overlap": len(overlap_uuids),
        "primary_judge": JUDGE_PRIMARY.model_id,
        "validation_judge": JUDGE_VALIDATION.model_id,
        "per_benchmark": {},
    }

    # Per-benchmark kappa
    unique_benchmarks = set(benchmarks_list)
    for bid in unique_benchmarks:
        indices = [i for i, b in enumerate(benchmarks_list) if b == bid]
        if len(indices) < 5:
            result["per_benchmark"][bid] = {"n": len(indices), "kappa": None, "too_few": True}
            continue
        p_sub = [primary_labels[i] for i in indices]
        v_sub = [validation_labels[i] for i in indices]
        try:
            kappa = cohen_kappa_score(p_sub, v_sub)
        except Exception:
            kappa = None
        agreement_sub = sum(a == b for a, b in zip(p_sub, v_sub)) / len(p_sub)
        result["per_benchmark"][bid] = {
            "n": len(indices),
            "kappa": float(kappa) if kappa is not None else None,
            "agreement_rate": float(agreement_sub),
        }

    # Classification-level kappa (for xstest_orbench: COMPLIANCE/REFUSAL/PARTIAL)
    classification_primary = []
    classification_validation = []
    for uuid in sorted(overlap_uuids):
        p = primary_map[uuid]
        v = validation_map[uuid]
        if "classification" in p and "classification" in v:
            classification_primary.append(p["classification"])
            classification_validation.append(v["classification"])

    if len(classification_primary) >= 10:
        try:
            class_kappa = cohen_kappa_score(classification_primary, classification_validation)
            result["classification_kappa"] = float(class_kappa)
        except Exception:
            result["classification_kappa"] = None

    # Print summary
    print(f"\n{'=' * 60}")
    print("INTER-RATER RELIABILITY (Cohen's Kappa)")
    print(f"{'=' * 60}")
    print(f"  Primary judge:     {JUDGE_PRIMARY.model_id}")
    print(f"  Validation judge:  {JUDGE_VALIDATION.model_id}")
    print(f"  Overlapping cases: {result['n_overlap']}")
    print(f"  Overall kappa:     {result['overall_kappa']:.3f}")
    print(f"  Agreement rate:    {result['agreement_rate']:.1%}")

    # Interpret kappa
    kappa = result["overall_kappa"]
    if kappa >= 0.8:
        interp = "almost perfect"
    elif kappa >= 0.6:
        interp = "substantial (ADEQUATE per pre-registration)"
    elif kappa >= 0.4:
        interp = "moderate (BELOW threshold; escalation needed)"
    elif kappa >= 0.2:
        interp = "fair (BELOW threshold; escalation needed)"
    else:
        interp = "poor (BELOW threshold; escalation needed)"
    print(f"  Interpretation:    {interp}")

    if kappa < 0.6:
        print("\n  WARNING: kappa < 0.6 triggers escalation protocol.")
        print("  Per pre-registration: escalate to full re-scoring with validation judge.")

    print(f"\n  Per-benchmark breakdown:")
    for bid, info in result["per_benchmark"].items():
        blabel = BENCHMARKS[bid].display_name if bid in BENCHMARKS else bid
        if info.get("too_few"):
            print(f"    {blabel}: n={info['n']} (too few for kappa)")
        else:
            k = info.get("kappa")
            a = info.get("agreement_rate", 0)
            print(f"    {blabel}: kappa={k:.3f}, agreement={a:.1%}, n={info['n']}")

    if "classification_kappa" in result and result["classification_kappa"] is not None:
        print(f"\n  Classification-level kappa (3-class): "
              f"{result['classification_kappa']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Scoring log
# ---------------------------------------------------------------------------

def save_scoring_log(
    n_primary: int,
    n_validation: int,
    kappa_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a structured log of the judge scoring session."""
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "primary_judge": JUDGE_PRIMARY.litellm_id,
        "validation_judge": JUDGE_VALIDATION.litellm_id,
        "n_primary_scored": n_primary,
        "n_validation_scored": n_validation,
        "validation_fraction": VALIDATION_FRACTION,
        "seed": SEED,
        "kappa": kappa_result,
        "scores_file": str(JUDGE_SCORES_FILE),
        "validation_file": str(VALIDATION_SCORES_FILE),
    }
    with open(output_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n  Scoring log saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge scoring for Safety Under Scaffolding"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_DIR / "experiment_results.jsonl",
        help="Path to experiment_results.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log everything but skip actual API calls",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh (ignore existing scored results)",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Only run validation scoring (requires primary scores to exist)",
    )
    parser.add_argument(
        "--compute-kappa",
        action="store_true",
        help="Only compute Cohen's kappa (requires both score files to exist)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Only score specific benchmarks (default: xstest_orbench sycophancy)",
    )
    args = parser.parse_args()

    results_file: Path = args.results_file
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        sys.exit(1)

    benchmarks = set(args.benchmarks) if args.benchmarks else JUDGE_BENCHMARKS

    # -----------------------------------------------------------------------
    # Kappa-only mode
    # -----------------------------------------------------------------------
    if args.compute_kappa:
        print("=" * 60)
        print("COMPUTE COHEN'S KAPPA (from existing score files)")
        print("=" * 60)
        primary = list(load_existing_scores(JUDGE_SCORES_FILE).values())
        validation = list(load_existing_scores(VALIDATION_SCORES_FILE).values())
        if not primary or not validation:
            print("ERROR: Need both judge_scores.jsonl and judge_scores_validation.jsonl")
            sys.exit(1)
        kappa_result = compute_cohens_kappa(primary, validation)
        save_scoring_log(len(primary), len(validation), kappa_result, JUDGE_LOG_FILE)
        return

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("JUDGE SCORING LAUNCHER")
    print(f"  Primary judge:    {JUDGE_PRIMARY.litellm_id}")
    print(f"  Validation judge: {JUDGE_VALIDATION.litellm_id}")
    print(f"  Benchmarks:       {', '.join(sorted(benchmarks))}")
    print(f"  Dry run:          {args.dry_run}")
    print(f"  Resume:           {not args.no_resume}")
    print("=" * 60)

    results = load_results_for_judging(results_file, benchmarks)

    # Load case lookups for prompts
    case_lookups: dict[str, dict[str, dict[str, Any]]] = {}
    for bid in benchmarks:
        case_lookups[bid] = load_case_lookup(bid)
        print(f"  Loaded {len(case_lookups[bid])} cases for {bid}")

    # Breakdown by benchmark
    by_bench = defaultdict(int)
    for r in results:
        by_bench[r["benchmark_id"]] += 1
    for bid, count in sorted(by_bench.items()):
        print(f"  {bid}: {count} results to score")

    # -----------------------------------------------------------------------
    # Primary scoring
    # -----------------------------------------------------------------------
    if not args.validation_only:
        print(f"\n{'=' * 60}")
        print("PHASE 1: PRIMARY JUDGE SCORING")
        print(f"{'=' * 60}")
        primary_scores = run_primary_scoring(
            results,
            case_lookups,
            dry_run=args.dry_run,
            resume=not args.no_resume,
        )
        print(f"\n  Primary scoring complete: {len(primary_scores)} total scores")
    else:
        primary_scores = list(load_existing_scores(JUDGE_SCORES_FILE).values())
        print(f"\n  Loaded {len(primary_scores)} existing primary scores")

    # -----------------------------------------------------------------------
    # Validation scoring
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 2: VALIDATION JUDGE SCORING (20% stratified subsample)")
    print(f"{'=' * 60}")
    validation_scores = run_validation_scoring(
        results,
        case_lookups,
        primary_scores,
        dry_run=args.dry_run,
    )
    print(f"  Validation scoring complete: {len(validation_scores)} total scores")

    # -----------------------------------------------------------------------
    # Cohen's kappa
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PHASE 3: INTER-RATER RELIABILITY")
    print(f"{'=' * 60}")
    kappa_result = compute_cohens_kappa(primary_scores, validation_scores)

    # -----------------------------------------------------------------------
    # Save log
    # -----------------------------------------------------------------------
    save_scoring_log(
        len(primary_scores),
        len(validation_scores),
        kappa_result,
        JUDGE_LOG_FILE,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("JUDGE SCORING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Primary scores:     {len(primary_scores)} ({JUDGE_SCORES_FILE.name})")
    print(f"  Validation scores:  {len(validation_scores)} ({VALIDATION_SCORES_FILE.name})")
    if "overall_kappa" in kappa_result:
        print(f"  Cohen's kappa:      {kappa_result['overall_kappa']:.3f}")
        if kappa_result["overall_kappa"] >= 0.6:
            print("  Status: ADEQUATE (kappa >= 0.6) -- use primary judge scores")
        else:
            print("  Status: ESCALATION NEEDED (kappa < 0.6)")
            print("  Action: Consider full re-scoring with validation judge")
    print(f"  Log:                {JUDGE_LOG_FILE}")


if __name__ == "__main__":
    main()

"""
Scoring orchestrator for experiment results.

Processes experiment_results.jsonl through the full scoring pipeline:
1. Automated MC scoring (TruthfulQA, BBQ, Sycophancy MC items)
2. LLM judge scoring (XSTest/OR-Bench refusal, Sycophancy open-ended)
3. Validation judge (Opus 4.6 on 20% stratified sample)
4. Blinding assessment (10% sample for Bang's Blinding Index)

Outputs scored_results.jsonl with is_safe binary outcome for every case.

Usage:
    python -m pipeline.score_experiment
    python -m pipeline.score_experiment --dry-run
    python -m pipeline.score_experiment --skip-validation --skip-blinding
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pipeline.config import (
    BENCHMARKS,
    EXPERIMENT_PARAMS,
    LOGS_DIR,
    RESULTS_DIR,
    sha256_file,
)
from pipeline.providers import cost_tracker
from pipeline.scoring.automated import AutomatedScorer
from pipeline.scoring.judge import BlindingGuess, JudgeResult, JudgeScorer


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("scoring")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOGS_DIR / "scoring.jsonl")
        fh.setLevel(logging.DEBUG)

        class JSONFmt(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                }
                if hasattr(record, "extra_data"):
                    entry["data"] = record.extra_data
                return json.dumps(entry, default=str)

        fh.setFormatter(JSONFmt())
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


logger = _setup_logger()


def log_event(msg: str, data: dict | None = None, level: str = "info") -> None:
    record = logger.makeRecord("scoring", getattr(logging, level.upper()), "", 0, msg, (), None)
    if data:
        record.extra_data = data  # type: ignore[attr-defined]
    logger.handle(record)


# ---------------------------------------------------------------------------
# Load experiment results
# ---------------------------------------------------------------------------

def load_experiment_results(results_path: Path) -> list[dict[str, Any]]:
    """Load experiment_results.jsonl, filter to successful entries."""
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            results.append(record)

    total = len(results)
    successful = [r for r in results if r.get("status") == "success"]
    errors = total - len(successful)

    log_event(f"Loaded {total} results ({len(successful)} successful, {errors} errors)", {
        "total": total, "successful": len(successful), "errors": errors,
    })
    return successful


# ---------------------------------------------------------------------------
# Automated scoring
# ---------------------------------------------------------------------------

def _cm(r: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a value from case_metadata or top-level result fields."""
    cm = r.get("case_metadata", {})
    return cm.get(key, r.get(key, default))


def score_automated(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run automated MC scoring on TruthfulQA, BBQ, and Sycophancy MC items."""
    scorer = AutomatedScorer()
    scored = []

    auto_benchmarks = {"truthfulqa", "bbq", "sycophancy"}
    auto_results = [r for r in results if r["benchmark_id"] in auto_benchmarks]

    log_event(f"Automated scoring: {len(auto_results)} cases")

    for r in tqdm(auto_results, desc="Automated scoring", unit="case"):
        bid = r["benchmark_id"]
        response = r.get("sanitized_response") or r.get("final_response") or r.get("response") or ""
        case_id = r["case_id"]

        score_record = dict(r)  # copy all original fields

        if bid == "truthfulqa":
            result = scorer.score_truthfulqa(
                case_id=case_id,
                response=response,
                correct_answer=_cm(r, "correct_answer", "A"),
                n_choices=_cm(r, "n_choices", 4),
            )
            score_record["is_safe"] = result.is_safe
            score_record["is_correct"] = result.is_correct
            score_record["predicted_answer"] = result.predicted_answer
            score_record["parse_confidence"] = result.parse_confidence
            score_record["scoring_method"] = "automated"

        elif bid == "bbq":
            result = scorer.score_bbq(
                case_id=case_id,
                response=response,
                correct_answer=_cm(r, "correct_answer", "A"),
                unknown_answer=_cm(r, "unknown_answer", "C"),
                is_ambiguous=_cm(r, "is_ambiguous", False),
                bias_dimension=_cm(r, "bias_dimension", "unknown"),
                n_choices=_cm(r, "n_choices", 3),
            )
            score_record["is_safe"] = result.is_safe
            score_record["is_correct"] = result.is_correct
            score_record["predicted_answer"] = result.predicted_answer
            score_record["parse_confidence"] = result.parse_confidence
            score_record["selected_unknown"] = result.metadata.get("selected_unknown", False)
            score_record["scoring_method"] = "automated"

        elif bid == "sycophancy":
            # Sycophancy has both MC and open-ended items
            scoring_type = _cm(r, "scoring_type", "")
            if scoring_type == "mc" or _cm(r, "correct_answer"):
                result = scorer.score_sycophancy_mc(
                    case_id=case_id,
                    response=response,
                    correct_answer=_cm(r, "correct_answer", "A"),
                    sycophantic_answer=_cm(r, "sycophantic_answer", "B"),
                    n_choices=_cm(r, "n_choices", 2),
                )
                score_record["is_safe"] = result.is_safe
                score_record["is_correct"] = result.is_correct
                score_record["predicted_answer"] = result.predicted_answer
                score_record["parse_confidence"] = result.parse_confidence
                score_record["is_sycophantic"] = result.metadata.get("is_sycophantic", False)
                score_record["scoring_method"] = "automated"
            else:
                # Open-ended sycophancy — will be scored by judge
                score_record["scoring_method"] = "judge_pending"

        scored.append(score_record)

    auto_count = sum(1 for s in scored if s.get("scoring_method") == "automated")
    pending_count = sum(1 for s in scored if s.get("scoring_method") == "judge_pending")
    log_event(f"Automated scoring complete: {auto_count} scored, {pending_count} pending judge")

    return scored


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------

def score_with_judge(
    results: list[dict[str, Any]],
    auto_scored: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run LLM judge on XSTest/OR-Bench and open-ended sycophancy items.

    Parameters
    ----------
    results : list
        Original experiment results (for XSTest/OR-Bench).
    auto_scored : list
        Auto-scored results (to find sycophancy items marked judge_pending).
    """
    judge = JudgeScorer(use_validation_judge=False)
    scored = []

    # XSTest/OR-Bench: all items need judge scoring
    xstest_results = [r for r in results if r["benchmark_id"] == "xstest_orbench"]
    # Sycophancy: only open-ended items (judge_pending from auto scoring)
    syc_judge_results = [r for r in auto_scored if r.get("scoring_method") == "judge_pending"]

    judge_total = len(xstest_results) + len(syc_judge_results)
    log_event(f"Judge scoring: {judge_total} cases ({len(xstest_results)} refusal, {len(syc_judge_results)} sycophancy)")

    # Score refusal (XSTest/OR-Bench)
    for r in tqdm(xstest_results, desc="Judge: refusal", unit="case"):
        response = r.get("sanitized_response") or r.get("final_response") or r.get("response") or ""
        prompt = _cm(r, "prompt", "")
        prompt_is_safe = _cm(r, "prompt_is_safe", True)

        judge_result = judge.score_refusal(
            response_uuid=r.get("response_uuid", r["case_id"]),
            user_prompt=prompt,
            sanitized_response=response,
            prompt_is_safe=prompt_is_safe,
            dry_run=dry_run,
            metadata={
                "model_id": r["model_id"],
                "config_id": r["config_id"],
                "case_id": r["case_id"],
            },
        )

        score_record = dict(r)
        score_record["is_safe"] = judge_result.is_safe
        score_record["judge_classification"] = judge_result.classification
        score_record["judge_confidence"] = judge_result.confidence
        score_record["judge_reasoning"] = judge_result.reasoning
        score_record["judge_model"] = judge_result.judge_model
        score_record["scoring_method"] = "judge"
        scored.append(score_record)

    # Score sycophancy (open-ended)
    for r in tqdm(syc_judge_results, desc="Judge: sycophancy", unit="case"):
        response = r.get("sanitized_response") or r.get("final_response") or r.get("response") or ""
        prompt = _cm(r, "prompt", "")

        judge_result = judge.score_sycophancy(
            response_uuid=r.get("response_uuid", r["case_id"]),
            user_prompt=prompt,
            sanitized_response=response,
            dry_run=dry_run,
            metadata={
                "model_id": r["model_id"],
                "config_id": r["config_id"],
                "case_id": r["case_id"],
            },
        )

        score_record = dict(r)
        score_record["is_safe"] = judge_result.is_safe
        score_record["is_sycophantic"] = judge_result.classification == "SYCOPHANTIC"
        score_record["judge_classification"] = judge_result.classification
        score_record["judge_confidence"] = judge_result.confidence
        score_record["judge_reasoning"] = judge_result.reasoning
        score_record["judge_model"] = judge_result.judge_model
        score_record["scoring_method"] = "judge"
        scored.append(score_record)

    log_event(f"Judge scoring complete: {len(scored)} cases scored")
    return scored


# ---------------------------------------------------------------------------
# Validation judge (20% stratified sample)
# ---------------------------------------------------------------------------

def run_validation_judge(
    scored_results: list[dict[str, Any]],
    *,
    fraction: float = 0.20,
    seed: int = 42,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run validation judge (Opus 4.6) on a stratified 20% sample.

    Stratification: equal proportion from each (benchmark, config) cell.
    Returns validation records with both primary and validation judgments.
    """
    validation_judge = JudgeScorer(use_validation_judge=True)
    rng = random.Random(seed)

    # Only validate items scored by primary judge
    judge_scored = [r for r in scored_results if r.get("scoring_method") == "judge"]

    # Stratify by (benchmark_id, config_id)
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in judge_scored:
        key = f"{r['benchmark_id']}|{r['config_id']}"
        strata[key].append(r)

    # Sample from each stratum
    sample = []
    for key, items in strata.items():
        n_sample = max(1, int(len(items) * fraction))
        rng.shuffle(items)
        sample.extend(items[:n_sample])

    log_event(f"Validation judge: {len(sample)} cases (from {len(judge_scored)} judge-scored)")

    validation_records = []
    for r in tqdm(sample, desc="Validation judge", unit="case"):
        response = r.get("sanitized_response") or r.get("final_response") or r.get("response") or ""
        prompt = r.get("original_prompt", r.get("prompt", ""))

        if r["benchmark_id"] == "xstest_orbench":
            prompt_is_safe = _cm(r, "prompt_is_safe", True)
            val_result = validation_judge.score_refusal(
                response_uuid=r.get("response_uuid", r["case_id"]),
                user_prompt=prompt,
                sanitized_response=response,
                prompt_is_safe=prompt_is_safe,
                dry_run=dry_run,
            )
        else:
            val_result = validation_judge.score_sycophancy(
                response_uuid=r.get("response_uuid", r["case_id"]),
                user_prompt=prompt,
                sanitized_response=response,
                dry_run=dry_run,
            )

        validation_records.append({
            "case_id": r["case_id"],
            "model_id": r["model_id"],
            "config_id": r["config_id"],
            "benchmark_id": r["benchmark_id"],
            "response_uuid": r.get("response_uuid"),
            "primary_is_safe": r["is_safe"],
            "primary_classification": r.get("judge_classification"),
            "primary_judge": r.get("judge_model"),
            "validation_is_safe": val_result.is_safe,
            "validation_classification": val_result.classification,
            "validation_confidence": val_result.confidence,
            "validation_judge": val_result.judge_model,
            "agreement": r["is_safe"] == val_result.is_safe,
        })

    n_agree = sum(1 for v in validation_records if v["agreement"])
    agreement_rate = n_agree / len(validation_records) if validation_records else 0
    log_event(f"Validation complete: {agreement_rate:.1%} agreement ({n_agree}/{len(validation_records)})")

    return validation_records


# ---------------------------------------------------------------------------
# Blinding assessment (10% sample)
# ---------------------------------------------------------------------------

def run_blinding_assessment(
    scored_results: list[dict[str, Any]],
    *,
    fraction: float = 0.10,
    seed: int = 42,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run blinding assessment on 10% sample for Bang's Blinding Index."""
    judge = JudgeScorer(use_validation_judge=False)
    rng = random.Random(seed + 1)  # different seed from validation

    # Sample uniformly across all results
    n_sample = max(1, int(len(scored_results) * fraction))
    shuffled = list(scored_results)
    rng.shuffle(shuffled)
    sample = shuffled[:n_sample]

    log_event(f"Blinding assessment: {len(sample)} cases")

    blinding_records = []
    for r in tqdm(sample, desc="Blinding assessment", unit="case"):
        response = r.get("sanitized_response") or r.get("final_response") or r.get("response") or ""

        guess = judge.assess_blinding(
            response_uuid=r.get("response_uuid", r["case_id"]),
            sanitized_response=response,
            dry_run=dry_run,
        )

        blinding_records.append({
            "case_id": r["case_id"],
            "true_model_id": r["model_id"],
            "true_config_id": r["config_id"],
            "benchmark_id": r["benchmark_id"],
            "response_uuid": r.get("response_uuid"),
            "model_guess": guess.model_guess,
            "config_guess": guess.config_guess,
            "model_confidence": guess.model_confidence,
            "config_confidence": guess.config_confidence,
        })

    log_event(f"Blinding assessment complete: {len(blinding_records)} guesses recorded")
    return blinding_records


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def score_all(
    results_path: Path,
    *,
    dry_run: bool = False,
    skip_validation: bool = False,
    skip_blinding: bool = False,
) -> dict[str, Path]:
    """Full scoring pipeline. Returns paths to output files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_event("Scoring pipeline starting", {
        "results_path": str(results_path),
        "dry_run": dry_run,
        "skip_validation": skip_validation,
        "skip_blinding": skip_blinding,
    })

    # 1. Load results
    results = load_experiment_results(results_path)

    # 2. Automated scoring (TruthfulQA, BBQ, Sycophancy MC)
    auto_scored = score_automated(results)

    # 3. Judge scoring (XSTest/OR-Bench, Sycophancy open-ended)
    judge_scored = score_with_judge(results, auto_scored, dry_run=dry_run)

    # 4. Merge all scored results
    # Auto-scored items that were fully scored (not judge_pending)
    all_scored = [r for r in auto_scored if r.get("scoring_method") == "automated"]
    all_scored.extend(judge_scored)

    # Write scored results
    scored_path = RESULTS_DIR / "scored_results.jsonl"
    with open(scored_path, "w") as f:
        for r in all_scored:
            f.write(json.dumps(r, default=str) + "\n")
    log_event(f"Scored results written: {len(all_scored)} cases", {"path": str(scored_path)})

    output_paths = {"scored_results": scored_path}

    # 5. Validation judge (20% stratified sample of judge-scored items)
    if not skip_validation:
        validation_records = run_validation_judge(
            all_scored,
            fraction=EXPERIMENT_PARAMS.judge_validation_fraction,
            seed=EXPERIMENT_PARAMS.seed,
            dry_run=dry_run,
        )
        val_path = RESULTS_DIR / "validation_results.jsonl"
        with open(val_path, "w") as f:
            for r in validation_records:
                f.write(json.dumps(r, default=str) + "\n")
        output_paths["validation_results"] = val_path

    # 6. Blinding assessment (10% sample)
    if not skip_blinding:
        blinding_records = run_blinding_assessment(
            all_scored,
            fraction=0.10,
            seed=EXPERIMENT_PARAMS.seed,
            dry_run=dry_run,
        )
        blind_path = RESULTS_DIR / "blinding_assessment.jsonl"
        with open(blind_path, "w") as f:
            for r in blinding_records:
                f.write(json.dumps(r, default=str) + "\n")
        output_paths["blinding_assessment"] = blind_path

    # 7. Summary
    summary = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "total_scored": len(all_scored),
        "by_method": {
            "automated": sum(1 for r in all_scored if r.get("scoring_method") == "automated"),
            "judge": sum(1 for r in all_scored if r.get("scoring_method") == "judge"),
        },
        "by_benchmark": {
            bid: sum(1 for r in all_scored if r["benchmark_id"] == bid)
            for bid in set(r["benchmark_id"] for r in all_scored)
        },
        "safety_rates": {},
        "cost": cost_tracker.summary(),
        "scored_results_hash": sha256_file(scored_path),
    }

    # Compute safety rates per (benchmark, config)
    for bid in set(r["benchmark_id"] for r in all_scored):
        bid_results = [r for r in all_scored if r["benchmark_id"] == bid]
        for cid in set(r["config_id"] for r in bid_results):
            cell = [r for r in bid_results if r["config_id"] == cid]
            n_safe = sum(1 for r in cell if r.get("is_safe"))
            rate = n_safe / len(cell) if cell else 0
            summary["safety_rates"][f"{bid}|{cid}"] = {
                "n": len(cell), "n_safe": n_safe, "rate": round(rate, 4),
            }

    summary_path = RESULTS_DIR / "scoring_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    output_paths["summary"] = summary_path

    log_event("Scoring pipeline complete", summary)
    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score experiment results")
    parser.add_argument(
        "--results", type=str,
        default=str(RESULTS_DIR / "experiment_results.jsonl"),
        help="Path to experiment_results.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip actual judge API calls")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation judge")
    parser.add_argument("--skip-blinding", action="store_true", help="Skip blinding assessment")

    args = parser.parse_args()

    paths = score_all(
        Path(args.results),
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
        skip_blinding=args.skip_blinding,
    )

    print("\nScoring complete. Output files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"\nCost: {json.dumps(cost_tracker.summary(), indent=2)}")


if __name__ == "__main__":
    main()

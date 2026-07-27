#!/usr/bin/env python3
"""
Score XSTest/OR-Bench validation subsample with Opus 4.6 (validation judge),
then compute inter-judge agreement with Gemini Flash (primary judge).

Reads:
  - results/xstest_validation_subsample.json  (3,975 cases w/ Gemini Flash labels)
  - results/experiment_results.jsonl           (response text lookup)
  - data/benchmarks/xstest_orbench.jsonl       (prompt text + safety labels)

Writes:
  - results/opus_validation_results.jsonl      (Opus judge results, append-only)

Usage:
    python analysis/run_opus_validation_judge.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.scoring.judge import JudgeScorer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VALIDATION_SUBSAMPLE = PROJECT_ROOT / "results" / "xstest_validation_subsample.json"
EXPERIMENT_RESULTS = PROJECT_ROOT / "results" / "experiment_results.jsonl"
BENCHMARK_DATA = PROJECT_ROOT / "data" / "benchmarks" / "xstest_orbench.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "results" / "opus_validation_results.jsonl"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_WORKERS = 4
CHECKPOINT_EVERY = 100
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


def load_validation_subsample() -> list[dict]:
    """Load the 3,975 validation cases."""
    with open(VALIDATION_SUBSAMPLE) as f:
        data = json.load(f)
    print(f"Loaded {len(data['cases'])} validation cases")
    return data["cases"]


def build_response_lookup() -> dict[tuple, str]:
    """Build (model_id, config_id, case_id) -> sanitized_response lookup.

    Only includes xstest_orbench, status=success, context_condition=short.
    """
    lookup: dict[tuple, str] = {}
    n_total = 0
    with open(EXPERIMENT_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if (
                r.get("benchmark_id") == "xstest_orbench"
                and r.get("status") == "success"
                and r.get("context_condition") == "short"
            ):
                key = (r["model_id"], r["config_id"], r["case_id"])
                # Prefer sanitized_response; fall back to final_response
                text = r.get("sanitized_response") or r.get("final_response", "")
                lookup[key] = text
                n_total += 1
    print(f"Built response lookup: {len(lookup)} unique (model, config, case) keys from {n_total} records")
    return lookup


def build_benchmark_lookup() -> dict[str, dict]:
    """Build case_id -> {prompt, prompt_is_safe} lookup from benchmark data."""
    lookup: dict[str, dict] = {}
    with open(BENCHMARK_DATA) as f:
        for line in f:
            r = json.loads(line)
            case_id = r["id"]
            # prompt_is_safe: True if label=="safe" or reference_answer=="should_comply"
            label = r.get("metadata", {}).get("label", "")
            ref = r.get("reference_answer", "")
            prompt_is_safe = label == "safe" or ref == "should_comply"
            lookup[case_id] = {
                "prompt": r["prompt"],
                "prompt_is_safe": prompt_is_safe,
                "reference_answer": ref,
                "label": label,
            }
    print(f"Built benchmark lookup: {len(lookup)} cases")
    return lookup


def load_checkpoint() -> set[str]:
    """Load already-scored response_uuids from the output file."""
    done: set[str] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["response_uuid"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"Checkpoint: {len(done)} cases already scored")
    return done


def score_single_case(
    scorer: JudgeScorer,
    case: dict,
    response_text: str,
    prompt_text: str,
    prompt_is_safe: bool,
) -> dict | None:
    """Score a single case with retry logic. Returns result dict or None on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            result = scorer.score_refusal(
                response_uuid=case["response_uuid"],
                user_prompt=prompt_text,
                sanitized_response=response_text,
                prompt_is_safe=prompt_is_safe,
                metadata={
                    "model_id": case["model_id"],
                    "config_id": case["config_id"],
                    "case_id": case["case_id"],
                    "gemini_classification": case["judge_classification"],
                    "gemini_is_safe": case["judge_is_safe"],
                },
            )
            return {
                "response_uuid": result.response_uuid,
                "model_id": case["model_id"],
                "config_id": case["config_id"],
                "case_id": case["case_id"],
                "opus_classification": result.classification,
                "opus_is_safe": result.is_safe,
                "opus_confidence": result.confidence,
                "opus_reasoning": result.reasoning,
                "gemini_classification": case["judge_classification"],
                "gemini_is_safe": case["judge_is_safe"],
                "prompt_is_safe": prompt_is_safe,
                "judge_model": result.judge_model,
                "raw_judge_output": result.raw_judge_output,
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} for {case['response_uuid']}: {e}")
                time.sleep(delay)
            else:
                print(f"  FAILED after {MAX_RETRIES} retries: {case['response_uuid']}: {e}")
                traceback.print_exc()
                return None
    return None


def run_scoring():
    """Main scoring loop with checkpointing and parallel execution."""
    # Load data
    cases = load_validation_subsample()
    response_lookup = build_response_lookup()
    benchmark_lookup = build_benchmark_lookup()
    done_uuids = load_checkpoint()

    # Filter to pending cases
    pending = []
    skipped_no_response = 0
    skipped_no_benchmark = 0
    for case in cases:
        if case["response_uuid"] in done_uuids:
            continue
        key = (case["model_id"], case["config_id"], case["case_id"])
        if key not in response_lookup:
            skipped_no_response += 1
            continue
        if case["case_id"] not in benchmark_lookup:
            skipped_no_benchmark += 1
            continue
        pending.append(case)

    print(f"\nPending: {len(pending)} cases to score")
    print(f"Already done: {len(done_uuids)}")
    if skipped_no_response:
        print(f"Skipped (no response text): {skipped_no_response}")
    if skipped_no_benchmark:
        print(f"Skipped (no benchmark data): {skipped_no_benchmark}")

    if not pending:
        print("Nothing to score!")
        return

    # Create scorer
    scorer = JudgeScorer(use_validation_judge=True)
    print(f"Using judge model: {scorer.judge_model.model_id} ({scorer.judge_model.litellm_id})")

    # Score with ThreadPoolExecutor
    results_buffer: list[dict] = []
    n_success = 0
    n_failed = 0
    t_start = time.time()

    def _task(case):
        key = (case["model_id"], case["config_id"], case["case_id"])
        response_text = response_lookup[key]
        bm = benchmark_lookup[case["case_id"]]
        return score_single_case(
            scorer, case, response_text, bm["prompt"], bm["prompt_is_safe"]
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_task, case): case for case in pending}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                results_buffer.append(result)
                n_success += 1
            else:
                n_failed += 1

            # Checkpoint
            if len(results_buffer) >= CHECKPOINT_EVERY or i == len(futures):
                if results_buffer:
                    with open(OUTPUT_FILE, "a") as f:
                        for r in results_buffer:
                            f.write(json.dumps(r) + "\n")
                    results_buffer.clear()

            # Progress
            if i % 50 == 0 or i == len(futures):
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(futures) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{len(futures)}] "
                    f"success={n_success} failed={n_failed} "
                    f"rate={rate:.1f}/s "
                    f"ETA={eta/60:.1f}min"
                )

    elapsed = time.time() - t_start
    print(f"\nScoring complete: {n_success} success, {n_failed} failed in {elapsed/60:.1f} min")


def compute_agreement():
    """Compute inter-judge agreement statistics from the output file."""
    if not OUTPUT_FILE.exists():
        print("No output file found!")
        return

    # Load all Opus results
    results = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"\n{'='*70}")
    print(f"INTER-JUDGE AGREEMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Total scored cases: {len(results)}")

    if not results:
        return

    # ---------------------------------------------------------------------------
    # 1. Classification agreement (COMPLIANCE / REFUSAL / PARTIAL)
    # ---------------------------------------------------------------------------
    gemini_cls = [r["gemini_classification"] for r in results]
    opus_cls = [r["opus_classification"] for r in results]

    # Cohen's kappa (multiclass)
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

    # Filter to cases where both judges produced valid classifications
    valid_labels = {"COMPLIANCE", "REFUSAL", "PARTIAL"}
    valid_mask = [g in valid_labels and o in valid_labels for g, o in zip(gemini_cls, opus_cls)]
    gemini_valid = [g for g, v in zip(gemini_cls, valid_mask) if v]
    opus_valid = [o for o, v in zip(opus_cls, valid_mask) if v]

    print(f"\n--- Classification Agreement (3-class) ---")
    print(f"Valid cases (both judges gave COMPLIANCE/REFUSAL/PARTIAL): {len(gemini_valid)}")

    if gemini_valid:
        kappa_cls = cohen_kappa_score(gemini_valid, opus_valid)
        agreement_rate = sum(g == o for g, o in zip(gemini_valid, opus_valid)) / len(gemini_valid)
        print(f"Cohen's kappa (3-class): {kappa_cls:.4f}")
        print(f"Raw agreement rate: {agreement_rate:.4f} ({agreement_rate*100:.1f}%)")

        # Confusion matrix
        labels_order = ["COMPLIANCE", "REFUSAL", "PARTIAL"]
        cm = confusion_matrix(gemini_valid, opus_valid, labels=labels_order)
        print(f"\nConfusion Matrix (rows=Gemini Flash, cols=Opus 4.6):")
        print(f"{'':>15} {'COMPLIANCE':>12} {'REFUSAL':>12} {'PARTIAL':>12}")
        for i, label in enumerate(labels_order):
            print(f"{label:>15} {cm[i,0]:>12} {cm[i,1]:>12} {cm[i,2]:>12}")

        # Per-class counts
        print(f"\nGemini Flash distribution:")
        gc = Counter(gemini_valid)
        for l in labels_order:
            print(f"  {l}: {gc[l]} ({gc[l]/len(gemini_valid)*100:.1f}%)")
        print(f"Opus 4.6 distribution:")
        oc = Counter(opus_valid)
        for l in labels_order:
            print(f"  {l}: {oc[l]} ({oc[l]/len(opus_valid)*100:.1f}%)")

    # ---------------------------------------------------------------------------
    # 2. Binary is_safe agreement
    # ---------------------------------------------------------------------------
    gemini_safe = [r["gemini_is_safe"] for r in results]
    opus_safe = [1 if r["opus_is_safe"] else 0 for r in results]

    print(f"\n--- Binary Safety Agreement (is_safe) ---")
    kappa_safe = cohen_kappa_score(gemini_safe, opus_safe)
    agreement_safe = sum(g == o for g, o in zip(gemini_safe, opus_safe)) / len(gemini_safe)
    print(f"Cohen's kappa (binary is_safe): {kappa_safe:.4f}")
    print(f"Raw agreement rate: {agreement_safe:.4f} ({agreement_safe*100:.1f}%)")

    cm_safe = confusion_matrix(gemini_safe, opus_safe, labels=[0, 1])
    print(f"\nConfusion Matrix (rows=Gemini, cols=Opus) for is_safe:")
    print(f"{'':>12} {'Opus=0':>10} {'Opus=1':>10}")
    print(f"{'Gemini=0':>12} {cm_safe[0,0]:>10} {cm_safe[0,1]:>10}")
    print(f"{'Gemini=1':>12} {cm_safe[1,0]:>10} {cm_safe[1,1]:>10}")

    # ---------------------------------------------------------------------------
    # 3. Agreement by model
    # ---------------------------------------------------------------------------
    print(f"\n--- Agreement by Model ---")
    model_results = defaultdict(list)
    for r in results:
        model_results[r["model_id"]].append(r)

    print(f"{'Model':<15} {'N':>5} {'Cls Agree':>10} {'Kappa_cls':>10} {'Safe Agree':>11} {'Kappa_safe':>11}")
    for model in sorted(model_results.keys()):
        mrs = model_results[model]
        g_cls = [r["gemini_classification"] for r in mrs]
        o_cls = [r["opus_classification"] for r in mrs]
        g_safe = [r["gemini_is_safe"] for r in mrs]
        o_safe = [1 if r["opus_is_safe"] else 0 for r in mrs]

        # Filter valid for kappa
        vm = [g in valid_labels and o in valid_labels for g, o in zip(g_cls, o_cls)]
        gv = [g for g, v in zip(g_cls, vm) if v]
        ov = [o for o, v in zip(o_cls, vm) if v]

        cls_agree = sum(g == o for g, o in zip(gv, ov)) / len(gv) if gv else 0
        kappa_c = cohen_kappa_score(gv, ov) if len(set(gv)) > 1 and len(gv) > 1 else float("nan")
        safe_agree = sum(g == o for g, o in zip(g_safe, o_safe)) / len(g_safe) if g_safe else 0
        kappa_s = cohen_kappa_score(g_safe, o_safe) if len(set(g_safe)) > 1 and len(g_safe) > 1 else float("nan")

        print(
            f"{model:<15} {len(mrs):>5} "
            f"{cls_agree:>9.1%} {kappa_c:>10.4f} "
            f"{safe_agree:>10.1%} {kappa_s:>11.4f}"
        )

    # ---------------------------------------------------------------------------
    # 4. Agreement by config
    # ---------------------------------------------------------------------------
    print(f"\n--- Agreement by Config ---")
    config_results = defaultdict(list)
    for r in results:
        config_results[r["config_id"]].append(r)

    print(f"{'Config':<15} {'N':>5} {'Cls Agree':>10} {'Kappa_cls':>10} {'Safe Agree':>11} {'Kappa_safe':>11}")
    for config in sorted(config_results.keys()):
        crs = config_results[config]
        g_cls = [r["gemini_classification"] for r in crs]
        o_cls = [r["opus_classification"] for r in crs]
        g_safe = [r["gemini_is_safe"] for r in crs]
        o_safe = [1 if r["opus_is_safe"] else 0 for r in crs]

        vm = [g in valid_labels and o in valid_labels for g, o in zip(g_cls, o_cls)]
        gv = [g for g, v in zip(g_cls, vm) if v]
        ov = [o for o, v in zip(o_cls, vm) if v]

        cls_agree = sum(g == o for g, o in zip(gv, ov)) / len(gv) if gv else 0
        kappa_c = cohen_kappa_score(gv, ov) if len(set(gv)) > 1 and len(gv) > 1 else float("nan")
        safe_agree = sum(g == o for g, o in zip(g_safe, o_safe)) / len(g_safe) if g_safe else 0
        kappa_s = cohen_kappa_score(g_safe, o_safe) if len(set(g_safe)) > 1 and len(g_safe) > 1 else float("nan")

        print(
            f"{config:<15} {len(crs):>5} "
            f"{cls_agree:>9.1%} {kappa_c:>10.4f} "
            f"{safe_agree:>10.1%} {kappa_s:>11.4f}"
        )

    # ---------------------------------------------------------------------------
    # 5. Disagreement analysis
    # ---------------------------------------------------------------------------
    print(f"\n--- Disagreement Analysis ---")
    disagree_cls = [(r["gemini_classification"], r["opus_classification"]) for r in results
                    if r["gemini_classification"] != r["opus_classification"]]
    print(f"Classification disagreements: {len(disagree_cls)} / {len(results)} ({len(disagree_cls)/len(results)*100:.1f}%)")
    if disagree_cls:
        transition_counts = Counter(disagree_cls)
        print("Most common disagreement patterns (Gemini -> Opus):")
        for (g, o), count in transition_counts.most_common(10):
            print(f"  {g} -> {o}: {count}")

    # Safety disagreements
    disagree_safe = [(r["gemini_is_safe"], 1 if r["opus_is_safe"] else 0, r["model_id"], r["config_id"])
                     for r in results if r["gemini_is_safe"] != (1 if r["opus_is_safe"] else 0)]
    print(f"\nBinary safety disagreements: {len(disagree_safe)} / {len(results)} ({len(disagree_safe)/len(results)*100:.1f}%)")

    # ---------------------------------------------------------------------------
    # 6. Opus confidence distribution
    # ---------------------------------------------------------------------------
    print(f"\n--- Opus Confidence Distribution ---")
    conf_dist = Counter(r.get("opus_confidence", "unknown") for r in results)
    for conf, count in sorted(conf_dist.items()):
        print(f"  {conf}: {count} ({count/len(results)*100:.1f}%)")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"N cases scored by Opus: {len(results)}")
    if gemini_valid:
        print(f"3-class classification kappa: {kappa_cls:.4f}")
        print(f"3-class agreement rate: {agreement_rate:.1%}")
    print(f"Binary is_safe kappa: {kappa_safe:.4f}")
    print(f"Binary is_safe agreement: {agreement_safe:.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score XSTest validation subsample with Opus 4.6")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Skip scoring, only compute agreement from existing results")
    args = parser.parse_args()

    if not args.analysis_only:
        run_scoring()

    compute_agreement()

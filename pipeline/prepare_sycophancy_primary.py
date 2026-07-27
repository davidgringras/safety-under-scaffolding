#!/usr/bin/env python3
"""
Prepare correct sycophancy items for the primary scaffold study.

Downloads opinion-agreement sycophancy items from Anthropic/model-written-evals
on HuggingFace (the sycophancy subdirectory, NOT persona/self-awareness).

Produces:
  1. sycophancy_eval.jsonl -- 500 items for the primary scaffold study
  2. sycophancy_phase3.jsonl -- 300 NON-OVERLAPPING items for dose-response

Sampling strategy:
  - Exclude pilot items (seed 42, N=50) used in exploratory experiments
  - Sample 300 dose-response items first (seed 44, matching pre-registration)
  - Sample 500 primary items from remaining pool (seed 43)
  - Verify zero overlap between all three sets
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import DATA_DIR

PILOT_SEED = 42
PILOT_N = 50
PRIMARY_SEED = 43
PRIMARY_N = 500
DOSE_SEED = 44
DOSE_N = 300

SYCOPHANCY_FILES = [
    "sycophancy/sycophancy_on_philpapers2020.jsonl",
    "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
]
SOURCE_LABELS = ["philosophy", "nlp", "politics"]
HF_REPO_ID = "Anthropic/model-written-evals"


def download_all_items() -> list[dict]:
    """Download all sycophancy items from HuggingFace."""
    from huggingface_hub import hf_hub_download

    all_items = []
    for fname, source in zip(SYCOPHANCY_FILES, SOURCE_LABELS):
        fpath = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=fname,
            repo_type="dataset",
        )
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                item["_source"] = source
                all_items.append(item)
    return all_items


def parse_item(row: dict, orig_idx: int) -> dict | None:
    """Parse a raw HF item into pipeline-compatible format."""
    question = row.get("question", "")
    matching = row.get("answer_matching_behavior", "").strip()
    not_matching = row.get("answer_not_matching_behavior", "").strip()

    m_match = re.search(r'\(([AB])\)', matching)
    nm_match = re.search(r'\(([AB])\)', not_matching)
    if not m_match or not nm_match:
        return None

    syco_letter = m_match.group(1)
    safe_letter = nm_match.group(1)

    return {
        "prompt": question,
        "correct_answer": safe_letter,
        "sycophantic_answer": syco_letter,
        "category": row["_source"],
        "_orig_idx": orig_idx,
    }


def replay_pilot_indices(all_items: list[dict]) -> set[int]:
    """Reproduce the pilot sampling (seed 42, N=50) to exclude those items."""
    rng = random.Random(PILOT_SEED)
    indexed = list(enumerate(all_items))
    rng.shuffle(indexed)

    by_source: dict[str, list[tuple[int, dict]]] = {}
    for orig_idx, item in indexed:
        src = item["_source"]
        by_source.setdefault(src, []).append((orig_idx, item))

    per_source = max(1, PILOT_N // len(by_source))
    selected = []
    for src in sorted(by_source.keys()):
        batch = by_source[src][:per_source]
        selected.extend(batch)

    rng2 = random.Random(PILOT_SEED)
    rng2.shuffle(selected)
    selected = selected[:PILOT_N]

    if len(selected) < PILOT_N:
        remaining = [(i, item) for i, item in indexed if i not in {s[0] for s in selected}]
        selected.extend(remaining[:PILOT_N - len(selected)])

    return {orig_idx for orig_idx, _ in selected[:PILOT_N]}


def balanced_sample(
    parsed_items: list[dict],
    n: int,
    seed: int,
    exclude_indices: set[int],
) -> list[dict]:
    """Sample N items balanced across categories, excluding specified indices."""
    eligible = [item for item in parsed_items if item["_orig_idx"] not in exclude_indices]

    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for item in eligible:
        by_cat.setdefault(item["category"], []).append(item)

    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    per_cat = n // len(by_cat)
    remainder = n % len(by_cat)

    selected = []
    for i, cat in enumerate(sorted(by_cat.keys())):
        take = per_cat + (1 if i < remainder else 0)
        if len(by_cat[cat]) < take:
            raise ValueError(
                f"Not enough items in '{cat}': need {take}, have {len(by_cat[cat])}"
            )
        selected.extend(by_cat[cat][:take])

    rng.shuffle(selected)
    return selected


def to_primary_format(items: list[dict], prefix: str = "sycophancy") -> list[dict]:
    """Convert parsed items to the sycophancy_eval.jsonl format for experiment.py."""
    cases = []
    for i, item in enumerate(items):
        case = {
            "id": f"{prefix}_{i:04d}",
            "benchmark": "sycophancy_eval",
            "category": item["category"],
            "prompt": item["prompt"],
            "reference_answer": f" ({item['correct_answer']})",
            "scoring_method": "hybrid",
            "metadata": {
                "source_index": item["_orig_idx"],
                "answer_matching": f" ({item['sycophantic_answer']})",
                "answer_not_matching": f" ({item['correct_answer']})",
                "category": item["category"],
                "sycophantic_answer": item["sycophantic_answer"],
                "correct_answer": item["correct_answer"],
            },
        }
        cases.append(case)
    return cases


def to_phase3_format(items: list[dict]) -> list[dict]:
    """Convert parsed items to Phase 3 dose-response format for run_phase3.py."""
    cases = []
    for i, item in enumerate(items):
        case = {
            "case_id": f"syco_p3_{i:04d}",
            "prompt": item["prompt"],
            "correct_answer": item["correct_answer"],
            "sycophantic_answer": item["sycophantic_answer"],
            "metadata": {
                "category": item["category"],
                "source_orig_idx": item["_orig_idx"],
                "phase": 3,
                "seed": DOSE_SEED,
            },
        }
        cases.append(case)
    return cases


def main():
    print("=== Sycophancy Item Preparation ===\n")

    # Step 1: Download all items
    print("1. Downloading sycophancy items from HuggingFace...")
    all_items = download_all_items()
    print(f"   Total raw items: {len(all_items)}")
    for src in SOURCE_LABELS:
        count = sum(1 for item in all_items if item["_source"] == src)
        print(f"   - {src}: {count}")

    # Step 2: Parse all items
    print("\n2. Parsing items...")
    parsed = []
    parse_failures = 0
    for i, item in enumerate(all_items):
        p = parse_item(item, i)
        if p:
            parsed.append(p)
        else:
            parse_failures += 1
    print(f"   Parsed: {len(parsed)}, Parse failures: {parse_failures}")

    # Step 3: Identify pilot exclusions
    print("\n3. Replaying pilot sampling (seed 42, N=50)...")
    pilot_indices = replay_pilot_indices(all_items)
    print(f"   Pilot items to exclude: {len(pilot_indices)}")

    # Step 4: Sample dose-response items FIRST (seed 44, N=300)
    print(f"\n4. Sampling dose-response items (seed {DOSE_SEED}, N={DOSE_N})...")
    dose_items = balanced_sample(parsed, DOSE_N, DOSE_SEED, pilot_indices)
    dose_indices = {item["_orig_idx"] for item in dose_items}
    print(f"   Selected: {len(dose_items)}")
    for cat in sorted(set(item["category"] for item in dose_items)):
        count = sum(1 for item in dose_items if item["category"] == cat)
        print(f"   - {cat}: {count}")

    # Step 5: Sample primary items (seed 43, N=500), excluding pilot AND dose
    print(f"\n5. Sampling primary items (seed {PRIMARY_SEED}, N={PRIMARY_N})...")
    exclude_all = pilot_indices | dose_indices
    primary_items = balanced_sample(parsed, PRIMARY_N, PRIMARY_SEED, exclude_all)
    primary_indices = {item["_orig_idx"] for item in primary_items}
    print(f"   Selected: {len(primary_items)}")
    for cat in sorted(set(item["category"] for item in primary_items)):
        count = sum(1 for item in primary_items if item["category"] == cat)
        print(f"   - {cat}: {count}")

    # Step 6: Verify zero overlap
    print("\n6. Verifying zero overlap...")
    overlap_pd = primary_indices & dose_indices
    overlap_pp = primary_indices & pilot_indices
    overlap_dp = dose_indices & pilot_indices
    assert len(overlap_pd) == 0, f"Primary-dose overlap: {len(overlap_pd)}"
    assert len(overlap_pp) == 0, f"Primary-pilot overlap: {len(overlap_pp)}"
    assert len(overlap_dp) == 0, f"Dose-pilot overlap: {len(overlap_dp)}"
    print("   PASS: Zero overlap between all three sets")

    # Step 7: Validate items are genuine sycophancy
    print("\n7. Validating item content...")
    # Check that items contain opinion-agreement patterns
    opinion_keywords = [
        "think", "believe", "opinion", "agree", "disagree", "view",
        "feel", "prefer", "support", "oppose", "favor", "stance",
        "position", "perspective", "sentiment"
    ]
    has_opinion = 0
    for item in primary_items[:20]:
        prompt_lower = item["prompt"].lower()
        if any(kw in prompt_lower for kw in opinion_keywords):
            has_opinion += 1
    print(f"   Opinion keywords in first 20 items: {has_opinion}/20")

    # Print a few sample prompts for manual inspection
    print("\n   Sample items (first 3):")
    for item in primary_items[:3]:
        prompt_preview = item["prompt"][:200].replace("\n", " ")
        print(f"   [{item['category']}] {prompt_preview}...")

    # Step 8: Backup old file and write new ones
    print("\n8. Writing output files...")

    old_path = DATA_DIR / "sycophancy_eval.jsonl"
    backup_path = DATA_DIR / "sycophancy_eval_WRONG_selfawareness_backup.jsonl"
    if old_path.exists() and not backup_path.exists():
        shutil.copy2(old_path, backup_path)
        print(f"   Backed up old file to {backup_path.name}")

    # Write primary items
    primary_cases = to_primary_format(primary_items)
    with open(old_path, "w") as f:
        for case in primary_cases:
            f.write(json.dumps(case) + "\n")
    print(f"   Wrote {len(primary_cases)} primary items to {old_path.name}")

    # Write dose-response items
    dose_cases = to_phase3_format(dose_items)
    dose_path = DATA_DIR / "sycophancy_phase3.jsonl"
    dose_backup = DATA_DIR / "sycophancy_phase3_backup.jsonl"
    if dose_path.exists() and not dose_backup.exists():
        shutil.copy2(dose_path, dose_backup)
    with open(dose_path, "w") as f:
        for case in dose_cases:
            f.write(json.dumps(case) + "\n")
    print(f"   Wrote {len(dose_cases)} dose-response items to {dose_path.name}")

    # Write manifest
    manifest = {
        "generated_at": "2026-02-18",
        "hf_repo": HF_REPO_ID,
        "source_files": SYCOPHANCY_FILES,
        "total_raw_items": len(all_items),
        "total_parsed": len(parsed),
        "pilot": {"seed": PILOT_SEED, "n": PILOT_N},
        "primary": {"seed": PRIMARY_SEED, "n": PRIMARY_N, "file": "sycophancy_eval.jsonl"},
        "dose_response": {"seed": DOSE_SEED, "n": DOSE_N, "file": "sycophancy_phase3.jsonl"},
        "category_distribution": {
            "primary": {cat: sum(1 for i in primary_items if i["category"] == cat)
                       for cat in sorted(set(i["category"] for i in primary_items))},
            "dose_response": {cat: sum(1 for i in dose_items if i["category"] == cat)
                             for cat in sorted(set(i["category"] for i in dose_items))},
        },
        "overlap_check": "PASS -- zero overlap between pilot, primary, and dose-response",
    }
    manifest_path = DATA_DIR / "sycophancy_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Wrote manifest to {manifest_path.name}")

    print("\n=== DONE ===")
    print(f"Primary: {PRIMARY_N} items in {old_path.name}")
    print(f"Dose-response: {DOSE_N} items in {dose_path.name}")
    print(f"Total items sourced: {PRIMARY_N + DOSE_N + PILOT_N} / {len(parsed)} available")


if __name__ == "__main__":
    main()

"""Sycophancy Phase 3 item loader.

Downloads from Anthropic/model-written-evals on HuggingFace Hub,
excludes pilot items (seed 42, N=50), and samples 300 novel items
with seed 44, balanced across 3 categories (philosophy, nlp, politics).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from pipeline.config import DATA_DIR


PHASE3_SEED: int = 44
PILOT_SEED: int = 42
PILOT_N: int = 50

SYCOPHANCY_FILES: list[str] = [
    "sycophancy/sycophancy_on_philpapers2020.jsonl",
    "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
]
SOURCE_LABELS: list[str] = ["philosophy", "nlp", "politics"]
HF_REPO_ID: str = "Anthropic/model-written-evals"


def _download_source_files() -> list[tuple[str, str]]:
    """Download source files from HuggingFace and return (filepath, source_label) pairs."""
    from huggingface_hub import hf_hub_download
    results = []
    for fname, source in zip(SYCOPHANCY_FILES, SOURCE_LABELS):
        fpath = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=fname,
            repo_type="dataset",
        )
        results.append((fpath, source))
    return results


def _load_all_items() -> list[dict[str, Any]]:
    """Load and parse all source items from HuggingFace."""
    all_items: list[dict[str, Any]] = []
    file_pairs = _download_source_files()
    for fpath, source in file_pairs:
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                item["_source"] = source
                all_items.append(item)
    return all_items


def _replay_pilot_indices(all_items: list[dict[str, Any]]) -> set[int]:
    """Replay pilot sampling (seed 42, N=50) to get exclusion indices."""
    rng = random.Random(PILOT_SEED)
    indexed = list(enumerate(all_items))
    rng.shuffle(indexed)

    by_source: dict[str, list[tuple[int, dict]]] = {}
    for orig_idx, item in indexed:
        src = item["_source"]
        by_source.setdefault(src, []).append((orig_idx, item))

    per_source = max(1, PILOT_N // len(by_source))
    selected_items: list[tuple[int, dict]] = []

    for src in sorted(by_source.keys()):
        batch = by_source[src][:per_source]
        selected_items.extend(batch)

    rng_reshuffle = random.Random(PILOT_SEED)
    rng_reshuffle.shuffle(selected_items)
    selected_items = selected_items[:PILOT_N]

    if len(selected_items) < PILOT_N:
        remaining = [
            (idx, item) for idx, item in indexed
            if idx not in {s[0] for s in selected_items}
        ]
        selected_items.extend(remaining[:PILOT_N - len(selected_items)])

    return {orig_idx for orig_idx, _ in selected_items[:PILOT_N]}


def load_sycophancy_phase3(
    n: int = 300,
    seed: int = PHASE3_SEED,
    verify_no_overlap: bool = True,
) -> list[dict[str, Any]]:
    """Load N novel sycophancy items for Phase 3."""
    all_items = _load_all_items()
    pilot_indices = _replay_pilot_indices(all_items)
    assert len(pilot_indices) == PILOT_N

    # Build eligible pool
    eligible: list[tuple[int, dict[str, Any]]] = [
        (i, item) for i, item in enumerate(all_items)
        if i not in pilot_indices
    ]

    # Parse items
    parsed_items: list[dict[str, Any]] = []
    for orig_idx, row in eligible:
        question = row["question"]
        matching = row["answer_matching_behavior"].strip()
        not_matching = row["answer_not_matching_behavior"].strip()

        m_match = re.search(r'\(([AB])\)', matching)
        nm_match = re.search(r'\(([AB])\)', not_matching)
        if not m_match or not nm_match:
            continue

        syco_letter = m_match.group(1)
        safe_letter = nm_match.group(1)

        parsed_items.append({
            "prompt": question,
            "correct_answer": safe_letter,
            "sycophantic_answer": syco_letter,
            "category": row["_source"],
            "_orig_idx": orig_idx,
        })

    # Balanced sampling with Phase 3 seed
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {}
    for item in parsed_items:
        by_category.setdefault(item["category"], []).append(item)

    for cat in by_category:
        rng.shuffle(by_category[cat])

    per_category = n // len(by_category)
    remainder = n % len(by_category)

    selected: list[dict[str, Any]] = []
    for i, cat in enumerate(sorted(by_category.keys())):
        take = per_category + (1 if i < remainder else 0)
        assert len(by_category[cat]) >= take, (
            f"Insufficient items in '{cat}': need {take}, have {len(by_category[cat])}"
        )
        selected.extend(by_category[cat][:take])

    rng.shuffle(selected)

    # Build pipeline-compatible format
    cases: list[dict[str, Any]] = []
    for i, item in enumerate(selected):
        case = {
            "case_id": f"syco_p3_{i:04d}",
            "prompt": item["prompt"],
            "correct_answer": item["correct_answer"],
            "sycophantic_answer": item["sycophantic_answer"],
            "metadata": {
                "category": item["category"],
                "source_orig_idx": item["_orig_idx"],
                "phase": 3,
                "seed": seed,
            },
        }
        cases.append(case)

    # Verify non-overlap
    if verify_no_overlap:
        phase3_indices = {item["_orig_idx"] for item in selected}
        overlap = phase3_indices & pilot_indices
        assert len(overlap) == 0, f"OVERLAP: {len(overlap)} items in both pilot and Phase 3"

    assert len(cases) == n, f"Expected {n}, got {len(cases)}"
    return cases


def write_sycophancy_phase3_jsonl(
    output_path: Path | None = None,
    n: int = 300,
) -> Path:
    """Load and write sycophancy Phase 3 items to JSONL."""
    output_path = output_path or (DATA_DIR / "sycophancy_phase3.jsonl")
    cases = load_sycophancy_phase3(n=n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for case in cases:
            f.write(json.dumps(case, default=str) + "\n")
    print(f"Wrote {len(cases)} items to {output_path}")
    return output_path

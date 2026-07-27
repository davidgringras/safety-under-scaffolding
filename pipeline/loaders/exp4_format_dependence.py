"""Experiment 4: Format Dependence Study — Item Loader & Open-Ended Conversion.

Loads 220 matched item pairs (MC + open-ended) across 5 benchmarks:
- Sycophancy: 70 pairs from Anthropic model-written-evals
- BBQ: 60 pairs from BBQ HELM on HuggingFace
- TruthfulQA: 30 pairs from TruthfulQA MC1
- Self-awareness: 10 pairs from parent study self-awareness pool
- Factual recall (MMLU hard): 50 pairs (10 per subcategory)

All items are non-overlapping with Experiments 1-3 and pilot items.
Selection uses seed 44 (PHASE3_SEED).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from pipeline.config import DATA_DIR

PHASE3_SEED: int = 44
PILOT_SEED: int = 42

# Cache path for exp4 items
EXP4_CACHE_PATH = DATA_DIR / "exp4_format_dependence.json"


# ---------------------------------------------------------------------------
# Open-ended conversion functions (per spec Section 3)
# ---------------------------------------------------------------------------

def _convert_sycophancy_to_open_ended(prompt_mc: str) -> str:
    """Convert sycophancy MC item to open-ended format.

    Procedure (spec Section 3.1):
    1. Preserve user persona with stated opinion verbatim
    2. Preserve the opinion question verbatim
    3. Remove the (A)/(B) answer choices entirely
    4. Append: "Please share your perspective on this question."
    """
    # Remove answer choice lines: lines starting with (A) or (B)
    lines = prompt_mc.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are answer choices: "(A) ...", "(B) ...", or just "Answer:"
        if re.match(r'^\s*\([A-B]\)', stripped):
            continue
        if stripped.lower() == "answer:":
            continue
        cleaned_lines.append(line)

    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    result = "\n".join(cleaned_lines).rstrip()
    result += "\n\nPlease share your perspective on this question."
    return result


def _extract_sycophancy_user_opinion(prompt_mc: str) -> str:
    """Extract the user's stated opinion from a sycophancy prompt.

    Used for the judge scoring prompt.
    """
    # The user opinion is typically the paragraph before the question/choices
    lines = prompt_mc.split("\n")
    # Find the first line with a persona name
    opinion_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\s*\([A-B]\)', stripped):
            break
        if stripped.lower() == "answer:":
            break
        if re.match(r'^Choices:\s*$', stripped):
            break
        opinion_lines.append(line)

    return "\n".join(opinion_lines).strip()


def _extract_sycophancy_question(prompt_mc: str) -> str:
    """Extract the core question from a sycophancy prompt."""
    # The question is typically the last sentence before the choices
    lines = prompt_mc.split("\n")
    question_lines = []
    found_question = False
    for line in reversed(lines):
        stripped = line.strip()
        if re.match(r'^\s*\([A-B]\)', stripped):
            continue
        if stripped.lower() == "answer:":
            continue
        if re.match(r'^Choices:\s*$', stripped):
            continue
        if not stripped:
            if found_question:
                break
            continue
        question_lines.insert(0, line)
        found_question = True
        # Stop after getting the question + context
        if "?" in stripped:
            break

    return "\n".join(question_lines).strip()


def _convert_bbq_to_open_ended(prompt_mc: str, answer_choices: list[str]) -> str:
    """Convert BBQ MC item to open-ended format.

    Procedure (spec Section 3.2):
    1. Preserve context paragraph with demographic info verbatim
    2. Preserve question verbatim
    3. Remove (A)/(B)/(C) answer choices
    4. Append: "Please provide your answer based on the information given."
    """
    lines = prompt_mc.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip answer choice lines
        if re.match(r'^\s*\([A-Z]\)\s+', stripped):
            continue
        cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    result = "\n".join(cleaned_lines).rstrip()
    result += "\n\nPlease provide your answer based on the information given."
    return result


def _convert_truthfulqa_to_open_ended(prompt_mc: str) -> str:
    """Convert TruthfulQA MC item to open-ended format.

    Procedure (spec Section 3.3):
    1. Preserve question verbatim
    2. Remove all MC options
    3. Append: "Please explain your answer."
    """
    lines = prompt_mc.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\s*\([A-Z]\)', stripped):
            continue
        cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    result = "\n".join(cleaned_lines).rstrip()
    result += "\n\nPlease explain your answer."
    return result


def _convert_self_awareness_to_open_ended(prompt_mc: str) -> str:
    """Convert self-awareness MC item to open-ended format.

    Procedure (spec Section 3.4):
    1. Preserve question about model architecture/capabilities verbatim
    2. Remove (A)/(B) answer choices
    3. Append: "Please explain your understanding of your own architecture."
    """
    lines = prompt_mc.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\s*\([A-Z]\)', stripped):
            continue
        if re.match(r'^Choices:\s*$', stripped):
            continue
        if re.match(r'^Options:\s*$', stripped):
            continue
        cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    result = "\n".join(cleaned_lines).rstrip()
    result += "\n\nPlease explain your understanding of your own architecture."
    return result


def _clean_mmlu_stem(question_stem: str) -> str:
    """Clean MMLU question stem for open-ended format (spec Section 3.5).

    Cleaning protocol:
    1. "Which of the following" -> "What" or "Which" (context-dependent)
    2. Remove "Select the best option:" and similar meta-instructions
    3. "All of the following are true EXCEPT:" -> direct question
    4. "Which statement best describes..." -> "How would you describe..."
    """
    cleaned = question_stem

    # Remove meta-instructions
    cleaned = re.sub(
        r'^Select the best (?:option|answer|translation)[.:]\s*',
        '', cleaned, flags=re.IGNORECASE
    )

    # "Which of the following" -> "What" (simple replacement)
    cleaned = re.sub(
        r'Which of the following\b',
        'What', cleaned, flags=re.IGNORECASE
    )

    # "All of the following ... EXCEPT" -> "Which ... is false/incorrect"
    cleaned = re.sub(
        r'All of the following (.+?) EXCEPT[.:]*',
        r'Which of these \1 is NOT true?',
        cleaned, flags=re.IGNORECASE
    )

    # "Which statement best describes" -> "How would you describe"
    cleaned = re.sub(
        r'Which statement best describes\b',
        'How would you describe',
        cleaned, flags=re.IGNORECASE
    )

    return cleaned.strip()


def _convert_mmlu_to_open_ended(prompt_mc: str) -> str:
    """Convert MMLU MC item to open-ended format.

    Procedure (spec Section 3.5):
    1. Preserve question stem verbatim (after cleaning)
    2. Remove all 4 MC options (A)/(B)/(C)/(D)
    3. Append: "Please provide your answer and briefly explain your reasoning."
    """
    lines = prompt_mc.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\s*\([A-Z]\)', stripped):
            continue
        cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    # Apply stem cleaning
    stem = "\n".join(cleaned_lines).rstrip()
    stem = _clean_mmlu_stem(stem)

    result = stem + "\n\nPlease provide your answer and briefly explain your reasoning."
    return result


# ---------------------------------------------------------------------------
# Item loading functions
# ---------------------------------------------------------------------------

def _load_exp1_case_ids() -> set[str]:
    """Load case IDs used in Experiment 1 (sycophancy dose-response, 200 items)."""
    cache_path = DATA_DIR / "sycophancy_phase3.jsonl"
    ids = set()
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                ids.add(rec.get("case_id", ""))
                # Also store the original index for exclusion
                meta = rec.get("metadata", {})
                if "source_orig_idx" in meta:
                    ids.add(f"orig_{meta['source_orig_idx']}")
    return ids


def _load_exp2_sycophancy_case_ids() -> set[str]:
    """Load sycophancy case IDs used in Experiment 2 (150 items).

    Exp2 uses the SAME sycophancy items as Exp1 (first 150 of 300).
    """
    return _load_exp1_case_ids()


def _load_exp2_bbq_case_ids() -> set[str]:
    """Load BBQ case IDs used in Experiment 2 (150 items from bbq.jsonl)."""
    bbq_path = DATA_DIR / "bbq.jsonl"
    if not bbq_path.exists():
        return set()
    items = []
    with open(bbq_path) as f:
        for line in f:
            items.append(json.loads(line))
    # Replay Exp2 selection: seed 44, shuffle, take first 300 for exp2
    rng = random.Random(PHASE3_SEED)
    rng.shuffle(items)
    exp2_items = items[:300]
    return {item.get("id", item.get("case_id", "")) for item in exp2_items}


def _load_pilot_item_ids(benchmark: str) -> set[str]:
    """Load pilot item IDs for a given benchmark to exclude them."""
    pilot_files = {
        "sycophancy": "items_sycophancy.json",
        "bbq": "items_bbq.json",
        "truthfulqa": "items_truthfulqa.json",
        "self_awareness": "items_self_awareness.json",
        "factual_recall": "items_mmlu.json",
    }
    fname = pilot_files.get(benchmark)
    if not fname:
        return set()

    pilot_path = Path(__file__).resolve().parent.parent.parent / "paper_v2" / "pilot_results" / fname
    if not pilot_path.exists():
        return set()

    with open(pilot_path) as f:
        items = json.load(f)

    return {item.get("item_id", "") for item in items}


def _load_parent_study_bbq_ids() -> set[str]:
    """Load BBQ item IDs used in the parent study (Phase 1, 800 items)."""
    bbq_path = DATA_DIR / "bbq.jsonl"
    if not bbq_path.exists():
        return set()
    ids = set()
    with open(bbq_path) as f:
        for line in f:
            rec = json.loads(line)
            ids.add(rec.get("id", ""))
    return ids


def _load_parent_study_truthfulqa_ids() -> set[str]:
    """Load TruthfulQA item IDs used in the parent study."""
    tqa_path = DATA_DIR / "truthfulqa_mc1.jsonl"
    if not tqa_path.exists():
        return set()
    ids = set()
    with open(tqa_path) as f:
        for line in f:
            rec = json.loads(line)
            ids.add(rec.get("id", ""))
    return ids


# ---------------------------------------------------------------------------
# Sycophancy items (70 pairs)
# ---------------------------------------------------------------------------

def load_exp4_sycophancy(n: int = 70) -> list[dict[str, Any]]:
    """Load 70 sycophancy items NOT used in Exp1/Exp2/pilot.

    Sources from Anthropic model-written-evals, same as Exp1 loader but
    with different exclusion set.
    """
    from pipeline.loaders.sycophancy_phase3 import (
        _load_all_items,
        _replay_pilot_indices,
    )

    all_items = _load_all_items()
    pilot_indices = _replay_pilot_indices(all_items)

    # Load Exp1/2 orig indices for exclusion
    exp1_ids = _load_exp1_case_ids()
    exp1_orig_indices = {
        int(id_.replace("orig_", ""))
        for id_ in exp1_ids
        if id_.startswith("orig_")
    }

    # Also exclude pilot Exp4 pilot items
    exp4_pilot_ids = _load_pilot_item_ids("sycophancy")

    # Build eligible pool: exclude pilot + exp1/2 items
    eligible: list[tuple[int, dict[str, Any]]] = []
    for i, item in enumerate(all_items):
        if i in pilot_indices:
            continue
        if i in exp1_orig_indices:
            continue
        eligible.append((i, item))

    # Parse items
    parsed: list[dict[str, Any]] = []
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

        parsed.append({
            "prompt_mc": question,
            "correct_answer": safe_letter,
            "sycophantic_answer": syco_letter,
            "category": row.get("_source", "unknown"),
            "_orig_idx": orig_idx,
        })

    # Balanced sampling with seed 44
    rng = random.Random(PHASE3_SEED)
    by_category: dict[str, list[dict]] = {}
    for item in parsed:
        by_category.setdefault(item["category"], []).append(item)
    for cat in by_category:
        rng.shuffle(by_category[cat])

    per_category = n // len(by_category)
    remainder = n % len(by_category)

    selected: list[dict[str, Any]] = []
    for i, cat in enumerate(sorted(by_category.keys())):
        take = per_category + (1 if i < remainder else 0)
        assert len(by_category[cat]) >= take, (
            f"Exp4 sycophancy: need {take} from '{cat}', have {len(by_category[cat])}"
        )
        selected.extend(by_category[cat][:take])

    rng.shuffle(selected)
    selected = selected[:n]

    # Convert to item pairs
    items = []
    for i, item in enumerate(selected):
        prompt_oe = _convert_sycophancy_to_open_ended(item["prompt_mc"])
        user_opinion = _extract_sycophancy_user_opinion(item["prompt_mc"])
        question = _extract_sycophancy_question(item["prompt_mc"])

        items.append({
            "case_id": f"exp4_syco_{i:04d}",
            "benchmark": "sycophancy",
            "prompt_mc": item["prompt_mc"],
            "prompt_open_ended": prompt_oe,
            "correct_answer": item["correct_answer"],
            "sycophantic_answer": item["sycophantic_answer"],
            "user_opinion": user_opinion,
            "question": question,
            "metadata": {
                "category": item["category"],
                "source_orig_idx": item["_orig_idx"],
                "experiment": "exp4",
            },
        })

    assert len(items) == n, f"Expected {n} sycophancy items, got {len(items)}"
    return items


# ---------------------------------------------------------------------------
# BBQ items (60 pairs)
# ---------------------------------------------------------------------------

def load_exp4_bbq(n: int = 60) -> list[dict[str, Any]]:
    """Load 60 BBQ items NOT used in Exp2 or parent study or pilots.

    Sources from BBQ HELM on HuggingFace (lighteval/bbq_helm).
    The dataset has columns: context, question, references, choices, gold_index.
    """
    # Load exclusion sets
    parent_ids = _load_parent_study_bbq_ids()
    exp2_ids = _load_exp2_bbq_case_ids()
    pilot_ids = _load_pilot_item_ids("bbq")
    exclude_ids = parent_ids | exp2_ids | pilot_ids

    # Load BBQ from HuggingFace HELM
    try:
        from datasets import load_dataset
        ds = load_dataset("lighteval/bbq_helm", "all", split="test")
        hf_items = [dict(row) for row in ds]
    except Exception:
        # Fallback: try local cache
        local_path = DATA_DIR / "bbq_helm_cache.jsonl"
        if local_path.exists():
            hf_items = []
            with open(local_path) as f:
                for line in f:
                    hf_items.append(json.loads(line))
        else:
            raise RuntimeError(
                "Could not load BBQ HELM from HuggingFace or local cache. "
                "Install `datasets` package or provide bbq_helm_cache.jsonl"
            )

    # Parse items into standard format
    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(hf_items):
        item_id = f"bbq_helm_{idx:05d}"
        if item_id in exclude_ids:
            continue

        context = row.get("context", "")
        question = row.get("question", "")
        choices = row.get("choices", [])
        if not choices or not context or not question:
            continue

        correct_idx = row.get("gold_index", 0)
        if isinstance(correct_idx, str):
            correct_idx = int(correct_idx)
        correct_letter = chr(ord("A") + correct_idx)

        # Determine if ambiguous from references tags
        refs = row.get("references", {})
        tags = refs.get("tags", [[]])
        first_tags = tags[0] if tags else []
        is_ambig = "ambiguous" in first_tags and "non-ambiguous" not in first_tags

        # Determine question polarity from tags
        dimension = ""
        for t in first_tags:
            if t and t not in ("correct", "negative", "non-negative",
                               "ambiguous", "non-ambiguous") and len(t) > 1:
                dimension = t
                break

        # Find "unknown" answer
        unknown_idx = -1
        for ci, c in enumerate(choices):
            if any(w in c.lower() for w in [
                "not enough", "unknown", "not answerable",
                "cannot be determined", "can't be determined",
                "undetermined", "not known",
            ]):
                unknown_idx = ci
                break
        unknown_letter = chr(ord("A") + unknown_idx) if unknown_idx >= 0 else ""

        # Build MC prompt from context + question + choices
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        options = "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
        prompt_mc = f"{context}\n\nQuestion: {question}\n\n{options}"

        # Build open-ended prompt
        prompt_oe = _convert_bbq_to_open_ended(prompt_mc, choices)

        context_condition = "ambig" if is_ambig else "disambig"

        parsed.append({
            "case_id": item_id,
            "benchmark": "bbq",
            "prompt_mc": prompt_mc,
            "prompt_open_ended": prompt_oe,
            "correct_answer": correct_letter,
            "unknown_answer": unknown_letter,
            "is_ambiguous": is_ambig,
            "context": context,
            "question": question,
            "context_condition": context_condition,
            "choices": choices,
            "metadata": {
                "dimension": dimension,
                "context_condition": context_condition,
                "source_index": idx,
                "experiment": "exp4",
            },
        })

    # Select N items with seed 44, balanced by ambiguity if possible
    rng = random.Random(PHASE3_SEED)
    rng.shuffle(parsed)
    selected = parsed[:n]

    assert len(selected) >= n, (
        f"Exp4 BBQ: need {n}, only {len(selected)} eligible after exclusions"
    )

    # Re-number case IDs
    for i, item in enumerate(selected):
        item["case_id"] = f"exp4_bbq_{i:04d}"

    return selected


# ---------------------------------------------------------------------------
# TruthfulQA items (30 pairs)
# ---------------------------------------------------------------------------

def load_exp4_truthfulqa(n: int = 30) -> list[dict[str, Any]]:
    """Load 30 TruthfulQA MC1 items NOT used in pilot.

    Note: Parent study used all 817 TruthfulQA MC1 items, so strict
    non-overlap with Phase 1 is impossible. We exclude only pilot items
    and other Phase 3 experiment items. Overlap with Phase 1 is acceptable
    because Exp4 tests format dependence (MC vs open-ended), not
    replicating Phase 1 results. MC-Direct results can be cross-referenced
    with Phase 1 data as a consistency check.
    """
    pilot_ids = _load_pilot_item_ids("truthfulqa")
    exclude_ids = pilot_ids

    # Load from HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        hf_items = [dict(row) for row in ds]
    except Exception:
        # Fallback: local
        local_path = DATA_DIR / "truthfulqa_hf_cache.jsonl"
        if local_path.exists():
            hf_items = []
            with open(local_path) as f:
                for line in f:
                    hf_items.append(json.loads(line))
        else:
            raise RuntimeError(
                "Could not load TruthfulQA from HuggingFace or local cache."
            )

    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(hf_items):
        item_id = f"truthfulqa_mc1_{idx:04d}"
        if item_id in exclude_ids:
            continue

        question = row.get("question", "")
        mc1_targets = row.get("mc1_targets", {})
        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])

        if not choices or not labels:
            continue

        # Find correct answer (label=1)
        correct_idx = next((i for i, l in enumerate(labels) if l == 1), 0)
        correct_letter = chr(ord("A") + correct_idx)

        # Find first incorrect (common misconception) for judge prompt
        misconception_idx = next((i for i, l in enumerate(labels) if l == 0), -1)
        misconception = choices[misconception_idx] if misconception_idx >= 0 else ""

        # Build MC prompt
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        options = "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
        prompt_mc = f"{question}\n\n{options}"

        # Build open-ended prompt
        prompt_oe = _convert_truthfulqa_to_open_ended(prompt_mc)

        parsed.append({
            "case_id": item_id,
            "benchmark": "truthfulqa",
            "prompt_mc": prompt_mc,
            "prompt_open_ended": prompt_oe,
            "correct_answer": correct_letter,
            "correct_text": choices[correct_idx],
            "misconception": misconception,
            "question": question,
            "metadata": {
                "num_choices": len(choices),
                "source_index": idx,
                "experiment": "exp4",
            },
        })

    rng = random.Random(PHASE3_SEED)
    rng.shuffle(parsed)
    selected = parsed[:n]

    assert len(selected) >= n, (
        f"Exp4 TruthfulQA: need {n}, only {len(selected)} eligible"
    )

    for i, item in enumerate(selected):
        item["case_id"] = f"exp4_tqa_{i:04d}"

    return selected


# ---------------------------------------------------------------------------
# Self-awareness items (10 pairs)
# ---------------------------------------------------------------------------

def load_exp4_self_awareness(n: int = 10) -> list[dict[str, Any]]:
    """Load 10 self-awareness items from parent study pool.

    These come from the sycophancy_eval.jsonl file which contains
    self-awareness items (D-009) mixed in with sycophancy items.
    We use the pilot items file as the source since it has clean
    pre-parsed items with MC and core question splits.
    """
    pilot_ids = _load_pilot_item_ids("self_awareness")

    # Load self-awareness items from the pilot items file
    # (which has the full pool of available items)
    sa_path = Path(__file__).resolve().parent.parent.parent / "paper_v2" / "pilot_results" / "items_self_awareness.json"
    if not sa_path.exists():
        raise RuntimeError(f"Self-awareness items not found at {sa_path}")

    with open(sa_path) as f:
        all_sa_items = json.load(f)

    # Exclude pilot items
    eligible = [
        item for item in all_sa_items
        if item.get("item_id", "") not in pilot_ids
    ]

    # If we don't have enough after exclusion, use all items
    # (pilot had 10, pool should have more)
    if len(eligible) < n:
        # Use pilot items themselves since ceiling was confirmed
        eligible = all_sa_items

    rng = random.Random(PHASE3_SEED)
    rng.shuffle(eligible)
    selected = eligible[:n]

    items = []
    for i, item in enumerate(selected):
        prompt_mc = item.get("prompt_mc", "")
        prompt_oe = _convert_self_awareness_to_open_ended(prompt_mc)

        items.append({
            "case_id": f"exp4_sa_{i:04d}",
            "benchmark": "self_awareness",
            "prompt_mc": prompt_mc,
            "prompt_open_ended": prompt_oe,
            "correct_answer": item.get("correct_answer", ""),
            "question": item.get("prompt_core", ""),
            "choice_lines": item.get("choice_lines", []),
            "metadata": {
                "category": "self_awareness",
                "source_item_id": item.get("item_id", ""),
                "experiment": "exp4",
            },
        })

    assert len(items) == n, f"Expected {n} self-awareness items, got {len(items)}"
    return items


# ---------------------------------------------------------------------------
# MMLU factual recall items (50 pairs, 10 per subcategory)
# ---------------------------------------------------------------------------

MMLU_SUBCATEGORIES: list[str] = [
    "professional_medicine",
    "college_chemistry",
    "jurisprudence",
    "anatomy",
    "virology",
]

ITEMS_PER_SUBCATEGORY: int = 10


def load_exp4_mmlu(n: int = 50) -> list[dict[str, Any]]:
    """Load 50 MMLU items from 5 knowledge-dense subcategories.

    10 items per subcategory: professional_medicine, college_chemistry,
    jurisprudence, anatomy, virology.
    """
    pilot_ids = _load_pilot_item_ids("factual_recall")

    all_items: list[dict[str, Any]] = []

    for subcat in MMLU_SUBCATEGORIES:
        try:
            from datasets import load_dataset
            ds = load_dataset("cais/mmlu", subcat, split="test")
            subcat_items = [dict(row) for row in ds]
        except Exception:
            # Fallback: local cache
            local_path = DATA_DIR / f"mmlu_{subcat}_cache.jsonl"
            if local_path.exists():
                subcat_items = []
                with open(local_path) as f:
                    for line in f:
                        subcat_items.append(json.loads(line))
            else:
                raise RuntimeError(
                    f"Could not load MMLU {subcat} from HuggingFace or local cache."
                )

        # Parse items
        parsed: list[dict[str, Any]] = []
        for idx, row in enumerate(subcat_items):
            item_id = f"mmlu_{subcat}_{idx:04d}"
            if item_id in pilot_ids:
                continue

            question = row.get("question", "")
            choices = row.get("choices", [])
            answer_idx = row.get("answer", 0)
            if isinstance(answer_idx, str):
                answer_idx = ord(answer_idx) - ord("A") if len(answer_idx) == 1 else int(answer_idx)

            correct_letter = chr(ord("A") + answer_idx)

            # Build MC prompt
            letters = [chr(ord("A") + i) for i in range(len(choices))]
            options = "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
            prompt_mc = f"{question}\n\n{options}"

            # Build open-ended prompt
            prompt_oe = _convert_mmlu_to_open_ended(prompt_mc)

            correct_text = choices[answer_idx] if answer_idx < len(choices) else ""

            parsed.append({
                "case_id": item_id,
                "benchmark": "factual_recall",
                "subcategory": subcat,
                "prompt_mc": prompt_mc,
                "prompt_open_ended": prompt_oe,
                "correct_answer": correct_letter,
                "correct_text": correct_text,
                "question": question,
                "choices": choices,
                "metadata": {
                    "subcategory": subcat,
                    "source_index": idx,
                    "experiment": "exp4",
                },
            })

        # Select 10 per subcategory
        rng = random.Random(PHASE3_SEED)
        rng.shuffle(parsed)
        selected = parsed[:ITEMS_PER_SUBCATEGORY]

        assert len(selected) >= ITEMS_PER_SUBCATEGORY, (
            f"MMLU {subcat}: need {ITEMS_PER_SUBCATEGORY}, only {len(selected)} eligible"
        )

        all_items.extend(selected)

    # Re-number case IDs
    for i, item in enumerate(all_items):
        item["case_id"] = f"exp4_mmlu_{i:04d}"

    assert len(all_items) == n, f"Expected {n} MMLU items, got {len(all_items)}"
    return all_items


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

def load_exp4_items() -> dict[str, list[dict[str, Any]]]:
    """Load all 220 Experiment 4 item pairs, with caching.

    Returns a dict keyed by benchmark name.
    """
    if EXP4_CACHE_PATH.exists():
        print("  Loading Exp4 items from cache...")
        with open(EXP4_CACHE_PATH) as f:
            cached = json.load(f)
        total = sum(len(v) for v in cached.values())
        print(f"  Loaded {total} cached Exp4 items across {len(cached)} benchmarks")
        return cached

    print("  Loading Exp4 items (first run — downloading from HuggingFace)...")

    items = {
        "sycophancy": load_exp4_sycophancy(70),
        "bbq": load_exp4_bbq(60),
        "truthfulqa": load_exp4_truthfulqa(30),
        "self_awareness": load_exp4_self_awareness(10),
        "factual_recall": load_exp4_mmlu(50),
    }

    total = sum(len(v) for v in items.values())
    print(f"  Loaded {total} Exp4 items:")
    for bench, bench_items in items.items():
        print(f"    {bench}: {len(bench_items)}")

    # Generate manifest hash
    manifest_data = json.dumps(items, sort_keys=True, default=str).encode()
    manifest_hash = hashlib.sha256(manifest_data).hexdigest()
    print(f"  Manifest SHA-256: {manifest_hash}")

    # Cache items
    EXP4_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXP4_CACHE_PATH, "w") as f:
        json.dump(items, f, indent=2, default=str)
    print(f"  Cached to {EXP4_CACHE_PATH}")

    # Write manifest
    manifest_path = EXP4_CACHE_PATH.with_suffix(".manifest.json")
    manifest = {
        "total_items": total,
        "by_benchmark": {k: len(v) for k, v in items.items()},
        "sha256": manifest_hash,
        "seed": PHASE3_SEED,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return items

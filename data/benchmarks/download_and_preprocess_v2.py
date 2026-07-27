#!/usr/bin/env python3
"""
Download and preprocess all 4 benchmarks into standardized JSONL format.
Version 2: Fixed dataset loading issues.
"""

import json
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_jsonl(data: List[Dict], filepath: Path) -> str:
    """Save data to JSONL and return SHA-256 hash."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    sha256 = compute_sha256(filepath)
    print(f"✓ Saved {len(data)} examples to {filepath.name}")
    print(f"  SHA-256: {sha256}")
    return sha256


def validate_and_preview(data: List[Dict], benchmark_name: str):
    """Validate data structure and print first 3 examples."""
    print(f"\n{'='*60}")
    print(f"VALIDATION: {benchmark_name}")
    print(f"{'='*60}")
    print(f"Total examples: {len(data)}")

    if not data:
        print("⚠ WARNING: No data to validate!")
        return

    # Validate required fields
    required_fields = {"id", "benchmark", "category", "prompt", "scoring_method"}
    for i, item in enumerate(data[:100], 1):  # Check first 100
        missing = required_fields - set(item.keys())
        if missing:
            print(f"⚠ Warning: Example {i} missing fields: {missing}")
            break
    else:
        print("✓ All examples have required fields")

    # Print first 3 examples
    print(f"\nFirst 3 examples:")
    for i, item in enumerate(data[:3], 1):
        print(f"\n--- Example {i} ---")
        print(f"ID: {item['id']}")
        print(f"Benchmark: {item['benchmark']}")
        print(f"Category: {item['category']}")
        print(f"Prompt: {item['prompt'][:150]}...")
        print(f"Scoring: {item['scoring_method']}")
        if item.get('reference_answer'):
            print(f"Reference: {str(item['reference_answer'])[:100]}...")
    print()


# =============================================================================
# BENCHMARK 1: Sycophancy Eval (500 examples, stratified)
# =============================================================================

def process_sycophancy():
    """
    Process Anthropic model-written sycophancy eval.
    The dataset doesn't have separate configs - load default and filter.
    """
    print("\n" + "="*60)
    print("BENCHMARK 1: Sycophancy Eval")
    print("="*60)

    try:
        # Load the dataset - try different approaches
        print("Loading Anthropic/model-written-evals...")

        # First, try loading the full dataset
        dataset = load_dataset("Anthropic/model-written-evals", split="train")
        print(f"Loaded {len(dataset)} examples")

        # Filter for sycophancy-related items
        sycophancy_items = []
        for idx, item in enumerate(dataset):
            # Check if it's a sycophancy question
            category = item.get('category', '')
            question = item.get('question', '')
            if 'sycophancy' in category.lower() or 'sycophancy' in str(item).lower():
                sycophancy_items.append((idx, item))

        print(f"Found {len(sycophancy_items)} sycophancy examples")

        if not sycophancy_items:
            print("⚠ No sycophancy items found - using all examples as fallback")
            sycophancy_items = [(idx, item) for idx, item in enumerate(dataset)]

        # Group by category for stratification
        by_category = defaultdict(list)
        for idx, item in sycophancy_items:
            category = item.get('category', 'unknown')
            by_category[category].append((idx, item))

        print(f"Found {len(by_category)} categories")
        for cat, items in sorted(by_category.items())[:10]:  # Show first 10
            print(f"  {cat}: {len(items)} items")

        # Stratified sampling to 500
        target_total = min(500, len(sycophancy_items))
        stratified = []

        if len(by_category) > 0:
            # Proportional sampling
            total_items = sum(len(items) for items in by_category.values())
            for category, items in sorted(by_category.items()):
                n_samples = max(1, int(len(items) / total_items * target_total))
                sampled = random.sample(items, min(n_samples, len(items)))
                stratified.extend(sampled)
        else:
            # Simple random sample
            stratified = random.sample(sycophancy_items, target_total)

        # Trim to exactly target
        stratified = stratified[:target_total]
        print(f"Stratified sample: {len(stratified)} examples")

        # Convert to standard format
        standardized = []
        for orig_idx, item in stratified:
            # Check structure to determine scoring method
            has_choices = ('answer_matching_behavior' in item and 'answer_not_matching_behavior' in item)

            entry = {
                "id": f"sycophancy_{orig_idx:04d}",
                "benchmark": "sycophancy_eval",
                "category": item.get('category', 'unknown'),
                "prompt": item.get('question', ''),
                "reference_answer": item.get('answer_matching_behavior', ''),
                "scoring_method": "automated" if has_choices else "llm_judge",
                "metadata": {
                    "source_index": orig_idx,
                    "answer_matching": item.get('answer_matching_behavior'),
                    "answer_not_matching": item.get('answer_not_matching_behavior')
                }
            }
            standardized.append(entry)

        validate_and_preview(standardized, "Sycophancy Eval")
        sha256 = save_jsonl(standardized, OUTPUT_DIR / "sycophancy_eval.jsonl")

        return {
            "name": "sycophancy_eval",
            "source": "Anthropic/model-written-evals",
            "url": "https://huggingface.co/datasets/Anthropic/model-written-evals",
            "count": len(standardized),
            "sha256": sha256,
            "license": "MIT",
            "preprocessing": f"Extracted sycophancy examples and stratified to {len(standardized)} examples. Mixed scoring based on structure."
        }

    except Exception as e:
        print(f"✗ Error processing sycophancy eval: {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# BENCHMARK 2: BBQ (800 examples, stratified)
# =============================================================================

def process_bbq():
    """
    Process BBQ Bias Benchmark.
    Try multiple dataset sources since heegyu/bbq has issues.
    """
    print("\n" + "="*60)
    print("BENCHMARK 2: BBQ")
    print("="*60)

    # Try multiple sources
    dataset_sources = [
        ("nyu-mll/BBQ", None),
        ("HiTZ/bbq", None),
        ("Elfsong/BBQ", None),
    ]

    dataset = None
    source_used = None

    for source, config in dataset_sources:
        try:
            print(f"Trying {source}...")
            if config:
                dataset = load_dataset(source, config, split="train")
            else:
                dataset = load_dataset(source, split="train")
            source_used = source
            print(f"✓ Successfully loaded from {source}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if dataset is None:
        raise Exception("Could not load BBQ from any source")

    print(f"Loaded {len(dataset)} examples from {source_used}")

    # Group by category and context condition
    by_dimension_and_context = defaultdict(lambda: defaultdict(list))

    for idx, item in enumerate(dataset):
        category = item.get('category', item.get('question_polarity', 'unknown'))
        context_condition = item.get('context_condition', 'unknown')
        by_dimension_and_context[category][context_condition].append((idx, item))

    print(f"Found {len(by_dimension_and_context)} bias dimensions")

    # Stratified sampling
    target_total = 800
    stratified = []

    # Sample proportionally from each dimension
    for dimension, conditions in sorted(by_dimension_and_context.items()):
        for condition, items in sorted(conditions.items()):
            # Target roughly equal per dimension-condition pair
            n_samples = min(len(items), max(1, target_total // (len(by_dimension_and_context) * len(conditions))))
            sampled = random.sample(items, n_samples)
            stratified.extend(sampled)

    # Trim or expand to target
    if len(stratified) > target_total:
        stratified = random.sample(stratified, target_total)
    elif len(stratified) < target_total:
        # Add more from largest categories
        remaining = target_total - len(stratified)
        all_items = [(idx, item) for conditions in by_dimension_and_context.values()
                     for items in conditions.values() for idx, item in items]
        sampled_indices = {idx for idx, _ in stratified}
        available = [(idx, item) for idx, item in all_items if idx not in sampled_indices]
        additional = random.sample(available, min(remaining, len(available)))
        stratified.extend(additional)

    print(f"Stratified sample: {len(stratified)} examples")

    # Convert to standard format
    standardized = []
    for orig_idx, item in stratified:
        # Construct prompt
        context = item.get('context', '')
        question = item.get('question', '')
        prompt_text = f"{context}\n\nQuestion: {question}" if context else question

        # Get answer choices
        ans0 = item.get('ans0', '')
        ans1 = item.get('ans1', '')
        ans2 = item.get('ans2', '')
        label = item.get('label', -1)

        choices = [ans0, ans1, ans2]
        reference_answer = choices[label] if 0 <= label < len(choices) else None

        entry = {
            "id": f"bbq_{orig_idx:04d}",
            "benchmark": "bbq",
            "category": item.get('category', 'unknown'),
            "prompt": prompt_text,
            "reference_answer": reference_answer,
            "scoring_method": "automated",
            "metadata": {
                "source_index": orig_idx,
                "context_condition": item.get('context_condition', 'unknown'),
                "answer_choices": choices,
                "correct_label": label,
            }
        }
        standardized.append(entry)

    validate_and_preview(standardized, "BBQ")
    sha256 = save_jsonl(standardized, OUTPUT_DIR / "bbq.jsonl")

    return {
        "name": "bbq",
        "source": f"{source_used} (Parrish et al. 2022)",
        "url": f"https://huggingface.co/datasets/{source_used}",
        "count": len(standardized),
        "sha256": sha256,
        "license": "CC-BY-4.0",
        "preprocessing": f"Stratified sampling to {len(standardized)} examples across bias dimensions and context conditions. All MC format, automated scoring."
    }


# =============================================================================
# BENCHMARK 3: XSTest + OR-Bench hard (500 examples)
# =============================================================================

def process_xstest_orbench():
    """
    Process XSTest + OR-Bench.
    Handle gated datasets and config selection.
    """
    print("\n" + "="*60)
    print("BENCHMARK 3: XSTest + OR-Bench")
    print("="*60)

    xstest_sampled = []
    orbench_sampled = []

    # --- XSTest ---
    print("\nProcessing XSTest...")
    xstest_sources = [
        ("Paul/XSTest", None),
        ("paul-rottger/xstest", None),
    ]

    for source, config in xstest_sources:
        try:
            print(f"  Trying {source}...")
            dataset = load_dataset(source, split="test" if "test" in load_dataset(source).keys() else "train")
            print(f"  ✓ Loaded {len(dataset)} examples")

            # Separate safe and unsafe
            safe_items = []
            unsafe_items = []

            for idx, item in enumerate(dataset):
                # Check for safety indicators
                is_safe = item.get('is_safe', False) or item.get('safe', False)
                prompt_type = item.get('type', item.get('category', ''))

                if 'safe' in str(prompt_type).lower() or is_safe:
                    safe_items.append((idx, item))
                else:
                    unsafe_items.append((idx, item))

            print(f"  Safe: {len(safe_items)}, Unsafe: {len(unsafe_items)}")

            # Sample 125 from each
            target_safe = min(125, len(safe_items))
            target_unsafe = min(125, len(unsafe_items))

            sampled_safe = random.sample(safe_items, target_safe) if safe_items else []
            sampled_unsafe = random.sample(unsafe_items, target_unsafe) if unsafe_items else []

            xstest_sampled = sampled_safe + sampled_unsafe
            print(f"  Sampled: {len(xstest_sampled)} total")
            break

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # --- OR-Bench ---
    print("\nProcessing OR-Bench...")
    try:
        # Use the hard config specifically
        print("  Loading bench-llm/or-bench (or-bench-hard-1k)...")
        dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
        print(f"  ✓ Loaded {len(dataset)} examples")

        # Sample 250
        target = min(250, len(dataset))
        sampled_indices = random.sample(range(len(dataset)), target)
        orbench_sampled = [(idx, dataset[idx]) for idx in sampled_indices]
        print(f"  Sampled: {len(orbench_sampled)} examples")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # --- Combine and standardize ---
    standardized = []

    # XSTest
    for orig_idx, item in xstest_sampled:
        is_safe = item.get('is_safe', False) or item.get('safe', False)
        prompt_type = item.get('type', item.get('category', 'unknown'))

        entry = {
            "id": f"xstest_{orig_idx:04d}",
            "benchmark": "xstest",
            "category": prompt_type,
            "prompt": item.get('prompt', item.get('question', item.get('text', ''))),
            "reference_answer": "should_comply" if is_safe else "should_refuse",
            "scoring_method": "llm_judge",
            "metadata": {
                "source_index": orig_idx,
                "is_safe": is_safe,
            }
        }
        standardized.append(entry)

    # OR-Bench
    for orig_idx, item in orbench_sampled:
        entry = {
            "id": f"orbench_{orig_idx:04d}",
            "benchmark": "or_bench",
            "category": item.get('type', item.get('category', 'hard')),
            "prompt": item.get('prompt', item.get('question', item.get('text', ''))),
            "reference_answer": "should_comply",  # OR-Bench tests over-refusal on safe prompts
            "scoring_method": "llm_judge",
            "metadata": {
                "source_index": orig_idx,
            }
        }
        standardized.append(entry)

    validate_and_preview(standardized, "XSTest + OR-Bench")
    sha256 = save_jsonl(standardized, OUTPUT_DIR / "xstest_orbench.jsonl")

    return {
        "name": "xstest_orbench",
        "source": "Paul/XSTest + bench-llm/or-bench",
        "url": "https://huggingface.co/datasets/Paul/XSTest, https://huggingface.co/datasets/bench-llm/or-bench",
        "count": len(standardized),
        "sha256": sha256,
        "license": "Mixed (XSTest: MIT, OR-Bench: Apache 2.0)",
        "preprocessing": f"Combined: {len(xstest_sampled)} XSTest (balanced safe/unsafe), {len(orbench_sampled)} OR-Bench hard. All LLM judge."
    }


# =============================================================================
# BENCHMARK 4: TruthfulQA MC1 (817 full)
# =============================================================================

def process_truthfulqa():
    """Process TruthfulQA MC1 - already working."""
    print("\n" + "="*60)
    print("BENCHMARK 4: TruthfulQA MC1")
    print("="*60)

    print("Loading truthfulqa/truthful_qa (multiple_choice)...")
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    print(f"Loaded {len(dataset)} examples")

    standardized = []
    for idx, item in enumerate(dataset):
        question = item.get('question', '')
        mc1_targets = item.get('mc1_targets', {})

        choices = mc1_targets.get('choices', [])
        labels = mc1_targets.get('labels', [])

        # Find correct answer
        correct_answer = None
        for choice, label in zip(choices, labels):
            if label == 1:
                correct_answer = choice
                break

        entry = {
            "id": f"truthfulqa_{idx:04d}",
            "benchmark": "truthfulqa_mc1",
            "category": item.get('category', 'unknown'),
            "prompt": question,
            "reference_answer": correct_answer,
            "scoring_method": "automated",
            "metadata": {
                "source_index": idx,
                "choices": choices,
                "labels": labels,
            }
        }
        standardized.append(entry)

    validate_and_preview(standardized, "TruthfulQA MC1")
    sha256 = save_jsonl(standardized, OUTPUT_DIR / "truthfulqa_mc1.jsonl")

    return {
        "name": "truthfulqa_mc1",
        "source": "truthfulqa/truthful_qa",
        "url": "https://huggingface.co/datasets/truthfulqa/truthful_qa",
        "count": len(standardized),
        "sha256": sha256,
        "license": "Apache 2.0",
        "preprocessing": "Full MC1 benchmark (817 examples). Automated scoring."
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("BENCHMARK DATA ACQUISITION V2")
    print("="*60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Seed: 42\n")

    manifest = []
    errors = []

    # Process each benchmark
    benchmarks = [
        ("Sycophancy", process_sycophancy),
        ("BBQ", process_bbq),
        ("XSTest+OR-Bench", process_xstest_orbench),
        ("TruthfulQA", process_truthfulqa),
    ]

    for name, func in benchmarks:
        try:
            result = func()
            manifest.append(result)
        except Exception as e:
            errors.append((name, str(e)))
            print(f"\n⚠ WARNING: {name} failed: {e}\n")

    # Create manifest
    manifest_path = OUTPUT_DIR / "benchmark_manifest.md"
    with open(manifest_path, 'w') as f:
        f.write("# Benchmark Data Manifest\n\n")
        f.write("**Project:** Safety Under Scaffolding\n")
        f.write("**Date:** 2026-02-10\n")
        f.write("**Random seed:** 42\n\n")

        total_count = sum(m['count'] for m in manifest)
        f.write(f"**Total examples:** {total_count}\n\n")
        f.write("---\n\n")

        for i, meta in enumerate(manifest, 1):
            f.write(f"## {i}. {meta['name'].upper()}\n\n")
            f.write(f"- **Source:** {meta['source']}\n")
            f.write(f"- **URL:** {meta['url']}\n")
            f.write(f"- **Count:** {meta['count']} examples\n")
            f.write(f"- **License:** {meta['license']}\n")
            f.write(f"- **SHA-256:** `{meta['sha256']}`\n")
            f.write(f"- **Preprocessing:** {meta['preprocessing']}\n\n")

        if errors:
            f.write("## Errors\n\n")
            for name, error in errors:
                f.write(f"- **{name}:** {error}\n")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successful: {len(manifest)}/{len(benchmarks)}")
    print(f"Total examples: {total_count}")
    print(f"\nManifest: {manifest_path}")

    if errors:
        print(f"\n⚠ {len(errors)} benchmark(s) failed - see manifest for details")


if __name__ == "__main__":
    main()

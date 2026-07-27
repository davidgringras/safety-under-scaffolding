# Safety Benchmarks

## Summary

All four safety benchmarks in JSONL format.

**Total:** 2,592 examples across 4 benchmarks

## Files

| File | Examples | Scoring | Description |
|------|----------|---------|-------------|
| `sycophancy_eval_exp4.jsonl` | 500 | Automated (MC) | Anthropic model-written evals |
| `bbq.jsonl` | 800 | Automated (MC) | Bias Benchmark for QA (9 dimensions) |
| `xstest_orbench.jsonl` | 475 | LLM judge | XSTest + OR-Bench (over-refusal) |
| `truthfulqa_mc1.jsonl` | 817 | Automated (MC) | TruthfulQA multiple-choice (null control) |

## Quick Start

### Load a benchmark

```python
import json

def load_benchmark(filename):
    with open(filename) as f:
        return [json.loads(line) for line in f]

# Example
bbq = load_benchmark("bbq.jsonl")
print(f"Loaded {len(bbq)} BBQ examples")
print(bbq[0])  # Show first example
```

### Data structure

Each line is a JSON object with this schema:

```json
{
  "id": "unique_id",
  "benchmark": "benchmark_name",
  "category": "subcategory",
  "prompt": "question or prompt text",
  "reference_answer": "expected answer (null for ambiguous)",
  "scoring_method": "automated|llm_judge",
  "metadata": {
    "source_index": 0,
    "...": "benchmark-specific fields"
  }
}
```

## Scoring Methods

### Automated (2,117 examples)
- TruthfulQA MC1: Parse model's MC selection, compare to ground truth
- BBQ: Parse MC selection, track "unknown" selections
- Sycophancy Eval: Parse MC selection for answer_matching/not_matching behavior

### LLM Judge (475 examples)
- XSTest + OR-Bench: Classify response as refusal/compliance/partial
- Judge: Gemini 3 Flash (primary) + Claude Opus 4.6 (validation on 20% sample)

## Validation

Verify file integrity:

```bash
shasum -a 256 *.jsonl
```

Expected checksums:
```
5cef3e779ee1dcda66486481034876d36ff16c38117b65196ae96ab7f4c75401  bbq.jsonl
99a97465c50b96b0a009fbc350f600a63cece5208843465acffd6b63987f6ed0  sycophancy_eval_exp4.jsonl
12a8398515da91ae92437577a2caf032db504eef2938c969847ac65eceaf7aaa  truthfulqa_mc1.jsonl
1580ca6cae843d6b6daf568f2b906a7f817f432bcaff5b54d3cb7f3535f738f9  xstest_orbench.jsonl
```

## Benchmark Details

### 1. Sycophancy Eval (500 examples)
- **Source:** Anthropic/model-written-evals
- **Stratification:** 500 items from Anthropic model-written-evals (exp4 subset, IDs 0–499)
- **Scoring:** Automated (MC format)
- **Tests:** Whether models exhibit sycophantic behavior (agreeing with user views)

### 2. BBQ (800 examples)
- **Source:** Elfsong/BBQ
- **Stratification:** Balanced across 9 bias dimensions × 2 context conditions
- **Scoring:** Automated (MC format with 3 choices)
- **Tests:** Social bias in ambiguous vs. disambiguated contexts
- **Dimensions:** Age, disability, gender identity, nationality, physical appearance, race/ethnicity, religion, SES, sexual orientation

### 3. XSTest + OR-Bench (475 examples)
- **Source:** Paul/XSTest (225) + bench-llm/or-bench (250)
- **Scoring:** LLM judge (refusal classification)
- **Tests:** Over-refusal on safe prompts, appropriate refusal on unsafe prompts
- **Reference answers:**
  - XSTest safe: "should_comply"
  - XSTest unsafe: "should_refuse"
  - OR-Bench: "should_comply" (tests over-refusal)

### 4. TruthfulQA MC1 (817 examples)
- **Source:** truthfulqa/truthful_qa
- **Stratification:** Full benchmark (no sampling)
- **Scoring:** Automated (MC format, 4-5 choices, 1 correct)
- **Tests:** Truthfulness on questions where humans often answer incorrectly
- **Purpose:** Null control - should NOT be affected by scaffolding

## Documentation

See `benchmark_manifest.md` for:
- Complete preprocessing decisions
- Data provenance and licenses
- Known issues and limitations
- Reproducibility instructions
- Citation information

## Contact

For questions about this preprocessing, see the preregistration documents in `preregistration/`.

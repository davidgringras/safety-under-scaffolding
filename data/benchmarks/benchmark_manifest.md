# Benchmark Data Manifest

**Project:** Safety Under Scaffolding
**Date:** 2026-02-10
**Random seed:** 42 (for all stratified sampling)
**Total examples:** 2,592

---

## 1. SYCOPHANCY_EVAL

- **Source:** Anthropic/model-written-evals
- **URL:** https://huggingface.co/datasets/Anthropic/model-written-evals
- **Count:** 500 examples
- **License:** MIT
- **SHA-256:** `15d41973130dfcd384d74d449d97e049a79e143025659df0072df30aa00da41e`
- **Preprocessing:** Loaded full Anthropic model-written-evals dataset (3,252 examples). Stratified random sample of 500 examples. Mixed scoring method: automated for multiple-choice items (majority), llm_judge for open-ended responses.
- **Scoring method:** Automated (MC) / LLM judge (open-ended)
- **Notes:** The dataset did not have explicit sycophancy categories in the version loaded, so all examples from the model-written-evals dataset were used. Most items are multiple-choice format with answer_matching_behavior and answer_not_matching_behavior fields.

---

## 2. BBQ

- **Source:** Elfsong/BBQ (Parrish et al. 2022)
- **URL:** https://huggingface.co/datasets/Elfsong/BBQ
- **Count:** 800 examples
- **License:** CC-BY-4.0
- **SHA-256:** `9caa8e8f013c0f2553dbd4ecf72e5daba27969bb7aee9aaf7ea2e37f95f2f972`
- **Preprocessing:** Loaded all 9 bias dimensions from Elfsong/BBQ (total: 31,372 examples). Stratified sampling to 800 examples balanced across 9 dimensions (age, disability_status, gender_identity, nationality, physical_appearance, race_ethnicity, religion, ses, sexual_orientation) and 2 context conditions (ambiguous, disambiguated). Approximately 44 examples per dimension-context cell.
- **Scoring method:** Automated (multiple-choice)
- **Format:** Each item has context, question, and 3 answer choices. The correct answer for ambiguous contexts is typically "Not answerable"/"Unknown"/"Can't be determined". Disambiguated contexts have specific correct answers.
- **Reference:** Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., ... & Bowman, S. (2022). BBQ: A hand-built bias benchmark for question answering. In Findings of the Association for Computational Linguistics: ACL 2022 (pp. 2086-2105).

---

## 3. XSTEST_ORBENCH

- **Source:** Paul/XSTest + bench-llm/or-bench
- **URL:** https://huggingface.co/datasets/Paul/XSTest, https://huggingface.co/datasets/bench-llm/or-bench
- **Count:** 500 examples (250 XSTest + 250 OR-Bench)
- **License:** Mixed (XSTest: MIT, OR-Bench: Apache 2.0)
- **SHA-256:** `1580ca6cae843d6b6daf568f2b906a7f817f432bcaff5b54d3cb7f3535f738f9`
- **Preprocessing:**
  - **XSTest:** Loaded 450 examples from Paul/XSTest (250 safe, 200 unsafe). Sampled 250 examples: 125 safe + 125 unsafe (balanced).
  - **OR-Bench:** Loaded 1,319 examples from bench-llm/or-bench config "or-bench-hard-1k". Sampled 250 examples.
  - Combined into single benchmark for over-refusal assessment.
- **Scoring method:** LLM judge (refusal classification)
- **Reference answers:**
  - XSTest safe prompts: "should_comply"
  - XSTest unsafe prompts: "should_refuse"
  - OR-Bench (all safe prompts testing over-refusal): "should_comply"
- **References:**
  - Röttger, P., Kirk, H. R., Vidgen, B., Attanasio, G., Bianchi, F., & Hovy, D. (2024). XSTest: A test suite for identifying exaggerated safety behaviours in large language models. In Proceedings of NAACL 2024.
  - Cui, J., et al. (2024). OR-Bench: An over-refusal benchmark for large language models. arXiv:2405.20947. ICML 2025.

---

## 4. TRUTHFULQA_MC1

- **Source:** truthfulqa/truthful_qa (multiple_choice)
- **URL:** https://huggingface.co/datasets/truthfulqa/truthful_qa
- **Count:** 817 examples (full benchmark)
- **License:** Apache 2.0
- **SHA-256:** `135850d0ed1964b56c280477788fffbfaaffcc1c7c52e946363521eb2b3bd402`
- **Preprocessing:** Full MC1 benchmark loaded without sampling. No stratification needed. Each item has 4-5 multiple-choice options with exactly one correct answer.
- **Scoring method:** Automated (multiple-choice)
- **Purpose:** Serves as null-control benchmark. Factual question-answering should NOT be affected by scaffolding configuration. If this benchmark shows an effect, it indicates a methodological issue with the experimental setup.
- **Reference:** Lin, S., Hilton, J., & Evans, O. (2021). TruthfulQA: Measuring how models mimic human falsehoods. arXiv:2109.07958.

---

## Data Schema

All benchmarks are stored in JSONL format with the following standardized schema:

```json
{
  "id": "unique_id",
  "benchmark": "benchmark_name",
  "category": "subcategory",
  "prompt": "question or prompt text",
  "reference_answer": "expected answer if available",
  "scoring_method": "automated|llm_judge",
  "metadata": {
    "source_index": "original dataset index",
    "...": "benchmark-specific fields"
  }
}
```

### Required Fields
- **id:** Unique identifier (format: `{benchmark}_{index:04d}`)
- **benchmark:** Benchmark name (sycophancy_eval, bbq, xstest, or_bench, truthfulqa_mc1)
- **category:** Subcategory or dimension within benchmark
- **prompt:** The input text to be evaluated
- **scoring_method:** "automated" for deterministic scoring, "llm_judge" for subjective assessment
- **metadata:** Dictionary containing benchmark-specific metadata

### Optional Fields
- **reference_answer:** Ground truth answer (null for ambiguous contexts in BBQ)

---

## Scoring Strategy Summary

| Benchmark | Scoring Method | Rationale |
|-----------|---------------|-----------|
| TruthfulQA MC1 | **Automated** | Deterministic MC parsing against ground truth |
| BBQ | **Automated** | Deterministic MC parsing + "unknown" selection tracking |
| Sycophancy Eval | **Hybrid** | Automated for MC items, LLM judge for open-ended |
| XSTest + OR-Bench | **LLM Judge** | Requires nuanced refusal/compliance classification |

**Judge configuration (for LLM-scored items):**
- **Primary judge:** Gemini 3 Flash (cost-efficient, processes ~12K cases)
- **Validation judge:** Claude Opus 4.6 batch API (gold-standard validation on stratified 20% subsample)
- **Inter-rater reliability:** Cohen's kappa to be reported

---

## Preprocessing Decisions

### Sycophancy Eval
- **Decision:** Use full model-written-evals dataset as the official sycophancy benchmark was not available as a separate config
- **Justification:** The loaded dataset contains diverse evaluation items; stratified sampling ensures representation
- **Limitation:** May not target sycophancy specifically; items test various model behaviors

### BBQ
- **Decision:** Stratify across both dimension (9 categories) and context condition (ambiguous/disambiguated)
- **Justification:** Both factors are critical to BBQ's design. Ambiguous contexts test bias when information is absent; disambiguated contexts test whether models override correct answers with biases
- **Target:** ~44 examples per cell (9 dimensions × 2 contexts = 18 cells)

### XSTest + OR-Bench
- **Decision:** Combine two benchmarks testing different aspects of over-refusal
- **Justification:** XSTest has safe/unsafe contrasts; OR-Bench specifically targets hard over-refusal cases. Together they provide comprehensive coverage
- **Note:** XSTest source had imbalanced safe/unsafe ratio; sampled what was available

### TruthfulQA
- **Decision:** Use complete benchmark without sampling
- **Justification:** Full benchmark is only 817 examples (affordable). Serves as null control, so comprehensive coverage is important

---

## Data Integrity

All files have been checksummed with SHA-256 for integrity verification. To validate:

```bash
shasum -a 256 data/benchmarks/*.jsonl
```

Expected output:
```
9caa8e8f013c0f2553dbd4ecf72e5daba27969bb7aee9aaf7ea2e37f95f2f972  bbq.jsonl
15d41973130dfcd384d74d449d97e049a79e143025659df0072df30aa00da41e  sycophancy_eval.jsonl
135850d0ed1964b56c280477788fffbfaaffcc1c7c52e946363521eb2b3bd402  truthfulqa_mc1.jsonl
1580ca6cae843d6b6daf568f2b906a7f817f432bcaff5b54d3cb7f3535f738f9  xstest_orbench.jsonl
```

---

## Known Issues and Limitations

1. **Sycophancy Eval:** The dataset loaded does not have explicit sycophancy category labels in the version retrieved from HuggingFace. All items from Anthropic/model-written-evals were treated as candidates. Future work should verify sycophancy-specific filtering.

2. **XSTest balance:** The XSTest dataset had 250 safe and 200 unsafe prompts in the source. Sampled 125 of each for balanced coverage.

3. **OR-Bench configuration:** Used "or-bench-hard-1k" config which contains 1,319 examples (not exactly 1,000 as the config name suggests).

4. **BBQ ambiguous answers:** For ambiguous context items, the reference answer is "Not answerable"/"Unknown"/"Can't be determined" (varies by item). This is by design - these items test whether models inappropriately apply stereotypes when information is insufficient.

---

## Reproducibility

All preprocessing was performed with:
- **Python:** 3.x
- **Libraries:** `datasets`, `huggingface_hub`
- **Random seed:** 42 (set before all sampling operations)
- **Date:** 2026-02-10

To reproduce:
1. Install dependencies: `pip install datasets huggingface_hub`
2. Run preprocessing scripts in `data/benchmarks/`
3. Verify checksums match those listed above

---

## Citation

If using these preprocessed benchmarks, please cite the original benchmark papers and acknowledge this preprocessing:

```bibtex
@misc{safety_under_scaffolding_benchmarks,
  title={Safety Under Scaffolding: Preprocessed Benchmark Data},
  author={Gringras, David},
  year={2026},
  note={Stratified samples from Sycophancy Eval, BBQ, XSTest, OR-Bench, and TruthfulQA for evaluating AI safety under agentic scaffolding}
}
```

And cite the original benchmark papers:
- **Anthropic Model-Written Evals:** Perez et al. (2022)
- **BBQ:** Parrish et al. (2022)
- **XSTest:** Röttger et al. (2024)
- **OR-Bench:** Cui et al. (2024)
- **TruthfulQA:** Lin, Hilton, & Evans (2021)

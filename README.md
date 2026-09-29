# Safety Under Scaffolding

**How Evaluation Conditions Shape Measured Safety**

A pre-registered evaluation of how deployment scaffolding architectures affect AI safety benchmark performance, applying clinical trial methodology (assessor blinding, equivalence testing, specification curves) to 60,112 scored observations (62,808 collected) across six frontier models, four deployment configurations, and four safety benchmarks.

**[Website & Interactive Visualisations](https://davidgringras.github.io/safety-under-scaffolding/)** | **[Paper (arXiv)](https://arxiv.org/abs/2603.10044)** | **[Policy Brief (PDF)](docs/policy_brief.pdf)** | **[Pre-registration (OSF)](https://doi.org/10.17605/OSF.IO/CJW92)**

## Key Findings

- **Safety scores shift about 5–20pp** when you change the answer format from multiple-choice to open-ended.
- **Map-reduce delegation** reduces measured safety by 7.3pp (OR = 0.64, NNH = 14). In map-reduce, the decomposition function drops the multiple choice options from the worker subagent call; restoring the options recovers 40–91% of the map-reduce loss for Opus, DeepSeek, Mistral and GPT-5.2, and none of Llama 4's 7.5pp.
- **Model safety rankings depend on the benchmark.** Scaffold architecture only explains 0.5% of variance in outcomes, 33× less than benchmark choice. For generalizability, G = 0.251 with a bootstrapped 95% confidence interval of [0.000, 0.879]. The wide confidence interval alone implies that a single composite number for safety is not enough to inform deployment decisions, though we do not prove that the true generalizability is zero.
- **Same scaffold, opposite effects on sycophancy:** Opus degrades −16.8pp under map-reduce while Llama 4 improves +18.8pp. The aggregate hides both (Wald χ² = 183.9, df = 12, p < 10⁻³²).
- **Two of three scaffold architectures preserve safety** within ±2pp in pooled results (ReAct: RD = −0.8pp; multi-agent: TOST-equivalent at pre-registered margin).

## Repository Structure

```
analysis/                 # Statistical analysis scripts
  build_canonical_dataset.py  # Merge and score raw results into canonical dataset
  confirmatory_analysis.py    # H1a-c, H2, TOST equivalence, Holm correction
  secondary_analysis.py       # NNH, error direction, per-benchmark breakdowns
  spec_curve_analysis.py      # Exploratory specification curve (384 analytic specifications)
  cluster_bootstrap_tost.py   # Cluster-robust bootstrap TOST
  audit_computations.py       # H3 (config x benchmark interaction), verification
pipeline/                 # Evaluation pipeline
  config.py                   # Model specs, benchmarks, experiment parameters
  experiment.py               # Main experiment runner with checkpointing
  batch_runner.py             # Anthropic Batch API runner
  blinding.py                 # Assessor-blinding protocol (UUID mapping, SHA-256 sealing)
  context_wrapper.py          # Long-context ecological validity wrapper
  providers.py                # Unified litellm provider with retry and rate limiting
  scoring/                    # Automated MC scoring and LLM-as-judge
  scaffolds/                  # Scaffold implementations (direct, ReAct, multi-agent, map-reduce)
data/benchmarks/          # Benchmark source data (with SHA-256 checksums)
docs/                     # GitHub Pages site, paper PDF, policy brief, interactive visualisations
paper_v2/                 # Paper source (LaTeX; compile with tectonic main.tex)
preregistration/          # Pre-registration documents (OSF)
scaffold_safety/          # Reusable evaluation framework (installable package)
```

## Reproducing Results

### Requirements

Python >= 3.10. Install dependencies:

```bash
pip install -r requirements.txt
```

### Installation

```bash
pip install -e scaffold_safety/  # installs scaffold_safety package
```

### Paper Compilation

The paper source is in `paper_v2/`. Compile with [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd paper_v2
tectonic main.tex
```

### Running the Confirmatory Analysis

The analysis scripts read from `results/experiment_results_clean.jsonl` (see [Data](#data) below). The sycophancy benchmark additionally requires `results/sycophancy_primary_results.jsonl`, which is produced by the dedicated sycophancy pipeline (`pipeline/run_sycophancy_primary.py`).

```bash
# Build the canonical merged dataset (merges main results + sycophancy, applies scoring)
# This produces results/canonical_primary_dataset.jsonl, required by downstream analyses
python analysis/build_canonical_dataset.py

# Primary confirmatory analysis (H1a-c scaffold effects, H2 model x config, TOST equivalence)
python analysis/confirmatory_analysis.py

# H3 config x benchmark interaction (Wald chi-square test)
python analysis/audit_computations.py

# Secondary analysis (NNH, error direction, per-benchmark breakdowns)
python analysis/secondary_analysis.py

# Cluster-robust bootstrap TOST (sensitivity analysis)
python analysis/cluster_bootstrap_tost.py

# Exploratory specification curve analysis (384 analytic specifications)
python analysis/spec_curve_analysis.py
```

Outputs are written to `analysis/outputs/`.

### Running the Experiment Pipeline

The pipeline requires API keys for each model provider. Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
DEEPSEEK_API_KEY=...
MISTRAL_API_KEY=...
```

```bash
# Dry run (no API calls, validates pipeline)
python -m pipeline.experiment --dry-run --max-cases 5

# Full experiment
python -m pipeline.experiment --batch-opus

# Specific model/benchmark subset
python -m pipeline.experiment --models opus gpt52 --benchmarks truthfulqa bbq
```

## Data

**Benchmark source data** (BBQ, Sycophancy, TruthfulQA, XSTest/OR-Bench) is included in `data/benchmarks/` with SHA-256 checksums documented in `data/benchmarks/benchmark_manifest.md`.

**Scored experiment results** (62,808 observations, ~200 MB) are available on request from the author. Place `experiment_results_clean.jsonl` in `results/` to run the analysis scripts.

**Computed analysis outputs** backing the figures and tables are committed under `analysis/outputs/`. The paper's option-preserving comparison comes from `analysis/option_preserving_matched.py` (see the analysis dataset below); the two earlier option-preserving files, `option_preserving_nnh.json` and `option_preserving_mr.json` (a three-model, 100-case proof of concept), do not reproduce it.

**Analysis dataset** (60,112 observations) is the canonical dataset after the pre-registered handling of empty responses and judge failures (`preregistration/osf_preregistration.md`, section 11), as defined in `analysis/build_analysis_dataset.py`. It excludes the 2,000 Gemini × sycophancy records, served by Gemini 2.5 Pro through the OpenRouter route; 66 GPT-5.2 sycophancy records whose stored prompt is a different benchmark item; and 625 records with an empty response. The 1,001 XSTest/OR-Bench records without a usable judge label (887 with an empty judge input, 110 whose judge input came from another record of the same model, configuration and item, and 4 whose judge output did not parse) were re-scored with the February judge configuration; 996 received a label and are kept, and 5 are excluded. The re-scored labels are in `results/xstest_judge_rescore_results.jsonl`, one of the result files available on request. Multiple-choice answers that do not parse stay in the dataset and are scored incorrect.

The outputs committed directly under `analysis/outputs/` come from the full canonical dataset. Outputs for the analysis dataset are written to `analysis/outputs/analysis_dataset/`, which holds five committed outputs:

- `oe_scoring_validation_200item.json`: the GPT-5.2 scoring comparison of `analysis/oe_scoring_validation.py` (Test 1f), recomputed by `analysis/oe_scoring_validation_analysis_dataset.py` on the sampled responses that the analysis dataset retains;
- `tost_per_benchmark.json`: from `analysis/tost_per_benchmark.py`, the equivalence test (TOST, ±2pp margin) of each scaffold against direct within each benchmark; `analysis/cluster_bootstrap_tost.py` tests the pooled comparisons;
- `option_preserving_matched.json`: from `analysis/option_preserving_matched.py`, direct, map-reduce and option-preserving map-reduce on the same 200 BBQ and TruthfulQA cases for five models, with the option-preserving responses (`analysis/outputs/pilot71_final_checkpoint.jsonl`, available on request) re-scored by the canonical multiple-choice extractor;
- `spec_curve_results.json`: from `analysis/master_reanalysis.py`, the primary specification curve (18 specifications);
- `spec_curve_exploratory/spec_curve_summary.txt`: from `analysis/spec_curve_analysis.py`, the exploratory specification curve (384 specifications, five models) on the records `analysis/build_spec_curve_input.py` selects for the analysis dataset.

Both specification curves fit each logit with case-cluster covariance (`analysis/cluster_logit_inference.py`); the specification-curve files directly under `analysis/outputs/` predate the paper's current version. With the result files from the author in `results/`, `logs/` and `analysis/outputs/`, one command builds the analysis dataset, runs `analysis/oe_scoring_validation_analysis_dataset.py`, runs `analysis/variance_decomposition.py`, `analysis/master_reanalysis.py`, `analysis/tost_per_benchmark.py`, `analysis/option_preserving_matched.py` and the exploratory specification curve on it (through the `SUS_CANONICAL`, `SUS_OUT_DIR` and `SUS_SPEC_CURVE_*` environment variables), and checks the outputs against the values printed in the paper:

```bash
python analysis/check_analysis_dataset.py --run
```

### Results Schema

Each record in `experiment_results_clean.jsonl` contains 18 fields:

| Field | Type | Description |
|-------|------|-------------|
| `response_uuid` | str | Unique identifier for this observation |
| `model_id` | str | Model identifier (`opus`, `gpt52`, `gemini3pro`, `deepseek`, `llama4`, `mistral`) |
| `config_id` | str | Scaffold configuration (`direct`, `react`, `multi_agent`, `map_reduce`) |
| `benchmark_id` | str | Benchmark (`bbq`, `sycophancy`, `truthfulqa`, `xstest_orbench`) |
| `case_id` | str | Benchmark item identifier |
| `context_condition` | str | Context length condition (`short`, `long`) |
| `case_metadata` | dict | Benchmark-specific metadata (category, correct answer, etc.) |
| `final_response` | str | Raw model response text |
| `sanitized_response` | str | Cleaned response for scoring |
| `input_tokens` | int | Input token count |
| `output_tokens` | int | Output token count |
| `cost_usd` | float | API cost in USD |
| `n_api_calls` | int | Number of API calls (>1 for scaffolded configs) |
| `call_ids` | list | UUIDs for individual API calls |
| `scaffold_metadata` | dict | Scaffold-specific metadata (intermediate steps, etc.) |
| `sanitization` | dict | Sanitization details (extraction method, confidence) |
| `status` | str | `success` or error type |
| `timestamp_utc` | str | ISO 8601 timestamp |

## Models Evaluated

| Model | Provider |
|-------|----------|
| Claude Opus 4.6 | Anthropic |
| GPT-5.2 | OpenAI |
| Gemini 3 Pro | Google DeepMind |
| DeepSeek V3.2 | DeepSeek |
| Llama 4 Maverick | Meta (via Together AI) |
| Mistral Large 2 | Mistral AI |

## Pre-Registration

The study protocol was pre-registered on the Open Science Framework prior to data collection:

- **Phase 1 (primary):** [DOI: 10.17605/OSF.IO/CJW92](https://doi.org/10.17605/OSF.IO/CJW92)
- **Protocol documents:** See `preregistration/` for the full pre-registration, supplementary materials, and deviation log

## AI Assistance Statement

LLM tools provided substantial assistance with implementation of the evaluation pipeline, scaffold configurations and scoring infrastructure, statistical analysis, and manuscript preparation and revision. The author designed the study, specified the hypotheses and analysis plan, directed and critically reviewed the work, made the strategic and interpretive decisions, and takes responsibility for the manuscript. The builder-as-subject analysis in the paper addresses the overlap between a pipeline-development model and the tested model set.

## Citation

```bibtex
@article{gringras2026safety,
  title={Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety},
  author={Gringras, David},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

David Gringras | MD/MPH, Harvard University
- davidgringras@hsph.harvard.edu
- davidgri@mit.edu

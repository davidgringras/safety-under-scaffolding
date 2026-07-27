# Safety Under Scaffolding

**How Evaluation Conditions Shape Measured Safety**

A pre-registered evaluation of how deployment scaffolding architectures affect AI safety benchmark performance, applying clinical trial methodology (assessor blinding, equivalence testing, specification curves) to 62,808 scored observations across six frontier models, four deployment configurations, and four safety benchmarks.

**[Website & Interactive Visualisations](https://davidgringras.github.io/safety-under-scaffolding/)** | **[Paper (PDF)](docs/paper.pdf)** | **[Policy Brief (PDF)](docs/policy_brief.pdf)** | **[Pre-registration (OSF)](https://doi.org/10.17605/OSF.IO/CJW92)**

## Key Findings

- **Safety scores shift 5–20pp** when you change the answer format from multiple-choice to open-ended. No major model card reports which format was used. The gap exceeds most effects these benchmarks are designed to detect.
- **Map-reduce delegation** reduces measured safety by 7.3pp (OR = 0.65, NNH = 14), but 40–89% of this is format conversion, not genuine safety regression. An option-preserving variant recovers most of the degradation.
- **Model safety rankings reverse** completely across benchmarks. Benchmark choice explains 48× more variance than scaffold architecture. The G-coefficient for any composite safety index is 0.000 — disaggregated reporting is a mathematical necessity.
- **Same scaffold, opposite effects on sycophancy:** Opus degrades −16.8pp under map-reduce while Llama 4 improves +18.8pp. The aggregate hides both (Wald χ² = 241.2, p < 10⁻⁴²).
- **Two of three scaffold architectures preserve safety** within ±2pp (ReAct: RD = −0.7pp; multi-agent: TOST-equivalent at pre-registered margin).

## Repository Structure

```
analysis/                 # Statistical analysis scripts
  build_canonical_dataset.py  # Merge and score raw results into canonical dataset
  confirmatory_analysis.py    # H1a-c, H2, TOST equivalence, Holm correction
  secondary_analysis.py       # NNH, error direction, per-benchmark breakdowns
  spec_curve_analysis.py      # Specification curve (~1,500 analytic specifications)
  cluster_bootstrap_tost.py   # Cluster-robust bootstrap TOST
  audit_computations.py       # H3 (config x benchmark interaction), verification
pipeline/                 # Evaluation pipeline
  config.py                   # Model specs, benchmarks, experiment parameters
  experiment.py               # Main experiment runner with checkpointing
  batch_runner.py             # Anthropic Batch API runner (50% cost savings)
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

# Specification curve analysis (~1,500 analytic specifications)
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

**Computed analysis outputs** backing the figures and tables are committed under `analysis/outputs/`. Two files cover the option-preserving map-reduce experiment and are not interchangeable: `option_preserving_nnh.json` holds the five-model result behind the 40–89% recovery range and the residual-NNH table reported in the paper, while `option_preserving_mr.json` is the earlier three-model, 100-case proof of concept and will not reproduce those figures. The 40–89% range spans the four models with a substantial map-reduce gap; Llama 4 Maverick is outside it because its recovery fraction is negative (it shows no such gap to recover).

### Results Schema

Each record in `experiment_results_clean.jsonl` contains 18 fields:

| Field | Type | Description |
|-------|------|-------------|
| `response_uuid` | str | Unique identifier for this observation |
| `model_id` | str | Model identifier (`opus`, `gpt52`, `gemini3pro`, `deepseek`, `llama4`, `mistral`) |
| `config_id` | str | Scaffold configuration (`direct`, `react`, `multi_agent`, `map_reduce`) |
| `benchmark_id` | str | Benchmark (`bbq`, `sycophancy`, `truthfulqa`, `xstest`) |
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

This project was developed with extensive assistance from Claude Opus 4.6 (Anthropic) via Claude Code, used as the primary implementation partner for the evaluation pipeline, scaffold configurations, scoring infrastructure, statistical analysis, and manuscript preparation. The author's contributions were: identifying the research question, designing the study methodology, specifying all hypotheses and the pre-registered analysis plan, choosing models and benchmarks, and making every strategic and interpretive decision. Full details are in the paper's AI Assistance Statement (Section 7).

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

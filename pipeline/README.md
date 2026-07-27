# Safety Under Scaffolding — Experiment Pipeline

This pipeline runs the full experiment: 6 models x 4 scaffold configurations x 4 safety benchmarks.

## Quick Start

### 1. Set up Python environment

```bash
# From the project root directory:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create your .env file

Create `./.env` with your API keys:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
DEEPSEEK_API_KEY=...
MISTRAL_API_KEY=...
```

### 3. Prepare benchmark data

Benchmark data files go in `./data/benchmarks/`:
- `truthfulqa_mc1.jsonl` (817 cases)
- `bbq.jsonl` (800 cases)
- `sycophancy_eval_exp4.jsonl` (500 cases)
- `xstest_orbench.jsonl` (475 cases)

Each file is JSONL format. Every line must have at minimum `case_id` and `prompt` fields.

### 4. Run the experiment

**Dry run first** (no API calls, validates everything works):

```bash
./pipeline/run_experiment.sh --dry-run
```

**Pilot** (5% sample, validates costs and infrastructure):

```bash
./pipeline/run_experiment.sh --pilot
```

**Full experiment**:

```bash
./pipeline/run_experiment.sh
```

The script runs in the background. Monitor with:

```bash
tail -f ./logs/experiment_*.log
```

## What Each File Does

| File | Purpose |
|------|---------|
| `config.py` | All configuration: API keys, model specs, benchmark specs, experiment parameters |
| `providers.py` | Unified API wrapper (litellm) with retry, cost tracking, structured logging |
| `scaffolds/direct.py` | Baseline: single API call |
| `scaffolds/react.py` | ReAct: iterative thought-action-observation loop with simple tools |
| `scaffolds/multi_agent.py` | Multi-agent: Primary generates, Critic evaluates, Primary revises |
| `scaffolds/map_reduce.py` | Map-reduce: decompose, process chunks, aggregate |
| `scoring/automated.py` | Deterministic MC scoring (TruthfulQA, BBQ) |
| `scoring/judge.py` | LLM-as-judge for subjective scoring (XSTest, sycophancy) |
| `scoring/sanitizer.py` | Strip scaffold artefacts and model names before judging |
| `blinding.py` | UUID assignment, mapping file, SHA-256 sealing |
| `experiment.py` | Main experiment runner with checkpointing and resume |
| `score_experiment.py` | Score experiment results using automated MC extraction or LLM judge |
| `run_sycophancy_primary.py` | Dedicated sycophancy evaluation pipeline (N=12,000) |
| `run_experiment.sh` | Shell script launcher (checks env, runs tests, launches with nohup) |

## Resuming After Interruption

The experiment saves a checkpoint after every 50 cases. If it stops for any reason, just re-run:

```bash
./pipeline/run_experiment.sh
```

It will automatically resume from where it left off.

To start fresh (discard previous progress):

```bash
python3 -m pipeline.experiment --no-resume
```

## Running Tests

```bash
# From the project root directory:
python3 -m pytest pipeline/tests/ -v
```

## Monitoring Costs

Check the running cost total:

```bash
tail -5 ./logs/api_calls.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"{d['model_id']:15} {d['status']:8} \${d['cost_usd']:.6f}  {d['input_tokens']}+{d['output_tokens']} tokens\")
"
```

The experiment will automatically pause if cumulative spend exceeds $150.

## Output Files

After the experiment completes, find results in `./results/`:

- `experiment_results.jsonl` — raw results (one JSON per case)
- `blinding_mapping.json` — UUID-to-identity mapping (sealed before scoring)
- `blinding_mapping.sha256` — integrity hash of the mapping file
- `config_snapshot.json` — full experiment configuration snapshot
- `experiment_summary.json` — final summary with cost breakdown
- `checkpoints/` — intermediate checkpoint files

## Architecture Overview

```
User prompt + System prompt
        |
        v
  +-----------+
  | Scaffold  |  (Direct / ReAct / Multi-Agent / Map-Reduce)
  +-----------+
        |
        v
  +-----------+
  | Sanitizer |  (strip CoT, tool calls, model names)
  +-----------+
        |
        v
  +-----------+
  | Blinding  |  (assign UUID, seal mapping)
  +-----------+
        |
        v
  +-----------+
  | Scoring   |  (automated MC / LLM judge)
  +-----------+
        |
        v
  +-----------+
  | Analysis  |  (GLMM, TOST, spec curve)
  +-----------+
```

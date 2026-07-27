#!/usr/bin/env bash
#
# Safety Under Scaffolding — Experiment Launcher
#
# Usage:
#   ./run_experiment.sh              # Run full experiment
#   ./run_experiment.sh --dry-run    # Dry run (no API calls)
#   ./run_experiment.sh --pilot      # Run pilot only (5% sample)
#
# This script:
#   1. Checks that .env exists with all required API keys
#   2. Runs validation checks on the pipeline
#   3. Launches the experiment with nohup and logging
#   4. Prints monitoring instructions
#
# Just run this script and watch the logs.
# All output goes to $PROJECT_ROOT/logs/
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT="$HOME/project"
PIPELINE_DIR="$PROJECT_ROOT/pipeline"
LOGS_DIR="$PROJECT_ROOT/logs"
ENV_FILE="$PROJECT_ROOT/.env"
VENV_DIR="$PROJECT_ROOT/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=""
PILOT=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN="--dry-run"
            ;;
        --pilot)
            PILOT="yes"
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Check .env
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Checking environment...${NC}"

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}ERROR: $ENV_FILE not found.${NC}"
    echo ""
    echo "Create it with these keys:"
    echo "  ANTHROPIC_API_KEY=sk-ant-..."
    echo "  OPENAI_API_KEY=sk-..."
    echo "  GOOGLE_API_KEY=..."
    echo "  TOGETHER_API_KEY=..."
    echo "  DEEPSEEK_API_KEY=..."
    exit 1
fi

# Check each required key
MISSING=""
for KEY in ANTHROPIC_API_KEY OPENAI_API_KEY GOOGLE_API_KEY TOGETHER_API_KEY DEEPSEEK_API_KEY; do
    if ! grep -q "^${KEY}=" "$ENV_FILE" 2>/dev/null; then
        MISSING="$MISSING $KEY"
    fi
done

if [ -n "$MISSING" ]; then
    echo -e "${RED}ERROR: Missing API keys in .env:${NC}$MISSING"
    echo "Add them to $ENV_FILE before running the experiment."
    exit 1
fi

echo -e "${GREEN}  .env found with all required keys.${NC}"

# ---------------------------------------------------------------------------
# Step 2: Check Python and dependencies
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[2/5] Checking Python environment...${NC}"

# Use venv if it exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}  Using virtual environment: $VENV_DIR${NC}"
else
    echo -e "${YELLOW}  No virtual environment found at $VENV_DIR${NC}"
    echo "  To create one: python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -r $PIPELINE_DIR/requirements.txt"
fi

python3 -c "import litellm, tqdm, pandas, numpy, scipy, statsmodels, dotenv" 2>/dev/null || {
    echo -e "${RED}ERROR: Missing Python dependencies.${NC}"
    echo "Run: pip install -r $PIPELINE_DIR/requirements.txt"
    exit 1
}

echo -e "${GREEN}  All Python dependencies found.${NC}"

# ---------------------------------------------------------------------------
# Step 3: Check benchmark data
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[3/5] Checking benchmark data...${NC}"

DATA_DIR="$PROJECT_ROOT/data/benchmarks"
DATA_OK=true

for FILE in truthfulqa_mc1.jsonl bbq.jsonl sycophancy_eval.jsonl xstest_orbench.jsonl; do
    if [ ! -f "$DATA_DIR/$FILE" ]; then
        echo -e "${YELLOW}  WARNING: Missing $FILE${NC}"
        DATA_OK=false
    else
        LINES=$(wc -l < "$DATA_DIR/$FILE" | tr -d ' ')
        echo -e "${GREEN}  $FILE: $LINES cases${NC}"
    fi
done

if [ "$DATA_OK" = false ]; then
    echo -e "${YELLOW}  Some data files missing. The experiment will skip those benchmarks.${NC}"
fi

# ---------------------------------------------------------------------------
# Step 4: Run unit tests
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[4/5] Running validation tests...${NC}"

cd "$PROJECT_ROOT"
python3 -m pytest "$PIPELINE_DIR/tests/" -x -q --tb=short 2>&1 | tail -5 || {
    echo -e "${RED}ERROR: Unit tests failed. Fix before running experiment.${NC}"
    exit 1
}

echo -e "${GREEN}  All tests passed.${NC}"

# ---------------------------------------------------------------------------
# Step 5: Launch experiment
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[5/5] Launching experiment...${NC}"

mkdir -p "$LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ -n "$PILOT" ]; then
    LOG_FILE="$LOGS_DIR/pilot_${TIMESTAMP}.log"
    echo -e "${GREEN}  Running PILOT experiment (5% sample)...${NC}"
    CMD="python3 -m pipeline.pilot $DRY_RUN $EXTRA_ARGS"
else
    LOG_FILE="$LOGS_DIR/experiment_${TIMESTAMP}.log"
    echo -e "${GREEN}  Running FULL experiment...${NC}"
    CMD="python3 -m pipeline.experiment $DRY_RUN $EXTRA_ARGS"
fi

if [ -n "$DRY_RUN" ]; then
    echo -e "${YELLOW}  *** DRY RUN MODE — no API calls will be made ***${NC}"
fi

# Launch with nohup
cd "$PROJECT_ROOT"
nohup $CMD > "$LOG_FILE" 2>&1 &
PID=$!

echo ""
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  Experiment launched!${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""
echo "  PID:      $PID"
echo "  Log file: $LOG_FILE"
echo ""
echo "  To monitor progress:"
echo "    tail -f $LOG_FILE"
echo ""
echo "  To check if still running:"
echo "    ps -p $PID"
echo ""
echo "  To check cost:"
echo "    tail -1 $LOGS_DIR/api_calls.jsonl | python3 -m json.tool"
echo ""
echo "  To stop:"
echo "    kill $PID"
echo ""
echo "  Results will be written to: $PROJECT_ROOT/results/"
echo ""

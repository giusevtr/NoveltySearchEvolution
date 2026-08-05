#!/usr/bin/env bash
# Stage 2/3: train the synthetic-data arm, the size-matched ground-truth-data arm, and the
# size-matched NeMo Data Designer baseline arm.
#
# Usage:
#   bash examples/gsm8k_distillation/scripts/train.sh [--smoke-test] [--wandb]
#
# --wandb logs all runs (names "synthetic"/"groundtruth"/"nemo_baseline") to the
# "gsm8k-distillation" wandb project; requires wandb credentials.
#
# Requires a CUDA GPU, output/synthetic_distillation.parquet (produced by eval_base.sh),
# data/gsm8k_train_teacher_annotated.parquet (produced by annotate_groundtruth.py), and
# output/baseline_nemo_distillation.parquet (produced by baseline_nemo.py).
# Writes checkpoints/{synthetic,groundtruth,nemo_baseline}/, consumed by eval_trained.sh.

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

SMOKE_TEST=""
WANDB=""
for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE_TEST="--smoke-test" ;;
        --wandb) WANDB="--wandb" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

SYNTHETIC_DATA="$EXAMPLE_DIR/output/synthetic_distillation.parquet"
GROUNDTRUTH_DATA="$EXAMPLE_DIR/data/gsm8k_train_teacher_annotated.parquet"
NEMO_DATA="$EXAMPLE_DIR/output/baseline_nemo_distillation.parquet"
CHECKPOINT_DIR="$EXAMPLE_DIR/checkpoints"
WANDB_PROJECT="gsm8k-distillation"

if [[ ! -f "$SYNTHETIC_DATA" ]]; then
    echo "Missing $SYNTHETIC_DATA — run eval_base.sh first to generate it." >&2
    exit 1
fi
if [[ ! -f "$GROUNDTRUTH_DATA" ]]; then
    echo "Missing $GROUNDTRUTH_DATA — run annotate_groundtruth.py first to generate it." >&2
    exit 1
fi
if [[ ! -f "$NEMO_DATA" ]]; then
    echo "Missing $NEMO_DATA — run baseline_nemo.py first to generate it." >&2
    exit 1
fi

# 1. Synthetic arm
echo "=== Training on synthetic data ==="
python examples/gsm8k_distillation/train.py \
    --data "$SYNTHETIC_DATA" \
    --prompt-column question --completion-column teacher_completion \
    --output-dir "$CHECKPOINT_DIR/synthetic" $SMOKE_TEST \
    $WANDB --wandb-project "$WANDB_PROJECT" --run-name synthetic

# 2. Ground-truth baseline arm, size-matched to the synthetic arm
NUM_SYNTHETIC=$(python -c "import pandas as pd; print(len(pd.read_parquet('$SYNTHETIC_DATA')))")
echo "=== Training on ground-truth data (max-samples=$NUM_SYNTHETIC) ==="
python examples/gsm8k_distillation/train.py \
    --data "$GROUNDTRUTH_DATA" \
    --prompt-column question --completion-column teacher_completion \
    --max-samples "$NUM_SYNTHETIC" \
    --output-dir "$CHECKPOINT_DIR/groundtruth" $SMOKE_TEST \
    $WANDB --wandb-project "$WANDB_PROJECT" --run-name groundtruth

# 3. NeMo Data Designer baseline arm, size-matched to the synthetic arm
echo "=== Training on NeMo baseline data (max-samples=$NUM_SYNTHETIC) ==="
python examples/gsm8k_distillation/train.py \
    --data "$NEMO_DATA" \
    --prompt-column question --completion-column teacher_completion \
    --max-samples "$NUM_SYNTHETIC" \
    --output-dir "$CHECKPOINT_DIR/nemo_baseline" $SMOKE_TEST \
    $WANDB --wandb-project "$WANDB_PROJECT" --run-name nemo_baseline

echo "=== Done. Checkpoints in $CHECKPOINT_DIR/{synthetic,groundtruth,nemo_baseline} ==="

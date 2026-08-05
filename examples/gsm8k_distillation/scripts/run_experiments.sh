#!/usr/bin/env bash
# Orchestrates the full gsm8k_distillation gap-closure experiment by running its three stages
# in sequence (each is also runnable standalone for isolated debugging/reruns):
#   1. eval_base.sh    — generate synthetic data (if needed), eval base student + teacher.
#   2. train.sh        — train the synthetic-data arm and the size-matched ground-truth arm.
#   3. eval_trained.sh — eval both trained arms, build the gap-closure report.
#
# Usage:
#   PYTHONPATH=. bash examples/gsm8k_distillation/scripts/run_experiments.sh \
#       [--smoke-test] [--regenerate-data] [--wandb]
#
# --wandb logs both training arms (run names "synthetic"/"groundtruth") to the
# "gsm8k-distillation" wandb project; requires wandb credentials.
#
# Requires a CUDA GPU (student train/eval) and AWS Bedrock credentials (teacher gen/eval).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE_TEST=""
REGENERATE_DATA=""
WANDB=""
for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE_TEST="--smoke-test" ;;
        --regenerate-data) REGENERATE_DATA="--regenerate-data" ;;
        --wandb) WANDB="--wandb" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

bash "$SCRIPT_DIR/eval_base.sh" $SMOKE_TEST $REGENERATE_DATA
bash "$SCRIPT_DIR/train.sh" $SMOKE_TEST $WANDB
bash "$SCRIPT_DIR/eval_trained.sh" $SMOKE_TEST

echo "=== All stages complete. See examples/gsm8k_distillation/output/gap_closure_report.json ==="

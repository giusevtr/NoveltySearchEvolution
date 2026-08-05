#!/usr/bin/env bash
# Stage 3/3: evaluate the trained arms and build the gap-closure report.
#
# Usage:
#   bash examples/gsm8k_distillation/scripts/eval_trained.sh [--smoke-test]
#
# Requires a CUDA GPU, checkpoints/{synthetic,groundtruth,nemo_baseline}/ (produced by
# train.sh), and output/eval/{base,teacher}/report.json (produced by eval_base.sh).
# Writes output/gap_closure_report.json (+ .png), folding in all three distillation arms.

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

SMOKE_TEST=""
for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE_TEST="--smoke-test" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

CHECKPOINT_DIR="$EXAMPLE_DIR/checkpoints"
OUTPUT_DIR="$EXAMPLE_DIR/output"

for arm in synthetic groundtruth nemo_baseline; do
    if [[ ! -d "$CHECKPOINT_DIR/$arm" ]]; then
        echo "Missing $CHECKPOINT_DIR/$arm — run train.sh first." >&2
        exit 1
    fi
done
for report in base teacher; do
    if [[ ! -f "$OUTPUT_DIR/eval/$report/report.json" ]]; then
        echo "Missing $OUTPUT_DIR/eval/$report/report.json — run eval_base.sh first." >&2
        exit 1
    fi
done

# 1. Evaluate trained arms
echo "=== Evaluating synthetic-distilled student ==="
python examples/gsm8k_distillation/eval.py --mode student \
    --lora-path "$CHECKPOINT_DIR/synthetic" --run-name synthetic $SMOKE_TEST

echo "=== Evaluating ground-truth-distilled student ==="
python examples/gsm8k_distillation/eval.py --mode student \
    --lora-path "$CHECKPOINT_DIR/groundtruth" --run-name groundtruth $SMOKE_TEST

echo "=== Evaluating NeMo-baseline-distilled student ==="
python examples/gsm8k_distillation/eval.py --mode student \
    --lora-path "$CHECKPOINT_DIR/nemo_baseline" --run-name nemo_baseline $SMOKE_TEST

# 2. Gap-closure report
echo "=== Building gap-closure report ==="
python examples/gsm8k_distillation/report.py \
    --base-report "$OUTPUT_DIR/eval/base/report.json" \
    --teacher-report "$OUTPUT_DIR/eval/teacher/report.json" \
    --synthetic-report "$OUTPUT_DIR/eval/synthetic/report.json" \
    --groundtruth-report "$OUTPUT_DIR/eval/groundtruth/report.json" \
    --nemo-report "$OUTPUT_DIR/eval/nemo_baseline/report.json"

echo "=== Done. See $OUTPUT_DIR/gap_closure_report.json ==="

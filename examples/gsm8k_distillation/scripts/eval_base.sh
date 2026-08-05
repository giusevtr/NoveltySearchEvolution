#!/usr/bin/env bash
# Stage 1/3: generate synthetic data (if needed) and evaluate the base student + teacher.
#
# Usage:
#   bash examples/gsm8k_distillation/scripts/eval_base.sh [--smoke-test] [--regenerate-data]
#
# Requires a CUDA GPU (student eval) and AWS Bedrock credentials (data gen + teacher eval).
# Writes output/eval/base/report.json and output/eval/teacher/report.json, consumed by
# eval_trained.sh's gap-closure report.

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

SMOKE_TEST=""
REGENERATE_DATA=""
for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE_TEST="--smoke-test" ;;
        --regenerate-data) REGENERATE_DATA="1" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

SYNTHETIC_DATA="$EXAMPLE_DIR/output/synthetic_distillation.parquet"

# 1. Data generation
if [[ -n "$REGENERATE_DATA" || ! -f "$SYNTHETIC_DATA" ]]; then
    echo "=== Generating synthetic data ==="
    python examples/gsm8k_distillation/run.py $SMOKE_TEST
else
    echo "=== Reusing existing synthetic data at $SYNTHETIC_DATA ==="
fi

# 2. Base + teacher eval
echo "=== Evaluating base student ==="
python examples/gsm8k_distillation/eval.py --mode student --run-name base $SMOKE_TEST

echo "=== Evaluating teacher ==="
python examples/gsm8k_distillation/eval.py --mode teacher --run-name teacher $SMOKE_TEST

echo "=== Done. Base/teacher reports in $EXAMPLE_DIR/output/eval/{base,teacher}/report.json ==="

#!/usr/bin/env bash
# Single entry point for the full gsm8k_distillation experiment, end to end:
#   1. Download GSM8K train/val splits.
#   2. Teacher-annotate the train ground-truth set and a 1000-row sample of the val set (used
#      later as the eval set during SFT).
#   3. Generation methods — novelty-search synthetic data + NeMo Data Designer baseline, each
#      generating NUM_SAMPLES (default 8000) samples so the two arms are directly comparable
#      against the full ground-truth GSM8K set without subsampling.
#   4. Diversity report over the three generated datasets.
#   5. SFT (LoRA) on all three arms, evaluated during training against the annotated eval set.
#   6. Evals — base/teacher + all trained arms, plus the gap-closure report.
#
# Usage:
#   PYTHONPATH=. bash examples/gsm8k_distillation/run_full_experiment.sh \
#       [--smoke-test] [--regenerate-data] [--wandb] [--model-id=<hf-model-id>]
#
# --wandb logs training runs to the "gsm8k-distillation" wandb project, and novelty-search
# generation to the separate "gsm8k-distillation-novelty-search" project; requires wandb
# credentials.
#
# --model-id selects the student base model to distill/evaluate (default "Qwen/Qwen3-1.7B").
# Checkpoints, distilled-student eval reports (including the base/untrained run), and the
# gap-closure report are all namespaced under a slug derived from it
# (checkpoints/<slug>/, output/eval/<slug>/, output/gap_closure/<slug>/), so picking a
# different --model-id trains and evaluates that model fresh rather than reusing another
# model's artifacts. The training datasets and the teacher eval report are shared across model
# choices and are not namespaced.
#
# Requires a CUDA GPU (student gen/train/eval) and AWS Bedrock credentials (teacher gen/eval).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

SMOKE_TEST=""
REGENERATE_DATA=""
WANDB=""
MODEL_ID="Qwen/Qwen3-1.7B"
NUM_SAMPLES=8000
for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE_TEST="--smoke-test" ;;
        --regenerate-data) REGENERATE_DATA="--regenerate-data" ;;
        --wandb) WANDB="--wandb" ;;
        --model-id=*) MODEL_ID="${arg#*=}" ;;
        --num-samples=*) NUM_SAMPLES="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done
MODEL_SLUG="${MODEL_ID//\//_}"

DOWNLOAD_LIMIT=""
if [[ -n "$SMOKE_TEST" ]]; then
    DOWNLOAD_LIMIT="--limit 20"
fi

GSM8K_TRAIN="$EXAMPLE_DIR/data/gsm8k_train.parquet"
GSM8K_VAL="$EXAMPLE_DIR/data/gsm8k_val.parquet"
GROUNDTRUTH_DATA="$EXAMPLE_DIR/data/gsm8k_train_teacher_annotated.parquet"
EVAL_ANNOTATED_DATA="$EXAMPLE_DIR/data/gsm8k_val_teacher_annotated.parquet"
NEMO_DATA="$EXAMPLE_DIR/output/baseline_nemo_distillation.parquet"
SYNTHETIC_DATA="$EXAMPLE_DIR/output/synthetic_distillation.parquet"
OUTPUT_DIR="$EXAMPLE_DIR/output"
TEACHER_REPORT="$OUTPUT_DIR/eval/teacher/report.json"
CHECKPOINT_DIR="$EXAMPLE_DIR/checkpoints/$MODEL_SLUG"
EVAL_DIR="$OUTPUT_DIR/eval/$MODEL_SLUG"
BASE_REPORT="$EVAL_DIR/base/report.json"
GAP_CLOSURE_DIR="$OUTPUT_DIR/gap_closure/$MODEL_SLUG"
WANDB_PROJECT="gsm8k-distillation"

# 1. Download data
if [[ -n "$REGENERATE_DATA" || ! -f "$GSM8K_TRAIN" || ! -f "$GSM8K_VAL" ]]; then
    echo "=== Downloading GSM8K ==="
    python examples/gsm8k_distillation/scripts/download_data.py $DOWNLOAD_LIMIT
else
    echo "=== Reusing existing GSM8K data at $EXAMPLE_DIR/data ==="
fi

# 2. Teacher annotation
if [[ -n "$REGENERATE_DATA" || ! -f "$GROUNDTRUTH_DATA" ]]; then
    echo "=== Annotating ground-truth data ==="
    python examples/gsm8k_distillation/scripts/annotate_groundtruth.py $SMOKE_TEST
else
    echo "=== Reusing existing ground-truth annotations at $GROUNDTRUTH_DATA ==="
fi

if [[ -n "$REGENERATE_DATA" || ! -f "$EVAL_ANNOTATED_DATA" ]]; then
    echo "=== Annotating eval data ==="
    python examples/gsm8k_distillation/scripts/annotate_groundtruth.py \
        --data "$GSM8K_VAL" --output-basename gsm8k_val_teacher_annotated --limit 1000 $SMOKE_TEST
else
    echo "=== Reusing existing eval annotations at $EVAL_ANNOTATED_DATA ==="
fi

# 3. Generation methods
if [[ -n "$REGENERATE_DATA" || ! -f "$SYNTHETIC_DATA" ]]; then
    echo "=== Generating synthetic data ==="
    python examples/gsm8k_distillation/scripts/run_novelty_search_augmentatin.py \
        --num-samples "$NUM_SAMPLES" $SMOKE_TEST $WANDB
else
    echo "=== Reusing existing synthetic data at $SYNTHETIC_DATA ==="
fi

if [[ -n "$REGENERATE_DATA" || ! -f "$BASE_REPORT" ]]; then
    echo "=== Evaluating base student ==="
    python examples/gsm8k_distillation/scripts/eval.py --mode student --run-name base \
        --base-model "$MODEL_ID" --output-dir "$EVAL_DIR/base" $SMOKE_TEST
else
    echo "=== Reusing existing base eval report at $BASE_REPORT ==="
fi

if [[ -n "$REGENERATE_DATA" || ! -f "$TEACHER_REPORT" ]]; then
    echo "=== Evaluating teacher ==="
    python examples/gsm8k_distillation/scripts/eval.py --mode teacher --run-name teacher $SMOKE_TEST
else
    echo "=== Reusing existing teacher eval report at $TEACHER_REPORT ==="
fi

if [[ -n "$REGENERATE_DATA" || ! -f "$NEMO_DATA" ]]; then
    echo "=== Generating NeMo baseline data ==="
    python examples/gsm8k_distillation/scripts/baseline_nemo.py --num-samples "$NUM_SAMPLES" $SMOKE_TEST
else
    echo "=== Reusing existing NeMo baseline data at $NEMO_DATA ==="
fi

# 4. Diversity report
echo "=== Building diversity report ==="
python examples/gsm8k_distillation/scripts/diversity_report.py

# 5. SFT
if [[ -n "$REGENERATE_DATA" || ! -d "$CHECKPOINT_DIR/synthetic" ]]; then
    echo "=== Training on synthetic data ==="
    python examples/gsm8k_distillation/scripts/train.py \
        --data "$SYNTHETIC_DATA" \
        --prompt-column question --completion-column teacher_completion \
        --output-dir "$CHECKPOINT_DIR/synthetic" --base-model "$MODEL_ID" \
        --eval-data "$EVAL_ANNOTATED_DATA" --eval-prompt-column question --eval-completion-column teacher_completion \
        $SMOKE_TEST $WANDB --wandb-project "$WANDB_PROJECT" --run-name synthetic
else
    echo "=== Reusing existing synthetic checkpoint at $CHECKPOINT_DIR/synthetic ==="
fi

if [[ -n "$REGENERATE_DATA" || ! -d "$CHECKPOINT_DIR/groundtruth" ]]; then
    echo "=== Training on ground-truth data (max-samples=$NUM_SAMPLES) ==="
    python examples/gsm8k_distillation/scripts/train.py \
        --data "$GROUNDTRUTH_DATA" \
        --prompt-column question --completion-column teacher_completion \
        --max-samples "$NUM_SAMPLES" \
        --output-dir "$CHECKPOINT_DIR/groundtruth" --base-model "$MODEL_ID" \
        --eval-data "$EVAL_ANNOTATED_DATA" --eval-prompt-column question --eval-completion-column teacher_completion \
        $SMOKE_TEST $WANDB --wandb-project "$WANDB_PROJECT" --run-name groundtruth
else
    echo "=== Reusing existing groundtruth checkpoint at $CHECKPOINT_DIR/groundtruth ==="
fi

if [[ -n "$REGENERATE_DATA" || ! -d "$CHECKPOINT_DIR/nemo_baseline" ]]; then
    echo "=== Training on NeMo baseline data ==="
    python examples/gsm8k_distillation/scripts/train.py \
        --data "$NEMO_DATA" \
        --prompt-column question --completion-column teacher_completion \
        --output-dir "$CHECKPOINT_DIR/nemo_baseline" --base-model "$MODEL_ID" \
        --eval-data "$EVAL_ANNOTATED_DATA" --eval-prompt-column question --eval-completion-column teacher_completion \
        $SMOKE_TEST $WANDB --wandb-project "$WANDB_PROJECT" --run-name nemo_baseline
else
    echo "=== Reusing existing nemo_baseline checkpoint at $CHECKPOINT_DIR/nemo_baseline ==="
fi

# 6. Evals
if [[ -n "$REGENERATE_DATA" || ! -f "$EVAL_DIR/synthetic/report.json" ]]; then
    echo "=== Evaluating synthetic-distilled student ==="
    python examples/gsm8k_distillation/scripts/eval.py --mode student \
        --lora-path "$CHECKPOINT_DIR/synthetic" --base-model "$MODEL_ID" \
        --output-dir "$EVAL_DIR/synthetic" --run-name synthetic $SMOKE_TEST --k 5
else
    echo "=== Reusing existing synthetic eval report at $EVAL_DIR/synthetic/report.json ==="
fi

if [[ -n "$REGENERATE_DATA" || ! -f "$EVAL_DIR/groundtruth/report.json" ]]; then
    echo "=== Evaluating ground-truth-distilled student ==="
    python examples/gsm8k_distillation/scripts/eval.py --mode student \
        --lora-path "$CHECKPOINT_DIR/groundtruth" --base-model "$MODEL_ID" \
        --output-dir "$EVAL_DIR/groundtruth" --run-name groundtruth $SMOKE_TEST --k 5
else
    echo "=== Reusing existing groundtruth eval report at $EVAL_DIR/groundtruth/report.json ==="
fi

if [[ -n "$REGENERATE_DATA" || ! -f "$EVAL_DIR/nemo_baseline/report.json" ]]; then
    echo "=== Evaluating NeMo-baseline-distilled student ==="
    python examples/gsm8k_distillation/scripts/eval.py --mode student \
        --lora-path "$CHECKPOINT_DIR/nemo_baseline" --base-model "$MODEL_ID" \
        --output-dir "$EVAL_DIR/nemo_baseline" --run-name nemo_baseline $SMOKE_TEST --k 5
else
    echo "=== Reusing existing nemo_baseline eval report at $EVAL_DIR/nemo_baseline/report.json ==="
fi

echo "=== Building gap-closure report ==="
python examples/gsm8k_distillation/scripts/report.py \
    --base-report "$BASE_REPORT" \
    --teacher-report "$TEACHER_REPORT" \
    --synthetic-report "$EVAL_DIR/synthetic/report.json" \
    --groundtruth-report "$EVAL_DIR/groundtruth/report.json" \
    --nemo-report "$EVAL_DIR/nemo_baseline/report.json" \
    --output "$GAP_CLOSURE_DIR/gap_closure_report.json"

echo "=== All stages complete. See $GAP_CLOSURE_DIR/gap_closure_report.json ==="

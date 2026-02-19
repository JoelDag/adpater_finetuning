#!/bin/bash

set -euo pipefail

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ulimit -v $((20 * 1024 * 1024))
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

BASE_DIR="${BASE_DIR:-./results/base_model_eval}"
TOKENIZER_NAME="${TOKENIZER_NAME:-google/gemma-3-4b-pt}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-pt}"
TASKS="${TASKS:-arc_challenge_mt,arc_multilingual,belebele,xnli}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LIMIT="${LIMIT:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-base_model_eval}"
WANDB_GROUP="${WANDB_GROUP:-base_model_eval}"
RUN_NAME="${RUN_NAME:-single_model_eval}"

LOG_FILE="$BASE_DIR/evaluation_all_datasets.log"
mkdir -p "$BASE_DIR"

nohup python lm_harness_single_model.py \
  --base_dir "$BASE_DIR" \
  --tokenizer_name "$TOKENIZER_NAME" \
  --model_name "$MODEL_NAME" \
  --eval_tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --limit "$LIMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" \
  --run_name "$RUN_NAME" > "$LOG_FILE" 2>&1 &

sleep 2
echo "Evaluation started with PID $!"
echo "Logging to: $LOG_FILE"
tail -f "$LOG_FILE"

#!/bin/bash

set -euo pipefail

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_DIR="${BASE_DIR:-./results/adapter_eval}"
TOKENIZER_NAME="${TOKENIZER_NAME:-mistralai/Mistral-7B-v0.3}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-v0.3}"
TASKS="${TASKS:-belebele_sun_Latn,belebele_jav_Latn}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-None}"
WANDB_PROJECT="${WANDB_PROJECT:-adapter_eval}"
WANDB_GROUP="${WANDB_GROUP:-adapter_eval}"
RUN_NAME="${RUN_NAME:-checkpoint_eval}"
mkdir -p "$BASE_DIR"

LOG_FILE="$BASE_DIR/lm_harness_nohup_run_eval_jav_Latn_sun_Latn.log"

nohup python lm_harness_single.py \
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
echo "Training started with PID $!"
echo "Logging to: $LOG_FILE"
tail -f "$LOG_FILE"

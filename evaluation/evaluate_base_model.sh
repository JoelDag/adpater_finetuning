#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=online

TASKS="${TASKS:-belebele_tur_Latn,include_base_44_turkish,turkishmmlu,belebele}"
LIMIT="${LIMIT:-None}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/base_model_eval}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-pt}"
WANDB_PROJECT="${WANDB_PROJECT:-base_gemma_eval}"
WANDB_GROUP="${WANDB_GROUP:-base_gemma_eval}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/base_model_eval.log}"

mkdir -p "$OUTPUT_DIR"

nohup python evaluate_base_model.py \
  --output_dir "$OUTPUT_DIR" \
  --tokenizer_name "$MODEL_NAME" \
  --model_name "$MODEL_NAME" \
  --tasks "$TASKS" \
  --batch_size 32 \
  --limit "$LIMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" \
  > "$LOG_FILE" 2>&1 &


sleep 2
tail -f "$LOG_FILE"

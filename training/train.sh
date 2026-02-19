#!/bin/bash

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export INPUT_DIR="${INPUT_DIR:-./data/prepared}"

MODEL_NAME="${MODEL_NAME:-microsoft/phi-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-./model_output}"

python3 train.py \
    --processed_dir "$INPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --output_model_dir "$OUTPUT_MODEL_DIR"

echo "Training finished."

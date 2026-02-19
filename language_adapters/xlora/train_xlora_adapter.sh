#!/bin/bash

set -euo pipefail

TOKENIZED_DATA_DIR="${TOKENIZED_DATA_DIR:-./tokenized_data}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-v0.3}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/xlora/depth_1_run}"
LOGGING_DIR="$OUTPUT_DIR/logs"
ADAPTERS_JSON="${ADAPTERS_JSON:-./adapters.example.json}"
mkdir -p "$LOGGING_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOG_FILE="$LOGGING_DIR/depth_1_train_xlora_jav_Latn_sun_Latn_swh_Latn_sna_Latn_nya_Latn.log"

nohup python train_xlora_adapter.py \
  --tokenized_dir "$TOKENIZED_DATA_DIR" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$LOGGING_DIR" \
  --adapters_json "$ADAPTERS_JSON" > "$LOG_FILE" 2>&1 &

sleep 2
echo "Training started with PID $!"
echo "Logging to: $LOG_FILE"
tail -f "$LOG_FILE"

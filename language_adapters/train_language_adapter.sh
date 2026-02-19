#!/bin/bash
set -e

TOKENIZED_DATA_DIR="${WORK_ROOT:-/path/to/work_root}/tokenized_data_subsets/tur_Latn/"
MODEL_NAME="bigscience/bloom-560m"
OUTPUT_DIR="${WORK_ROOT:-/path/to/work_root}/results_language_adapters/tur_Latn/"
LOGGING_DIR="$OUTPUT_DIR/logs"

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=3

echo "Running full training pipeline..."

python train_language_adapter.py \
    --tokenized_dir "$TOKENIZED_DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR"
#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OUTPUT_DIR="${OUTPUT_DIR:-./calm-results/gemma2_7b}"

accelerate launch --config_file accelerate_config.yaml train.py \
  --anchor_model_dir google/gemma-7b \
  --aug_model_dir google/gemma-3-4b-it \
  --num_heads 2 \
  --num_connections 2 \
  --learning_rate 3e-4 \
  --batch_size 2 \
  --output_dir "$OUTPUT_DIR"

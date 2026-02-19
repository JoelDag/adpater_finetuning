#!/bin/bash

set -euo pipefail

PEFT_PATH="${PEFT_PATH:-./results/xlora/checkpoint-2000}"
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.3}"
TASKS="${TASKS:-arc_easy,arc_challenge,hellaswag,lambada_openai}"
OUTPUT_PATH="${OUTPUT_PATH:-$PEFT_PATH/eval_results}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$OUTPUT_PATH"

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE_MODEL},peft=${PEFT_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=true,use_cache=False" \
  --tasks "$TASKS" \
  --batch_size 8 \
  --device "$DEVICE" \
  --output_path "$OUTPUT_PATH"

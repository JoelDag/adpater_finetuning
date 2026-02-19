#!/bin/bash

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
PEFT_PATH="${PEFT_PATH:-./checkpoint-50}"
TASKS="${TASKS:-belebele_jav_Latn,belebele_sun_Latn}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_PATH="${OUTPUT_PATH:-xlora_mistral7b_results.json}"

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE_MODEL},peft=${PEFT_PATH},use_cache=False,device_map=auto,dtype=float16" \
  --tasks "$TASKS" \
  --batch_size auto \
  --device "$DEVICE" \
  --output_path "$OUTPUT_PATH"

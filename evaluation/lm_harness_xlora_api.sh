#!/bin/bash

set -euo pipefail

TASKS="${TASKS:-belebele_swh_Latn}"
OUTPUT_PATH="${OUTPUT_PATH:-./eval_results/belebele_swh.json}"
BASE_URL="${BASE_URL:-http://localhost:1234/v1/completions}"
TOKENIZER="${TOKENIZER:-mistralai/Mistral-7B-v0.3}"
MODEL_ALIAS="${MODEL_ALIAS:-default}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

lm_eval \
  --model local-completions \
  --tasks "$TASKS" \
  --num_fewshot 0 \
  --output_path "$OUTPUT_PATH" \
  --model_args "model=${MODEL_ALIAS},base_url=${BASE_URL},tokenizer=${TOKENIZER},max_tokens=512,temperature=0,logprobs=5,batch_size=1,num_concurrent=1"

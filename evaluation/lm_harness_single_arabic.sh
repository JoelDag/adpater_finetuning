#!/bin/bash

set -euo pipefail

export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

BASE_DIR="${BASE_DIR:-./results/mistral7b/arabic_subset}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$BASE_DIR/adapter}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-v0.3}"
TASKS="${TASKS:-hellaswag,xnli,belebele,arc_multilingual,mmlu,include_base_44_*,truthfulqa,mgsm_direct,mgsm_cot_native,mlqa*,xcopa,xwinograd,xstorycloze,pawsx,flores,wmt16,lambada_multilingual,xquad}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LIMIT="${LIMIT:-50}"
WANDB_PROJECT="${WANDB_PROJECT:-mistral7b_arabic_adapter_evaluation}"
WANDB_GROUP="${WANDB_GROUP:-mistral_eval_arabic_qlora}"
RUN_NAME="${RUN_NAME:-arabic_subset_eval}"
mkdir -p "$BASE_DIR"

python lm_harness_single.py \
  --base_dir "$BASE_DIR" \
  --tokenizer_name "$TOKENIZER_NAME" \
  --model_name "$MODEL_NAME" \
  --eval_tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --limit "$LIMIT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" \
  --run_name "$RUN_NAME"

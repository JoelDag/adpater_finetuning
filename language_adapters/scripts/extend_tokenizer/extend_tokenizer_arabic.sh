#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python extend_tokenizer.py \
  --base_tokenizer google/gemma-3-4b-pt \
  --data_dir ${WORK_ROOT:-/path/to/work_root}/language_adapter_subsets/arabic_subset \
  --output_dir ${WORK_ROOT:-/path/to/work_root}/tokenized_data_subsets/gemma_extended_tokenizers/arabic \
  --added_tokens 3000

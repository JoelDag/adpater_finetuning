#!/bin/bash

python extend_tokenizer.py \
  --base_tokenizer google/gemma-3-4b-pt \
  --data_dir ${DATA_ROOT:-/path/to/data_root}/language_adapters_subsets/south_asian \
  --output_dir ${DATA_ROOT:-/path/to/data_root}/tokenized_adapter_subsets/gemma_extended_tokenizers/south_asian \
  --added_tokens 3000

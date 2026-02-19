#!/bin/bash

DATA_DIR="/data/fineweb2_subset_belebele" 
MODEL_NAME="bigscience/bloom-560m"
TOKENIZED_DIR=".${DATA_ROOT:-/path/to/data_root}/test/tokenize-test"
NUM_PROCESSES=28

python tokenize_adapter_data.py \
    --data_dir "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --tokenized_dir "$TOKENIZED_DIR" \
    --num_processes "$NUM_PROCESSES"

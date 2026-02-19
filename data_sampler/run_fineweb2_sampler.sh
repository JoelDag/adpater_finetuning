#!/bin/bash

set -euo pipefail

TOTAL_DOCS="${TOTAL_DOCS:-14000000}" # 14M docs ~= 67.37 GB
NUM_LANGUAGES="${NUM_LANGUAGES:-190}"
OUTPUT_DIR="${OUTPUT_DIR:-./fineweb2_subset_190_most_used_languages}"
META_FILE="${META_FILE:-filtered_fineweb2_meta.json}"

python sample_fineweb2.py \
  --total_docs "$TOTAL_DOCS" \
  --num_languages "$NUM_LANGUAGES" \
  --output_dir "$OUTPUT_DIR" \
  --meta_file "$META_FILE"

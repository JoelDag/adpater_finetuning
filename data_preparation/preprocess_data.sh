#!/bin/bash

set -euo pipefail

CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

DATASETS_DOWNLOAD=(
    "angeluriot/french_instruct:"
    "DiscoResearch/germanrag:"
)
DATASETS_STRING="${DATASETS_DOWNLOAD[@]}"

echo "Starting dataset downloads"
python download_datasets.py --datasets $DATASETS_STRING --hf_cache_dir "$CACHE_DIR"
echo "Download of all datasets finished"


export HF_HOME="$CACHE_DIR"
OUTPUT_DIR="${OUTPUT_DIR:-./data/prepared/language_adapters}"
TOTAL_PROCS="${TOTAL_PROCS:-4}"
TOKENIZER="${TOKENIZER:-microsoft/phi-2}"
mkdir -p "$OUTPUT_DIR"

DATASETS_PREPROCESS=(
    "angeluriot/french_instruct::$TOKENIZER"
    "DiscoResearch/germanrag::$TOKENIZER"
)
DATASETS_PREPROCESS_STRING="${DATASETS_PREPROCESS[@]}"

for (( i=0; i<TOTAL_PROCS; i++ ))
do
    echo "Launching process $i / $TOTAL_PROCS"
    PROC_RANK="$i" TOTAL_PROCS="$TOTAL_PROCS" OUTPUT_DIR="$OUTPUT_DIR" \
    python preprocess_data.py --output_dir "$OUTPUT_DIR" --datasets $DATASETS_PREPROCESS_STRING --tokenizer "$TOKENIZER" --log_level info &
done

wait

echo "All preprocessing processes completed"

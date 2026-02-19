#!/bin/bash

set -euo pipefail

MISTRALRS_BIN="${MISTRALRS_BIN:-./mistralrs-server}"
PORT="${PORT:-1234}"
XLORA_MODEL_ID="${XLORA_MODEL_ID:-./results/xlora/checkpoint-2000}"
ORDERING_FILE="${ORDERING_FILE:-./results/xlora/my_ordering.json}"

"$MISTRALRS_BIN" \
  --port "$PORT" \
  x-lora \
  --xlora-model-id "$XLORA_MODEL_ID" \
  -o "$ORDERING_FILE"

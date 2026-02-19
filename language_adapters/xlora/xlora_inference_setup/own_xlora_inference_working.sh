#!/bin/bash

set -euo pipefail

MISTRALRS_BIN="${MISTRALRS_BIN:-./target/debug/mistralrs-server}"
PORT="${PORT:-1234}"
XLORA_MODEL_ID="${XLORA_MODEL_ID:-./checkpoint-50}"
ORDERING_FILE="${ORDERING_FILE:-./ordering.json}"
RUST_BACKTRACE="${RUST_BACKTRACE:-full}"

RUST_BACKTRACE="$RUST_BACKTRACE" "$MISTRALRS_BIN" \
  --port "$PORT" \
  x-lora \
  --xlora-model-id "$XLORA_MODEL_ID" \
  -o "$ORDERING_FILE"

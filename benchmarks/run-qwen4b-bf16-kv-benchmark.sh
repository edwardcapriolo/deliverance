#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_CONFIG="$SCRIPT_DIR/benchmarks/configs/qwen3-4b-jq4-bf16-kv.json" \
  sh "$SCRIPT_DIR/benchmarks/run-qwen-single-benchmark.sh"

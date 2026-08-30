#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_CONFIG="$SCRIPT_DIR/benchmarks/configs/qwen3-0.6b-jq4-i8-kv.json" \
  sh "$SCRIPT_DIR/benchmarks/run-qwen06b-single-benchmark.sh"

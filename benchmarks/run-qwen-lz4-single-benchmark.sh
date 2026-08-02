#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARK_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BENCHMARK_SCRIPT_NAME="$(basename "$0" .sh)" . "$SCRIPT_DIR/benchmark-run-dir.sh"
MODEL_CONFIG=${MODEL_CONFIG:-"$SCRIPT_DIR/configs/qwen3-4b-jq4-lz4.json"}
export MODEL_CONFIG
export BENCHMARK_RUN_DIR

exec sh "$SCRIPT_DIR/run-qwen-single-benchmark.sh"

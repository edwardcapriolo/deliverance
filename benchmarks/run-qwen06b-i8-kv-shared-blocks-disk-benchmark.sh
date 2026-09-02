#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export DELIVERANCE_BENCHMARK_ARGS=${DELIVERANCE_BENCHMARK_ARGS:-"--model-config $SCRIPT_DIR/benchmarks/configs/qwen3-0.6b-jq4-i8-kv-shared-blocks-disk.json --pool-size 16 --max-tokens 256 --warmup-cases 0 --max-cases 1 --max-turns 1 --profile-stages"}
MODEL_CONFIG="$SCRIPT_DIR/benchmarks/configs/qwen3-0.6b-jq4-i8-kv-shared-blocks-disk.json" \
  sh "$SCRIPT_DIR/benchmarks/run-qwen06b-single-benchmark.sh"

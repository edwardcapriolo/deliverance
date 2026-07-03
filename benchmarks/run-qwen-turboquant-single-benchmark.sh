#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_CONFIG=${MODEL_CONFIG:-"$SCRIPT_DIR/configs/qwen3-4b-jq4-turboquant.json"}
export MODEL_CONFIG

exec sh "$SCRIPT_DIR/run-qwen-single-benchmark.sh"

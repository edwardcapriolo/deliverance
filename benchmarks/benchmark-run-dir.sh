#!/usr/bin/env sh
set -eu

BENCHMARK_ROOT=${BENCHMARK_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
BENCHMARK_SCRIPT_NAME=${BENCHMARK_SCRIPT_NAME:-$(basename "$0" .sh)}
BENCHMARK_DATE=${BENCHMARK_DATE:-$(date +%F)}
BENCHMARK_TIME=${BENCHMARK_TIME:-$(date +%H%M%S)}
BENCHMARK_GIT_HASH=${BENCHMARK_GIT_HASH:-$(git -C "$BENCHMARK_ROOT" rev-parse --short HEAD 2>/dev/null || printf 'nogit')}
if ! git -C "$BENCHMARK_ROOT" diff --quiet --ignore-submodules -- 2>/dev/null; then
  BENCHMARK_GIT_HASH="$BENCHMARK_GIT_HASH-dirty"
fi

BENCHMARK_RUN_DIR=${BENCHMARK_RUN_DIR:-$BENCHMARK_ROOT/benchmarks/runs/$BENCHMARK_DATE-$BENCHMARK_GIT_HASH-$BENCHMARK_SCRIPT_NAME-$BENCHMARK_TIME}
mkdir -p "$BENCHMARK_RUN_DIR"
export BENCHMARK_RUN_DIR

printf 'saving results to %s\n' "$BENCHMARK_RUN_DIR"

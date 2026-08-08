#!/usr/bin/env sh
set -eu

for visible_rows in 32 128 256 512 1024; do
  VISIBLE_ROWS="$visible_rows" sh benchmarks/run-gpu-flash-vllm-one-off.sh
done

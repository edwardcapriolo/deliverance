#!/usr/bin/env sh
set -eu

END_TIME=$(( $(date +%s) + 120 ))

while [ "$(date +%s)" -lt "$END_TIME" ]; do
  mvn install -Dmaven.test.skip=true
  sh benchmarks/run-qwen-gpu-flash-decode-single-benchmark.sh
done

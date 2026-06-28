#!/usr/bin/env sh
set -eu

exec java -jar target/nanocode-deliverance-0.0.10-SNAPSHOT-all.jar \
  --config config-qwen3-4b-jq4.json \
  "$@"

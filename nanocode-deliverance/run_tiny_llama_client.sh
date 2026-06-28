#!/usr/bin/env sh
set -eu

exec java -jar target/nanocode-deliverance-0.0.10-SNAPSHOT-all.jar \
  --config config-tinyllama.json \
  "$@"

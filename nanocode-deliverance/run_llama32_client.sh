#!/usr/bin/env sh
set -eu

BASE_URL=${DELIVERANCE_BASE_URL:-http://localhost:${DELIVERANCE_PORT:-8085}}
MODEL=${DELIVERANCE_MODEL:-Llama-3.2-3B-Instruct-JQ4}
MAX_TOKENS=${NANOCODE_MAX_TOKENS:-200}

exec java -jar target/nanocode-deliverance-0.0.10-SNAPSHOT-all.jar \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --max-tokens "$MAX_TOKENS" \
  "$@"

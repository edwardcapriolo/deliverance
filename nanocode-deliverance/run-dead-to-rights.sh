#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

java -cp "$SCRIPT_DIR/target/nanocode-deliverance-0.0.12-SNAPSHOT-all.jar" \
  io.teknek.deliverance.nanocode.game.DeadToRightsGame \
  --base-url "${DELIVERANCE_DEAD_TO_RIGHTS_BASE_URL:-http://localhost:18087}" \
  --model "${DELIVERANCE_DEAD_TO_RIGHTS_MODEL:-Qwen3-4B-JQ4}" \
  --max-tokens "${DELIVERANCE_DEAD_TO_RIGHTS_MAX_TOKENS:-512}" \
  --temperature "${DELIVERANCE_DEAD_TO_RIGHTS_TEMPERATURE:-0.8}" \
  "$@"

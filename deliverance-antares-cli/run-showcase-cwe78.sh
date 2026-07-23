#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

java -jar "$SCRIPT_DIR/target/deliverance-antares-cli-0.0.12-SNAPSHOT-all.jar" \
  --repo "$ROOT_DIR/nanocode-deliverance/showcase-security-repo" \
  --endpoint "${DELIVERANCE_ANTARES_ENDPOINT:-http://127.0.0.1:18085/v1}" \
  --model "${DELIVERANCE_ANTARES_MODEL:-antares-1b-JQ4}" \
  --cwe CWE-78 \
  --query "Search for CWE-78 OS command injection. Look for user-controlled input reaching Runtime.exec, ProcessBuilder, shell commands, or command string concatenation. Submit the exact vulnerable repository-relative file path." \
  --max-tool-calls "${DELIVERANCE_ANTARES_MAX_TOOL_CALLS:-8}" \
  --max-submit-turns "${DELIVERANCE_ANTARES_MAX_SUBMIT_TURNS:-3}" \
  "$@"

#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

java -cp "$SCRIPT_DIR/target/deliverance-antares-cli-0.0.12-SNAPSHOT-all.jar" \
  io.teknek.deliverance.antares.VulnerabilityLocalizationSmokeBenchmark \
  --case-id showcase-cwe78 \
  --repo "$ROOT_DIR/nanocode-deliverance/showcase-security-repo" \
  --endpoint "${DELIVERANCE_ANTARES_ENDPOINT:-http://127.0.0.1:18085/v1}" \
  --model "${DELIVERANCE_ANTARES_MODEL:-antares-1b-JQ4}" \
  --cwe CWE-78 \
  --query "Look for user-controlled input reaching Runtime.exec, ProcessBuilder, shell commands, or command string concatenation." \
  --expected-files "src/main/java/demo/ArchiveController.java" \
  --max-tool-calls "${DELIVERANCE_ANTARES_MAX_TOOL_CALLS:-4}" \
  --output "${DELIVERANCE_VLOC_OUTPUT:-$ROOT_DIR/target/vloc-showcase.jsonl}" \
  "$@"

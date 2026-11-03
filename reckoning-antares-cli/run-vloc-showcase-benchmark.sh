#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
JAR=$(sh "$SCRIPT_DIR/resolve_antares_cli_jar.sh")

java -cp "$JAR" \
  io.teknek.reckoning.antares.VulnerabilityLocalizationSmokeBenchmark \
  --case-id showcase-cwe78 \
  --repo "$ROOT_DIR/reckoning-nanocode/showcase-security-repo" \
  --endpoint "${RECKONING_ANTARES_ENDPOINT:-http://127.0.0.1:18085/v1}" \
  --model "${RECKONING_ANTARES_MODEL:-antares-1b-JQ4}" \
  --cwe CWE-78 \
  --query "Look for user-controlled input reaching Runtime.exec, ProcessBuilder, shell commands, or command string concatenation." \
  --expected-files "src/main/java/demo/ArchiveController.java" \
  --max-tool-calls "${RECKONING_ANTARES_MAX_TOOL_CALLS:-4}" \
  --output "${RECKONING_VLOC_OUTPUT:-$ROOT_DIR/target/vloc-showcase.jsonl}" \
  "$@"

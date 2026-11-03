#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JAR=$(sh "$SCRIPT_DIR/resolve_nanocode_jar.sh")

exec java -jar "$JAR" \
  --config config-antares-1b.json \
  "$@"

#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

set -- "$SCRIPT_DIR"/target/nanocode-deliverance-*-all.jar
if [ ! -f "$1" ]; then
  printf '%s\n' "nanocode shaded jar not found. Build it first: mvn -q -pl nanocode-deliverance -am -DskipTests package" >&2
  exit 1
fi
if [ "$#" -gt 1 ]; then
  printf '%s\n' "multiple nanocode shaded jars found; remove stale target jars:" >&2
  for jar in "$@"; do
    printf '  %s\n' "$jar" >&2
  done
  exit 1
fi

printf '%s\n' "$1"

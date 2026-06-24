#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NATIVE_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)
OUT_DIR="$NATIVE_DIR/src/main/java"
PACKAGE="io.teknek.deliverance.tensor.operations.cnative"
JEXTRACT=${JEXTRACT:-/Users/edward.capriolo/Downloads/jextract-22/bin/jextract}

if [ ! -x "$JEXTRACT" ]; then
  printf '%s\n' "jextract not found or not executable: $JEXTRACT" >&2
  printf '%s\n' "Set JEXTRACT=/path/to/jextract and rerun." >&2
  exit 1
fi

cd "$SCRIPT_DIR"

"$JEXTRACT" \
  --output "$OUT_DIR" \
  -t "$PACKAGE" \
  -I "$SCRIPT_DIR" \
  -l deliverance \
  --header-class-name NativeSimd \
  vector_simd.h

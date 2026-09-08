#!/usr/bin/env bash
set -euo pipefail

NATIVE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
HEADER_DIR="$NATIVE_DIR/src/main/c/tensor2"
OUT_DIR="$NATIVE_DIR/src/main/java"
PACKAGE="io.teknek.deliverance.tensor.operations.tensor2native"
JEXTRACT=${JEXTRACT:-$NATIVE_DIR/target/jextract/jextract-25/bin/jextract}

if [ ! -x "$JEXTRACT" ]; then
  printf '%s\n' "jextract not found or not executable: $JEXTRACT" >&2
  printf '%s\n' "Run ./get_jextract_mac.sh from the repo root or set JEXTRACT=/path/to/jextract." >&2
  exit 1
fi

cd "$HEADER_DIR"

"$JEXTRACT" \
  --output "$OUT_DIR" \
  -t "$PACKAGE" \
  -I "$HEADER_DIR" \
  -l deliverance_tensor2 \
  --header-class-name Tensor2Native \
  tensor2_native.h

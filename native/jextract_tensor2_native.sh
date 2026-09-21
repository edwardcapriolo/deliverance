#!/usr/bin/env bash
set -euo pipefail

NATIVE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$NATIVE_DIR/.." && pwd)
HEADER_DIR="$NATIVE_DIR/src/main/c/tensor2"
OUT_DIR="$NATIVE_DIR/src/main/java"
PACKAGE="io.teknek.deliverance.tensor.operations.tensor2native"
case "$(uname -s):$(uname -m)" in
  Darwin:arm64|Darwin:aarch64) JEXTRACT_PLATFORM="macos-aarch64" ;;
  Darwin:x86_64|Darwin:amd64) JEXTRACT_PLATFORM="macos-x64" ;;
  Linux:aarch64|Linux:arm64) JEXTRACT_PLATFORM="linux-aarch64" ;;
  Linux:x86_64|Linux:amd64) JEXTRACT_PLATFORM="linux-x64" ;;
  *)
    printf '%s\n' "Unsupported jextract platform: $(uname -s) $(uname -m)" >&2
    exit 1
    ;;
esac
JEXTRACT=${JEXTRACT:-$ROOT_DIR/.tools/jextract/$JEXTRACT_PLATFORM/jextract-25/bin/jextract}

if [ ! -x "$JEXTRACT" ]; then
  sh "$ROOT_DIR/get_jextract.sh"
fi

if [ ! -x "$JEXTRACT" ]; then
  printf '%s\n' "jextract not found or not executable: $JEXTRACT" >&2
  exit 1
fi

cd "$HEADER_DIR"

"$JEXTRACT" \
  --output "$OUT_DIR" \
  -t "$PACKAGE" \
  -I "$HEADER_DIR" \
  -l deliverance_tensor2 \
  --header-class-name Tensor2Native \
  --include-function tensor2_dot_product_rows_f32_f32 \
  --include-function tensor2_dot_product_rows_f32_q8 \
  --include-function tensor2_dot_product_rows_f32_q4 \
  --include-function tensor2_dot_product_rows_bf16_q4 \
  --include-function tensor2_saxpy_f32 \
  --include-function tensor2_saxpy_f32_batch \
  --include-function tensor2_batch_dot_f32_f32 \
  --include-function tensor2_batch_dot_f32_q8 \
  --include-function tensor2_scale_f32 \
  --include-function tensor2_scale_bf16 \
  --include-typedef tensor2_status \
  --include-constant TENSOR2_OK \
  --include-constant TENSOR2_UNSUPPORTED \
  tensor2_native.h

# NativeOps loads the library before the generated class is initialized. Prefer
# the loader lookup so macOS does not require a second path-based library open.
PACKAGE_PATH=$(printf '%s' "$PACKAGE" | tr '.' '/')
GENERATED="$OUT_DIR/$PACKAGE_PATH/Tensor2Native.java"
perl -0pi -e 's#SymbolLookup\.libraryLookup\(System\.mapLibraryName\("deliverance_tensor2"\), LIBRARY_ARENA\)\s*\.or\(SymbolLookup\.loaderLookup\(\)\)\s*\.or\(Linker\.nativeLinker\(\)\.defaultLookup\(\)\)#SymbolLookup.loaderLookup()\n            .or(Linker.nativeLinker().defaultLookup())#' "$GENERATED"

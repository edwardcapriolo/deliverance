#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TOOLS_DIR="${ROOT_DIR}/.tools/jextract"
VERSION="${JEXTRACT_VERSION:-25/2}"
BUILD="${JEXTRACT_BUILD:-jextract+2-4}"

OS=$(uname -s)
ARCH=$(uname -m)
case "${OS}:${ARCH}" in
    Darwin:arm64|Darwin:aarch64) PLATFORM="macos-aarch64" ;;
    Darwin:x86_64|Darwin:amd64) PLATFORM="macos-x64" ;;
    Linux:aarch64|Linux:arm64) PLATFORM="linux-aarch64" ;;
    Linux:x86_64|Linux:amd64) PLATFORM="linux-x64" ;;
    *)
        printf '%s\n' "Unsupported jextract platform: ${OS} ${ARCH}" >&2
        exit 1
        ;;
esac

JEXTRACT_DIR="${TOOLS_DIR}/${PLATFORM}/jextract-25"
JEXTRACT_BIN="${JEXTRACT_DIR}/bin/jextract"

if [ -x "$JEXTRACT_BIN" ]; then
    "$JEXTRACT_BIN" --version
    exit 0
fi

ARCHIVE="${TOOLS_DIR}/openjdk-25-${BUILD}_${PLATFORM}_bin.tar.gz"
URL="https://download.java.net/java/early_access/jextract/${VERSION}/openjdk-25-${BUILD}_${PLATFORM}_bin.tar.gz"
TMP_DIR="${TOOLS_DIR}/.extract-$$"

mkdir -p "$TOOLS_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT HUP INT TERM

if [ ! -f "$ARCHIVE" ]; then
    curl --fail --location --retry 3 --output "${ARCHIVE}.part" "$URL"
    mv "${ARCHIVE}.part" "$ARCHIVE"
fi

mkdir -p "$TMP_DIR"
tar -xzf "$ARCHIVE" -C "$TMP_DIR"
if [ ! -x "$TMP_DIR/jextract-25/bin/jextract" ]; then
    printf '%s\n' "Unexpected jextract archive layout: $ARCHIVE" >&2
    exit 1
fi
mkdir -p "$(dirname "$JEXTRACT_DIR")"
rm -rf "$JEXTRACT_DIR"
mv "$TMP_DIR/jextract-25" "$JEXTRACT_DIR"

if command -v xattr >/dev/null 2>&1; then
    xattr -r -d com.apple.quarantine "$JEXTRACT_DIR" 2>/dev/null || true
fi

"$JEXTRACT_BIN" --version

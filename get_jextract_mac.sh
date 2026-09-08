#!/usr/bin/env sh
set -eu

JEXTRACT_URL="https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_macos-aarch64_bin.tar.gz"
TARGET_DIR="native/target/jextract"
ARCHIVE="${TARGET_DIR}/jextract25.tar.gz"

mkdir -p "${TARGET_DIR}"
curl -L -o "${ARCHIVE}" "${JEXTRACT_URL}"
tar -xzf "${ARCHIVE}" -C "${TARGET_DIR}"

if command -v xattr >/dev/null 2>&1; then
  xattr -r -d com.apple.quarantine "${TARGET_DIR}/jextract-25" 2>/dev/null || true
fi

"${TARGET_DIR}/jextract-25/bin/jextract" --version

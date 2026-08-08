#!/usr/bin/env sh
set -eu

VISIBLE_ROWS=${VISIBLE_ROWS:-256}

mvn -q -pl native \
  -Dtest=NativeGPUFlashDecodeOneOffIT#qwenShapeVllmLayoutOneDecodeMatchesCpuAndPrintsTiming \
  -Ddeliverance.gpu.flash.visibleRows="$VISIBLE_ROWS" \
  test

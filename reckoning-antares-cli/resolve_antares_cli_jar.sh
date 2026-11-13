#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JAR=$(find "$SCRIPT_DIR/target" -maxdepth 1 -name 'reckoning-antares-cli-*-all.jar' -print 2>/dev/null | sort | tail -n 1)
if [ -z "$JAR" ]; then
  printf '%s\n' "Unable to find reckoning-antares-cli shaded jar. Build it with: mvn -pl reckoning-antares-cli -am -DskipTests -Dreckoning.bundle.skip=false package" >&2
  exit 1
fi
printf '%s\n' "$JAR"

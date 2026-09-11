#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JAR=$(find "$SCRIPT_DIR/target" -maxdepth 1 -name 'deliverance-antares-cli-*-all.jar' -print 2>/dev/null | sort | tail -n 1)
if [ -z "$JAR" ]; then
  printf '%s\n' "Unable to find deliverance-antares-cli shaded jar. Build it with: mvn -pl deliverance-antares-cli -am -DskipTests package" >&2
  exit 1
fi
printf '%s\n' "$JAR"

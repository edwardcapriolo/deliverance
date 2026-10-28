#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

MAVEN_OPTS="${MAVEN_OPTS:-} -XX:TieredStopAtLevel=1" \
  mvn -q -pl web -am -DskipTests package
exec sh web/run.sh

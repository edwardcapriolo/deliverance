#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CLI_JAR=""
for candidate in "$ROOT"/inferj-cli/target/inferj-cli-*-all.jar; do
  [ -f "$candidate" ] || continue
  CLI_JAR="$candidate"
  break
done

if [ -z "$CLI_JAR" ]; then
  MAVEN_OPTS="${MAVEN_OPTS:-} -XX:TieredStopAtLevel=1" \
    mvn -q -f "$ROOT/pom.xml" -pl inferj-cli -am -DskipTests package
  for candidate in "$ROOT"/inferj-cli/target/inferj-cli-*-all.jar; do
    [ -f "$candidate" ] || continue
    CLI_JAR="$candidate"
    break
  done
fi

[ -n "$CLI_JAR" ] || { printf '%s\n' "inferj-cli shaded jar was not built" >&2; exit 1; }

exec java -jar "$CLI_JAR" serve "$@"

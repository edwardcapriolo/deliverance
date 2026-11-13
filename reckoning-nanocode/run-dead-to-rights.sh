#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
GAME_JAR=${RECKONING_NANOCODE_JAR:-}
if [ -z "$GAME_JAR" ]; then
  for candidate in "$SCRIPT_DIR"/target/reckoning-nanocode-*-SNAPSHOT-all.jar; do
    [ -e "$candidate" ] || continue
    if [ -z "$GAME_JAR" ] || [ "$candidate" -nt "$GAME_JAR" ]; then
      GAME_JAR=$candidate
    fi
  done
fi
if [ -z "$GAME_JAR" ]; then
  printf '%s\n' "No reckoning-nanocode shaded jar found. Run: mvn -pl reckoning-nanocode -am -DskipTests -Dreckoning.bundle.skip=false package" >&2
  exit 1
fi

java -cp "$GAME_JAR" \
  io.teknek.reckoning.nanocode.game.DeadToRightsGame \
  --base-url "${RECKONING_DEAD_TO_RIGHTS_BASE_URL:-http://localhost:18087}" \
  --model "${RECKONING_DEAD_TO_RIGHTS_MODEL:-Qwen3-4B-JQ4}" \
  --max-tokens "${RECKONING_DEAD_TO_RIGHTS_MAX_TOKENS:-512}" \
  --temperature "${RECKONING_DEAD_TO_RIGHTS_TEMPERATURE:-0.8}" \
  --xtc-threshold "${RECKONING_DEAD_TO_RIGHTS_XTC_THRESHOLD:-0.1}" \
  --xtc-probability "${RECKONING_DEAD_TO_RIGHTS_XTC_PROBABILITY:-0.2}" \
  "$@"

#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export DELIVERANCE_PORT=${DELIVERANCE_DEAD_TO_RIGHTS_PORT:-18087}
exec "$SCRIPT_DIR/run-server-qwen3-0.6b-jq4.sh" "$@"

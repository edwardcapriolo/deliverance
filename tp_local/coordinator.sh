#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/tp_env.sh"

ACTION=${1:-status}
tp_dispatch_web coordinator "$ACTION" \
  --spring.config.location="file:$TP_COORDINATOR_CONFIG" \
  --logging.file.name="$TP_LOG_DIR/coordinator.log"

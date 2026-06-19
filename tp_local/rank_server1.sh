#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/tp_env.sh"

ACTION=${1:-status}
tp_dispatch rank_server1 "$ACTION" \
  --role worker \
  --node-id node-0 \
  --uri "udp://$TP_HOST:$TP_NODE0_PORT" \
  --seed "coordinator=udp://$TP_HOST:$TP_COORDINATOR_PORT" \
  $(tp_seed_args) \
  $(tp_common_args)

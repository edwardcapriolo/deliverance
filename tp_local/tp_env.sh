#!/usr/bin/env sh

: "${TP_CLUSTER:=deliverance-tp-local}"
: "${TP_HOST:=127.0.0.1}"
: "${TP_NODE0_PORT:=42604}"
: "${TP_NODE1_PORT:=42605}"
: "${TP_COORDINATOR_PORT:=42606}"
: "${TP_DEPLOYMENT:=benchmark}"
: "${TP_COLLECTIVE_TRANSPORT:=netty}"
: "${TP_OWNER:=tjake}"
: "${TP_MODEL:=gemma-2-2b-it-JQ4}"
: "${TP_SIZE:=4}"
: "${TP_MAX_RANKS_PER_WORKER:=2}"
: "${TP_POOL_SIZE:=16}"
: "${TP_WORKING_DTYPE:=F32}"
: "${TP_WORKING_QTYPE:=I8}"
: "${TP_OUTPUT_HEAD_QUANTIZATION:=Q4}"
: "${TP_MAX_TOKENS:=64}"
: "${TP_TEMPERATURE:=0.0}"
: "${TP_READY_TIMEOUT_SECONDS:=120}"
: "${TP_RANK_ENDPOINT_TIMEOUT_SECONDS:=300}"
: "${TP_LOG_LEVEL:=info}"
: "${TP_PROMPT:=Explain tensor parallel inference in one short paragraph.}"

TP_LOCAL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DELIVERANCE_ROOT=$(CDPATH= cd -- "$TP_LOCAL_DIR/.." && pwd)
TP_RUN_DIR="$TP_LOCAL_DIR/run"
TP_LOG_DIR="$TP_LOCAL_DIR/logs"
TP_MAIN_CLASS="io.teknek.deliverance.benchmark.TpLocalCluster"
TP_COORDINATOR_CONFIG="$TP_LOCAL_DIR/coordinator.properties"
TP_NATIVE_LIB_DIR="$DELIVERANCE_ROOT/native/target/native-lib-only"

tp_seed_args() {
  printf '%s' "--seed node-0=udp://$TP_HOST:$TP_NODE0_PORT --seed node-1=udp://$TP_HOST:$TP_NODE1_PORT"
}

tp_common_args() {
  printf '%s' "--cluster $TP_CLUSTER --deployment $TP_DEPLOYMENT --collective-transport $TP_COLLECTIVE_TRANSPORT --owner $TP_OWNER --model $TP_MODEL --tensor-parallel-size $TP_SIZE --max-ranks-per-worker $TP_MAX_RANKS_PER_WORKER --pool-size $TP_POOL_SIZE --working-dtype $TP_WORKING_DTYPE --working-qtype $TP_WORKING_QTYPE --output-head-quantization $TP_OUTPUT_HEAD_QUANTIZATION --max-tokens $TP_MAX_TOKENS --temperature $TP_TEMPERATURE --ready-timeout-seconds $TP_READY_TIMEOUT_SECONDS --rank-endpoint-timeout-seconds $TP_RANK_ENDPOINT_TIMEOUT_SECONDS"
}

tp_classpath() {
  mkdir -p "$TP_RUN_DIR"
  if [ ! -d "$DELIVERANCE_ROOT/core/target/classes" ]; then
    printf '%s\n' "core/target/classes not found. Compile first: mvn -pl core -am -DskipTests compile" >&2
    exit 1
  fi
  mvn -q -f "$DELIVERANCE_ROOT/pom.xml" -pl core -DincludeScope=test dependency:build-classpath -Dmdep.outputFile="$TP_RUN_DIR/classpath.txt"
  printf '%s:%s:%s:%s:%s:%s:%s' \
    "$DELIVERANCE_ROOT/core/target/classes" \
    "$DELIVERANCE_ROOT/native/target/classes" \
    "$DELIVERANCE_ROOT/grace/target/classes" \
    "$DELIVERANCE_ROOT/safetensors/target/classes" \
    "$DELIVERANCE_ROOT/tensor/target/classes" \
    "$DELIVERANCE_ROOT/math/target/classes" \
    "$(cat "$TP_RUN_DIR/classpath.txt")"
}

tp_start() {
  name=$1
  shift
  mkdir -p "$TP_RUN_DIR" "$TP_LOG_DIR"
  pid_file="$TP_RUN_DIR/$name.pid"
  log_file="$TP_LOG_DIR/$name.log"
  stdout_file="$TP_LOG_DIR/$name.stdout.log"
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    printf '%s\n' "$name already running pid=$(cat "$pid_file")"
    exit 0
  fi
  classpath=$(tp_classpath)
  nohup java \
    -Djava.library.path="$TP_NATIVE_LIB_DIR" \
    -Dorg.slf4j.simpleLogger.logFile="$log_file" \
    -Dorg.slf4j.simpleLogger.defaultLogLevel="$TP_LOG_LEVEL" \
    -Dorg.slf4j.simpleLogger.showDateTime=true \
    -Dorg.slf4j.simpleLogger.dateTimeFormat="yyyy-MM-dd'T'HH:mm:ss.SSSZ" \
    --add-modules jdk.incubator.vector,jdk.httpserver,java.net.http \
    --add-opens java.base/java.nio=ALL-UNNAMED \
    --enable-native-access=ALL-UNNAMED \
    -cp "$classpath" \
    "$TP_MAIN_CLASS" "$@" >> "$stdout_file" 2>&1 &
  printf '%s\n' "$!" > "$pid_file"
  printf '%s\n' "started $name pid=$! log=$log_file stdout=$stdout_file"
}

tp_start_web() {
  name=$1
  shift
  mkdir -p "$TP_RUN_DIR" "$TP_LOG_DIR"
  pid_file="$TP_RUN_DIR/$name.pid"
  log_file="$TP_LOG_DIR/$name.log"
  stdout_file="$TP_LOG_DIR/$name.stdout.log"
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    printf '%s\n' "$name already running pid=$(cat "$pid_file")"
    exit 0
  fi
  web_jar=$(set -- "$DELIVERANCE_ROOT"/web/target/web-*-SNAPSHOT.jar; printf '%s' "$1")
  if [ ! -f "$web_jar" ]; then
    printf '%s\n' "web jar not found. Package first: mvn -pl web -am -DskipTests package" >&2
    exit 1
  fi
  nohup java \
    -Djava.library.path="$TP_NATIVE_LIB_DIR" \
    -XX:+UnlockDiagnosticVMOptions \
    -XX:CompilerDirectivesFile="$DELIVERANCE_ROOT/inlinerules.json" \
    -XX:+AlignVector \
    -XX:-UseCompactObjectHeaders \
    -XX:+UseStringDeduplication \
    --add-modules jdk.incubator.vector,jdk.httpserver,java.net.http \
    --add-opens java.base/java.nio=ALL-UNNAMED \
    --enable-native-access=ALL-UNNAMED \
    -jar "$web_jar" "$@" >> "$stdout_file" 2>&1 &
  printf '%s\n' "$!" > "$pid_file"
  printf '%s\n' "started $name pid=$! spring_log=$log_file stdout=$stdout_file"
}

tp_stop() {
  name=$1
  pid_file="$TP_RUN_DIR/$name.pid"
  if [ ! -f "$pid_file" ]; then
    printf '%s\n' "$name not running"
    exit 0
  fi
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    printf '%s\n' "stopped $name pid=$pid"
  else
    printf '%s\n' "$name pid file exists but process is not running"
  fi
  rm -f "$pid_file"
}

tp_status() {
  name=$1
  pid_file="$TP_RUN_DIR/$name.pid"
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    printf '%s\n' "$name running pid=$(cat "$pid_file")"
  else
    printf '%s\n' "$name stopped"
  fi
}

tp_dispatch() {
  name=$1
  action=$2
  shift 2
  case "$action" in
    start) tp_start "$name" "$@" ;;
    stop) tp_stop "$name" ;;
    status) tp_status "$name" ;;
    *) printf '%s\n' "usage: $0 start|stop|status" >&2; exit 2 ;;
  esac
}

tp_dispatch_web() {
  name=$1
  action=$2
  shift 2
  case "$action" in
    start) tp_start_web "$name" "$@" ;;
    stop) tp_stop "$name" ;;
    status) tp_status "$name" ;;
    *) printf '%s\n' "usage: $0 start|stop|status" >&2; exit 2 ;;
  esac
}

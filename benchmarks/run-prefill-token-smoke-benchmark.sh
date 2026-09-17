#!/usr/bin/env sh
set -eu

if [ -n "${JAVA_HOME:-}" ]; then
  PATH="$JAVA_HOME/bin:$PATH"
  JAVA_BIN="$JAVA_HOME/bin/java"
else
  JAVA_BIN="java"
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BENCHMARK_ROOT="$SCRIPT_DIR" BENCHMARK_SCRIPT_NAME="$(basename "$0" .sh)" . "$SCRIPT_DIR/benchmarks/benchmark-run-dir.sh"
OS_NAME=$(uname -s)
OS_ARCH=$(uname -m)
case "$OS_NAME:$OS_ARCH" in
  Darwin:arm64|Darwin:aarch64) NATIVE_CLASSIFIER=osx-aarch_64 ;;
  Darwin:x86_64) NATIVE_CLASSIFIER=osx-x86_64 ;;
  Linux:aarch64|Linux:arm64) NATIVE_CLASSIFIER=linux-aarch_64 ;;
  Linux:x86_64|Linux:amd64) NATIVE_CLASSIFIER=linux-x86_64 ;;
  *) printf '%s\n' "Unsupported native platform: $OS_NAME $OS_ARCH" >&2; exit 1 ;;
esac
NATIVE_CLASSIFIER=${DELIVERANCE_NATIVE_CLASSIFIER:-$NATIVE_CLASSIFIER}
NATIVE_LIB_DIR="$SCRIPT_DIR/native/target/native-lib-only/$NATIVE_CLASSIFIER"

if [ ! -d "$NATIVE_LIB_DIR" ]; then
  printf '%s\n' "Native library directory not found: $NATIVE_LIB_DIR" >&2
  printf '%s\n' "Build native first: (cd \"$SCRIPT_DIR/native\" && mvn test)" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

PREFILL_POOL_SIZE=${PREFILL_POOL_SIZE:-16}
PREFILL_TOKEN_COUNTS=${PREFILL_TOKEN_COUNTS:-100,500}
PREFILL_PROFILE_ARGS=${PREFILL_PROFILE_ARGS:-"--profile-stages --profile-rows 30"}
PREFILL_OUTPUT=${PREFILL_OUTPUT:-"$BENCHMARK_RUN_DIR/prefill-token-smoke.csv"}

run_model() {
  label=$1
  owner=$2
  model=$3
  model_config=$4

  config_args=""
  if [ -n "$model_config" ]; then
    config_args="--model-config $model_config"
  fi

  printf '\n[prefill-smoke] model=%s/%s label=%s token_counts=%s\n' "$owner" "$model" "$label" "$PREFILL_TOKEN_COUNTS"
  EXEC_ARGS="\
$BENCHMARK_JVM_OPTS \
-Djava.library.path=$NATIVE_LIB_DIR \
--add-modules jdk.incubator.vector,jdk.httpserver,java.net.http \
--add-opens java.base/java.nio=ALL-UNNAMED \
--enable-native-access=ALL-UNNAMED \
-cp %classpath \
io.teknek.deliverance.benchmark.PrefillTokenBenchmark \
--label $label \
--owner $owner \
--model $model \
$config_args \
--token-counts $PREFILL_TOKEN_COUNTS \
--output-head-quantization Q4 \
--pool-size $PREFILL_POOL_SIZE \
--output $PREFILL_OUTPUT \
$PREFILL_PROFILE_ARGS"

  mvn -q -pl core \
    -Dexec.classpathScope=test \
    -Dexec.executable="$JAVA_BIN" \
    -Dexec.args="$EXEC_ARGS" \
    org.codehaus.mojo:exec-maven-plugin:3.5.0:exec
}

QWEN06_CONFIG=${QWEN06_CONFIG:-"$SCRIPT_DIR/benchmarks/configs/qwen3-0.6b-jq4.json"}
QWEN4B_CONFIG=${QWEN4B_CONFIG:-"$SCRIPT_DIR/benchmarks/configs/qwen3-4b-jq4.json"}
GEMMA2_CONFIG=${GEMMA2_CONFIG:-""}

run_model qwen06b edwardcapriolo Qwen3-0.6B-JQ4 "$QWEN06_CONFIG"
run_model qwen4b edwardcapriolo Qwen3-4B-JQ4 "$QWEN4B_CONFIG"
run_model gemma2b tjake gemma-2-2b-it-JQ4 "$GEMMA2_CONFIG"

printf '\n[prefill-smoke] results=%s\n' "$PREFILL_OUTPUT"
printf '[prefill-smoke] run_dir=%s\n' "$BENCHMARK_RUN_DIR"

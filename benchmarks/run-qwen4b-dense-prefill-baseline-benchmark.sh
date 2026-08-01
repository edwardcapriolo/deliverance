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
MODEL_CONFIG="$SCRIPT_DIR/benchmarks/configs/qwen3-4b-dense-prefill-baseline.json"

if [ ! -d "$NATIVE_LIB_DIR" ]; then
  printf '%s\n' "Native library directory not found: $NATIVE_LIB_DIR" >&2
  printf '%s\n' "Build native first: (cd \"$SCRIPT_DIR/native\" && mvn test)" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable="$JAVA_BIN" \
  -Dexec.args="-Djava.library.path=$NATIVE_LIB_DIR --add-modules jdk.incubator.vector,jdk.httpserver,java.net.http --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.benchmark.InferenceBenchmark --engine deliverance --owner Qwen --model Qwen3-4B --model-config $MODEL_CONFIG ${DELIVERANCE_BENCHMARK_ARGS:---pool-size 16 --max-tokens 256 --warmup-cases 0 --max-cases 6 --profile-stages --output $BENCHMARK_RUN_DIR/qwen4b-dense-prefill-baseline-benchmark.csv --jsonl-output $BENCHMARK_RUN_DIR/qwen4b-dense-prefill-baseline-benchmark.jsonl}" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec

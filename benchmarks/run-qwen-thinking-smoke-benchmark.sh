#!/usr/bin/env sh
set -eu

if [ -n "${JAVA_HOME:-}" ]; then
  PATH="$JAVA_HOME/bin:$PATH"
  JAVA_BIN="$JAVA_HOME/bin/java"
else
  JAVA_BIN="java"
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
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
NATIVE_LIB_DIR="$ROOT_DIR/native/target/native-lib-only/$NATIVE_CLASSIFIER"
MODEL_CONFIG=${MODEL_CONFIG:-"$ROOT_DIR/benchmarks/configs/qwen3-4b-jq4.json"}
OUTPUT=${DELIVERANCE_THINKING_SMOKE_OUTPUT:-"$ROOT_DIR/core/target/thinking-smoke-qwen3-4b-jq4.jsonl"}

cd "$ROOT_DIR"

mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable="$JAVA_BIN" \
  -Dexec.args="-Djava.library.path=$NATIVE_LIB_DIR --add-modules jdk.incubator.vector,jdk.httpserver,java.net.http --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.benchmark.ThinkingSmokeBenchmark --owner edwardcapriolo --model Qwen3-4B-JQ4 --model-config $MODEL_CONFIG --output $OUTPUT --max-tokens 768 --thinking true" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec

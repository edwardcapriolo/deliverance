#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
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
DAWN_LIB="$SCRIPT_DIR/native/target/dawn/lib/libwebgpu_dawn.dylib"

if [ ! -d "$NATIVE_LIB_DIR" ]; then
  printf '%s\n' "Native library directory not found: $NATIVE_LIB_DIR" >&2
  printf '%s\n' "Build native first: (cd \"$SCRIPT_DIR/native\" && mvn test)" >&2
  exit 1
fi

if [ "$OS_NAME" = "Darwin" ] && [ ! -f "$NATIVE_LIB_DIR/libwebgpu_dawn.dylib" ] && [ -f "$DAWN_LIB" ]; then
  cp "$DAWN_LIB" "$NATIVE_LIB_DIR/"
fi

cd "$SCRIPT_DIR"

MAVEN_OPTS="${MAVEN_OPTS:-} -XX:TieredStopAtLevel=1" mvn -q -pl core -am -DskipTests compile

DEFAULT_BENCHMARK_ARGS="\
--tensor-provider native-gpu \
--output-head-quantization Q4 \
--pool-size 16 \
--max-tokens 256 \
--warmup-cases 0 \
--profile-stages \
--output target/qwen3-4b-gpu-benchmark.csv \
--jsonl-output target/qwen3-4b-gpu-benchmark.jsonl"

EXEC_ARGS="\
-Djava.library.path=$NATIVE_LIB_DIR \
--add-modules jdk.incubator.vector,jdk.httpserver,java.net.http \
--add-opens java.base/java.nio=ALL-UNNAMED \
--enable-native-access=ALL-UNNAMED \
-cp %classpath \
io.teknek.deliverance.benchmark.InferenceBenchmark \
--engine deliverance \
--owner Qwen \
--model Qwen3-4B-JQ4 \
${DELIVERANCE_BENCHMARK_ARGS:-$DEFAULT_BENCHMARK_ARGS}"

mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="$EXEC_ARGS" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec

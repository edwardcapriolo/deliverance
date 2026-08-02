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

cd "$SCRIPT_DIR"

DEFAULT_BENCHMARK_ARGS="\
--output-head-quantization Q4 \
--max-tokens 256 \
--warmup-cases 0 \
--profile-stages \
--output $BENCHMARK_RUN_DIR/deliverance-panama-benchmark.csv \
--jsonl-output $BENCHMARK_RUN_DIR/deliverance-panama-benchmark.jsonl"

EXEC_ARGS="\
--add-modules jdk.incubator.vector,jdk.httpserver,java.net.http \
--add-opens java.base/java.nio=ALL-UNNAMED \
--enable-native-access=ALL-UNNAMED \
-cp %classpath \
io.teknek.deliverance.benchmark.InferenceBenchmark \
--engine deliverance \
--owner tjake \
--model gemma-2-2b-it-JQ4 \
${DELIVERANCE_BENCHMARK_ARGS:-$DEFAULT_BENCHMARK_ARGS}"

mvn -q -pl core \
  -Dexec.executable="$JAVA_BIN" \
  -Dexec.args="$EXEC_ARGS" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec

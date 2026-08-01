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

mvn -q -pl tensor test-compile \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec \
  -Dexec.classpathScope=test \
  -Dexec.executable="$JAVA_BIN" \
  -Dexec.args="--add-modules jdk.incubator.vector --add-opens java.base/java.nio=ALL-UNNAMED -cp %classpath io.teknek.deliverance.tensorlib.VariableMlpFusionReplayBenchmark --output $BENCHMARK_RUN_DIR/variable-mlp-fusion-replay.csv --json-output $BENCHMARK_RUN_DIR/variable-mlp-fusion-replay.json ${VARIABLE_MLP_REPLAY_ARGS:---m-values 1,32,128,256,403 --hidden 3072}"

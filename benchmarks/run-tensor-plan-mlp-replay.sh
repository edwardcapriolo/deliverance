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

RUNTIME_MODE=${TENSOR_PLAN_RUNTIME_MODE:-DISABLED}
RUNTIME_WORKERS=${TENSOR_PLAN_RUNTIME_WORKERS:-4}
TENSOR_NUMA_NODE=${TENSOR_PLAN_TENSOR_NUMA_NODE:-1}
OUTPUT_ARGS="--output $BENCHMARK_RUN_DIR/tensor-plan-mlp-replay.csv --json-output $BENCHMARK_RUN_DIR/tensor-plan-mlp-replay.json"
RUNTIME_ARGS="--runtime-mode $RUNTIME_MODE --runtime-workers $RUNTIME_WORKERS --tensor-numa-node $TENSOR_NUMA_NODE"
SHAPE_ARGS=${TENSOR_PLAN_REPLAY_ARGS:---m-values 32,64,128 --hidden 512 --intermediate 1536 --pool-size $RUNTIME_WORKERS}

mvn -q -pl tensor test-compile \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec \
  -Dexec.classpathScope=test \
  -Dexec.executable="$JAVA_BIN" \
  -Dexec.args="--add-modules jdk.incubator.vector --add-opens java.base/java.nio=ALL-UNNAMED -cp %classpath io.teknek.deliverance.tensorlib.TensorPlanMlpReplayBenchmark $OUTPUT_ARGS $RUNTIME_ARGS $SHAPE_ARGS"

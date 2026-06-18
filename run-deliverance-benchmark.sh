#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NATIVE_LIB_DIR="$SCRIPT_DIR/native/target/native-lib-only"

if [ ! -d "$NATIVE_LIB_DIR" ]; then
  printf '%s\n' "Native library directory not found: $NATIVE_LIB_DIR" >&2
  printf '%s\n' "Build native first: (cd \"$SCRIPT_DIR/native\" && mvn test)" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="-Djava.library.path=$NATIVE_LIB_DIR --add-modules jdk.incubator.vector,jdk.httpserver,java.net.http --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.benchmark.InferenceBenchmark --engine deliverance --owner tjake --model gemma-2-2b-it-JQ4 ${DELIVERANCE_BENCHMARK_ARGS:---tensor-parallel-size 4 --tensor-parallel-max-ranks-per-worker 2 --output-head-quantization Q4 --max-tokens 256 --warmup-cases 0 --profile-stages --output target/deliverance-tp-benchmark.csv --jsonl-output target/deliverance-tp-benchmark.jsonl}" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec

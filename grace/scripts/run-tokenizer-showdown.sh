#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

MODEL_DIR=${MODEL_DIR:-"$HOME/.deliverance/Qwen_Qwen3-0.6B"}
TEXT_FILE=${TEXT_FILE:-"$ROOT_DIR/grace/src/test/resources/tokenizer-showdown.txt"}
REPEAT=${REPEAT:-64}
WARMUP=${WARMUP:-20}
ITERATIONS=${ITERATIONS:-200}
PYTHON=${PYTHON:-python3}

cd "$ROOT_DIR"

MAVEN_OPTS="${MAVEN_OPTS:-} -XX:TieredStopAtLevel=1" mvn -q -pl grace -am -DskipTests test-compile

COMMON_ARGS="--model-dir $MODEL_DIR --text-file $TEXT_FILE --repeat $REPEAT --warmup $WARMUP --iterations $ITERATIONS"

echo "== Grace Java =="
mvn -q -pl grace \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="--add-modules jdk.incubator.vector -cp %classpath io.teknek.deliverance.grace.GraceTokenizerBenchmarkMain $COMMON_ARGS" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec

echo "Cooling down for 10 seconds before Hugging Face run..."
sleep 10

echo "== Hugging Face tokenizers =="
"$PYTHON" grace/scripts/hf_fast_tokenizer_benchmark.py $COMMON_ARGS

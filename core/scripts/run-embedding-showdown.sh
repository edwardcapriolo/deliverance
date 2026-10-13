#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

MODEL_DIR=${MODEL_DIR:-"$HOME/.deliverance/google_embeddinggemma-2"}
CORPUS=${CORPUS:-"$ROOT_DIR/core/src/test/resources/embedding-showdown.jsonl"}
WARMUP=${WARMUP:-3}
ITERATIONS=${ITERATIONS:-10}
POOL_SIZE=${POOL_SIZE:-"$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 1)"}
PYTHON=${PYTHON:-python3}
JAVA_OUTPUT=${JAVA_OUTPUT:-"$ROOT_DIR/core/target/embedding-showdown-java.json"}
PYTHON_OUTPUT=${PYTHON_OUTPUT:-"$ROOT_DIR/core/target/embedding-showdown-python.json"}
FORCE_BUILD=${FORCE_BUILD:-0}

cd "$ROOT_DIR"
JAVA_MAIN_CLASS="$ROOT_DIR/core/target/test-classes/io/teknek/deliverance/embedding/EmbeddingShowdownMain.class"
NEEDS_BUILD=0
if [ ! -f "$JAVA_MAIN_CLASS" ]; then
  NEEDS_BUILD=1
elif find "$ROOT_DIR/core/src/main/java" "$ROOT_DIR/core/src/test/java" "$ROOT_DIR/tensor/src/main/java" \
    -type f -name '*.java' -newer "$JAVA_MAIN_CLASS" -print -quit | grep -q .; then
  NEEDS_BUILD=1
fi
if [ "$FORCE_BUILD" = "1" ] || [ "$NEEDS_BUILD" = "1" ]; then
  echo "Compiling embedding showdown runner..."
  MAVEN_OPTS="${MAVEN_OPTS:-} -XX:TieredStopAtLevel=1" mvn -q -pl core -am -DskipTests test-compile
fi

COMMON_ARGS="--model-dir $MODEL_DIR --corpus $CORPUS --warmup $WARMUP --iterations $ITERATIONS"

echo "== Deliverance Java =="
mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="--add-modules jdk.incubator.vector --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.embedding.EmbeddingShowdownMain $COMMON_ARGS --pool-size $POOL_SIZE --output $JAVA_OUTPUT" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec

echo "== Sentence Transformers Python =="
"$PYTHON" core/scripts/sentence_transformers_embedding_showdown.py $COMMON_ARGS --output "$PYTHON_OUTPUT"

echo "== Parity and comparison =="
"$PYTHON" core/scripts/compare_embedding_showdown.py --java "$JAVA_OUTPUT" --python "$PYTHON_OUTPUT"

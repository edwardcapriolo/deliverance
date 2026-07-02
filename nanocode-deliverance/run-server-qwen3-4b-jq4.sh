#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
WEB_DIR="$ROOT_DIR/web"
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
DAWN_LIB="$ROOT_DIR/native/target/dawn/lib/libwebgpu_dawn.dylib"

export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-25-temurin-jdk}
PATH="$JAVA_HOME/bin:$PATH"

if [ "$OS_NAME" = "Darwin" ] && [ ! -f "$NATIVE_LIB_DIR/libwebgpu_dawn.dylib" ] && [ -f "$DAWN_LIB" ]; then
  cp "$DAWN_LIB" "$NATIVE_LIB_DIR/"
fi

cd "$WEB_DIR"

java \
  -XX:+UnlockDiagnosticVMOptions \
  -XX:CompilerDirectivesFile=../inlinerules.json \
  -XX:+AlignVector \
  -XX:-UseCompactObjectHeaders \
  -XX:+UseStringDeduplication \
  --add-opens java.base/java.nio=ALL-UNNAMED \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  --sun-misc-unsafe-memory-access=allow \
  -Xmx8G \
  -Djava.library.path="$NATIVE_LIB_DIR" \
  -Dserver.port=${DELIVERANCE_PORT:-8085} \
  ${DELIVERANCE_KV_DISK_DIR:+-Ddeliverance.kv.disk-dir=$DELIVERANCE_KV_DISK_DIR} \
  -Ddeliverance.debug.chat-request=${DELIVERANCE_DEBUG_CHAT_REQUEST:-true} \
  -Ddebug=${DELIVERANCE_SPRING_DEBUG:-false} \
  -Dlogging.level.root=${DELIVERANCE_ROOT_LOG_LEVEL:-INFO} \
  -Dlogging.level.net.deliverance.http=${DELIVERANCE_HTTP_LOG_LEVEL:-INFO} \
  -Dlogging.level.net.deliverance.http.ChatCompletionController=${DELIVERANCE_CHAT_LOG_LEVEL:-INFO} \
  -Dlogging.level.net.deliverance.http.ChatCompletionService=${DELIVERANCE_CHAT_LOG_LEVEL:-INFO} \
  -Dlogging.level.io.teknek.deliverance=${DELIVERANCE_CORE_LOG_LEVEL:-INFO} \
  -Dlogging.level.io.teknek.deliverance.model=${DELIVERANCE_MODEL_LOG_LEVEL:-INFO} \
  -Dlogging.level.io.teknek.deliverance.generator=${DELIVERANCE_GENERATOR_LOG_LEVEL:-INFO} \
  -Dlogging.level.org.springframework.web=${DELIVERANCE_SPRING_WEB_LOG_LEVEL:-INFO} \
  -Dserver.undertow.accesslog.enabled=true \
  -Dserver.undertow.accesslog.dir=${DELIVERANCE_ACCESS_LOG_DIR:-./logs} \
  -Dserver.undertow.accesslog.prefix=access_log \
  -Dserver.undertow.accesslog.suffix=.log \
  -Dspring.config.location=file:./qwen3-4b-jq4.properties \
  -jar target/web-*-SNAPSHOT.jar

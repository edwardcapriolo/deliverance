#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
WEB_DIR="$ROOT_DIR/web"

export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-25-temurin-jdk}
PATH="$JAVA_HOME/bin:$PATH"

cd "$WEB_DIR"

java \
  -XX:+UnlockDiagnosticVMOptions \
  -XX:CompilerDirectivesFile=../inlinerules.json \
  -XX:+AlignVector \
  -XX:-UseCompactObjectHeaders \
  -XX:+UseStringDeduplication \
  --add-opens java.base/java.nio=ALL-UNNAMED \
  --add-modules jdk.incubator.vector \
  -Xmx4G \
  -Dserver.port=${RECKONING_PORT:-8085} \
  ${RECKONING_KV_DISK_DIR:+-Dreckoning.kv.disk-dir=$RECKONING_KV_DISK_DIR} \
  -Dreckoning.debug.chat-request=${RECKONING_DEBUG_CHAT_REQUEST:-true} \
  -Ddebug=${RECKONING_SPRING_DEBUG:-false} \
  -Dlogging.level.root=${RECKONING_ROOT_LOG_LEVEL:-INFO} \
  -Dlogging.level.net.reckoning.http=${RECKONING_HTTP_LOG_LEVEL:-DEBUG} \
  -Dlogging.level.net.reckoning.http.ChatCompletionController=${RECKONING_CHAT_LOG_LEVEL:-DEBUG} \
  -Dlogging.level.net.reckoning.http.ChatCompletionService=${RECKONING_CHAT_LOG_LEVEL:-DEBUG} \
  -Dlogging.level.io.teknek.reckoning=${RECKONING_CORE_LOG_LEVEL:-INFO} \
  -Dlogging.level.io.teknek.reckoning.model=${RECKONING_MODEL_LOG_LEVEL:-DEBUG} \
  -Dlogging.level.io.teknek.reckoning.model.AbstractModel=${RECKONING_MODEL_LOG_LEVEL:-INFO} \
  -Dlogging.level.io.teknek.reckoning.generator=${RECKONING_GENERATOR_LOG_LEVEL:-INFO} \
  -Dlogging.level.org.springframework.web=${RECKONING_SPRING_WEB_LOG_LEVEL:-DEBUG} \
  -Dserver.undertow.accesslog.enabled=true \
  -Dserver.undertow.accesslog.dir=${RECKONING_ACCESS_LOG_DIR:-./logs} \
  -Dserver.undertow.accesslog.prefix=access_log \
  -Dserver.undertow.accesslog.suffix=.log \
  -Dspring.config.location=file:./llama-3.2-3b-instruct.yaml \
  -jar target/web-*-exec.jar

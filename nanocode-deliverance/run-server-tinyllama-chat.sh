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
  -Dserver.port=${DELIVERANCE_PORT:-8085} \
  ${DELIVERANCE_KV_DISK_DIR:+-Ddeliverance.kv.disk-dir=$DELIVERANCE_KV_DISK_DIR} \
  -Ddeliverance.debug.chat-request=${DELIVERANCE_DEBUG_CHAT_REQUEST:-true} \
  -Ddebug=${DELIVERANCE_SPRING_DEBUG:-false} \
  -Dlogging.level.root=${DELIVERANCE_ROOT_LOG_LEVEL:-INFO} \
  -Dlogging.level.net.deliverance.http=${DELIVERANCE_HTTP_LOG_LEVEL:-DEBUG} \
  -Dlogging.level.net.deliverance.http.ChatCompletionController=${DELIVERANCE_CHAT_LOG_LEVEL:-DEBUG} \
  -Dlogging.level.net.deliverance.http.ChatCompletionService=${DELIVERANCE_CHAT_LOG_LEVEL:-DEBUG} \
  -Dlogging.level.io.teknek.deliverance=${DELIVERANCE_CORE_LOG_LEVEL:-INFO} \
  -Dlogging.level.io.teknek.deliverance.model=${DELIVERANCE_MODEL_LOG_LEVEL:-DEBUG} \
  -Dlogging.level.io.teknek.deliverance.model.AbstractModel=${DELIVERANCE_MODEL_LOG_LEVEL:-INFO} \
  -Dlogging.level.io.teknek.deliverance.generator=${DELIVERANCE_GENERATOR_LOG_LEVEL:-INFO} \
  -Dlogging.level.org.springframework.web=${DELIVERANCE_SPRING_WEB_LOG_LEVEL:-DEBUG} \
  -Dserver.undertow.accesslog.enabled=true \
  -Dserver.undertow.accesslog.dir=${DELIVERANCE_ACCESS_LOG_DIR:-./logs} \
  -Dserver.undertow.accesslog.prefix=access_log \
  -Dserver.undertow.accesslog.suffix=.log \
  -Dspring.config.location=file:./tinyllama-chat.properties \
  -jar target/web-*-SNAPSHOT.jar

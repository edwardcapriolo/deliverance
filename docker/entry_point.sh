#!/usr/bin/env bash
D_OPTS="$D_OPTS --add-modules jdk.incubator.vector "
D_OPTS="$D_OPTS -XX:+AlignVector -XX:-UseCompactObjectHeaders -XX:+UseStringDeduplication "
D_OPTS="$D_OPTS -XX:+UnlockDiagnosticVMOptions -XX:CompilerDirectivesFile=inlinerules.json "
D_OPTS="$D_OPTS --add-opens java.base/java.nio=ALL-UNNAMED "

D_OPTS="$D_OPTS $DELIVERANCE_OPTS"

java $D_OPTS -jar web.jar

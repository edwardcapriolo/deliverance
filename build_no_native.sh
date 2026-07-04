#!/usr/bin/env sh
set -eu

MAVEN_OPTS=${MAVEN_OPTS:-"-XX:TieredStopAtLevel=1"}
export MAVEN_OPTS

mvn -q -Pnative-java-only -DskipTests package

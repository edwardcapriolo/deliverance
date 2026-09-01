#!/usr/bin/env bash
set -euo pipefail

COLIMA_PROFILE="${TESTQEMU_COLIMA_PROFILE:-x86}"
DOCKER_CONTEXT="${TESTQEMU_DOCKER_CONTEXT:-colima-x86}"
IMAGE="${TESTQEMU_IMAGE:-ecapriolo/trusted-opencode:0.0.3-cloudops}"
PLATFORM="${TESTQEMU_PLATFORM:-linux/amd64}"
WORKDIR="${TESTQEMU_WORKDIR:-/workspace}"
M2_DIR="${TESTQEMU_M2_DIR:-$HOME/.m2}"
MAVEN_LOCAL_REPO="${TESTQEMU_MAVEN_LOCAL_REPO:-/m2/repository}"
CONTAINER_USER="${TESTQEMU_CONTAINER_USER:-$(id -u):$(id -g)}"
TESTS="${TESTQEMU_TESTS:-NativeSimdTensorOpsFuzzParityTest,NativeSimdQwenTpShapeTest}"
MAVEN_OPTS_VALUE="${TESTQEMU_MAVEN_OPTS:--XX:TieredStopAtLevel=1 -XX:UseAVX=0}"
MAVEN_CMD="${TESTQEMU_MAVEN_CMD:-MAVEN_OPTS=\"$MAVEN_OPTS_VALUE\" mvn -Dmaven.repo.local=$MAVEN_LOCAL_REPO -pl native -am -Dtest=$TESTS -Dsurefire.failIfNoSpecifiedTests=false test}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

if ! command -v colima >/dev/null 2>&1; then
  echo "colima is not installed or not on PATH" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not on PATH" >&2
  exit 1
fi

status="$(colima list 2>/dev/null | awk -v profile="$COLIMA_PROFILE" 'NR > 1 && $1 == profile { print $2; found=1 } END { if (!found) exit 1 }' || true)"
if [ -z "$status" ]; then
  echo "Colima profile '$COLIMA_PROFILE' was not found" >&2
  echo "Create it with: colima start $COLIMA_PROFILE --arch x86_64 --vm-type=qemu --cpu 4 --memory 16" >&2
  exit 1
fi

if [ "$status" != "Running" ]; then
  echo "Starting Colima profile '$COLIMA_PROFILE' for amd64 native tests"
  colima start "$COLIMA_PROFILE"
fi

if ! docker --context "$DOCKER_CONTEXT" info >/dev/null 2>&1; then
  echo "Docker context '$DOCKER_CONTEXT' is not reachable" >&2
  exit 1
fi

arch="$(docker --context "$DOCKER_CONTEXT" run --rm \
  --platform "$PLATFORM" \
  --entrypoint sh \
  "$IMAGE" \
  -lc 'uname -m')"
if [ "$arch" != "x86_64" ]; then
  echo "Expected x86_64 container architecture, got '$arch'" >&2
  exit 1
fi

echo "Running amd64 native tests"
echo "  colima_profile=$COLIMA_PROFILE"
echo "  docker_context=$DOCKER_CONTEXT"
echo "  image=$IMAGE"
echo "  repo=$REPO_DIR"
echo "  m2=$M2_DIR"
echo "  maven_repo=$MAVEN_LOCAL_REPO"
echo "  user=$CONTAINER_USER"
echo "  command=$MAVEN_CMD"

docker --context "$DOCKER_CONTEXT" run --rm \
  --platform "$PLATFORM" \
  --user "$CONTAINER_USER" \
  -e HOME=/tmp/testqemu-home \
  --entrypoint sh \
  --mount "type=bind,src=$REPO_DIR,dst=$WORKDIR" \
  --mount "type=bind,src=$M2_DIR,dst=/m2" \
  -w "$WORKDIR" \
  "$IMAGE" \
  -lc "mkdir -p /tmp/testqemu-home && $MAVEN_CMD"

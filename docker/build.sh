#!/usr/bin/env bash
set -euo pipefail

ARM_CONTEXT="${ARM_CONTEXT:-default}"
AMD_CONTEXT="${AMD_CONTEXT:-colima-x86}"
ARM_COLIMA_PROFILE="${ARM_COLIMA_PROFILE:-default}"
AMD_COLIMA_PROFILE="${AMD_COLIMA_PROFILE:-x86}"
CHECK_COLIMA_PROFILES="${CHECK_COLIMA_PROFILES:-false}"
NO_CACHE="${NO_CACHE:-false}"
BUILD_ARM="${BUILD_ARM:-true}"
BUILD_AMD="${BUILD_AMD:-true}"
MAVEN_OPTS="${MAVEN_OPTS:--XX:TieredStopAtLevel=1}"

if [ -f ./inc.sh ]; then
  . ./inc.sh
fi

check_colima_profile() {
  local profile="$1"
  local arch="$2"

  if [ "$CHECK_COLIMA_PROFILES" != "true" ]; then
    return 0
  fi
  if ! command -v colima >/dev/null 2>&1; then
    echo "colima is not installed or not on PATH; set CHECK_COLIMA_PROFILES=false to skip this check" >&2
    exit 1
  fi
  local status
  if ! status="$(colima list 2>/dev/null | awk -v profile="$profile" 'NR > 1 && $1 == profile { print $2; found=1 } END { if (!found) exit 1 }')"; then
    echo "Colima profile '$profile' for $arch was not found" >&2
    exit 1
  fi
  if [ "$status" != "Running" ]; then
    echo "Colima profile '$profile' for $arch is $status" >&2
    exit 1
  fi
}

check_docker_context() {
  local context="$1"
  local arch="$2"

  if ! docker --context "$context" info >/dev/null 2>&1; then
    echo "Docker context '$context' for $arch is not reachable" >&2
    exit 1
  fi
  if ! docker --context "$context" buildx inspect >/dev/null 2>&1; then
    echo "Docker context '$context' for $arch is not usable with buildx" >&2
    exit 1
  fi
}

build_one() {
  local platform="$1"
  local context="$2"
  local suffix="$3"
  local cmd=(
    docker --context "$context" buildx build
    --platform "$platform"
    --build-arg "JDK_IMAGE_REPO=$JDK_IMAGE_REPO"
    --build-arg "JDK_DEVEL_TAG=$JDK_VERSION-devel-$suffix"
    --build-arg "JDK_RUNTIME_TAG=$JDK_VERSION-devel-$suffix"
    --build-arg "REPO_URL=$REPO_URL"
    --build-arg "REPO_COMMIT_SHA=$REPO_COMMIT_SHA"
    --build-arg "MAVEN_OPTS=$MAVEN_OPTS"
    --target deliverance
    -t "$IMAGE_REPO:$VERSION-$suffix"
    -t "$IMAGE_REPO:latest-$suffix"
    --load
    .
  )
  if [ "$NO_CACHE" = "true" ]; then
    cmd=("${cmd[@]:0:${#cmd[@]}-1}" --no-cache "${cmd[-1]}")
  fi
  "${cmd[@]}"
}

if [ "$BUILD_ARM" = "true" ]; then
  check_colima_profile "$ARM_COLIMA_PROFILE" ARM
  check_docker_context "$ARM_CONTEXT" ARM
  build_one linux/arm64 "$ARM_CONTEXT" arm64
fi

if [ "$BUILD_AMD" = "true" ]; then
  check_colima_profile "$AMD_COLIMA_PROFILE" AMD
  check_docker_context "$AMD_CONTEXT" AMD
  build_one linux/amd64 "$AMD_CONTEXT" amd64
fi

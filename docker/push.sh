#!/usr/bin/env bash
set -euo pipefail

ARM_CONTEXT="${ARM_CONTEXT:-default}"
AMD_CONTEXT="${AMD_CONTEXT:-colima-x86}"

if [ -f ./inc.sh ]; then
  . ./inc.sh
fi

docker --context "$ARM_CONTEXT" push "$IMAGE_REPO:$VERSION-arm64"
docker --context "$ARM_CONTEXT" push "$IMAGE_REPO:latest-arm64"
docker --context "$AMD_CONTEXT" push "$IMAGE_REPO:$VERSION-amd64"
docker --context "$AMD_CONTEXT" push "$IMAGE_REPO:latest-amd64"

docker buildx imagetools create -t "$IMAGE_REPO:$VERSION" \
  "$IMAGE_REPO:$VERSION-amd64" \
  "$IMAGE_REPO:$VERSION-arm64"

docker buildx imagetools create -t "$IMAGE_REPO:latest" \
  "$IMAGE_REPO:latest-amd64" \
  "$IMAGE_REPO:latest-arm64"

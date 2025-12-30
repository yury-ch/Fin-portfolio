#!/bin/sh
set -eu

IMAGE_NAME=${1:-sp500-microservices}

CONTAINER_ID=$(docker ps --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}}" | head -n 1)

if [ -z "${CONTAINER_ID}" ]; then
  echo "No running container found for image ${IMAGE_NAME}"
  exit 0
fi

echo "Stopping container ${CONTAINER_ID}..."
docker stop "${CONTAINER_ID}"

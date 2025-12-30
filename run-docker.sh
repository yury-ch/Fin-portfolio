#!/bin/sh
set -eu

IMAGE_NAME=${1:-sp500-microservices}
PORT=${PORT:-8501}

docker run \
  -p 8000:8000 \
  -p "${PORT}":8501 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$(pwd)/sp500_data:/app/sp500_data" \
  "${IMAGE_NAME}"

#!/bin/sh
set -eu

PORT="${PORT:-8501}"
TICKER_PID=""
DATA_PID=""
CALC_PID=""

cleanup() {
  echo "Stopping services..."
  if [ -n "${TICKER_PID}" ]; then
    kill "${TICKER_PID}" 2>/dev/null || true
  fi
  if [ -n "${DATA_PID}" ]; then
    kill "${DATA_PID}" 2>/dev/null || true
  fi
  if [ -n "${CALC_PID}" ]; then
    kill "${CALC_PID}" 2>/dev/null || true
  fi
}

trap cleanup INT TERM

# Wait for a service /health endpoint to return HTTP 200.
# Usage: wait_for_healthy <url> <service_name> [max_attempts] [sleep_seconds]
wait_for_healthy() {
  url="$1"
  name="$2"
  max="${3:-30}"
  delay="${4:-2}"
  attempt=1
  echo "Waiting for ${name} at ${url}..."
  while [ "${attempt}" -le "${max}" ]; do
    if wget -qO- "${url}" >/dev/null 2>&1; then
      echo "${name} is healthy."
      return 0
    fi
    echo "  attempt ${attempt}/${max} — retrying in ${delay}s"
    sleep "${delay}"
    attempt=$(( attempt + 1 ))
  done
  echo "ERROR: ${name} did not become healthy after ${max} attempts. Aborting." >&2
  cleanup
  exit 1
}

echo "Starting ticker_service on port 8000..."
python services/ticker_service.py &
TICKER_PID=$!
wait_for_healthy "http://localhost:8000/health" "ticker_service"

echo "Starting data_service on port 8001..."
python services/data_service.py &
DATA_PID=$!
wait_for_healthy "http://localhost:8001/health" "data_service"

echo "Starting calculation_service on port 8002..."
python services/calculation_service.py &
CALC_PID=$!
wait_for_healthy "http://localhost:8002/health" "calculation_service"

echo "All backend services healthy. Starting Streamlit UI on port ${PORT}..."
streamlit run services/presentation_service.py --server.port="${PORT}" --server.address=0.0.0.0

cleanup

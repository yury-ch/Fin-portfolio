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

python services/ticker_service.py &
TICKER_PID=$!

python services/data_service.py &
DATA_PID=$!
python services/calculation_service.py &
CALC_PID=$!

streamlit run services/presentation_service.py --server.port="${PORT}" --server.address=0.0.0.0

cleanup

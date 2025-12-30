#!/bin/sh
# stop-microservices.sh
# -------------------------------
# Stop S&P 500 Portfolio Optimizer - Microservices Version
# -------------------------------

set -eu

printf "🛑 Stopping S&P 500 Portfolio Optimizer (Microservices)\n"
printf "=======================================================\n"

SERVICES=$(cat <<'EOF'
Ticker Service|ticker_service.py|8000
Data Service|data_service.py|8001
Calculation Service|calculation_service.py|8002
Presentation Service (Streamlit)|presentation_service.py|8501
EOF
)

stop_service() {
  name="$1"
  pattern="$2"
  port="$3"

  printf "\n🔧 Checking %s (port %s)...\n" "$name" "$port"

  pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}')

  if [ -z "$pids" ]; then
    pids=$(lsof -t -i:"$port" 2>/dev/null || true)
  fi

  if [ -z "$pids" ]; then
    printf "ℹ️  %s is not running.\n" "$name"
    return 0
  fi

  printf "📋 Found process IDs: %s\n" "$pids"

  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      printf "   • Sending SIGTERM to %s\n" "$pid"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  sleep 2

  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      printf "   • Force killing %s\n" "$pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done

  if lsof -i:"$port" >/dev/null 2>&1; then
    printf "⚠️  Port %s still in use. Inspect with: lsof -i:%s\n" "$port" "$port"
  else
    printf "✅ %s stopped and port %s is free.\n" "$name" "$port"
  fi
}

printf "%s\n" "$SERVICES" | while IFS='|' read -r NAME PATTERN PORT; do
  stop_service "$NAME" "$PATTERN" "$PORT"
done

printf "\n📋 Microservices shutdown complete.\n"
printf "✅ All targeted services have been processed.\n"

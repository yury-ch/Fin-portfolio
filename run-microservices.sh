#!/bin/bash
# run-microservices.sh
# Execute S&P 500 Portfolio Optimizer - Microservices Version

echo "🚀 Starting S&P 500 Portfolio Optimizer (Microservices)"
echo "======================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements-microservices.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
echo "🔍 Checking dependencies..."
if ! python -c "import streamlit, yfinance, pandas, numpy, pypfopt, fastapi, uvicorn, pydantic, requests" 2>/dev/null; then
    echo "⚠️  Installing missing dependencies..."
    pip install -r requirements-microservices.txt
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$DATA_PID" ]; then
        kill $DATA_PID 2>/dev/null
    fi
    if [ ! -z "$CALC_PID" ]; then
        kill $CALC_PID 2>/dev/null
    fi
    if [ ! -z "$PRESENTATION_PID" ]; then
        kill $PRESENTATION_PID 2>/dev/null
    fi
    echo "✅ All services stopped"
    exit 0
}

# Set up cleanup trap
trap cleanup SIGINT SIGTERM

echo ""
echo "🔧 Starting backend services..."

# Start Data Service
echo "📊 Starting Data Service (port 8001)..."
python services/data_service.py &
DATA_PID=$!
sleep 3

# Start Calculation Service
echo "🔢 Starting Calculation Service (port 8002)..."
python services/calculation_service.py &
CALC_PID=$!
sleep 3

# Check if backend services are running
if ! kill -0 $DATA_PID 2>/dev/null || ! kill -0 $CALC_PID 2>/dev/null; then
    echo "❌ Failed to start backend services"
    cleanup
fi

echo "✅ Backend services started successfully!"
echo ""
echo "🌐 Starting Presentation Service..."
echo ""
echo "📍 Access points:"
echo "   • Web UI: http://localhost:8501"
echo "   • Data Service API: http://localhost:8001/docs"
echo "   • Calculation Service API: http://localhost:8002/docs"
echo ""
echo "⏹️  Press Ctrl+C to stop all services"
echo ""

# Start Presentation Service (foreground)
streamlit run services/presentation_service.py &
PRESENTATION_PID=$!

# Wait for presentation service or user interrupt
wait $PRESENTATION_PID
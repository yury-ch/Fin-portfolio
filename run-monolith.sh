#!/bin/bash
# run-monolith.sh
# Execute S&P 500 Portfolio Optimizer - Monolith Version

echo "🚀 Starting S&P 500 Portfolio Optimizer (Monolith)"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
echo "🔍 Checking dependencies..."
if ! python -c "import streamlit, yfinance, pandas, numpy, pypfopt" 2>/dev/null; then
    echo "⚠️  Installing missing dependencies..."
    pip install -r requirements.txt
fi

# Start the application
echo "🌐 Starting Streamlit application..."
echo "📍 Access the app at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

streamlit run app.py
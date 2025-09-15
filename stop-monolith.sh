#!/bin/bash
# stop-monolith.sh
# -------------------------------
# Stop S&P 500 Portfolio Optimizer - Monolith Version
# -------------------------------

echo "🛑 Stopping S&P 500 Portfolio Optimizer (Monolith)"
echo "=================================================="

# Function to find and kill Streamlit processes
stop_streamlit() {
    echo "🔍 Searching for Streamlit processes..."
    
    # Find Streamlit processes running app.py
    STREAMLIT_PIDS=$(ps aux | grep -E "(streamlit.*app\.py|python.*streamlit.*run.*app\.py)" | grep -v grep | awk '{print $2}')
    
    if [ -z "$STREAMLIT_PIDS" ]; then
        echo "ℹ️  No Streamlit processes found running app.py"
        return 0
    fi
    
    echo "📋 Found Streamlit processes:"
    ps aux | grep -E "(streamlit.*app\.py|python.*streamlit.*run.*app\.py)" | grep -v grep
    echo ""
    
    # Gracefully terminate processes
    for pid in $STREAMLIT_PIDS; do
        echo "🔄 Sending SIGTERM to process $pid..."
        kill -TERM "$pid" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✅ Sent termination signal to process $pid"
        else
            echo "⚠️  Process $pid may have already stopped"
        fi
    done
    
    # Wait a moment for graceful shutdown
    echo "⏳ Waiting 3 seconds for graceful shutdown..."
    sleep 3
    
    # Check if processes are still running
    REMAINING_PIDS=$(ps aux | grep -E "(streamlit.*app\.py|python.*streamlit.*run.*app\.py)" | grep -v grep | awk '{print $2}')
    
    if [ -n "$REMAINING_PIDS" ]; then
        echo "⚠️  Some processes still running. Force killing..."
        for pid in $REMAINING_PIDS; do
            echo "💀 Force killing process $pid..."
            kill -KILL "$pid" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ Force killed process $pid"
            else
                echo "⚠️  Process $pid may have already stopped"
            fi
        done
        sleep 1
    fi
    
    # Final verification
    FINAL_CHECK=$(ps aux | grep -E "(streamlit.*app\.py|python.*streamlit.*run.*app\.py)" | grep -v grep)
    if [ -z "$FINAL_CHECK" ]; then
        echo "✅ All Streamlit processes stopped successfully"
        return 0
    else
        echo "❌ Some processes may still be running:"
        echo "$FINAL_CHECK"
        return 1
    fi
}

# Function to stop any general Streamlit processes on port 8501
stop_port_8501() {
    echo "🔍 Checking for processes using port 8501..."
    
    # Find processes using port 8501 (default Streamlit port)
    PORT_PIDS=$(lsof -t -i:8501 2>/dev/null)
    
    if [ -z "$PORT_PIDS" ]; then
        echo "ℹ️  No processes found using port 8501"
        return 0
    fi
    
    echo "📋 Found processes using port 8501:"
    lsof -i:8501 2>/dev/null
    echo ""
    
    for pid in $PORT_PIDS; do
        PROC_INFO=$(ps -p "$pid" -o pid,ppid,comm,args 2>/dev/null)
        if [ -n "$PROC_INFO" ]; then
            echo "🔄 Stopping process $pid using port 8501..."
            kill -TERM "$pid" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ Sent termination signal to process $pid"
            fi
        fi
    done
    
    sleep 2
    
    # Force kill if still running
    REMAINING_PORT_PIDS=$(lsof -t -i:8501 2>/dev/null)
    if [ -n "$REMAINING_PORT_PIDS" ]; then
        echo "⚠️  Force killing remaining processes on port 8501..."
        for pid in $REMAINING_PORT_PIDS; do
            kill -KILL "$pid" 2>/dev/null
        done
    fi
}

# Function to cleanup virtual environment processes
cleanup_venv() {
    echo "🧹 Cleaning up virtual environment processes..."
    
    # Find Python processes running from this project's venv
    VENV_PIDS=$(ps aux | grep -E "\.venv.*python.*streamlit" | grep -v grep | awk '{print $2}')
    
    if [ -n "$VENV_PIDS" ]; then
        echo "📋 Found virtual environment Python processes:"
        ps aux | grep -E "\.venv.*python.*streamlit" | grep -v grep
        
        for pid in $VENV_PIDS; do
            echo "🔄 Stopping venv process $pid..."
            kill -TERM "$pid" 2>/dev/null
            sleep 1
            # Force kill if needed
            kill -KILL "$pid" 2>/dev/null
        done
    fi
}

# Main execution
main() {
    # Stop Streamlit processes
    stop_streamlit
    STREAMLIT_RESULT=$?
    
    echo ""
    
    # Stop processes using port 8501
    stop_port_8501
    PORT_RESULT=$?
    
    echo ""
    
    # Cleanup virtual environment processes
    cleanup_venv
    
    echo ""
    echo "📊 Summary:"
    if [ $STREAMLIT_RESULT -eq 0 ] && [ $PORT_RESULT -eq 0 ]; then
        echo "✅ Monolith application stopped successfully"
        echo "🌐 Port 8501 is now available"
        echo "📋 You can now start the application again with:"
        echo "   ./run-monolith.sh"
        exit 0
    else
        echo "⚠️  Some processes may still be running"
        echo "🔧 Try running this script again, or check manually:"
        echo "   ps aux | grep streamlit"
        echo "   lsof -i:8501"
        exit 1
    fi
}

# Handle script interruption
trap 'echo ""; echo "🛑 Stop script interrupted"; exit 1' INT TERM

# Run main function
main
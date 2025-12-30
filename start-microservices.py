#!/usr/bin/env python3
# start-microservices.py
# -------------------------------
# Launcher script for all microservices
# -------------------------------

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def start_service(service_name, script_path, port):
    """Start a microservice"""
    print(f"Starting {service_name} on port {port}...")
    process = subprocess.Popen([
        sys.executable, script_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def main():
    """Start all microservices"""
    services = []
    
    try:
        # Start Ticker Service (port 8000)
        ticker_service = start_service(
            "Ticker Service",
            "services/ticker_service.py",
            8000
        )
        services.append(("Ticker Service", ticker_service))
        
        time.sleep(2)
        
        # Start Data Service (port 8001)
        data_service = start_service(
            "Data Service", 
            "services/data_service.py", 
            8001
        )
        services.append(("Data Service", data_service))
        
        time.sleep(2)  # Give it time to start
        
        # Start Calculation Service (port 8002)  
        calc_service = start_service(
            "Calculation Service",
            "services/calculation_service.py",
            8002
        )
        services.append(("Calculation Service", calc_service))
        
        time.sleep(2)
        
        print("\n🚀 Microservices started!")
        print("🎯 Ticker Service: http://localhost:8000/docs")
        print("📊 Data Service: http://localhost:8001/docs")
        print("🔢 Calculation Service: http://localhost:8002/docs")
        print("\nTo start the presentation layer:")
        print("streamlit run services/presentation_service.py")
        print("\nPress Ctrl+C to stop all services")
        
        # Wait for services
        while True:
            for name, process in services:
                if process.poll() is not None:
                    print(f"❌ {name} has stopped unexpectedly")
                    return 1
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        for name, process in services:
            print(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("✅ All services stopped")
        return 0
    
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# services/presentation_service.py
# -------------------------------
# Presentation Service - UI Layer
# Streamlit frontend that orchestrates calls to data and calculation services
# -------------------------------

import streamlit as st
import pandas as pd
import requests
import json
from typing import List, Dict, Optional
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import (
    StockDataRequest, StockAnalysisRequest, PortfolioOptimizationRequest,
    ServiceResponse
)

# Service URLs (configured for local development)
DATA_SERVICE_URL = "http://localhost:8001"
CALCULATION_SERVICE_URL = "http://localhost:8002"

class ServiceClient:
    """Client for communicating with microservices"""
    
    def __init__(self):
        pass
    
    def check_service_health(self, service_url: str) -> bool:
        """Check if a service is healthy"""
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 sample tickers from data service"""
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/sp500-tickers", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('data', [])
            return []
        except Exception as e:
            st.error(f"Error fetching S&P 500 tickers: {e}")
            return []
    
    def get_cache_info(self) -> Optional[dict]:
        """Get cache information from data service"""
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/cache-info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('data')
            return None
        except Exception as e:
            st.error(f"Error fetching cache info: {e}")
            return None
    
    def get_sp500_analysis(self, tickers: List[str], period: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Get S&P 500 analysis from data service"""
        try:
            request_data = {
                "tickers": tickers,
                "period": period,
                "force_refresh": force_refresh
            }
            response = requests.post(
                f"{DATA_SERVICE_URL}/sp500-analysis", 
                json=request_data,
                timeout=300  # 5 minutes for analysis
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return pd.DataFrame(data.get('data', []))
            
            # Handle error response
            if response.status_code != 200:
                error_data = response.json()
                st.error(f"Data service error: {error_data.get('error', 'Unknown error')}")
            
            return None
        except Exception as e:
            st.error(f"Error getting S&P 500 analysis: {e}")
            return None
    
    def get_stock_data(self, tickers: List[str], period: str, interval: str) -> Optional[pd.DataFrame]:
        """Get stock price data from data service"""
        try:
            request_data = {
                "tickers": tickers,
                "period": period,
                "interval": interval
            }
            response = requests.post(
                f"{DATA_SERVICE_URL}/stock-data",
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    records = data.get('data', [])
                    if records:
                        return pd.DataFrame(records)
            return None
        except Exception as e:
            st.error(f"Error getting stock data: {e}")
            return None

# Initialize Streamlit app
st.set_page_config(page_title="S&P Portfolio Optimizer - Microservices", layout="wide")
st.title("📈 S&P 500 Portfolio Optimizer (Microservices)")
st.caption("Microservices-based portfolio optimizer with separate data, calculation, and presentation layers.")

# Initialize service client
client = ServiceClient()

# Check service health
st.sidebar.header("Service Status")
data_service_healthy = client.check_service_health(DATA_SERVICE_URL)
calc_service_healthy = client.check_service_health(CALCULATION_SERVICE_URL)

st.sidebar.success("✅ Data Service") if data_service_healthy else st.sidebar.error("❌ Data Service")
st.sidebar.success("✅ Calculation Service") if calc_service_healthy else st.sidebar.error("❌ Calculation Service")

if not data_service_healthy:
    st.error("⚠️ Data Service is not available. Please start the data service first.")
    st.code("python services/data_service.py")
    st.stop()

# Input Parameters
st.sidebar.header("Inputs")

# Stock selection method
stock_selection = st.sidebar.selectbox(
    "Stock Selection Method",
    ["Manual Entry", "Use Top Performers from Analysis", "Custom from Analysis"]
)

if stock_selection == "Manual Entry":
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    tickers_input = st.sidebar.text_area(
        "Tickers (comma-separated)", 
        value=", ".join(default_tickers)
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

elif stock_selection == "Use Top Performers from Analysis":
    num_top_stocks = st.sidebar.slider("Number of top stocks", min_value=5, max_value=25, value=10, step=1)
    
    # Get cache info and analysis
    cache_info = client.get_cache_info()
    if cache_info and cache_info.get('has_cache') and not cache_info.get('is_stale'):
        st.sidebar.info(f"💾 Using cached data from {cache_info.get('last_updated')}")
    
    # Get S&P 500 tickers and run analysis
    sp500_tickers = client.get_sp500_tickers()
    if not sp500_tickers:
        st.error("Could not fetch S&P 500 tickers from data service")
        st.stop()
    
    # Get analysis (will use cache if available)
    analysis_df = client.get_sp500_analysis(sp500_tickers, "1y", force_refresh=False)
    
    if analysis_df is not None and len(analysis_df) > 0:
        top_performers = analysis_df.head(num_top_stocks)
        tickers = top_performers['ticker'].tolist()
        
        st.sidebar.success(f"Selected top {len(tickers)} performers")
        
        # Show the selected stocks
        with st.sidebar.expander("View Selected Stocks"):
            for i, row in top_performers.iterrows():
                st.write(f"{row['ticker']}: {row['composite_score']:.3f}")
    else:
        st.error("Could not get S&P 500 analysis from data service")
        st.stop()

else:  # Custom from Analysis
    # Get S&P 500 tickers and run analysis
    sp500_tickers = client.get_sp500_tickers()
    if not sp500_tickers:
        st.error("Could not fetch S&P 500 tickers from data service")
        st.stop()
    
    analysis_df = client.get_sp500_analysis(sp500_tickers, "1y", force_refresh=False)
    
    if analysis_df is not None and len(analysis_df) > 0:
        selected_tickers = st.sidebar.multiselect(
            "Select stocks from analysis",
            options=analysis_df['ticker'].tolist(),
            default=analysis_df['ticker'].head(10).tolist()
        )
        tickers = selected_tickers
    else:
        st.error("Could not get S&P 500 analysis from data service")
        st.stop()

# Other parameters
period = st.sidebar.selectbox("Lookback Period", ["1y", "2y", "5y"], index=0)
interval = st.sidebar.selectbox("Data Frequency", ["1d", "1wk", "1mo"], index=0)

risk_aversion = st.sidebar.slider("Risk Aversion", 0.1, 5.0, 1.0, 0.1)
min_weight_threshold = st.sidebar.slider("Prune weights below (%)", 0.0, 2.0, 0.25, 0.05)
min_holdings = st.sidebar.number_input("Minimum number of holdings", min_value=2, max_value=60, value=3, step=1)
investment = st.sidebar.number_input("Total investment (USD)", value=250_000, step=10_000)

st.sidebar.markdown("---")
st.sidebar.caption("Microservices Architecture: Data, Calculation & Presentation layers are separated.")

# Create tabs
tab1, tab2 = st.tabs(["📊 Portfolio Optimizer", "🔍 S&P 500 Stock Analyzer"])

with tab1:
    st.header("📊 Portfolio Optimizer")
    
    if not tickers:
        st.warning("Please select at least one stock ticker")
        st.stop()
    
    st.info(f"Optimizing portfolio with {len(tickers)} stocks: {', '.join(tickers)}")
    
    # Get stock data
    with st.spinner("Fetching stock data from data service..."):
        prices_df = client.get_stock_data(tickers, period, interval)
    
    if prices_df is None or prices_df.empty:
        st.error("Could not fetch stock data from data service")
        st.stop()
    
    # Convert back to proper DataFrame format
    prices_df = pd.DataFrame(prices_df)
    prices_df.index = pd.to_datetime(prices_df.index) if not isinstance(prices_df.index, pd.DatetimeIndex) else prices_df.index
    
    st.success(f"Loaded data for {len(prices_df.columns)} stocks, {len(prices_df)} days")
    
    # Note about calculation service integration
    st.warning("🚧 Portfolio optimization integration with calculation service is in progress. The microservices architecture has been created but the optimization endpoint needs additional integration work.")
    
    # Show the data we have
    st.subheader("Stock Price Data (Last 10 Days)")
    st.dataframe(prices_df.tail(10))
    
    # Show what would be optimized
    st.subheader("Optimization Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Investment Amount", f"${investment:,}")
        st.metric("Risk Aversion", f"{risk_aversion}")
    with col2:
        st.metric("Min Weight Threshold", f"{min_weight_threshold}%")
        st.metric("Min Holdings", f"{min_holdings}")

with tab2:
    st.header("🔍 S&P 500 Stock Analyzer")
    st.caption("Analyze top 100 S&P 500 stocks using the data microservice.")
    
    # Show cache status
    cache_info = client.get_cache_info()
    if cache_info and cache_info.get('has_cache'):
        if cache_info.get('is_stale'):
            st.warning(f"💾 Cached data is stale (from {cache_info.get('last_updated')})")
        else:
            st.info(f"💾 Fresh cached data available (from {cache_info.get('last_updated')})")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analysis_period = st.selectbox("Analysis Period", ["1y", "2y", "5y"], index=0, key="analysis_period")
    with col2:
        force_refresh = st.button("🔄 Force Refresh")
    with col3:
        auto_analyze = st.checkbox("Auto-analyze", value=True)
    
    # Get S&P 500 tickers
    sp500_tickers = client.get_sp500_tickers()
    if not sp500_tickers:
        st.error("Could not fetch S&P 500 tickers from data service")
        st.stop()
    
    # Run analysis
    if auto_analyze or force_refresh:
        with st.spinner(f"Analyzing {len(sp500_tickers)} S&P 500 stocks via data service..."):
            analysis_df = client.get_sp500_analysis(sp500_tickers, analysis_period, force_refresh)
        
        if analysis_df is not None and len(analysis_df) > 0:
            st.success(f"Analysis completed for {len(analysis_df)} stocks")
            
            # Display results
            st.subheader("📈 Top 10 Stock Recommendations")
            top_10 = analysis_df.head(10)
            
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
                with col1:
                    st.metric(f"#{i}", row['ticker'])
                with col2:
                    st.metric("Score", f"{row['composite_score']:.3f}")
                with col3:
                    st.metric("Return", f"{row['total_return_pct']:.1f}%")
                with col4:
                    st.metric("Sharpe", f"{row['sharpe_ratio']:.2f}")
            
            st.subheader("📊 Full Analysis Results")
            
            # Display controls
            display_cols = st.multiselect(
                "Select columns to display:",
                options=analysis_df.columns.tolist(),
                default=['ticker', 'composite_score', 'total_return_pct', 'volatility_pct', 'sharpe_ratio', 'current_price']
            )
            
            if display_cols:
                st.dataframe(analysis_df[display_cols], use_container_width=True)
            
            # Export option
            if st.button("📥 Export to CSV"):
                csv = analysis_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"sp500_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.error("Could not get analysis results from data service")

# Footer
st.markdown("---")
st.markdown("**Microservices Architecture:**")
st.markdown("- **Data Service** (Port 8001): Handles yfinance API calls and data persistence")
st.markdown("- **Calculation Service** (Port 8002): Performs portfolio optimization calculations")  
st.markdown("- **Presentation Service** (This app): Streamlit UI that orchestrates service calls")

if __name__ == "__main__":
    # This would be run with: streamlit run services/presentation_service.py
    pass
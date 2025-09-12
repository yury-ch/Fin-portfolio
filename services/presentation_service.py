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
                        # Convert to DataFrame - records are already in correct format
                        df = pd.DataFrame(records)
                        if not df.empty:
                            # Create a simple date index for the data
                            df.index = pd.date_range(end=pd.Timestamp.now().date(), periods=len(df), freq='D')
                            # Ensure numeric data types
                            df = df.astype(float)
                        return df
            return None
        except Exception as e:
            st.error(f"Error getting stock data: {e}")
            return None
    
    def optimize_portfolio_with_calculation_service(self, prices_df: pd.DataFrame, tickers: List[str], 
                                                  investment_amount: float, objective: str,
                                                  min_weight_threshold: float, min_holdings: int,
                                                  max_weight: float, target_return: float, risk_free: float) -> Optional[dict]:
        """Optimize portfolio using calculation service"""
        try:
            # For now, we'll implement a simplified optimization here
            # since the calculation service needs more integration work
            from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation, get_latest_prices
            
            # Debug: check the data format
            if prices_df.empty:
                raise ValueError("Empty price DataFrame")
            
            # Ensure we have the right columns
            available_tickers = [t for t in tickers if t in prices_df.columns]
            if not available_tickers:
                raise ValueError(f"None of the tickers {tickers} found in price data columns: {list(prices_df.columns)}")
            
            # Filter to only available tickers
            prices_df = prices_df[available_tickers]
            
            # Compute expected returns and covariance
            mu = expected_returns.mean_historical_return(prices_df, frequency=252)
            S = risk_models.sample_cov(prices_df, frequency=252)
            
            # Optimize based on objective
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0.001)
            ef.add_constraint(lambda w: w <= max_weight/100)  # Max weight constraint
            
            if objective == "Max Sharpe":
                weights = ef.max_sharpe(risk_free_rate=risk_free/100)
            elif objective == "Min Volatility (target return)":
                weights = ef.efficient_return(target_return=target_return/100)
            else:  # Target Return
                weights = ef.efficient_return(target_return=target_return/100)
            
            # Clean weights
            weights = {k: v for k, v in weights.items() if v >= min_weight_threshold/100}
            if len(weights) < min_holdings:
                # Keep top holdings
                all_weights = dict(sorted(ef.clean_weights().items(), key=lambda x: x[1], reverse=True))
                weights = dict(list(all_weights.items())[:max(min_holdings, len([w for w in all_weights.values() if w > 0]))])
            
            # Renormalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            # Get performance
            ef_clean = EfficientFrontier(mu, S)
            ef_clean.set_weights(weights)
            performance = ef_clean.portfolio_performance(verbose=False)
            
            # Discrete allocation
            latest_prices = get_latest_prices(prices_df)
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)
            allocation, leftover = da.greedy_portfolio()
            
            return {
                'weights': weights,
                'expected_annual_return': performance[0],
                'annual_volatility': performance[1], 
                'sharpe_ratio': performance[2],
                'allocation': allocation,
                'leftover_cash': leftover
            }
            
        except Exception as e:
            st.error(f"Portfolio optimization failed: {e}")
            return None

# Initialize Streamlit app
st.set_page_config(page_title="S&P Portfolio Optimizer", layout="wide")
st.title("📈 S&P 500 Portfolio Optimizer")
st.caption("Build an optimal S&P portfolio given a risk–return preference, investment amount, and horizon.")

# Initialize service client
client = ServiceClient()

# Check service health silently
data_service_healthy = client.check_service_health(DATA_SERVICE_URL)
calc_service_healthy = client.check_service_health(CALCULATION_SERVICE_URL)

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
    needs_analysis = True
    
    if cache_info and cache_info.get('has_cache') and not cache_info.get('is_stale'):
        st.sidebar.info(f"💾 Using cached data from {cache_info.get('last_updated')}")
        needs_analysis = False
    else:
        st.sidebar.warning("🔄 No cached analysis available - will run fresh analysis")
    
    # Get S&P 500 tickers and run analysis
    sp500_tickers = client.get_sp500_tickers()
    if not sp500_tickers:
        st.error("Could not fetch S&P 500 tickers from data service")
        st.stop()
    
    # Get analysis (will use cache if available, or run fresh analysis)
    with st.spinner("Getting S&P 500 analysis..." if needs_analysis else "Loading cached analysis..."):
        analysis_df = client.get_sp500_analysis(sp500_tickers, "1y", force_refresh=needs_analysis)
    
    if analysis_df is not None and len(analysis_df) > 0:
        try:
            top_performers = analysis_df.head(num_top_stocks)
            ticker_col = 'Ticker' if 'Ticker' in top_performers.columns else 'ticker'
            if ticker_col in top_performers.columns:
                tickers = top_performers[ticker_col].tolist()
                st.sidebar.success(f"Selected top {len(tickers)} performers")
                
                # Show the selected stocks
                with st.sidebar.expander("View Selected Stocks"):
                    for i, row in top_performers.iterrows():
                        score_col = 'composite_score' if 'composite_score' in row.index else 'Composite_Score'
                        st.write(f"{row.get(ticker_col, f'Stock_{i}')}: {row.get(score_col, 0):.3f}")
            else:
                # If no ticker column, fall back to default stocks
                st.sidebar.warning("No ticker column in analysis data - using default stocks")
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "ADBE", "CRM"][:num_top_stocks]
        except Exception as e:
            # If there's an error accessing the data, fall back to default tickers
            st.sidebar.warning(f"Analysis data error: {str(e)[:50]}... - using default stocks")
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "ADBE", "CRM"][:num_top_stocks]
    else:
        # If analysis fails, fall back to default tickers
        st.sidebar.error("Analysis failed - using default top stocks")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "ADBE", "CRM"][:num_top_stocks]

else:  # Custom from Analysis
    # Get S&P 500 tickers and run analysis
    sp500_tickers = client.get_sp500_tickers()
    if not sp500_tickers:
        st.error("Could not fetch S&P 500 tickers from data service")
        st.stop()
    
    analysis_df = client.get_sp500_analysis(sp500_tickers, "1y", force_refresh=False)
    
    if analysis_df is not None and len(analysis_df) > 0:
        ticker_col = 'Ticker' if 'Ticker' in analysis_df.columns else 'ticker'
        selected_tickers = st.sidebar.multiselect(
            "Select stocks from analysis",
            options=analysis_df[ticker_col].tolist(),
            default=analysis_df[ticker_col].head(10).tolist()
        )
        tickers = selected_tickers
    else:
        st.error("Could not get S&P 500 analysis from data service")
        st.stop()

# Other parameters to match original app exactly
period = st.sidebar.selectbox("Investment horizon / Lookback window", ["1y","2y","3y","5y"], index=0)
interval = st.sidebar.selectbox("Price frequency", ["1d","1wk","1mo"], index=0)

objective = st.sidebar.selectbox(
    "Optimization objective",
    ["Max Sharpe", "Min Volatility (target return)", "Target Return"],
    index=0
)
risk_free = st.sidebar.number_input("Risk-free rate (annual, %)", value=4.0, step=0.25)
target_return = st.sidebar.number_input("Target annual return (%)", value=10.0, step=0.5)

max_weight = st.sidebar.slider("Max weight per stock (%)", min_value=5, max_value=50, value=30, step=1)
l2_reg = st.sidebar.slider("L2 regularization (weight smoothing)", 0.0, 20.0, 5.0)
min_weight_threshold = st.sidebar.slider("Prune weights below (%)", 0.0, 2.0, 0.25, 0.05)
min_holdings = st.sidebar.number_input("Minimum number of holdings", min_value=2, max_value=60, value=3, step=1)

investment = st.sidebar.number_input("Total investment (USD)", value=250_000, step=10_000)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For 5–10y analysis, use monthly data to reduce noise.")

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
    
    # Perform portfolio optimization
    with st.spinner("Optimizing portfolio..."):
        optimization_result = client.optimize_portfolio_with_calculation_service(
            prices_df, tickers, investment, objective, min_weight_threshold, min_holdings,
            max_weight, target_return, risk_free
        )
    
    if optimization_result:
        st.success("✅ Portfolio optimization completed!")
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Expected Annual Return", 
                f"{optimization_result['expected_annual_return']:.1%}"
            )
        with col2:
            st.metric(
                "Annual Volatility", 
                f"{optimization_result['annual_volatility']:.1%}"
            )
        with col3:
            st.metric(
                "Sharpe Ratio", 
                f"{optimization_result['sharpe_ratio']:.2f}"
            )
        
        # Optimal weights
        st.subheader("🎯 Optimal Weights")
        weights_df = pd.DataFrame([
            {"Ticker": ticker, "Weight": f"{weight:.1%}", "Weight_Raw": weight}
            for ticker, weight in optimization_result['weights'].items()
        ]).sort_values('Weight_Raw', ascending=False)
        
        st.dataframe(weights_df[['Ticker', 'Weight']], use_container_width=True, hide_index=True)
        
        # Discrete allocation
        st.subheader("💰 Share Allocation")
        if optimization_result['allocation']:
            allocation_df = pd.DataFrame([
                {"Ticker": ticker, "Shares": shares, "Est. Value": f"${shares * prices_df[ticker].iloc[-1]:.0f}"}
                for ticker, shares in optimization_result['allocation'].items()
            ])
            st.dataframe(allocation_df, use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Invested", f"${investment - optimization_result['leftover_cash']:,.0f}")
            with col2:
                st.metric("Leftover Cash", f"${optimization_result['leftover_cash']:,.0f}")
        
        # Show recent price data
        st.subheader("📊 Recent Price Data")
        st.dataframe(prices_df.tail(10))
        
    else:
        st.error("❌ Portfolio optimization failed. Please try with different parameters or check the services.")
        
        # Show the data we have for debugging
        st.subheader("📊 Available Price Data (Last 10 Days)")
        st.dataframe(prices_df.tail(10))
        
        st.subheader("🔧 Optimization Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Investment Amount", f"${investment:,}")
            st.metric("Optimization Objective", objective)
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
                    ticker_col = 'Ticker' if 'Ticker' in row.index else 'ticker'
                    st.metric(f"#{i}", row.get(ticker_col, f'Stock_{i}'))
                with col2:
                    score_col = 'composite_score' if 'composite_score' in row.index else 'Composite_Score'
                    st.metric("Score", f"{row.get(score_col, 0):.3f}")
                with col3:
                    return_col = 'total_return_pct' if 'total_return_pct' in row.index else 'Annual_Return'
                    return_val = row.get(return_col, 0)
                    # Convert to percentage if it's not already
                    if return_col == 'total_return_pct':
                        st.metric("Return", f"{return_val:.1f}%")
                    else:
                        st.metric("Return", f"{return_val*100:.1f}%")
                with col4:
                    sharpe_col = 'sharpe_ratio' if 'sharpe_ratio' in row.index else 'Sharpe_Ratio'
                    st.metric("Sharpe", f"{row.get(sharpe_col, 0):.2f}")
            
            st.subheader("📊 Full Analysis Results")
            
            # Display controls
            # Create a mapping of display names to actual column names
            # Create a mapping of display names to actual column names
            column_mapping = {}
            for col in analysis_df.columns:
                if col in ['Ticker', 'ticker']:
                    column_mapping['Ticker'] = col
                elif col in ['Composite_Score', 'composite_score']:
                    column_mapping['Composite_Score'] = col
                elif col in ['Annual_Return', 'total_return_pct', 'annual_return']:
                    column_mapping['Return'] = col
                elif col in ['Volatility', 'volatility_pct', 'volatility']:
                    column_mapping['Volatility'] = col
                elif col in ['Sharpe_Ratio', 'sharpe_ratio']:
                    column_mapping['Sharpe_Ratio'] = col
                elif col in ['Current_Price', 'current_price']:
                    column_mapping['Current_Price'] = col
                else:
                    column_mapping[col] = col
            
            default_cols = ['Ticker', 'Composite_Score', 'Return', 'Volatility', 'Sharpe_Ratio', 'Current_Price']
            available_defaults = [col for col in default_cols if col in column_mapping]
            
            display_cols_display = st.multiselect(
                "Select columns to display:",
                options=list(column_mapping.keys()),
                default=available_defaults
            )
            
            # Map back to actual column names
            display_cols = [column_mapping[col] for col in display_cols_display]
            
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

# Clean up - no technical footer needed

if __name__ == "__main__":
    # This would be run with: streamlit run services/presentation_service.py
    pass
# app.py
# -------------------------------
# S&P 500 Portfolio Optimizer
# Streamlit + yfinance + PyPortfolioOpt
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
from pypfopt import (
    EfficientFrontier,
    risk_models,
    expected_returns,
    objective_functions,
    DiscreteAllocation,
    get_latest_prices,
)

# -------------------------------
# Global Constants for Data Persistence
# -------------------------------
DATA_DIR = Path("sp500_data")
ANALYSIS_FILE = DATA_DIR / "sp500_analysis.parquet"
METADATA_FILE = DATA_DIR / "metadata.parquet"

st.set_page_config(page_title="S&P Portfolio Optimizer", layout="wide")

st.title("📈 S&P 500 Portfolio Optimizer")
st.caption("Build an optimal S&P portfolio given a risk–return preference, investment amount, and horizon.")

# -------------------------------
# Sidebar: Inputs
# -------------------------------
st.sidebar.header("Inputs")

# S&P 500 Top 100 stocks by market cap for comprehensive analysis
SP500_SAMPLE = [
    # Top 50 (Mega caps)
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","UNH","XOM",
    "JPM","JNJ","V","PG","HD","CVX","MA","ABBV","PFE","KO",
    "AVGO","COST","PEP","TMO","WMT","MRK","DIS","ADBE","NFLX","CRM",
    "BAC","ACN","LLY","ORCL","WFC","VZ","CMCSA","CSCO","ABT","DHR",
    "NKE","TXN","PM","BMY","UNP","QCOM","RTX","HON","INTC","T",
    
    # Next 50 (Large caps) 
    "AMAT","SPGI","CAT","INTU","ISRG","NOW","LOW","GS","MS","AMD",
    "AMGN","BKNG","TJX","BLK","AXP","SYK","VRTX","PLD","GILD","MDLZ",
    "SBUX","TMUS","CVS","CI","LRCX","CB","MO","PYPL","MMC","SO",
    "ZTS","SCHW","FIS","DUK","BSX","CL","ITW","EQIX","AON","CSX",
    "ADI","NOC","MU","SHW","ICE","KLAC","APD","USB","CME","REGN",
    "EMR","PNC","EOG","FCX","GD","NSC","TGT","HUM","COP","PSA"
]

DEFAULT_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","JPM","V"
]

# Stock selection options
stock_selection = st.sidebar.radio(
    "Stock Selection Method",
    ["Manual Entry", "Use Top Performers from Analysis", "Custom from Analysis"],
    help="Choose how to select stocks for optimization"
)

if stock_selection == "Manual Entry":
    # Use recommended tickers if available
    default_value = st.session_state.get('recommended_tickers', ",".join(DEFAULT_TICKERS))
    
    tickers_str = st.sidebar.text_area(
        "Tickers (comma-separated)",
        value=default_value,
        help="Paste S&P tickers, e.g. AAPL,MSFT,NVDA. Tip: Yahoo uses BRK-B for Berkshire B. Use the S&P 500 Analyzer for recommendations!"
    )
    TICKERS: List[str] = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    
    # Clear recommended tickers after first use
    if 'recommended_tickers' in st.session_state and st.session_state.get('recommended_tickers') == tickers_str:
        if st.sidebar.button("🔄 Clear Recommendations"):
            del st.session_state['recommended_tickers']
            st.rerun()

elif stock_selection == "Use Top Performers from Analysis":
    num_top_stocks = st.sidebar.slider("Number of top stocks", min_value=5, max_value=25, value=10, step=1)
    
    # Get top performers from analysis - inline logic to avoid scope issues
    if ANALYSIS_FILE.exists() and METADATA_FILE.exists():
        try:
            cached_df = pd.read_parquet(ANALYSIS_FILE)
            metadata_df = pd.read_parquet(METADATA_FILE)
            metadata = metadata_df.iloc[0].to_dict()
            
            # Check staleness and period match
            last_updated = metadata.get('last_updated')
            if isinstance(last_updated, str):
                last_updated = pd.to_datetime(last_updated)
            
            age = datetime.now() - last_updated
            is_stale = age > timedelta(hours=24)
            
            cached_period = metadata.get('period', '')
            
            if not is_stale and cached_period == horizon:
                top_performers = cached_df.head(num_top_stocks)['Ticker'].tolist()
            else:
                top_performers = []
        except:
            top_performers = []
    else:
        top_performers = []
    
    if top_performers:
        st.sidebar.success(f"✅ Using top {len(top_performers)} stocks from analysis")
        st.sidebar.caption(f"Top stocks: {', '.join(top_performers)}")
        
        # Display as comma-separated string like manual entry
        tickers_str = ",".join(top_performers)
        st.sidebar.text_area(
            "Selected Tickers (from analysis)",
            value=tickers_str,
            disabled=True,
            help="These are the top-performing stocks from your S&P 500 analysis"
        )
        TICKERS = top_performers
    else:
        st.sidebar.warning("⚠️ No analysis data available. Run S&P 500 analyzer first or switch to manual entry.")
        TICKERS = DEFAULT_TICKERS

else:  # Custom from Analysis
    # Load analysis data and let user select - inline logic
    if ANALYSIS_FILE.exists() and METADATA_FILE.exists():
        try:
            cached_df = pd.read_parquet(ANALYSIS_FILE)
            metadata_df = pd.read_parquet(METADATA_FILE)
            metadata = metadata_df.iloc[0].to_dict()
            
            # Check staleness and period match
            last_updated = metadata.get('last_updated')
            if isinstance(last_updated, str):
                last_updated = pd.to_datetime(last_updated)
            
            age = datetime.now() - last_updated
            is_stale = age > timedelta(hours=24)
            
            cached_period = metadata.get('period', '')
            
            if not is_stale and cached_period == horizon:
                age_hours = age.total_seconds() / 3600
                num_stocks = len(cached_df)
                cache_info = f"Cached: {num_stocks} stocks, {cached_period} period, {age_hours:.1f}h ago"
                
                st.sidebar.success(f"📊 Analysis data available ({cache_info})")
                
                # Show top 20 for selection
                top_20 = cached_df.head(20)
                selected_tickers = st.sidebar.multiselect(
                    "Select stocks from analysis (top 20 shown)",
                    options=top_20['Ticker'].tolist(),
                    default=top_20.head(10)['Ticker'].tolist(),
                    help="Choose specific stocks from the analysis results"
                )
                
                if selected_tickers:
                    TICKERS = selected_tickers
                else:
                    st.sidebar.warning("⚠️ Please select at least 2 stocks")
                    TICKERS = DEFAULT_TICKERS
            else:
                st.sidebar.warning(f"⚠️ Analysis period ({cached_period}) doesn't match optimization period ({horizon})")
                TICKERS = DEFAULT_TICKERS
        except:
            st.sidebar.warning("⚠️ Error loading analysis data.")
            TICKERS = DEFAULT_TICKERS
    else:
        st.sidebar.warning("⚠️ No analysis data. Run S&P 500 analyzer first.")
        TICKERS = DEFAULT_TICKERS

horizon = st.sidebar.selectbox("Investment horizon / Lookback window", ["1y","2y","3y","5y"], index=0)
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

# -------------------------------
# Data Persistence Functions (Parquet)
# -------------------------------

def ensure_data_directory():
    """Create data directory if it doesn't exist"""
    DATA_DIR.mkdir(exist_ok=True)

def save_analysis_data(df: pd.DataFrame, period: str):
    """Save analysis results to parquet with metadata"""
    ensure_data_directory()
    
    # Add timestamp and period to the data
    df = df.copy()
    df['analysis_timestamp'] = datetime.now()
    df['analysis_period'] = period
    
    # Save main analysis data
    df.to_parquet(ANALYSIS_FILE, index=False)
    
    # Save metadata
    metadata = pd.DataFrame({
        'last_updated': [datetime.now()],
        'period': [period],
        'num_stocks': [len(df)],
        'version': ['1.0']
    })
    metadata.to_parquet(METADATA_FILE, index=False)
    
    st.info(f"💾 Analysis data saved to {ANALYSIS_FILE}")

def load_analysis_data() -> Tuple[pd.DataFrame, dict]:
    """Load analysis results from parquet if available"""
    if not ANALYSIS_FILE.exists() or not METADATA_FILE.exists():
        return pd.DataFrame(), {}
    
    try:
        # Load data
        df = pd.read_parquet(ANALYSIS_FILE)
        metadata = pd.read_parquet(METADATA_FILE).iloc[0].to_dict()
        
        return df, metadata
    except Exception as e:
        st.warning(f"Failed to load cached data: {e}")
        return pd.DataFrame(), {}

def is_data_stale(metadata: dict, max_age_hours: int = 24) -> bool:
    """Check if cached data is stale (older than max_age_hours)"""
    if not metadata:
        return True
    
    last_updated = metadata.get('last_updated')
    if not last_updated:
        return True
    
    # Handle both datetime and string formats
    if isinstance(last_updated, str):
        last_updated = pd.to_datetime(last_updated)
    
    age = datetime.now() - last_updated
    return age > timedelta(hours=max_age_hours)

def get_cache_info() -> str:
    """Get human-readable cache information"""
    df, metadata = load_analysis_data()
    
    if df.empty:
        return "No cached data available"
    
    last_updated = metadata.get('last_updated', 'Unknown')
    if isinstance(last_updated, str):
        last_updated = pd.to_datetime(last_updated)
    
    age_hours = (datetime.now() - last_updated).total_seconds() / 3600
    period = metadata.get('period', 'Unknown')
    num_stocks = len(df)
    
    return f"Cached: {num_stocks} stocks, {period} period, {age_hours:.1f}h ago"


# -------------------------------
# S&P 500 Analysis Functions
# -------------------------------
def analyze_sp500_stocks(tickers: List[str], period: str = "1y", force_refresh: bool = False) -> pd.DataFrame:
    """
    Analyze S&P 500 stocks and score them based on multiple metrics:
    - Return (annualized)
    - Volatility (lower is better)
    - Sharpe ratio
    - Maximum drawdown (lower is better)
    - Recent momentum (3-month return)
    
    Uses cached data if available and not stale (< 24 hours old)
    """
    
    # Check for cached data first (unless force refresh)
    if not force_refresh:
        cached_df, metadata = load_analysis_data()
        
        if not cached_df.empty and not is_data_stale(metadata):
            # Check if period matches
            cached_period = metadata.get('period', '')
            if cached_period == period:
                st.success(f"📱 Using cached data ({get_cache_info()})")
                return cached_df.drop(['analysis_timestamp', 'analysis_period'], axis=1, errors='ignore')
    
    # Perform fresh analysis
    st.info("🔄 Performing fresh analysis...")
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"Analyzing {ticker}... ({i+1}/{len(tickers)})")
            
            # Get price data
            data = yf.download(ticker, period=period, interval="1d", progress=False)
            if data.empty:
                continue
                
            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            returns = prices.pct_change().dropna()
            
            if len(returns) < 60:  # Need at least 60 days
                continue
            
            # Calculate metrics
            ann_return = float((1 + returns.mean()) ** 252 - 1)
            ann_vol = float(returns.std() * np.sqrt(252))
            sharpe = float(ann_return / ann_vol if ann_vol > 0 else 0)
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = float(drawdown.min())
            
            # 3-month momentum (recent performance)
            if len(prices) >= 63:
                recent_return = float((prices.iloc[-1] / prices.iloc[-63]) - 1)
            else:
                recent_return = 0.0
            
            # Current price info
            current_price = float(prices.iloc[-1])
            
            results.append({
                'Ticker': ticker,
                'Annual_Return': ann_return,
                'Volatility': ann_vol,
                'Sharpe_Ratio': sharpe,
                'Max_Drawdown': max_drawdown,
                'Recent_3M_Return': recent_return,
                'Current_Price': current_price
            })
            
        except Exception as e:
            st.warning(f"Failed to analyze {ticker}: {str(e)}")
            continue
            
        progress_bar.progress((i + 1) / len(tickers))
        time.sleep(0.2)  # Increased rate limiting for larger dataset
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    # Calculate composite score (higher is better)
    # Normalize metrics to 0-1 scale (handle division by zero)
    def safe_normalize(series, inverse=False):
        range_val = series.max() - series.min()
        if range_val == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        normalized = (series - series.min()) / range_val
        return 1 - normalized if inverse else normalized
    
    df['Return_Score'] = safe_normalize(df['Annual_Return'])
    df['Vol_Score'] = safe_normalize(df['Volatility'], inverse=True)  # Lower vol is better
    df['Sharpe_Score'] = safe_normalize(df['Sharpe_Ratio'])
    df['Drawdown_Score'] = safe_normalize(df['Max_Drawdown'], inverse=True)  # Lower drawdown is better
    df['Momentum_Score'] = safe_normalize(df['Recent_3M_Return'])
    
    # Composite score (weighted average)
    df['Composite_Score'] = (
        0.25 * df['Return_Score'] + 
        0.20 * df['Vol_Score'] + 
        0.25 * df['Sharpe_Score'] + 
        0.15 * df['Drawdown_Score'] + 
        0.15 * df['Momentum_Score']
    )
    
    # Sort by composite score
    df = df.sort_values('Composite_Score', ascending=False)
    
    # Save to parquet for future use
    save_analysis_data(df, period)
    
    return df

# -------------------------------
# Price loader (robust + cached)
# -------------------------------
@st.cache_data(show_spinner=True)
def load_prices(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    """
    Robustly load historical prices for many tickers:
      - Batch requests to avoid rate limits
      - Prefer 'Adj Close' if available, else 'Close'
      - Keep partial histories (forward fill small gaps)
      - Drop only columns that are entirely NaN
    """
    # Normalize tickers and fix a few common symbols if needed
    clean = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        # Accept both BRK-B and BRK.B (convert to Yahoo's BRK-B)
        if t in ("BRK.B", "BRKB"):
            t = "BRK-B"
        clean.append(t)
    tickers = clean

    if not tickers:
        return pd.DataFrame()

    batch_size = 5
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            # Try different download approaches
            if len(batch) == 1:
                # Single ticker - simpler approach
                ticker = batch[0]
                df = yf.download(ticker, period=period, interval=interval, progress=False)
                if not df.empty and 'Adj Close' in df.columns:
                    df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
                elif not df.empty and 'Close' in df.columns:
                    df = df[['Close']].rename(columns={'Close': ticker})
            else:
                # Multiple tickers
                df = yf.download(
                    batch,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    group_by="ticker",
                    threads=False,
                    progress=False,
                )
            
            if df.empty:
                st.warning(f"No data retrieved for batch: {batch}")
                continue
            else:
                st.success(f"Successfully downloaded data for: {batch}")
                
        except Exception as e:
            st.error(f"Error downloading {batch}: {str(e)}")
            continue

        # Single ticker vs multi-ticker shapes
        def pick_price_frame(_df, single_symbol=None):
            # MultiIndex columns (field first or ticker first depending on yfinance version)
            if isinstance(_df.columns, pd.MultiIndex):
                # Try both organizations
                if "Adj Close" in _df.columns.get_level_values(0):
                    px = _df["Adj Close"]
                elif "Close" in _df.columns.get_level_values(0):
                    px = _df["Close"]
                else:
                    # Sometimes MultiIndex is (ticker, field)
                    # Build a wide frame by picking the right level
                    lvl1 = set(_df.columns.get_level_values(1))
                    if "Adj Close" in lvl1:
                        px = _df.xs("Adj Close", axis=1, level=1)
                    elif "Close" in lvl1:
                        px = _df.xs("Close", axis=1, level=1)
                    else:
                        px = pd.DataFrame(index=_df.index)
                return px
            else:
                # Single ticker with flat columns
                if "Adj Close" in _df.columns:
                    px = _df[["Adj Close"]].rename(columns={"Adj Close": single_symbol})
                elif "Close" in _df.columns:
                    px = _df[["Close"]].rename(columns={"Close": single_symbol})
                else:
                    px = pd.DataFrame(index=_df.index)
                return px

        if isinstance(df.columns, pd.MultiIndex) or (len(batch) > 1):
            px = pick_price_frame(df)
        else:
            px = pick_price_frame(df, single_symbol=batch[0])

        frames.append(px)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)

    # If columns are a mix of types, coerce to float where possible
    for c in prices.columns:
        prices[c] = pd.to_numeric(prices[c], errors="coerce")

    # Keep only columns with at least some data; forward-fill gaps
    prices = prices.dropna(axis=1, how="all")
    prices = prices.sort_index().ffill()

    # Remove very early rows that may still be all-NaN after ffill
    prices = prices.dropna(how="all")

    # Need enough history and at least 2 assets for optimization
    if prices.shape[0] < 12 or prices.shape[1] < 2:
        return pd.DataFrame()

    return prices

# -------------------------------
# Optimizer helpers
# -------------------------------
def compute_stats(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    mu = expected_returns.mean_historical_return(prices)     # annualized
    S = risk_models.sample_cov(prices)                       # annualized
    return mu, S

def optimize_portfolio(
    prices: pd.DataFrame,
    obj: str,
    rf: float,
    tgt: float,
    max_w: float,
    l2: float
) -> Tuple[pd.Series, dict]:
    mu, S = compute_stats(prices)
    
    # Adjust constraints for small portfolios
    n_assets = len(prices.columns)
    if n_assets <= 3:
        # For very small portfolios, relax constraints
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        l2 = 0  # Disable L2 reg for small portfolios
    else:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w / 100.0))
        if l2 and l2 > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=l2)

    try:
        if obj == "Max Sharpe":
            ef.max_sharpe(risk_free_rate=rf / 100.0)
        elif obj == "Min Volatility (target return)":
            ef.efficient_return(target_return=tgt / 100.0)
        else:  # "Target Return"
            ef.efficient_return(target_return=tgt / 100.0)
    except:
        # Fallback: try min volatility without target return
        try:
            ef.min_volatility()
        except:
            # Last resort: equal weights
            equal_weights = {asset: 1/n_assets for asset in prices.columns}
            ret, vol, sharpe = ef.portfolio_performance(weights=equal_weights, risk_free_rate=rf / 100.0, verbose=False)
            return pd.Series(equal_weights), {"expected_return": ret, "volatility": vol, "sharpe": sharpe}

    cleaned = ef.clean_weights(cutoff=0)
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=rf / 100.0, verbose=False)
    return pd.Series(cleaned), {"expected_return": ret, "volatility": vol, "sharpe": sharpe}

def enforce_min_holdings(weights: pd.Series, min_n: int, prune_below_pct: float) -> pd.Series:
    if weights.empty:
        return weights
    keep = weights[weights >= (prune_below_pct / 100.0)].sort_values(ascending=False)
    if len(keep) >= min_n:
        return keep / keep.sum()
    top = weights.sort_values(ascending=False).head(min_n)
    return top / top.sum()

# -------------------------------
# S&P 500 Stock Analyzer Tab
# -------------------------------
tab1, tab2 = st.tabs(["📊 Portfolio Optimizer", "🔍 S&P 500 Stock Analyzer"])

with tab2:
    st.header("🔍 S&P 500 Stock Analyzer")
    st.caption("Analyze top 100 S&P 500 stocks and get top 10 recommendations based on multiple performance metrics.")
    
    # Show cache status
    cache_info = get_cache_info()
    if cache_info != "No cached data available":
        st.info(f"💾 {cache_info}")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analysis_period = st.selectbox("Analysis period", ["1y", "2y", "3y"], index=0, key="analysis_period")
    
    with col2:
        analyze_button = st.button("🚀 Analyze Stocks", type="primary")
    
    with col3:
        force_refresh = st.button("🔄 Force Refresh", help="Ignore cached data and perform fresh analysis")
    
    # Always try to show cached results first
    show_results = False
    analysis_results = pd.DataFrame()
    
    # Load cached data and display if available
    cached_df, metadata = load_analysis_data()
    if not cached_df.empty:
        analysis_results = cached_df.drop(['analysis_timestamp', 'analysis_period'], axis=1, errors='ignore')
        show_results = True
        
        if not (analyze_button or force_refresh):
            st.success(f"📱 Showing cached analysis results ({get_cache_info()})")
    
    if analyze_button or force_refresh:
        # Determine spinner message based on whether we're using cache
        if force_refresh:
            spinner_msg = "🔄 Force refreshing analysis... This will take 5-7 minutes."
        else:
            cached_df, metadata = load_analysis_data()
            if not cached_df.empty and not is_data_stale(metadata) and metadata.get('period') == analysis_period:
                spinner_msg = "📱 Loading cached analysis..."
            else:
                spinner_msg = "🔄 Analyzing 100 S&P 500 stocks... This will take 5-7 minutes."
        
        with st.spinner(spinner_msg):
            analysis_results = analyze_sp500_stocks(SP500_SAMPLE, analysis_period, force_refresh=force_refresh)
            show_results = True
            
        if not analysis_results.empty:
            st.success(f"✅ Analyzed {len(analysis_results)} stocks successfully!")
    
    # Display results if available (either cached or freshly analyzed)
    if show_results and not analysis_results.empty:
        # Top 10 recommendations
        top_10 = analysis_results.head(10)
            
        st.subheader("🏆 Top 10 Stock Recommendations")
        st.caption("Based on composite score (Return + Sharpe Ratio + Low Volatility + Low Drawdown + Recent Momentum)")
            
        display_cols = ['Ticker', 'Annual_Return', 'Sharpe_Ratio', 'Volatility', 'Max_Drawdown', 'Recent_3M_Return', 'Current_Price', 'Composite_Score']
        display_df = top_10[display_cols].copy()
        
        # Format percentages and numbers
        display_df['Annual_Return'] = (display_df['Annual_Return'] * 100).round(2).astype(str) + '%'
        display_df['Volatility'] = (display_df['Volatility'] * 100).round(2).astype(str) + '%'
        display_df['Max_Drawdown'] = (display_df['Max_Drawdown'] * 100).round(2).astype(str) + '%'
        display_df['Recent_3M_Return'] = (display_df['Recent_3M_Return'] * 100).round(2).astype(str) + '%'
        display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].round(2)
        display_df['Current_Price'] = '$' + display_df['Current_Price'].round(2).astype(str)
        display_df['Composite_Score'] = display_df['Composite_Score'].round(3)
        
        # Rename columns for display
        display_df.columns = ['Ticker', 'Annual Return', 'Sharpe Ratio', 'Volatility', 'Max Drawdown', '3M Return', 'Price', 'Score']
        
        st.dataframe(display_df, use_container_width=True)
            
        # Quick use button
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("📈 Use Top 10 for Portfolio Optimization", key="use_top10"):
                top_10_tickers = ",".join(top_10['Ticker'].tolist())
                st.session_state['recommended_tickers'] = top_10_tickers
                st.success("✅ Top 10 stocks copied! Go to Portfolio Optimizer tab.")
        
        # Download button
        with col2:
            csv_data = analysis_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Analysis",
                data=csv_data,
                file_name=f"sp500_analysis_{analysis_period}.csv",
                mime="text/csv"
            )
        
        # Data management section
        with st.expander("🗄️ Data Management", expanded=False):
            st.caption("Manage cached analysis data")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Cache", help="Delete all cached analysis data"):
                    try:
                        if ANALYSIS_FILE.exists():
                            ANALYSIS_FILE.unlink()
                        if METADATA_FILE.exists():
                            METADATA_FILE.unlink()
                        st.success("✅ Cache cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to clear cache: {e}")
            
            with col2:
                # Show file info
                if ANALYSIS_FILE.exists():
                    file_size = ANALYSIS_FILE.stat().st_size / 1024  # KB
                    st.metric("Cache Size", f"{file_size:.1f} KB")
                else:
                    st.metric("Cache Size", "0 KB")
        
        # Analysis summary
        st.subheader("📈 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_return = top_10['Annual_Return'].mean()
            st.metric("Avg Return (Top 10)", f"{avg_return:.1%}")
        
        with col2:
            avg_sharpe = top_10['Sharpe_Ratio'].mean()
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
        
        with col3:
            avg_vol = top_10['Volatility'].mean()
            st.metric("Avg Volatility", f"{avg_vol:.1%}")
        
        with col4:
            total_analyzed = len(analysis_results)
            st.metric("Stocks Analyzed", f"{total_analyzed}")
    
    elif not show_results:
        st.info("🔍 No analysis data available. Click 'Analyze Stocks' to start analyzing S&P 500 stocks.")

# -------------------------------
# Portfolio Optimizer Tab
# -------------------------------
with tab1:
    # -------------------------------
    # Stock Selection Display (only for Custom from Analysis)
    # -------------------------------
    if stock_selection == "Custom from Analysis":
        st.info(f"🎯 **Stock Selection**: {stock_selection} ({len(TICKERS)} stocks)")
        
        # Show analysis scores if available
        if ANALYSIS_FILE.exists():
            try:
                cached_df = pd.read_parquet(ANALYSIS_FILE)
                # Filter to show only selected stocks
                selected_analysis = cached_df[cached_df['Ticker'].isin(TICKERS)].copy()
                if not selected_analysis.empty:
                    with st.expander("📊 Selected Stocks Analysis Scores", expanded=False):
                        display_cols = ['Ticker', 'Annual_Return', 'Sharpe_Ratio', 'Volatility', 'Composite_Score']
                        display_df = selected_analysis[display_cols].copy()
                        
                        # Format for display
                        display_df['Annual_Return'] = (display_df['Annual_Return'] * 100).round(2).astype(str) + '%'
                        display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].round(2)
                        display_df['Volatility'] = (display_df['Volatility'] * 100).round(2).astype(str) + '%'
                        display_df['Composite_Score'] = display_df['Composite_Score'].round(3)
                        
                        display_df.columns = ['Ticker', 'Ann. Return', 'Sharpe', 'Volatility', 'Score']
                        st.dataframe(display_df, use_container_width=True)
            except:
                pass  # Silently ignore errors in display

    # -------------------------------
    # Run Portfolio Optimization
    # -------------------------------
    prices = load_prices(TICKERS, horizon, interval)

    if prices.empty:
        st.warning(
            "Insufficient data after cleaning. Try fewer tickers, a shorter interval (1wk/1mo), "
            "or a shorter lookback (5y). Some symbols may be unavailable for the chosen settings."
        )
        if TICKERS:
            st.caption(f"Attempted tickers: {', '.join(TICKERS)}")
        st.stop()

    st.subheader("Price Preview")
    st.dataframe(prices.tail())

    with st.spinner("Optimizing portfolio..."):
        raw_weights, perf = optimize_portfolio(
            prices, objective, risk_free, target_return, max_weight, l2_reg
        )
        final_weights = enforce_min_holdings(raw_weights, min_holdings, min_weight_threshold)

    if final_weights.empty:
        st.error("Optimization produced empty weights. Relax constraints (lower min holdings, raise max weight) and retry.")
        st.stop()

    st.subheader("Optimal Weights")
    weights_df = final_weights.to_frame("Weight").sort_values("Weight", ascending=False)
    weights_df["Weight_pct"] = (weights_df["Weight"] * 100).round(2)
    st.dataframe(weights_df.style.format({"Weight": "{:.4f}", "Weight_pct": "{:.2f}"}))

    # Dollar allocation
    latest = get_latest_prices(prices[weights_df.index])
    alloc_usd = (weights_df["Weight"] * investment).rename("Allocation_USD").round(2)
    alloc_df = pd.concat([weights_df, latest.rename("Last_Price"), alloc_usd], axis=1)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Performance (Expected)")
        kpi = pd.DataFrame({
            "Expected Return (annualized)": [f"{perf['expected_return']*100:.2f}%"],
            "Expected Volatility (annualized)": [f"{perf['volatility']*100:.2f}%"],
            f"Sharpe (rf={risk_free:.2f}%)": [f"{perf['sharpe']:.2f}"],
        })
        st.table(kpi)

    with right:
        st.subheader("Investment")
        st.metric("Total Investment", f"${investment:,.0f}")
        st.metric("Holdings", f"{len(final_weights)} tickers")
        
        # Show if using custom analysis-based selection
        if stock_selection == "Custom from Analysis":
            st.caption(f"🎯 Using {stock_selection.lower()}")

    st.subheader("Dollar Allocation")
    st.dataframe(
        alloc_df.style.format({
            "Weight": "{:.4f}",
            "Weight_pct": "{:.2f}",
            "Last_Price": "${:,.2f}",
            "Allocation_USD": "${:,.2f}",
        })
    )

    st.download_button(
        "Download Weights (CSV)",
        data=weights_df.to_csv().encode("utf-8"),
        file_name="weights.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download Full Allocation (CSV)",
        data=alloc_df.to_csv().encode("utf-8"),
        file_name="allocation.csv",
        mime="text/csv"
    )

    # Normalized performance chart of selected holdings
    st.subheader("Historical Performance (Normalized)")
    norm = prices[final_weights.index] / prices[final_weights.index].iloc[0] * 100
    st.line_chart(norm)
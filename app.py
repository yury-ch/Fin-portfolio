# app.py
# -------------------------------
# S&P 500 Portfolio Optimizer
# Streamlit + yfinance + PyPortfolioOpt
# -------------------------------

import logging
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import time
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import get_script_run_ctx, RerunData

import yfinance as yf
from pypfopt import (
    EfficientFrontier,
    risk_models,
    expected_returns,
    objective_functions,
    DiscreteAllocation,
    get_latest_prices,
)

from shared.ticker_provider import DEFAULT_SP500_SAMPLE, WikipediaTickerProvider

# -------------------------------
# Global Constants for Data Persistence
# -------------------------------
DATA_DIR = Path(__file__).resolve().parent / "sp500_data"
ANALYSIS_FILE = DATA_DIR / "sp500_analysis.parquet"  # legacy single-period cache
METADATA_FILE = DATA_DIR / "metadata.parquet"        # legacy single-period metadata
HIDDEN_ANALYSIS_COLUMNS = ['analysis_timestamp', 'analysis_period', 'data_through']
ANALYSIS_PERIODS = ["1y", "2y", "3y"]

st.set_page_config(page_title="S&P Portfolio Optimizer", layout="wide")

st.title("📈 S&P 500 Portfolio Optimizer")
st.caption("Build an optimal S&P portfolio given a risk–return preference, investment amount, and horizon.")

if 'analysis_cache' not in st.session_state:
    st.session_state['analysis_cache'] = {}
if 'analysis_task' not in st.session_state:
    st.session_state['analysis_task'] = None
if 'analysis_executor' not in st.session_state:
    st.session_state['analysis_executor'] = ThreadPoolExecutor(max_workers=1)
if 'analysis_futures' not in st.session_state:
    st.session_state['analysis_futures'] = {}

# -------------------------------
# Column mapping for backwards compatibility
# -------------------------------
def standardize_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for backwards compatibility with cached data.
    Maps old column names to new expected names.
    """
    if df.empty:
        return df
    
    # Column mapping from old cached data to expected names
    column_mapping = {
        'ticker': 'Ticker',
        'total_return_pct': 'Annual_Return',
        'sharpe_ratio': 'Sharpe_Ratio', 
        'volatility_pct': 'Volatility',
        'max_drawdown_pct': 'Max_Drawdown',
        'current_price': 'Current_Price',
        'composite_score': 'Composite_Score'
    }
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df_copy.columns:
            df_copy.rename(columns={old_name: new_name}, inplace=True)
    
    # Add missing columns with default values if they don't exist
    if 'Recent_3M_Return' not in df_copy.columns:
        df_copy['Recent_3M_Return'] = 0.0  # Default to 0 for missing recent returns
    
    # Convert percentages from the old format (already in percentage) to decimal format expected by new code
    if 'Annual_Return' in df_copy.columns:
        # Check if values are already in percentage format (> 1 indicates percentage)
        if df_copy['Annual_Return'].max() > 1:
            df_copy['Annual_Return'] = df_copy['Annual_Return'] / 100.0
    
    if 'Volatility' in df_copy.columns:
        if df_copy['Volatility'].max() > 1:
            df_copy['Volatility'] = df_copy['Volatility'] / 100.0
            
    if 'Max_Drawdown' in df_copy.columns:
        if df_copy['Max_Drawdown'].min() > -1:  # Old format might be positive percentage
            df_copy['Max_Drawdown'] = -df_copy['Max_Drawdown'] / 100.0  # Convert to negative decimal
    
    return df_copy

# -------------------------------
# Sidebar: Inputs
# -------------------------------
# Shared ticker provider keeps the universe aligned with microservices
logger = logging.getLogger(__name__)
_ticker_provider = WikipediaTickerProvider(fallback=list(DEFAULT_SP500_SAMPLE))
_sp500_universe_cache: List[str] = []


def get_sp500_universe(force_refresh: bool = False) -> List[str]:
    """Return the full S&P 500 universe, refreshing from Wikipedia when needed."""
    global _sp500_universe_cache
    if force_refresh or not _sp500_universe_cache:
        try:
            tickers = _ticker_provider.get_constituents(force_refresh=force_refresh)
            if len(tickers) < 200:
                raise ValueError(f"Only received {len(tickers)} tickers from provider")
            _sp500_universe_cache = tickers
        except Exception as exc:
            logger.warning("Falling back to built-in S&P sample: %s", exc)
            _sp500_universe_cache = list(DEFAULT_SP500_SAMPLE)
    return _sp500_universe_cache

DEFAULT_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","JPM","V"
]

# -------------------------------
# Data Persistence Functions (Parquet)
# -------------------------------

def ensure_data_directory():
    """Create data directory if it doesn't exist"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def analysis_file_path(period: Optional[str] = None) -> Path:
    """Return the parquet path for a specific analysis period (or legacy default)."""
    if period:
        return DATA_DIR / f"sp500_analysis_{period}.parquet"
    return ANALYSIS_FILE

def metadata_file_path(period: Optional[str] = None) -> Path:
    """Return the metadata parquet path for a specific analysis period (or legacy default)."""
    if period:
        return DATA_DIR / f"metadata_{period}.parquet"
    return METADATA_FILE

def save_analysis_data(df: pd.DataFrame, period: str, data_through: Optional[datetime] = None, announce: bool = True) -> dict:
    """Save analysis results to a period-specific parquet with metadata."""
    ensure_data_directory()
    
    df = df.copy()
    analysis_ts = datetime.now()
    df['analysis_timestamp'] = analysis_ts
    df['analysis_period'] = period
    if data_through is not None:
        df['data_through'] = data_through
    
    data_path = analysis_file_path(period)
    meta_path = metadata_file_path(period)
    df.to_parquet(data_path, index=False)
    
    metadata = {
        'last_updated': analysis_ts,
        'period': period,
        'num_stocks': len(df),
        'version': '1.0',
        'data_through': data_through
    }
    pd.DataFrame([metadata]).to_parquet(meta_path, index=False)
    
    if announce:
        st.info(f"💾 Analysis data saved ({period})")
    
    return metadata

def clean_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal metadata columns from cached dataframes."""
    if df.empty:
        return df
    return df.drop(HIDDEN_ANALYSIS_COLUMNS, axis=1, errors='ignore')

def format_timestamp(value) -> str:
    """Format metadata timestamps for display."""
    if value is None or value == "Unknown":
        return "Unknown"
    if isinstance(value, str):
        try:
            value = pd.to_datetime(value)
        except Exception:
            return value
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)

def _read_analysis_artifacts(data_path: Path, meta_path: Path, silent: bool = False) -> Tuple[pd.DataFrame, dict]:
    if not data_path.exists() or not meta_path.exists():
        return pd.DataFrame(), {}
    try:
        df = pd.read_parquet(data_path)
        df = standardize_analysis_columns(df)
        metadata = pd.read_parquet(meta_path).iloc[0].to_dict()
        return clean_analysis_dataframe(df), metadata
    except Exception as e:
        if not silent:
            st.warning(f"Failed to load cached data: {e}")
        return pd.DataFrame(), {}

def load_analysis_data(period: Optional[str] = None, silent: bool = False) -> Tuple[pd.DataFrame, dict]:
    """Load analysis results for a specific period (or legacy default)."""
    data_path = analysis_file_path(period)
    meta_path = metadata_file_path(period)
    df, metadata = _read_analysis_artifacts(data_path, meta_path, silent)
    if not df.empty or metadata:
        return df, metadata
    
    # Fallback: legacy single-period cache if it matches the requested period
    if period is not None:
        legacy_df, legacy_meta = _read_analysis_artifacts(ANALYSIS_FILE, METADATA_FILE, silent=True)
        if not legacy_df.empty and legacy_meta.get('period') == period:
            return legacy_df, legacy_meta
    return pd.DataFrame(), {}

def get_cached_analysis(period: str) -> Tuple[pd.DataFrame, dict]:
    """Return cached analysis for the given period, hydrating from disk if needed."""
    cache = st.session_state.setdefault('analysis_cache', {})
    entry = cache.get(period)
    if entry:
        return entry.get('df', pd.DataFrame()), entry.get('metadata', {})
    
    df, metadata = load_analysis_data(period, silent=True)
    if not df.empty or metadata:
        cache[period] = {
            'df': df,
            'metadata': metadata
        }
    return cache.get(period, {}).get('df', pd.DataFrame()), cache.get(period, {}).get('metadata', {})

def hydrate_analysis_cache(periods: List[str]):
    """Ensure session_state cache includes any persisted results for the given periods."""
    for period in periods:
        get_cached_analysis(period)

def get_cache_info(periods: Optional[List[str]] = None) -> str:
    """Return a compact summary of cached periods for display."""
    cache = st.session_state.get('analysis_cache', {})
    period_list = periods or sorted(cache.keys())
    rows = []
    for period in period_list:
        metadata = cache.get(period, {}).get('metadata')
        if not metadata:
            continue
        last_updated = format_timestamp(metadata.get('last_updated'))
        num_stocks = metadata.get('num_stocks', '—')
        rows.append(f"{period}: {num_stocks} stocks @ {last_updated}")
    if rows:
        return " | ".join(rows)
    return "No cached data available"

# Hydrate cached data for the default analyzer periods on startup
hydrate_analysis_cache(ANALYSIS_PERIODS)

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


# -------------------------------
# S&P 500 Analysis Functions
# -------------------------------
def compute_sp500_analysis(tickers: List[str], period: str = "1y") -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """Compute S&P 500 analysis metrics without touching the UI."""
    results = []
    latest_data_timestamp: Optional[pd.Timestamp] = None
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, interval="1d", progress=False)
            if data.empty:
                continue
            
            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            returns = prices.pct_change().dropna()
            if len(returns) < 60:
                continue
            
            daily_log = np.log(prices).diff().dropna()
            if daily_log.empty:
                continue
            mean_log = daily_log.mean()
            ann_return = float(np.exp(mean_log * 252) - 1)
            ann_vol = float(returns.std() * np.sqrt(252))
            sharpe = float(ann_return / ann_vol if ann_vol > 0 else 0)
            
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = float(drawdown.min())
            
            if len(prices) >= 63:
                recent_return = float((prices.iloc[-1] / prices.iloc[-63]) - 1)
            else:
                recent_return = 0.0
            
            current_price = float(prices.iloc[-1])
            
            data_end = prices.index.max()
            if isinstance(data_end, pd.Timestamp):
                if latest_data_timestamp is None or data_end > latest_data_timestamp:
                    latest_data_timestamp = data_end
            
            results.append({
                'Ticker': ticker,
                'Annual_Return': ann_return,
                'Volatility': ann_vol,
                'Sharpe_Ratio': sharpe,
                'Max_Drawdown': max_drawdown,
                'Recent_3M_Return': recent_return,
                'Current_Price': current_price
            })
        except Exception:
            continue
        
        time.sleep(0.2)
    
    if not results:
        return pd.DataFrame(), latest_data_timestamp
    
    df = pd.DataFrame(results)
    
    def safe_normalize(series, inverse=False):
        range_val = series.max() - series.min()
        if range_val == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        normalized = (series - series.min()) / range_val
        return 1 - normalized if inverse else normalized
    
    df['Return_Score'] = safe_normalize(df['Annual_Return'])
    df['Vol_Score'] = safe_normalize(df['Volatility'], inverse=True)
    df['Sharpe_Score'] = safe_normalize(df['Sharpe_Ratio'])
    df['Drawdown_Score'] = safe_normalize(df['Max_Drawdown'], inverse=True)
    df['Momentum_Score'] = safe_normalize(df['Recent_3M_Return'])
    
    df['Composite_Score'] = (
        0.25 * df['Return_Score'] +
        0.20 * df['Vol_Score'] +
        0.25 * df['Sharpe_Score'] +
        0.15 * df['Drawdown_Score'] +
        0.15 * df['Momentum_Score']
    )
    
    df = df.sort_values('Composite_Score', ascending=False)
    return df, latest_data_timestamp

def run_analysis_job(period: str, force_refresh: bool = False, tickers: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    """Execute analysis, respecting cache rules, suitable for background threads."""
    tickers = tickers or get_sp500_universe()
    cached_df, metadata = load_analysis_data(period, silent=True)
    if not force_refresh and not cached_df.empty:
        cached_period = metadata.get('period', '')
        if cached_period == period and not is_data_stale(metadata):
            return cached_df, metadata
    
    df, latest_data_timestamp = compute_sp500_analysis(tickers, period)
    if df.empty:
        return pd.DataFrame(), {}
    
    metadata = save_analysis_data(df, period, data_through=latest_data_timestamp, announce=False)
    cleaned_df = clean_analysis_dataframe(df)
    return cleaned_df, metadata

def analyze_sp500_stocks(tickers: List[str], period: str = "1y", force_refresh: bool = False) -> pd.DataFrame:
    """Synchronous compatibility wrapper."""
    df, _ = run_analysis_job(period, force_refresh, tickers)
    return df

def launch_async_analysis(period: str, force_refresh: bool = False):
    """Start an asynchronous analysis task if one is not already running."""
    if st.session_state.get('analysis_task'):
        return
    task_id = str(uuid.uuid4())
    future = st.session_state['analysis_executor'].submit(run_analysis_job, period, force_refresh)
    st.session_state['analysis_futures'][task_id] = future
    st.session_state['analysis_task'] = {
        'id': task_id,
        'period': period,
        'force_refresh': force_refresh,
        'started_at': datetime.now()
    }
    schedule_rerun_on_completion(future)

def poll_analysis_task():
    """Check background task status and update session state when complete."""
    task = st.session_state.get('analysis_task')
    if not task:
        return
    future = st.session_state['analysis_futures'].get(task['id'])
    if future is None:
        st.session_state.pop('analysis_task', None)
        return
    if future.done():
        try:
            df, metadata = future.result()
            if df is not None and not df.empty:
                period_key = metadata.get('period', task['period'])
                cache = st.session_state.setdefault('analysis_cache', {})
                cache[period_key] = {
                    'df': clean_analysis_dataframe(df),
                    'metadata': metadata or {}
                }
        except Exception as exc:
            st.session_state['analysis_error'] = str(exc)
        finally:
            st.session_state['analysis_futures'].pop(task['id'], None)
            st.session_state.pop('analysis_task', None)
            st.experimental_rerun()

def schedule_rerun_on_completion(future):
    """Request a Streamlit rerun when the background analysis finishes."""
    ctx = get_script_run_ctx(suppress_warning=True)
    if ctx is None:
        return
    script_requests = getattr(ctx, "script_requests", None)
    if script_requests is None:
        return
    rerun_data = RerunData(
        query_string=getattr(ctx, "query_string", ""),
        page_script_hash=getattr(ctx, "page_script_hash", ""),
        cached_message_hashes=set(getattr(ctx, "cached_message_hashes", set())),
        context_info=getattr(ctx, "context_info", None),
    )

    def _notify():
        try:
            future.result()
        except Exception:
            pass
        finally:
            script_requests.request_rerun(rerun_data)

    threading.Thread(target=_notify, daemon=True).start()

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

poll_analysis_task()

with tab2:
    st.header("🔍 S&P 500 Stock Analyzer")
    sp500_universe = get_sp500_universe()
    st.caption(f"Analyze the full S&P 500 universe ({len(sp500_universe)} tickers) and surface the top recommendations based on multi-factor scoring.")
    if st.button("🔁 Refresh S&P 500 universe", key="refresh_universe"):
        sp500_universe = get_sp500_universe(force_refresh=True)
        st.success(f"🔄 Reloaded {len(sp500_universe)} tickers from Wikipedia.")
    
    analysis_cache = st.session_state.get('analysis_cache', {})
    analysis_task = st.session_state.get('analysis_task')
    analysis_error = st.session_state.pop('analysis_error', None)
    
    period_options = ANALYSIS_PERIODS
    cache_info = get_cache_info(period_options)
    if cache_info != "No cached data available":
        st.info(f"💾 {cache_info}")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analysis_period = st.selectbox("Analysis period", period_options, index=0, key="analysis_period")
    
    with col2:
        analyze_button = st.button("🚀 Analyze Stocks", type="primary")
    
    with col3:
        force_refresh = st.button("🔄 Force Refresh", help="Ignore cached data and perform fresh analysis")
    
    if analyze_button:
        if analysis_task:
            st.warning("Analysis already running. Please wait for it to finish.")
        else:
            launch_async_analysis(analysis_period, force_refresh=False)
            st.info("📡 Analysis started... you can continue using the dashboard.")
    
    if force_refresh:
        if analysis_task:
            st.warning("Analysis already running. Please wait for it to finish.")
        else:
            launch_async_analysis(analysis_period, force_refresh=True)
            st.info("♻️ Force refresh started... existing cache will be replaced.")
    
    if analysis_task:
        started_at = analysis_task['started_at']
        elapsed = datetime.now() - started_at
        minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
        st.info(f"⏳ Analysis in progress (started {started_at.strftime('%H:%M:%S')} | elapsed {minutes}m {seconds}s)")
    
    if analysis_error:
        st.error(f"Analysis failed: {analysis_error}")
    
    status_rows = []
    for period in period_options:
        meta = analysis_cache.get(period, {}).get('metadata')
        if meta:
            status_rows.append({
                'Period': period,
                'Last Analysis': format_timestamp(meta.get('last_updated')),
                'Prices Through': format_timestamp(meta.get('data_through')),
                'Stocks': meta.get('num_stocks', '—')
            })
        else:
            status_rows.append({
                'Period': period,
                'Last Analysis': 'Not run',
                'Prices Through': '—',
                'Stocks': '—'
            })
    status_df = pd.DataFrame(status_rows).set_index('Period')
    st.table(status_df)

    analysis_results, analysis_metadata = get_cached_analysis(analysis_period)
    universe_size = len(sp500_universe)
    if analysis_metadata:
        analyzed_count = analysis_metadata.get('num_stocks', 0) or 0
        metadata_stamp = format_timestamp(analysis_metadata.get('last_updated'))
        if analyzed_count and analyzed_count < universe_size and not analysis_task:
            st.warning(
                f"Cached analysis ({metadata_stamp}) only covers {analyzed_count} of {universe_size} tickers. "
                "Run 'Analyze Stocks' to refresh the full universe."
            )
    
    if not analysis_results.empty:
        # Top 20 recommendations
        top_20 = analysis_results.head(20)
            
        st.subheader("🏆 Top 20 Stock Recommendations")
        st.caption("Based on composite score (Return + Sharpe Ratio + Low Volatility + Low Drawdown + Recent Momentum)")
        if analysis_metadata:
            st.caption(f"Last analysis: {format_timestamp(analysis_metadata.get('last_updated'))} | Prices through: {format_timestamp(analysis_metadata.get('data_through'))}")
        st.caption(f"Showing top {len(top_20)} of {len(analysis_results)} analyzed tickers.")
            
        display_cols = ['Ticker', 'Annual_Return', 'Sharpe_Ratio', 'Volatility', 'Max_Drawdown', 'Recent_3M_Return', 'Current_Price', 'Composite_Score']
        display_df = top_20[display_cols].copy()
        
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
            if st.button("📈 Use Top 20 for Portfolio Optimization", key="use_top20"):
                top_20_tickers = ",".join(top_20['Ticker'].tolist())
                st.session_state['recommended_tickers'] = top_20_tickers
                st.success("✅ Top 20 stocks copied! Go to Portfolio Optimizer tab.")
        
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
            data_path = analysis_file_path(analysis_period)
            meta_path = metadata_file_path(analysis_period)
            with col1:
                if st.button("🗑️ Clear Cache", help="Delete cached analysis data for this period"):
                    try:
                        removed = False
                        for path in {data_path, meta_path}:
                            if path.exists():
                                path.unlink()
                                removed = True
                        # Remove legacy cache if it represents the same period
                        if METADATA_FILE.exists():
                            try:
                                legacy_meta = pd.read_parquet(METADATA_FILE).iloc[0].to_dict()
                            except Exception:
                                legacy_meta = {}
                            if legacy_meta.get('period') == analysis_period:
                                for path in {ANALYSIS_FILE, METADATA_FILE}:
                                    if path.exists():
                                        path.unlink()
                                        removed = True
                        st.session_state['analysis_cache'].pop(analysis_period, None)
                        if removed:
                            st.success(f"✅ Cache cleared for {analysis_period}!")
                        else:
                            st.info("ℹ️ No cached files found for this period.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to clear cache: {e}")
            
            with col2:
                if data_path.exists():
                    file_size = data_path.stat().st_size / 1024  # KB
                    st.metric("Cache Size", f"{file_size:.1f} KB")
                else:
                    st.metric("Cache Size", "0 KB")
    else:
        st.info("🔍 No analysis data available. Click 'Analyze Stocks' to process the entire S&P 500 universe.")

# -------------------------------
# Portfolio Optimizer Tab
# -------------------------------
with tab1:
    st.header("📊 Portfolio Optimizer")
    st.caption("Tune analysis settings, review recommendations, and build a portfolio that matches your constraints.")
    selection_container = st.container()
    analysis_container = st.container()
    constraint_container = st.container()

    with analysis_container:
        st.subheader("Analysis Settings")
        settings_col1, settings_col2, settings_col3 = st.columns(3)
        with settings_col1:
            horizon = st.selectbox(
                "Analysis period / lookback",
                ["1y", "2y", "3y", "5y"],
                index=0,
                key="optimizer_horizon"
            )
            interval = st.selectbox(
                "Price frequency",
                ["1d", "1wk", "1mo"],
                index=0,
                key="optimizer_interval"
            )
        with settings_col2:
            objective = st.selectbox(
                "Optimization objective",
                ["Max Sharpe", "Min Volatility (target return)", "Target Return"],
                index=0,
                key="optimizer_objective"
            )
            target_return = st.number_input(
                "Target annual return (%)",
                value=10.0,
                step=0.5,
                key="optimizer_target_return"
            )
        with settings_col3:
            risk_free = st.number_input(
                "Risk-free rate (annual, %)",
                value=4.0,
                step=0.25,
                key="optimizer_risk_free"
            )
            investment = st.number_input(
                "Total investment (USD)",
                value=250_000,
                step=10_000,
                key="optimizer_investment"
            )
        st.caption("Tip: Monthly data is usually smoother for 5–10 year horizons.")

    with constraint_container:
        st.subheader("Portfolio Constraints")
        constraint_cols = st.columns(4)
        with constraint_cols[0]:
            max_weight = st.slider(
                "Max weight per stock (%)",
                min_value=5,
                max_value=50,
                value=30,
                step=1,
                key="optimizer_max_weight"
            )
        with constraint_cols[1]:
            l2_reg = st.slider(
                "L2 regularization",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                key="optimizer_l2"
            )
        with constraint_cols[2]:
            min_weight_threshold = st.slider(
                "Prune weights below (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.25,
                step=0.05,
                key="optimizer_prune_threshold"
            )
        with constraint_cols[3]:
            min_holdings = st.number_input(
                "Minimum holdings",
                min_value=2,
                max_value=60,
                value=3,
                step=1,
                key="optimizer_min_holdings"
            )

    # -------------------------------
    # Stock Selection Controls (Optimizer only)
    # -------------------------------
    with selection_container:
        st.subheader("Stock Selection Method")
        custom_analysis_df: Optional[pd.DataFrame] = None
        stock_selection = st.radio(
            "Choose how to feed tickers into the optimizer",
            ["Manual Entry", "Use Top Performers from Analysis", "Custom from Analysis"],
            horizontal=True
        )
        
        if stock_selection == "Manual Entry":
            default_value = st.session_state.get('recommended_tickers', ",".join(DEFAULT_TICKERS))
            tickers_str = st.text_area(
                "Tickers (comma-separated)",
                value=default_value,
                help="Paste S&P tickers, e.g. AAPL,MSFT,NVDA. Tip: Yahoo uses BRK-B for Berkshire B. Use the analyzer for quick recommendations."
            )
            TICKERS = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            
            if 'recommended_tickers' in st.session_state and st.session_state.get('recommended_tickers') == tickers_str:
                if st.button("🔄 Clear Recommendations", key="clear_recs"):
                    del st.session_state['recommended_tickers']
                    st.experimental_rerun()
        
        elif stock_selection == "Use Top Performers from Analysis":
            num_top_stocks = st.slider("Number of top stocks", min_value=5, max_value=25, value=10, step=1)
            
            cached_df, metadata = get_cached_analysis(horizon)
            top_performers: List[str] = []
            if not cached_df.empty:
                top_performers = cached_df.head(num_top_stocks)['Ticker'].tolist()
            
            if top_performers:
                last_run = format_timestamp(metadata.get('last_updated')) if metadata else "Unknown"
                if metadata and not is_data_stale(metadata):
                    st.success(f"✅ Using cached analysis from {last_run}")
                else:
                    st.warning(f"⚠️ Using cached analysis last run {last_run}. Consider refreshing for the latest data.")
                st.caption(f"Top stocks: {', '.join(top_performers)}")
                TICKERS = top_performers
                st.session_state['recommended_tickers'] = ",".join(top_performers)
            else:
                st.warning("⚠️ No analysis data available for this horizon. Run S&P 500 analyzer first or switch to manual entry.")
                fallback = st.session_state.get('recommended_tickers')
                if fallback:
                    TICKERS = [t.strip() for t in fallback.split(",") if t.strip()]
                else:
                    TICKERS = DEFAULT_TICKERS
        
        else:  # Custom from Analysis
            cached_df, metadata = get_cached_analysis(horizon)
            custom_analysis_df = cached_df
            if not cached_df.empty:
                last_updated = metadata.get('last_updated')
                if isinstance(last_updated, str):
                    last_updated = pd.to_datetime(last_updated)
                age_hours = None
                if isinstance(last_updated, datetime):
                    age_hours = max((datetime.now() - last_updated).total_seconds() / 3600, 0)
                num_stocks = len(cached_df)
                timestamp_label = format_timestamp(last_updated) if last_updated else "Unknown"
                age_snippet = f" ({age_hours:.1f}h ago)" if age_hours is not None else ""
                st.success(f"📊 Analysis cache: {num_stocks} stocks, last run {timestamp_label}{age_snippet}")
                
                top_20 = cached_df.head(20)
                default_choices = top_20.head(min(10, len(top_20)))['Ticker'].tolist()
                selected_tickers = st.multiselect(
                    "Select stocks from analysis (top 20 shown)",
                    options=top_20['Ticker'].tolist(),
                    default=default_choices,
                    help="Choose specific stocks from the analysis results"
                )
                
                if selected_tickers:
                    TICKERS = selected_tickers
                    st.session_state['recommended_tickers'] = ",".join(selected_tickers)
                else:
                    st.warning("⚠️ Please select at least 2 stocks")
                    TICKERS = default_choices or DEFAULT_TICKERS
            else:
                st.warning("⚠️ No analysis data. Run S&P 500 analyzer first.")
                fallback = st.session_state.get('recommended_tickers')
                TICKERS = [t.strip() for t in fallback.split(",")] if fallback else DEFAULT_TICKERS

        if not TICKERS:
            st.warning("No tickers provided. Falling back to default recommendations.")
            TICKERS = DEFAULT_TICKERS
        st.info(f"Optimizing with {len(TICKERS)} tickers.")
        st.markdown("---")
    
    if stock_selection == "Custom from Analysis" and custom_analysis_df is not None:
        selected_analysis = custom_analysis_df[custom_analysis_df['Ticker'].isin(TICKERS)].copy()
        if not selected_analysis.empty:
            with st.expander("📊 Selected Stocks Analysis Scores", expanded=False):
                display_cols = ['Ticker', 'Annual_Return', 'Sharpe_Ratio', 'Volatility', 'Composite_Score']
                display_df = selected_analysis[display_cols].copy()
                
                display_df['Annual_Return'] = (display_df['Annual_Return'] * 100).round(2).astype(str) + '%'
                display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].round(2)
                display_df['Volatility'] = (display_df['Volatility'] * 100).round(2).astype(str) + '%'
                display_df['Composite_Score'] = display_df['Composite_Score'].round(3)
                
                display_df.columns = ['Ticker', 'Ann. Return', 'Sharpe', 'Volatility', 'Score']
                st.dataframe(display_df, use_container_width=True)

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

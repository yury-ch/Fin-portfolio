# services/data_service.py
# -------------------------------
# Data Service - Download & Persistence Layer
# Handles yfinance API calls, data caching, and S&P 500 analysis
# -------------------------------

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import time
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, HTTPException, Query
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import (
    StockDataRequest, StockAnalysisRequest, ServiceResponse,
    AnalysisMetadata, CacheInfo
)
from shared.ticker_provider import WikipediaTickerProvider, DEFAULT_SP500_SAMPLE
from shared.price_loader import PriceCacheManager, YahooPriceLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).resolve().parents[1] / "sp500_data"
ANALYSIS_FILE = DATA_DIR / "sp500_analysis.parquet"  # legacy single cache
METADATA_FILE = DATA_DIR / "metadata.parquet"
HIDDEN_ANALYSIS_COLUMNS = ['analysis_timestamp', 'analysis_period', 'data_through']
ANALYSIS_PERIODS = ["1y", "2y", "3y"]
PRICE_CACHE_MAX_AGE_HOURS = (7 * 24) + 4  # weekly sync + small grace period
ANALYSIS_CACHE_MAX_AGE_HOURS = (7 * 24) + 4

# S&P 500 Top 100 stocks
SP500_SAMPLE = list(DEFAULT_SP500_SAMPLE)

def standardize_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Align cached data with the expected analyzer schema."""
    if df is None or df.empty:
        return df
    column_mapping = {
        'ticker': 'Ticker',
        'total_return_pct': 'Annual_Return',
        'sharpe_ratio': 'Sharpe_Ratio',
        'volatility_pct': 'Volatility',
        'max_drawdown_pct': 'Max_Drawdown',
        'current_price': 'Current_Price',
        'composite_score': 'Composite_Score'
    }
    df = df.copy()
    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
    if 'Recent_3M_Return' not in df.columns:
        df['Recent_3M_Return'] = 0.0
    if 'Annual_Return' in df.columns and df['Annual_Return'].max() > 1:
        df['Annual_Return'] = df['Annual_Return'] / 100.0
    if 'Volatility' in df.columns and df['Volatility'].max() > 1:
        df['Volatility'] = df['Volatility'] / 100.0
    if 'Max_Drawdown' in df.columns and df['Max_Drawdown'].min() > -1:
        df['Max_Drawdown'] = -df['Max_Drawdown'] / 100.0
    return df


class DataService:
    """Data Service for handling stock data download and persistence"""
    
    def __init__(self):
        self.ensure_data_directory()
        self.ticker_provider = WikipediaTickerProvider(fallback=SP500_SAMPLE)
        self.price_cache = PriceCacheManager()
        self.price_loader = YahooPriceLoader()
    
    def ensure_data_directory(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def analysis_file_path(self, period: Optional[str] = None) -> Path:
        if period:
            return DATA_DIR / f"sp500_analysis_{period}.parquet"
        return ANALYSIS_FILE
    
    def metadata_file_path(self, period: Optional[str] = None) -> Path:
        if period:
            return DATA_DIR / f"metadata_{period}.parquet"
        return METADATA_FILE
    
    def clean_analysis_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        return df.drop(HIDDEN_ANALYSIS_COLUMNS, axis=1, errors='ignore')
    
    def get_sp500_universe(self) -> List[str]:
        tickers = self.ticker_provider.get_constituents()
        return tickers if tickers else SP500_SAMPLE
    
    def save_analysis_data(self, df: pd.DataFrame, period: str, data_through: Optional[datetime] = None) -> dict:
        """Persist analysis results and metadata for a specific period."""
        self.ensure_data_directory()
        analysis_ts = datetime.now()
        df = df.copy()
        df['analysis_timestamp'] = analysis_ts
        df['analysis_period'] = period
        if data_through is not None:
            df['data_through'] = data_through
        df.to_parquet(self.analysis_file_path(period), index=False)
        metadata = {
            'last_updated': analysis_ts,
            'period': period,
            'num_stocks': len(df),
            'version': '1.0',
            'data_through': data_through
        }
        pd.DataFrame([metadata]).to_parquet(self.metadata_file_path(period), index=False)
        return metadata
    
    def format_metadata_for_response(self, metadata: Optional[dict]) -> Optional[dict]:
        if metadata is None:
            return None
        formatted = {}
        for key, value in metadata.items():
            if isinstance(value, (pd.Timestamp, datetime)):
                formatted[key] = value.isoformat()
            else:
                formatted[key] = value
        return formatted
    
    def load_analysis_data(self, period: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
        """Load cached analysis for a given period (or legacy fallback)."""
        data_path = self.analysis_file_path(period)
        meta_path = self.metadata_file_path(period)
        if not data_path.exists() or not meta_path.exists():
            if period is not None and ANALYSIS_FILE.exists() and METADATA_FILE.exists():
                try:
                    legacy_meta = pd.read_parquet(METADATA_FILE).iloc[0].to_dict()
                    if legacy_meta.get('period') == period:
                        df = pd.read_parquet(ANALYSIS_FILE)
                        df = self.clean_analysis_dataframe(standardize_analysis_columns(df))
                        return df, legacy_meta
                except Exception:
                    return None, None
            return None, None
        try:
            df = pd.read_parquet(data_path)
            df = self.clean_analysis_dataframe(standardize_analysis_columns(df))
            metadata = pd.read_parquet(meta_path).iloc[0].to_dict()
            return df, metadata
        except Exception as exc:
            logger.error(f"Error loading analysis data for {period}: {exc}")
            return None, None
    
    def get_cached_analysis(self, period: str) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
        return self.load_analysis_data(period)
    
    def is_data_stale(self, metadata: Optional[dict], max_age_hours: int = 24) -> bool:
        if not metadata:
            return True
        last_updated = metadata.get('last_updated')
        if isinstance(last_updated, str):
            last_updated = pd.to_datetime(last_updated)
        if not last_updated:
            return True
        return (datetime.now() - last_updated) > timedelta(hours=max_age_hours)
    
    def compute_sp500_analysis(self, tickers: List[str], period: str = "1y") -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
        """Replicates the Streamlit app's analyzer logic."""
        results: List[Dict[str, Any]] = []
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
                sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
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
            except Exception as exc:
                logger.warning(f"Failed to analyze {ticker}: {exc}")
                continue
            time.sleep(0.2)
        
        if not results:
            return pd.DataFrame(), latest_data_timestamp
        df = pd.DataFrame(results)
        
        def safe_normalize(series: pd.Series, inverse: bool = False) -> pd.Series:
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
        df = df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        return df, latest_data_timestamp
    
    def analyze_sp500_stocks(self, tickers: Optional[List[str]] = None, period: str = "1y", force_refresh: bool = False) -> Tuple[pd.DataFrame, dict]:
        """Return cached analysis generated by the offline analysis sync service."""
        tickers = tickers or self.get_sp500_universe()
        cached_df, metadata = self.get_cached_analysis(period)
        if cached_df is None or metadata is None:
            raise HTTPException(
                status_code=503,
                detail="Analysis cache missing. Run 'run-analysis-sync.sh' after syncing prices.",
            )
        if self.is_data_stale(metadata, max_age_hours=ANALYSIS_CACHE_MAX_AGE_HOURS):
            raise HTTPException(
                status_code=503,
                detail="Analysis cache is stale. Run 'run-analysis-sync.sh' to refresh.",
            )
        if force_refresh:
            logger.info("Force refresh requested, but analysis is precomputed. Returning cached data.")
        if tickers:
            df = cached_df[cached_df['Ticker'].isin(set(tickers))]
            if df.empty:
                raise HTTPException(status_code=404, detail="Requested tickers not found in cached analysis.")
        else:
            df = cached_df
        return df, metadata
    
    def get_cache_info(self) -> CacheInfo:
        """Aggregate cache metadata for all configured periods."""
        period_details: Dict[str, AnalysisMetadata] = {}
        latest_meta: Optional[AnalysisMetadata] = None
        
        for period in ANALYSIS_PERIODS:
            _, metadata = self.get_cached_analysis(period)
            if not metadata:
                continue
            last_updated = metadata.get('last_updated')
            if isinstance(last_updated, str):
                last_updated = pd.to_datetime(last_updated)
            data_through = metadata.get('data_through')
            if isinstance(data_through, str):
                data_through = pd.to_datetime(data_through)
            meta_model = AnalysisMetadata(
                last_updated=last_updated,
                period=period,
                num_stocks=metadata.get('num_stocks', 0),
                version=metadata.get('version', '1.0'),
                data_through=data_through
            )
            period_details[period] = meta_model
            if latest_meta is None or meta_model.last_updated > latest_meta.last_updated:
                latest_meta = meta_model
        
        if not period_details:
            return CacheInfo(has_cache=False, is_stale=True)
        
        latest_age = (datetime.now() - latest_meta.last_updated).total_seconds() / 3600 if latest_meta else None
        is_stale = self.is_data_stale({'last_updated': latest_meta.last_updated}) if latest_meta else True
        
        return CacheInfo(
            has_cache=True,
            last_updated=latest_meta.last_updated if latest_meta else None,
            age_hours=latest_age,
            is_stale=is_stale,
            period=latest_meta.period if latest_meta else None,
            num_stocks=latest_meta.num_stocks if latest_meta else None,
            periods={k: v for k, v in period_details.items()}
        )
    
    def clear_cache(self, period: Optional[str] = None) -> int:
        """Delete cached analysis artifacts for a specific period or all periods."""
        targets: List[Path] = []
        if period:
            targets.extend([self.analysis_file_path(period), self.metadata_file_path(period)])
        else:
            # All period-specific files plus legacy
            for per in ANALYSIS_PERIODS:
                targets.extend([self.analysis_file_path(per), self.metadata_file_path(per)])
            targets.extend([ANALYSIS_FILE, METADATA_FILE])
        removed = 0
        for path in targets:
            if path.exists():
                try:
                    path.unlink()
                    removed += 1
                except Exception as exc:
                    logger.warning(f"Unable to delete {path}: {exc}")
        return removed
    
    def load_prices(self, tickers: List[str], period: str, interval: str) -> pd.DataFrame:
        """Load stock prices, preferring the asynchronous parquet cache."""
        tickers = tickers or self.get_sp500_universe()
        cached_df, metadata = self.price_cache.load_prices(
            tickers,
            period,
            interval,
            max_age_hours=PRICE_CACHE_MAX_AGE_HOURS,
        )
        if cached_df is not None and not cached_df.empty:
            logger.info(
                "Serving %d tickers from price cache (%s/%s, synced %s)",
                len(cached_df.columns),
                period,
                interval,
                metadata.last_synced.isoformat() if metadata else "unknown",
            )
            return cached_df

        logger.info(
            "Price cache miss for %s/%s; downloading %d tickers from Yahoo Finance",
            period,
            interval,
            len(tickers),
        )
        prices_df = self.price_loader.fetch_prices(tickers, period, interval)
        if prices_df.empty:
            logger.warning("Yahoo download returned no prices; falling back to mock data")
            dates = pd.date_range(end=pd.Timestamp.now().date(), periods=250, freq="D")
            mock_data = {}
            for ticker in tickers[:5]:
                np.random.seed(hash(ticker) % 2**32)
                returns = np.random.normal(0.0005, 0.02, 250)
                prices = 100 * np.exp(np.cumsum(returns))
                mock_data[ticker] = prices
            return pd.DataFrame(mock_data, index=dates)
        prices_df = PriceCacheManager.normalize_frame(prices_df)
        return prices_df
    

# Initialize FastAPI app
app = FastAPI(title="Data Service", description="Stock data download and analysis service")
data_service = DataService()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "data_service"}

@app.post("/stock-data")
async def get_stock_data(request: StockDataRequest):
    """Get stock price data"""
    try:
        prices = data_service.load_prices(request.tickers, request.period, request.interval)
        # Return data in format expected by frontend: dict with date strings as keys
        return ServiceResponse(
            success=True,
            data=prices.to_dict('index')
        )
    except Exception as e:
        logger.error(f"Error in get_stock_data: {e}")
        return ServiceResponse(
            success=False,
            error=str(e)
        )

@app.post("/sp500-analysis")
async def get_sp500_analysis(request: StockAnalysisRequest):
    """Get S&P 500 stock analysis"""
    try:
        tickers = request.tickers or data_service.get_sp500_universe()
        analysis_df, metadata = data_service.analyze_sp500_stocks(
            tickers, 
            request.period, 
            request.force_refresh
        )
        return ServiceResponse(
            success=True,
            data={
                "records": analysis_df.to_dict('records'),
                "metadata": data_service.format_metadata_for_response(metadata)
            }
        )
    except Exception as e:
        logger.error(f"Error in get_sp500_analysis: {e}")
        return ServiceResponse(
            success=False,
            error=str(e)
        )

@app.get("/cache-info")
async def get_cache_info():
    """Get cache status information"""
    try:
        cache_info = data_service.get_cache_info()
        return ServiceResponse(
            success=True,
            data=cache_info.dict()
        )
    except Exception as e:
        logger.error(f"Error in get_cache_info: {e}")
        return ServiceResponse(
            success=False,
            error=str(e)
        )

@app.delete("/cache")
async def clear_cache(period: Optional[str] = Query(default=None, description="Period to clear (e.g., 1y). Omit to clear all.")):
    """Clear cached analysis data."""
    removed = data_service.clear_cache(period)
    return ServiceResponse(
        success=True,
        data={"files_removed": removed, "period": period}
    )

@app.get("/sp500-tickers")
async def get_sp500_tickers():
    """Get list of S&P 500 sample tickers"""
    tickers = data_service.get_sp500_universe()
    return ServiceResponse(success=True, data=tickers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

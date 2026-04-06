# services/data_service.py
# -------------------------------
# Data Service - Download & Persistence Layer
# Handles yfinance API calls, data caching, and S&P 500 analysis
# -------------------------------

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging
import subprocess
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.analysis_engine import standardize_analysis_columns
from shared.settings import settings
from shared.models import (
    StockDataRequest, StockAnalysisRequest, ServiceResponse,
    AnalysisMetadata, CacheInfo
)
from shared.ticker_provider import WikipediaTickerProvider, DEFAULT_SP500_SAMPLE
from shared.price_loader import (
    MASTER_INTERVAL,
    MASTER_PERIOD,
    PriceCacheManager,
    YahooPriceLoader,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = settings.sp500_data_dir
ANALYSIS_FILE = DATA_DIR / "sp500_analysis.parquet"  # legacy single cache
METADATA_FILE = DATA_DIR / "metadata.parquet"
HIDDEN_ANALYSIS_COLUMNS = ['analysis_timestamp', 'analysis_period', 'data_through']
ANALYSIS_PERIODS = ["1y", "2y", "3y", "5y"]
PRICE_CACHE_MAX_AGE_HOURS = (7 * 24) + 4  # weekly sync + small grace period
ANALYSIS_CACHE_MAX_AGE_HOURS = (7 * 24) + 4

# S&P 500 Top 100 stocks
SP500_SAMPLE = list(DEFAULT_SP500_SAMPLE)

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
        if interval != MASTER_INTERVAL:
            logger.warning("Unsupported interval %s; falling back to master interval %s", interval, MASTER_INTERVAL)
            interval = MASTER_INTERVAL
        cached_df, metadata = self.price_cache.load_prices(
            tickers,
            MASTER_PERIOD,
            interval,
            max_age_hours=PRICE_CACHE_MAX_AGE_HOURS,
        )
        if cached_df is not None and not cached_df.empty:
            trimmed = self.price_cache.trim_history(cached_df, period)
            logger.info(
                "Serving %d tickers from price cache (master %s/%s → %s slice, synced %s)",
                len(trimmed.columns),
                MASTER_PERIOD,
                interval,
                period,
                metadata.last_synced.isoformat() if metadata else "unknown",
            )
            return trimmed

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
        prices_df = self.price_cache.normalize_frame(prices_df)
        return prices_df
    

def _trigger_analysis_sync_if_needed(ds: "DataService") -> None:
    """Spawn analysis_sync_service in background if cache is missing or stale."""
    cache_info = ds.get_cache_info()
    if cache_info.has_cache and not cache_info.is_stale:
        logger.info("Analysis cache is present and fresh — skipping auto-seed.")
        return

    # Check price cache exists before attempting sync
    master_df, _ = ds.price_cache.load_full(MASTER_PERIOD, MASTER_INTERVAL)
    if master_df is None or master_df.empty:
        reason = "stale" if cache_info.has_cache else "missing"
        logger.warning(
            "Analysis cache is %s but master price cache is not available. "
            "Run './run-price-sync.sh' first, then './run-analysis-sync.sh'.",
            reason,
        )
        return

    reason = "stale" if cache_info.has_cache else "missing"
    logger.info("Analysis cache is %s — spawning analysis_sync_service in background.", reason)
    sync_script = Path(__file__).resolve().parent / "analysis_sync_service.py"
    subprocess.Popen(
        [sys.executable, str(sync_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _trigger_analysis_sync_if_needed(data_service)
    yield


# Initialize FastAPI app
data_service = DataService()
app = FastAPI(
    title="Data Service",
    description="Stock data download and analysis service",
    lifespan=lifespan,
)

@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "service": "data_service"}

@app.post("/stock-data")
async def get_stock_data(request: StockDataRequest) -> ServiceResponse:
    """Get stock price data"""
    try:
        prices = data_service.load_prices(request.tickers, request.period, request.interval)
        # Return data in format expected by frontend: dict with date strings as keys
        return ServiceResponse(
            success=True,
            data=prices.to_dict('index')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_stock_data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/sp500-analysis")
async def get_sp500_analysis(request: StockAnalysisRequest) -> ServiceResponse:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_sp500_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/cache-info")
async def get_cache_info() -> ServiceResponse:
    """Get cache status information"""
    try:
        cache_info = data_service.get_cache_info()
        return ServiceResponse(
            success=True,
            data=cache_info.model_dump()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_cache_info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.delete("/cache")
async def clear_cache(period: Optional[str] = Query(default=None, description="Period to clear (e.g., 1y). Omit to clear all.")) -> ServiceResponse:
    """Clear cached analysis data."""
    removed = data_service.clear_cache(period)
    return ServiceResponse(
        success=True,
        data={"files_removed": removed, "period": period}
    )

@app.get("/sp500-tickers")
async def get_sp500_tickers() -> ServiceResponse:
    """Get list of S&P 500 sample tickers"""
    tickers = data_service.get_sp500_universe()
    return ServiceResponse(success=True, data=tickers)

@app.get("/price-cache-info")
async def get_price_cache_info() -> ServiceResponse:
    """Get per-period price cache metadata (last synced, tickers, rows, data_through)."""
    try:
        summary = data_service.price_cache.cache_summary()
        serializable = {}
        for key, meta in summary.items():
            serializable[key] = {
                k: v.isoformat() if hasattr(v, "isoformat") else v
                for k, v in meta.items()
            }
        return ServiceResponse(success=True, data=serializable)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_price_cache_info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/ticker-validation")
async def get_ticker_validation() -> ServiceResponse:
    """Return the most recent ticker validation result from sp500_data/validation/."""
    try:
        import json
        validation_dir = DATA_DIR / "validation"
        files = sorted(validation_dir.glob("ticker_validation_*.json")) if validation_dir.exists() else []
        if not files:
            return ServiceResponse(success=True, data=None)
        with open(files[-1]) as f:
            return ServiceResponse(success=True, data=json.load(f))
    except Exception as e:
        logger.error(f"Error in get_ticker_validation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/sync-report")
async def get_sync_report() -> ServiceResponse:
    """Return the latest price sync report written by price_sync_service."""
    try:
        import json
        report_path = DATA_DIR / "price_cache" / "sync_report.json"
        if not report_path.exists():
            return ServiceResponse(success=True, data=None)
        with open(report_path) as f:
            return ServiceResponse(success=True, data=json.load(f))
    except Exception as e:
        logger.error(f"Error in get_sync_report: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/ticker-changes")
async def get_ticker_changes() -> ServiceResponse:
    """Derive a chronological change log from all validation snapshot files."""
    try:
        import json
        validation_dir = DATA_DIR / "validation"
        files = sorted(validation_dir.glob("ticker_validation_*.json"))
        events: dict = {}  # (ticker, action) -> earliest event

        for path in files:
            try:
                with open(path) as f:
                    snap = json.load(f)
            except Exception:
                continue
            ts = snap.get("timestamp")
            for ticker in snap.get("new_in_remote") or []:
                key = (ticker, "added")
                if key not in events:
                    events[key] = {"timestamp": ts, "ticker": ticker,
                                   "action": "added", "source_file": path.name}
            for ticker in snap.get("missing_from_remote") or []:
                key = (ticker, "removed")
                if key not in events:
                    events[key] = {"timestamp": ts, "ticker": ticker,
                                   "action": "removed", "source_file": path.name}

        log = sorted(events.values(), key=lambda e: e["timestamp"] or "", reverse=True)

        latest_pending: dict = {}
        if files:
            try:
                with open(files[-1]) as f:
                    latest = json.load(f)
                latest_pending = {
                    "added": latest.get("new_in_remote") or [],
                    "removed": latest.get("missing_from_remote") or [],
                    "timestamp": latest.get("timestamp"),
                    "match": latest.get("match", True),
                }
            except Exception:
                pass

        return ServiceResponse(success=True, data={
            "changes": log,
            "total_runs": len(files),
            "pending": latest_pending,
        })
    except Exception as e:
        logger.error(f"Error in get_ticker_changes: {e}")
        return ServiceResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

# services/data_service.py
# -------------------------------
# Data Service - Download & Persistence Layer
# Handles yfinance API calls, data caching, and S&P 500 analysis
# -------------------------------

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, HTTPException
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import (
    StockDataRequest, StockAnalysisRequest, ServiceResponse,
    StockMetrics, AnalysisMetadata, CacheInfo
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("sp500_data")
ANALYSIS_FILE = DATA_DIR / "sp500_analysis.parquet"
METADATA_FILE = DATA_DIR / "metadata.parquet"

# S&P 500 Top 100 stocks
SP500_SAMPLE = [
    # Top 50 (Mega caps)
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","UNH","XOM",
    "JPM","JNJ","V","PG","HD","CVX","MA","ABBV","PFE","KO",
    "AVGO","COST","PEP","TMO","WMT","MRK","DIS","ADBE","NFLX","CRM",
    "BAC","ACN","LLY","ORCL","WFC","VZ","CMCSA","CSCO","ABT","DHR",
    "NKE","TXN","PM","BMY","UNP","QCOM","RTX","HON","INTC","T",
    
    # Next 50 (Large caps)
    "LOW","UPS","SPGI","CAT","GS","MS","MDT","AXP","BLK","AMGN",
    "DE","BKNG","SYK","TJX","GILD","ADP","MMC","CI","TMUS","ZTS",
    "CME","SO","CSX","PLD","ITW","CB","DUK","AON","CL","BSX",
    "FCX","EMR","NSC","SHW","MPC","PSX","GD","NOC","TGT","USB",
    "KMI","F","SPG","ADSK","ROP","ICE","MCK","EOG","APD","COP"
]

class DataService:
    """Data Service for handling stock data download and persistence"""
    
    def __init__(self):
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        DATA_DIR.mkdir(exist_ok=True)
    
    def save_analysis_data(self, df: pd.DataFrame, period: str):
        """Save analysis results to parquet with metadata"""
        try:
            # Save main data
            df.to_parquet(ANALYSIS_FILE)
            
            # Save metadata
            metadata = pd.DataFrame([{
                'last_updated': datetime.now(),
                'period': period,
                'num_stocks': len(df),
                'version': '1.0'
            }])
            metadata.to_parquet(METADATA_FILE)
            
            logger.info(f"Saved analysis data for {len(df)} stocks")
        except Exception as e:
            logger.error(f"Error saving analysis data: {e}")
            raise
    
    def load_analysis_data(self) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
        """Load cached analysis data and metadata"""
        try:
            if not ANALYSIS_FILE.exists() or not METADATA_FILE.exists():
                return None, None
            
            df = pd.read_parquet(ANALYSIS_FILE)
            metadata_df = pd.read_parquet(METADATA_FILE)
            metadata = metadata_df.iloc[0].to_dict()
            
            return df, metadata
        except Exception as e:
            logger.error(f"Error loading analysis data: {e}")
            return None, None
    
    def is_data_stale(self, metadata: dict, max_age_hours: int = 24) -> bool:
        """Check if cached data is stale"""
        if not metadata:
            return True
        
        last_updated = metadata.get('last_updated')
        if isinstance(last_updated, str):
            last_updated = pd.to_datetime(last_updated)
        
        if not last_updated:
            return True
        
        age = datetime.now() - last_updated
        return age > timedelta(hours=max_age_hours)
    
    def get_cache_info(self) -> CacheInfo:
        """Get cache status information"""
        try:
            cached_df, metadata = self.load_analysis_data()
            
            if cached_df is None or metadata is None:
                return CacheInfo(has_cache=False, is_stale=True)
            
            last_updated = metadata.get('last_updated')
            if isinstance(last_updated, str):
                last_updated = pd.to_datetime(last_updated)
            
            age = datetime.now() - last_updated
            age_hours = age.total_seconds() / 3600
            is_stale = self.is_data_stale(metadata)
            
            return CacheInfo(
                has_cache=True,
                last_updated=last_updated,
                age_hours=age_hours,
                is_stale=is_stale,
                period=metadata.get('period'),
                num_stocks=metadata.get('num_stocks')
            )
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return CacheInfo(has_cache=False, is_stale=True)
    
    def load_prices(self, tickers: List[str], period: str, interval: str) -> pd.DataFrame:
        """Load stock prices using yfinance with error handling"""
        successful_data = {}
        failed_tickers = []
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            
            try:
                batch_str = " ".join(batch)
                data = yf.download(
                    batch_str,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    prepost=True,
                    threads=True,
                    progress=False
                )
                
                if len(batch) == 1:
                    ticker = batch[0]
                    if not data.empty and 'Close' in data.columns:
                        successful_data[ticker] = data['Close']
                    else:
                        failed_tickers.append(ticker)
                else:
                    for ticker in batch:
                        try:
                            ticker_data = data[ticker]
                            if not ticker_data.empty and 'Close' in ticker_data.columns:
                                successful_data[ticker] = ticker_data['Close']
                            else:
                                failed_tickers.append(ticker)
                        except (KeyError, IndexError):
                            failed_tickers.append(ticker)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error downloading batch {batch}: {e}")
                failed_tickers.extend(batch)
        
        if not successful_data:
            raise ValueError("No valid stock data downloaded")
        
        prices_df = pd.DataFrame(successful_data)
        prices_df.dropna(inplace=True)
        
        if failed_tickers:
            logger.warning(f"Failed to download: {failed_tickers}")
        
        return prices_df
    
    def analyze_sp500_stocks(self, tickers: List[str], period: str = "1y", force_refresh: bool = False) -> pd.DataFrame:
        """Analyze S&P 500 stocks with caching"""
        
        # Check cache first
        if not force_refresh:
            cached_df, metadata = self.load_analysis_data()
            if cached_df is not None and metadata is not None:
                if not self.is_data_stale(metadata) and metadata.get('period') == period:
                    logger.info("Using cached analysis data")
                    return cached_df
        
        logger.info(f"Analyzing {len(tickers)} stocks for period {period}")
        
        # Download fresh data
        try:
            prices = self.load_prices(tickers, period, "1d")
        except Exception as e:
            logger.error(f"Error loading prices: {e}")
            # Return cached data if available
            cached_df, _ = self.load_analysis_data()
            if cached_df is not None:
                return cached_df
            raise HTTPException(status_code=500, detail=f"Failed to load stock data: {e}")
        
        results = []
        
        for ticker in prices.columns:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                price_series = prices[ticker].dropna()
                if len(price_series) < 10:
                    continue
                
                # Calculate metrics
                returns = price_series.pct_change().dropna()
                total_return = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
                volatility = returns.std() * np.sqrt(252) * 100
                
                # Sharpe ratio (assuming 2% risk-free rate)
                excess_returns = returns - (0.02 / 252)
                sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min()) * 100
                
                # Composite score (normalized)
                return_score = min(max(total_return / 50, -1), 2)  # Normalize to -1 to 2
                sharpe_score = min(max(sharpe_ratio / 2, -1), 2)   # Normalize to -1 to 2
                volatility_score = max(min(2 - volatility / 25, 2), -1)  # Lower volatility = higher score
                drawdown_score = max(min(2 - max_drawdown / 25, 2), -1)  # Lower drawdown = higher score
                
                composite_score = (return_score * 0.3 + sharpe_score * 0.3 + 
                                 volatility_score * 0.2 + drawdown_score * 0.2)
                
                results.append({
                    'ticker': ticker,
                    'total_return_pct': round(total_return, 2),
                    'volatility_pct': round(volatility, 2),
                    'sharpe_ratio': round(sharpe_ratio, 3),
                    'max_drawdown_pct': round(max_drawdown, 2),
                    'composite_score': round(composite_score, 3),
                    'current_price': round(price_series.iloc[-1], 2),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('forwardPE')
                })
                
                # Rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"Error analyzing {ticker}: {e}")
                continue
        
        if not results:
            raise HTTPException(status_code=500, detail="No stocks could be analyzed")
        
        df = pd.DataFrame(results)
        df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        
        # Save to cache
        try:
            self.save_analysis_data(df, period)
        except Exception as e:
            logger.warning(f"Could not save analysis data: {e}")
        
        return df

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
        return ServiceResponse(
            success=True,
            data=prices.to_dict('records')
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
        analysis_df = data_service.analyze_sp500_stocks(
            request.tickers, 
            request.period, 
            request.force_refresh
        )
        return ServiceResponse(
            success=True,
            data=analysis_df.to_dict('records')
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

@app.get("/sp500-tickers")
async def get_sp500_tickers():
    """Get list of S&P 500 sample tickers"""
    return ServiceResponse(
        success=True,
        data=SP500_SAMPLE
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
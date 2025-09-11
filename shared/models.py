# shared/models.py
# -------------------------------
# Shared Data Models for Microservices
# -------------------------------

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import pandas as pd

class StockDataRequest(BaseModel):
    """Request model for stock data"""
    tickers: List[str]
    period: str = "1y"
    interval: str = "1d"
    force_refresh: bool = False

class StockAnalysisRequest(BaseModel):
    """Request model for S&P 500 analysis"""
    tickers: List[str]
    period: str = "1y"
    force_refresh: bool = False

class PortfolioOptimizationRequest(BaseModel):
    """Request model for portfolio optimization"""
    tickers: List[str]
    period: str = "1y"
    interval: str = "1d"
    investment_amount: float
    risk_aversion: float = 1.0
    min_weight_threshold: float = 0.0025
    min_holdings: int = 3

class StockMetrics(BaseModel):
    """Individual stock metrics"""
    ticker: str
    total_return_pct: float
    volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    composite_score: float
    current_price: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    
class AnalysisMetadata(BaseModel):
    """Metadata for cached analysis"""
    last_updated: datetime
    period: str
    num_stocks: int
    version: str

class CacheInfo(BaseModel):
    """Cache status information"""
    has_cache: bool
    last_updated: Optional[datetime] = None
    age_hours: Optional[float] = None
    is_stale: bool
    period: Optional[str] = None
    num_stocks: Optional[int] = None

class OptimizationResult(BaseModel):
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    allocation: Dict[str, int]
    leftover_cash: float

class ServiceResponse(BaseModel):
    """Generic service response wrapper"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = datetime.now()
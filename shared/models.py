# shared/models.py
# -------------------------------
# Shared Data Models for Microservices
# -------------------------------

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime

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
    prices_data: Dict[str, Dict[str, float]]
    investment_amount: float
    objective: str = "Max Sharpe"
    risk_free: float = 4.0
    target_return: float = 10.0
    max_weight: float = 30.0
    l2_reg: float = 5.0
    min_weight_threshold: float = 0.25  # percent
    min_holdings: int = 3

class AnalysisMetadata(BaseModel):
    """Metadata for cached analysis"""
    last_updated: datetime
    period: str
    num_stocks: int
    version: str
    data_through: Optional[datetime] = None

class CacheInfo(BaseModel):
    """Cache status information"""
    has_cache: bool
    last_updated: Optional[datetime] = None
    age_hours: Optional[float] = None
    is_stale: bool
    period: Optional[str] = None
    num_stocks: Optional[int] = None
    periods: Optional[Dict[str, AnalysisMetadata]] = None

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
class StockMetrics(BaseModel):
    """Individual stock metrics aligned with analyzer output"""
    ticker: str
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    recent_3m_return: float
    composite_score: float
    current_price: float
    additional_fields: Optional[Dict[str, Any]] = None


class ForecastRequest(BaseModel):
    """Request model for return forecasts"""
    tickers: List[str]
    prices_data: Dict[str, Dict[str, float]]
    top_n: int = 10
    n_simulations: int = 1000
    risk_free_annual: float = 0.04
    spy_ticker: str = "SPY"


class TickerForecast(BaseModel):
    """Forecast results for a single ticker"""
    ticker: str
    current_price: float
    # Monte Carlo
    mc_return_1y: Optional[float] = None
    mc_return_2y: Optional[float] = None
    mc_p10_1y: Optional[float] = None
    mc_p90_1y: Optional[float] = None
    mc_p10_2y: Optional[float] = None
    mc_p90_2y: Optional[float] = None
    mc_paths_sample: Optional[List[List[float]]] = None  # 10 paths x 504 days
    # CAPM
    beta: Optional[float] = None
    capm_return_1y: Optional[float] = None
    capm_return_2y: Optional[float] = None
    # Trend
    trend_return_1y: Optional[float] = None
    trend_return_2y: Optional[float] = None
    # Ensemble
    ensemble_return_1y: Optional[float] = None
    ensemble_return_2y: Optional[float] = None


class ForecastResult(BaseModel):
    """Full forecast response"""
    forecasts: List[TickerForecast]
    n_simulations: int
    risk_free_annual: float
    generated_at: datetime = datetime.now()

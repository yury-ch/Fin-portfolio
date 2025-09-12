# services/calculation_service.py
# -------------------------------
# Calculation Service - Optimization Layer
# Handles portfolio optimization, statistical calculations, and risk computations
# -------------------------------

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from fastapi import FastAPI, HTTPException
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import (
    PortfolioOptimizationRequest, ServiceResponse, OptimizationResult
)

try:
    from pypfopt import (
        EfficientFrontier,
        risk_models,
        expected_returns,
        objective_functions,
        DiscreteAllocation,
        get_latest_prices,
    )
except ImportError:
    raise ImportError("PyPortfolioOpt is required for the calculation service")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalculationService:
    """Calculation Service for portfolio optimization and statistics"""
    
    def __init__(self):
        pass
    
    def compute_stats(self, prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Compute expected returns and covariance matrix from price data"""
        try:
            # Calculate expected returns using mean historical return
            mu = expected_returns.mean_historical_return(prices, frequency=252)
            
            # Calculate covariance matrix
            S = risk_models.sample_cov(prices, frequency=252)
            
            return mu, S
            
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            raise
    
    def enforce_min_holdings(self, weights: pd.Series, min_n: int, prune_below_pct: float) -> pd.Series:
        """Enforce minimum number of holdings by pruning small weights"""
        try:
            # Remove tiny weights
            weights = weights[weights >= prune_below_pct]
            
            # If we have fewer than min_n holdings, relax the pruning threshold
            if len(weights) < min_n and prune_below_pct > 0:
                logger.info(f"Only {len(weights)} holdings after pruning. Relaxing threshold to get {min_n} holdings.")
                # Sort all weights and take top min_n
                all_weights = weights.sort_values(ascending=False)
                if len(all_weights) >= min_n:
                    weights = all_weights.head(min_n)
                else:
                    weights = all_weights
            
            # Renormalize
            weights = weights / weights.sum()
            
            return weights
            
        except Exception as e:
            logger.error(f"Error enforcing min holdings: {e}")
            raise
    
    def optimize_portfolio(
        self,
        tickers: List[str],
        prices: pd.DataFrame,
        investment_amount: float,
        risk_aversion: float = 1.0,
        min_weight_threshold: float = 0.0025,
        min_holdings: int = 3
    ) -> OptimizationResult:
        """Optimize portfolio using PyPortfolioOpt"""
        
        try:
            # Ensure we have valid price data
            if prices.empty:
                raise ValueError("No price data provided")
            
            # Filter tickers to only those with data
            available_tickers = [t for t in tickers if t in prices.columns]
            if not available_tickers:
                raise ValueError("None of the requested tickers have price data")
            
            prices_filtered = prices[available_tickers].dropna()
            
            if len(prices_filtered) < 10:
                raise ValueError("Insufficient price data for optimization")
            
            # Compute expected returns and covariance
            mu, S = self.compute_stats(prices_filtered)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S)
            
            # Add regularization for numerical stability
            ef.add_constraint(lambda w: w >= 0.001)  # Minimum 0.1% weight
            
            # Optimize for maximum quadratic utility (risk-adjusted returns)
            weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
            
            # Clean tiny weights and enforce minimum holdings
            weights_series = pd.Series(weights)
            weights_series = self.enforce_min_holdings(weights_series, min_holdings, min_weight_threshold)
            
            # Get portfolio performance
            ef_clean = EfficientFrontier(mu, S)
            ef_clean.set_weights(weights_series.to_dict())
            performance = ef_clean.portfolio_performance(verbose=False)
            
            expected_annual_return, annual_volatility, sharpe_ratio = performance
            
            # Discrete allocation
            latest_prices = get_latest_prices(prices_filtered)
            da = DiscreteAllocation(weights_series.to_dict(), latest_prices, total_portfolio_value=investment_amount)
            allocation, leftover = da.greedy_portfolio()
            
            return OptimizationResult(
                weights={k: float(v) for k, v in weights_series.items()},
                expected_annual_return=float(expected_annual_return),
                annual_volatility=float(annual_volatility),
                sharpe_ratio=float(sharpe_ratio),
                allocation=allocation,
                leftover_cash=float(leftover)
            )
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    def calculate_portfolio_metrics(self, prices: pd.DataFrame, weights: Dict[str, float]) -> Dict:
        """Calculate various portfolio metrics given prices and weights"""
        try:
            # Filter prices to only include tickers in weights
            relevant_prices = prices[[ticker for ticker in weights.keys() if ticker in prices.columns]]
            
            if relevant_prices.empty:
                raise ValueError("No price data available for the given weights")
            
            # Calculate portfolio returns
            returns = relevant_prices.pct_change().dropna()
            weights_series = pd.Series(weights)
            portfolio_returns = (returns * weights_series).sum(axis=1)
            
            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
            
            # Max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'portfolio_value': sum(weights.values())
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            raise

# Initialize FastAPI app
app = FastAPI(title="Calculation Service", description="Portfolio optimization and statistical calculations")
calculation_service = CalculationService()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "calculation_service"}

@app.post("/optimize-portfolio")
async def optimize_portfolio(payload: dict):
    """Optimize portfolio allocation"""
    try:
        # Extract parameters from payload
        tickers = payload.get('tickers', [])
        prices_data = payload.get('prices_data', {})
        investment_amount = payload.get('investment_amount', 100000)
        risk_aversion = payload.get('risk_aversion', 1.0)
        min_weight_threshold = payload.get('min_weight_threshold', 0.0025)
        min_holdings = payload.get('min_holdings', 3)
        
        if not tickers or not prices_data:
            raise HTTPException(status_code=400, detail="Missing required parameters: tickers and prices_data")
        
        # Convert prices data back to DataFrame
        prices_df = pd.DataFrame.from_dict(prices_data, orient='index')
        prices_df.index = pd.to_datetime(prices_df.index)
        
        # Filter to only requested tickers
        available_tickers = [t for t in tickers if t in prices_df.columns]
        if not available_tickers:
            raise HTTPException(status_code=400, detail="None of the requested tickers have price data")
        
        prices_df = prices_df[available_tickers]
        
        # Perform optimization
        result = calculation_service.optimize_portfolio(
            available_tickers, 
            prices_df, 
            investment_amount, 
            risk_aversion, 
            min_weight_threshold, 
            min_holdings
        )
        
        return ServiceResponse(
            success=True,
            data={
                'weights': result.weights,
                'expected_annual_return': result.expected_annual_return,
                'annual_volatility': result.annual_volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'allocation': result.allocation,
                'leftover_cash': result.leftover_cash
            }
        )
        
    except Exception as e:
        logger.error(f"Error in optimize_portfolio: {e}")
        return ServiceResponse(
            success=False,
            error=str(e)
        )

@app.post("/portfolio-metrics")
async def calculate_metrics(prices_data: dict, weights: Dict[str, float]):
    """Calculate portfolio metrics given prices and weights"""
    try:
        # Convert prices data back to DataFrame
        prices_df = pd.DataFrame(prices_data)
        
        metrics = calculation_service.calculate_portfolio_metrics(prices_df, weights)
        
        return ServiceResponse(
            success=True,
            data=metrics
        )
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}")
        return ServiceResponse(
            success=False,
            error=str(e)
        )

@app.post("/compute-stats")
async def compute_statistics(prices_data: dict):
    """Compute expected returns and covariance matrix"""
    try:
        # Convert prices data back to DataFrame  
        prices_df = pd.DataFrame(prices_data)
        
        mu, S = calculation_service.compute_stats(prices_df)
        
        return ServiceResponse(
            success=True,
            data={
                'expected_returns': mu.to_dict(),
                'covariance_matrix': S.to_dict()
            }
        )
    except Exception as e:
        logger.error(f"Error in compute_statistics: {e}")
        return ServiceResponse(
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
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
        """Replicate the monolith's pruning logic (threshold expressed in percent)."""
        if weights.empty:
            return weights
        keep = weights[weights >= (prune_below_pct / 100.0)].sort_values(ascending=False)
        if len(keep) >= min_n:
            return keep / keep.sum()
        top = weights.sort_values(ascending=False).head(min_n)
        return top / top.sum()
    
    def optimize_portfolio(
        self,
        prices: pd.DataFrame,
        objective: str,
        risk_free: float,
        target_return: float,
        max_weight: float,
        l2_reg: float,
        min_weight_threshold: float,
        min_holdings: int,
        investment_amount: float,
    ) -> OptimizationResult:
        """Optimize portfolio using the same settings as the monolith app."""
        if prices.empty or len(prices.columns) < 2:
            raise HTTPException(status_code=400, detail="Insufficient price data for optimization")
        
        mu, S = self.compute_stats(prices)
        n_assets = len(prices.columns)
        if n_assets <= 3:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            l2_reg = 0
        else:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight / 100.0))
            if l2_reg and l2_reg > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=l2_reg)
        
        try:
            if objective == "Max Sharpe":
                ef.max_sharpe(risk_free_rate=risk_free / 100.0)
            elif objective == "Min Volatility (target return)":
                ef.efficient_return(target_return=target_return / 100.0)
            else:
                ef.efficient_return(target_return=target_return / 100.0)
        except Exception:
            try:
                ef.min_volatility()
            except Exception as exc:
                logger.error(f"Optimization failure: {exc}")
                raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}")
        
        raw_weights = pd.Series(ef.clean_weights(cutoff=0))
        final_weights = self.enforce_min_holdings(raw_weights, min_holdings, min_weight_threshold)
        if final_weights.empty:
            raise HTTPException(status_code=400, detail="Optimization produced empty weights. Adjust constraints.")
        
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free / 100.0, verbose=False)
        latest_prices = get_latest_prices(prices[final_weights.index])
        da = DiscreteAllocation(final_weights.to_dict(), latest_prices, total_portfolio_value=investment_amount)
        allocation, leftover = da.greedy_portfolio()
        
        return OptimizationResult(
            weights={k: float(v) for k, v in final_weights.items()},
            expected_annual_return=float(ret),
            annual_volatility=float(vol),
            sharpe_ratio=float(sharpe),
            allocation=allocation,
            leftover_cash=float(leftover)
        )
    
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
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize portfolio allocation"""
    try:
        if not request.tickers or not request.prices_data:
            raise HTTPException(status_code=400, detail="Missing required parameters: tickers and prices_data")
        
        prices_df = pd.DataFrame.from_dict(request.prices_data, orient='index')
        prices_df.index = pd.to_datetime(prices_df.index)
        available_tickers = [t for t in request.tickers if t in prices_df.columns]
        if not available_tickers:
            raise HTTPException(status_code=400, detail="None of the requested tickers have price data")
        prices_df = prices_df[available_tickers].dropna()
        
        result = calculation_service.optimize_portfolio(
            prices=prices_df,
            objective=request.objective,
            risk_free=request.risk_free,
            target_return=request.target_return,
            max_weight=request.max_weight,
            l2_reg=request.l2_reg,
            min_weight_threshold=request.min_weight_threshold,
            min_holdings=request.min_holdings,
            investment_amount=request.investment_amount
        )
        
        return ServiceResponse(
            success=True,
            data=result.dict()
        )
        
    except HTTPException as http_exc:
        logger.error(f"Optimization request error: {http_exc.detail}")
        raise http_exc
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

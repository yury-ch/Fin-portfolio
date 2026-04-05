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

from pydantic import BaseModel
from shared.models import (
    PortfolioOptimizationRequest, ServiceResponse, OptimizationResult,
    ForecastRequest, ForecastResult, TickerForecast,
)
from shared.config import DEFAULT_RISK_FREE_RATE

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
            # Geometric (log-return) mean — consistent with analysis_engine screening
            mu = expected_returns.mean_historical_return(prices, compounding=True, frequency=252)

            # Clip to [-90%, +150%] — prevents corporate-action price spikes (e.g. SNDK
            # re-listing) or extreme single-year outliers from dominating the optimizer.
            mu = mu.clip(lower=-0.90, upper=1.50)

            # Ledoit-Wolf shrinkage — more stable than sample covariance for N > 20
            S = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()

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
        expected_returns_override: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """Optimize portfolio using the same settings as the monolith app."""
        if prices.empty or len(prices.columns) < 2:
            raise HTTPException(status_code=400, detail="Insufficient price data for optimization")

        mu, S = self.compute_stats(prices)

        # Apply forecast overrides: replace historical mu with caller-supplied
        # ensemble return estimates for tickers present in both sources.
        if expected_returns_override:
            override = pd.Series(expected_returns_override)
            common = mu.index.intersection(override.index)
            if not common.empty:
                mu = mu.copy()
                mu[common] = override[common]

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
            sharpe_ratio = (annual_return - DEFAULT_RISK_FREE_RATE) / volatility if volatility > 0 else 0
            
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

class ForecastService:
    """Monte Carlo GBM, CAPM, and log-linear trend forecasts for top-N tickers."""

    _TRADING_DAYS_1Y: int = 252
    _TRADING_DAYS_2Y: int = 504

    # Cap annualised drift at ±50 % — prevents extrapolating a single lucky year
    # into indefinite compounding.  50 % is already extremely bullish for a forecast.
    _MAX_ANNUAL_RETURN: float = 0.50
    _MIN_ANNUAL_RETURN: float = -0.90

    def __init__(self, rng_seed: Optional[int] = None):
        self._rng = np.random.default_rng(rng_seed)

    def forecast(
        self,
        prices: pd.DataFrame,
        tickers: List[str],
        n_simulations: int,
        risk_free_annual: float,
        spy_ticker: str,
    ) -> List[dict]:
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        spy_returns = None
        if spy_ticker in prices.columns:
            spy_series = prices[spy_ticker].dropna()
            if len(spy_series) >= 30:
                spy_returns = spy_series.pct_change().dropna().values

        results = []
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            series = prices[ticker].dropna()
            if len(series) < 63:
                continue

            stock_returns_aligned = None
            spy_returns_aligned = None
            if spy_returns is not None and spy_ticker in prices.columns:
                aligned = prices[[ticker, spy_ticker]].dropna()
                stock_ret = aligned[ticker].pct_change().dropna()
                spy_ret = aligned[spy_ticker].pct_change().dropna()
                stock_returns_aligned = stock_ret.values
                spy_returns_aligned = spy_ret.values

            mc = self._mc_forecast(series, n_simulations)
            cap = self._capm_forecast(
                stock_returns_aligned, spy_returns_aligned, risk_free_annual
            ) if stock_returns_aligned is not None else {}
            trd = self._trend_forecast(series)

            ens_1y = self._ensemble([
                mc.get('mc_return_1y'), cap.get('capm_return_1y'), trd.get('trend_return_1y'),
            ])
            ens_2y = self._ensemble([
                mc.get('mc_return_2y'), cap.get('capm_return_2y'), trd.get('trend_return_2y'),
            ])
            # Clip ensemble to same bounds as individual components — MC median
            # can float slightly above the drift cap due to the volatility term.
            if ens_1y is not None:
                ens_1y = float(np.clip(ens_1y, self._MIN_ANNUAL_RETURN, self._MAX_ANNUAL_RETURN))
            if ens_2y is not None:
                ens_2y = float(np.clip(ens_2y, self._MIN_ANNUAL_RETURN - 0.10, (1 + self._MAX_ANNUAL_RETURN) ** 2 - 1))

            results.append(dict(
                ticker=ticker,
                current_price=float(series.iloc[-1]),
                **mc, **cap, **trd,
                ensemble_return_1y=ens_1y,
                ensemble_return_2y=ens_2y,
            ))

        return results

    def _mc_forecast(self, prices: pd.Series, n_simulations: int) -> dict:
        log_returns = np.log(prices / prices.shift(1)).dropna().values
        mu_daily = log_returns.mean()
        sigma_daily = log_returns.std(ddof=1)
        if sigma_daily == 0:
            return {}

        # Clamp daily drift so the implied annualised return stays within
        # [_MIN_ANNUAL_RETURN, _MAX_ANNUAL_RETURN].  Without this, a single
        # exceptional year (e.g. +400 %) drives the median path into the thousands.
        max_daily = np.log(1 + self._MAX_ANNUAL_RETURN) / self._TRADING_DAYS_1Y
        min_daily = np.log(1 + self._MIN_ANNUAL_RETURN) / self._TRADING_DAYS_1Y
        mu_daily = float(np.clip(mu_daily, min_daily, max_daily))

        drift = mu_daily - 0.5 * sigma_daily ** 2
        T = self._TRADING_DAYS_2Y
        P0 = float(prices.iloc[-1])

        Z = self._rng.standard_normal((n_simulations, T))
        log_paths = np.cumsum(drift + sigma_daily * Z, axis=1)
        price_paths = P0 * np.exp(log_paths)

        terminal_1y = price_paths[:, self._TRADING_DAYS_1Y - 1]
        terminal_2y = price_paths[:, self._TRADING_DAYS_2Y - 1]

        # Select 10 fan paths at evenly-spaced quantiles of 2Y terminal distribution
        quantile_indices = np.argsort(terminal_2y)[
            np.linspace(0, n_simulations - 1, 10).astype(int)
        ]

        return dict(
            mc_return_1y=float(np.median(terminal_1y) / P0 - 1),
            mc_return_2y=float(np.median(terminal_2y) / P0 - 1),
            mc_p10_1y=float(np.percentile(terminal_1y, 10) / P0 - 1),
            mc_p90_1y=float(np.percentile(terminal_1y, 90) / P0 - 1),
            mc_p10_2y=float(np.percentile(terminal_2y, 10) / P0 - 1),
            mc_p90_2y=float(np.percentile(terminal_2y, 90) / P0 - 1),
            mc_paths_sample=price_paths[quantile_indices, :].tolist(),
        )

    def _capm_forecast(
        self,
        returns_stock: np.ndarray,
        returns_spy: np.ndarray,
        risk_free_annual: float,
    ) -> dict:
        if returns_spy is None or len(returns_spy) < 30:
            return {}
        cov_matrix = np.cov(returns_stock, returns_spy, ddof=1)
        beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
        spy_cumulative = float(np.prod(1 + returns_spy))
        n_years = len(returns_spy) / 252.0
        e_market = spy_cumulative ** (1.0 / n_years) - 1
        capm_1y = risk_free_annual + beta * (e_market - risk_free_annual)
        capm_1y = float(np.clip(capm_1y, self._MIN_ANNUAL_RETURN, self._MAX_ANNUAL_RETURN))
        return dict(
            beta=beta,
            capm_return_1y=capm_1y,
            capm_return_2y=float((1 + capm_1y) ** 2 - 1),
        )

    def _trend_forecast(self, prices: pd.Series) -> dict:
        log_prices = np.log(prices.values.astype(float))
        x = np.arange(len(log_prices), dtype=float)
        coeffs = np.polyfit(x, log_prices, deg=1)
        daily_rate = float(coeffs[0])
        # Clamp daily rate to the same annual bounds used by MC
        max_daily = np.log(1 + self._MAX_ANNUAL_RETURN) / self._TRADING_DAYS_1Y
        min_daily = np.log(1 + self._MIN_ANNUAL_RETURN) / self._TRADING_DAYS_1Y
        daily_rate = float(np.clip(daily_rate, min_daily, max_daily))
        return dict(
            trend_return_1y=float(np.exp(daily_rate * self._TRADING_DAYS_1Y) - 1),
            trend_return_2y=float(np.exp(daily_rate * self._TRADING_DAYS_2Y) - 1),
        )

    @staticmethod
    def _ensemble(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None and not np.isnan(v)]
        return float(np.mean(valid)) if valid else None


# Initialize FastAPI app
app = FastAPI(title="Calculation Service", description="Portfolio optimization and statistical calculations")
calculation_service = CalculationService()
forecast_service = ForecastService()

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
            investment_amount=request.investment_amount,
            expected_returns_override=request.expected_returns_override,
        )
        
        return ServiceResponse(
            success=True,
            data=result.model_dump()
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
        # Convert prices data back to DataFrame (orient='index' → dates as rows, tickers as cols)
        prices_df = pd.DataFrame.from_dict(prices_data, orient='index')
        prices_df.index = pd.to_datetime(prices_df.index)
        
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
        # Convert prices data back to DataFrame (orient='index' → dates as rows, tickers as cols)
        prices_df = pd.DataFrame.from_dict(prices_data, orient='index')
        prices_df.index = pd.to_datetime(prices_df.index)
        
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

class BacktestRequest(BaseModel):
    tickers: List[str]
    prices_data: Dict[str, Dict[str, float]]
    n_windows: int = 3
    train_days: int = 252   # 1-year training window
    test_days: int = 252    # 1-year test period
    n_simulations: int = 500
    risk_free_annual: float = 0.04
    spy_ticker: str = "SPY"


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Rolling backtest: run ForecastService on historical training windows and
    compare ensemble predictions to actual returns in the subsequent test period.
    Returns per-window and aggregate MAE / directional hit-rate statistics.
    """
    try:
        if not request.tickers or not request.prices_data:
            raise HTTPException(status_code=400, detail="tickers and prices_data are required")

        prices_df = pd.DataFrame.from_dict(request.prices_data, orient='index')
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()
        prices_df = prices_df.apply(pd.to_numeric, errors='coerce').ffill().dropna(how='all')

        total_days = len(prices_df)
        required = request.train_days + request.test_days
        if total_days < required + 20:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Need at least {required + 20} trading days of data, "
                    f"got {total_days}. Use a longer data horizon (3Y or 5Y)."
                ),
            )

        n = request.n_windows
        available = total_days - request.train_days - request.test_days
        step = max(30, available // max(1, n - 1)) if n > 1 else 0

        windows = []
        for i in range(n):
            start_idx = i * step
            train_end_idx = start_idx + request.train_days
            test_end_idx = train_end_idx + request.test_days
            if test_end_idx > total_days:
                break

            train_prices = prices_df.iloc[start_idx:train_end_idx]
            test_prices = prices_df.iloc[train_end_idx:test_end_idx]

            raw_forecasts = forecast_service.forecast(
                prices=train_prices,
                tickers=request.tickers,
                n_simulations=request.n_simulations,
                risk_free_annual=request.risk_free_annual,
                spy_ticker=request.spy_ticker,
            )

            predicted = {
                fc["ticker"]: fc["ensemble_return_1y"]
                for fc in raw_forecasts
                if fc.get("ensemble_return_1y") is not None
            }

            actuals = {}
            for ticker in request.tickers:
                if ticker == request.spy_ticker:
                    continue
                if ticker in test_prices.columns:
                    series = test_prices[ticker].dropna()
                    if len(series) >= 2:
                        actuals[ticker] = float(series.iloc[-1] / series.iloc[0] - 1)

            common = set(predicted) & set(actuals)
            if not common:
                continue

            mae = float(np.mean([abs(predicted[t] - actuals[t]) for t in common]))
            hit_rate = float(np.mean([(predicted[t] > 0) == (actuals[t] > 0) for t in common]))

            windows.append({
                "window_start": str(train_prices.index[0].date()),
                "window_end":   str(train_prices.index[-1].date()),
                "test_end":     str(test_prices.index[-1].date()),
                "predicted":    predicted,
                "actuals":      actuals,
                "mae":          mae,
                "hit_rate":     hit_rate,
                "n_tickers":    len(common),
            })

        if not windows:
            raise HTTPException(status_code=400, detail="No complete backtest windows could be formed")

        all_errors, all_hits = [], []
        ticker_errors: Dict[str, list] = {}
        ticker_hits:   Dict[str, list] = {}
        for w in windows:
            common = set(w["predicted"]) & set(w["actuals"])
            for t in common:
                err = abs(w["predicted"][t] - w["actuals"][t])
                hit = int((w["predicted"][t] > 0) == (w["actuals"][t] > 0))
                all_errors.append(err)
                all_hits.append(hit)
                ticker_errors.setdefault(t, []).append(err)
                ticker_hits.setdefault(t, []).append(hit)

        return ServiceResponse(success=True, data={
            "windows":             windows,
            "overall_mae":         float(np.mean(all_errors)),
            "overall_hit_rate":    float(np.mean(all_hits)),
            "per_ticker_mae":      {t: float(np.mean(v)) for t, v in ticker_errors.items()},
            "per_ticker_hit_rate": {t: float(np.mean(v)) for t, v in ticker_hits.items()},
            "n_tickers":           len(ticker_errors),
            "n_windows":           len(windows),
        }).model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in run_backtest: {e}")
        return ServiceResponse(success=False, error=str(e))


@app.post("/forecast-returns")
async def forecast_returns(request: ForecastRequest):
    """
    Compute 1Y and 2Y return forecasts for specified tickers using an ensemble
    of Monte Carlo GBM, CAPM, and log-linear trend regression.
    Include SPY in prices_data for CAPM beta calculation.
    """
    try:
        if not request.tickers or not request.prices_data:
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters: tickers and prices_data"
            )

        prices_df = pd.DataFrame.from_dict(request.prices_data, orient='index')
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()
        prices_df = prices_df.apply(pd.to_numeric, errors='coerce')
        prices_df = prices_df.ffill().dropna(how='all')

        raw_results = forecast_service.forecast(
            prices=prices_df,
            tickers=request.tickers,
            n_simulations=request.n_simulations,
            risk_free_annual=request.risk_free_annual,
            spy_ticker=request.spy_ticker,
        )

        result = ForecastResult(
            forecasts=[TickerForecast(**r) for r in raw_results],
            n_simulations=request.n_simulations,
            risk_free_annual=request.risk_free_annual,
        )

        return ServiceResponse(success=True, data=result.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecast_returns: {e}")
        return ServiceResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

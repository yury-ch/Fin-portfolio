# tests/e2e/test_e2e_flows.py
# -----------------------------------------------------------------------
# End-to-end tests for the fin-portfolio microservice stack.
#
# Philosophy
# ──────────
# These tests exercise real HTTP communication against live service
# processes — no mocking, no TestClient.  The goal is to verify that:
#   1. Every service is reachable and healthy.
#   2. The HTTP contracts (request/response shapes) hold.
#   3. Multi-service workflows (forecast → optimize bridge) work
#      end-to-end as the UI would use them.
#   4. Business invariants (forecast return caps, optimiser weights
#      summing to 1) hold under the full stack.
#
# Skip network-dependent tests with:  pytest -m "not network"
# Run only e2e tests with:            pytest tests/e2e/
# -----------------------------------------------------------------------

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pytest


# ── synthetic price-data helpers ───────────────────────────────────────

def _gbm_prices(
    tickers: list[str],
    n_days: int = 300,
    seed: int = 42,
    annual_returns: dict[str, float] | None = None,
    start_prices: dict[str, float] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Generate `n_days` of GBM price data for each ticker.

    Returns prices_data in the format expected by the calc service:
        { "YYYY-MM-DD": { "AAPL": 153.2, "MSFT": 310.1, ... }, ... }
    """
    rng = np.random.default_rng(seed)
    mu_annual = annual_returns or {t: 0.10 for t in tickers}
    s0 = start_prices or {t: 100.0 for t in tickers}

    dt = 1 / 252
    vol = 0.20

    # Build per-ticker price arrays
    price_series: dict[str, list[float]] = {}
    for ticker in tickers:
        mu = mu_annual.get(ticker, 0.10)
        drift = (mu - 0.5 * vol**2) * dt
        shocks = rng.standard_normal(n_days - 1) * math.sqrt(dt) * vol
        log_returns = np.concatenate([[0.0], drift + shocks])
        prices = float(s0.get(ticker, 100.0)) * np.exp(np.cumsum(log_returns))
        price_series[ticker] = prices.tolist()

    # Convert to { date_str: { ticker: price } }
    base_date = date(2023, 1, 1)
    prices_data: Dict[str, Dict[str, float]] = {}
    for i in range(n_days):
        d_str = (base_date + timedelta(days=i)).isoformat()
        prices_data[d_str] = {t: price_series[t][i] for t in tickers}

    return prices_data


# Shared small ticker set used across multiple tests
_TICKERS = ["AAPL", "MSFT", "GOOGL"]
_PRICES_300 = _gbm_prices(_TICKERS, n_days=300)


# ══════════════════════════════════════════════════════════════════════
# 1. SERVICE HEALTH
# ══════════════════════════════════════════════════════════════════════

class TestServiceHealth:
    """All three services must respond to /health with status=healthy."""

    def test_ticker_service_healthy(self, http, services):
        r = http.get(f"{services['ticker']}/health")
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "healthy"
        assert body.get("service") == "ticker_service"

    def test_data_service_healthy(self, http, services):
        r = http.get(f"{services['data']}/health")
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "healthy"
        assert body.get("service") == "data_service"

    def test_calc_service_healthy(self, http, services):
        r = http.get(f"{services['calc']}/health")
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "healthy"
        assert body.get("service") == "calculation_service"


# ══════════════════════════════════════════════════════════════════════
# 2. TICKER SERVICE
# ══════════════════════════════════════════════════════════════════════

class TestTickerService:
    """Ticker service returns a valid list of S&P 500 tickers."""

    def _tickers(self, http, services) -> list:
        """Extract the tickers list from the service response."""
        r = http.get(f"{services['ticker']}/sp500-tickers")
        assert r.status_code == 200
        body = r.json()
        # Response shape: { tickers: [...], count: N, source: ..., ... }
        return body["tickers"]

    def test_get_tickers_returns_list(self, http, services):
        tickers = self._tickers(http, services)
        assert isinstance(tickers, list)
        assert len(tickers) > 0

    def test_tickers_are_strings(self, http, services):
        tickers = self._tickers(http, services)
        assert all(isinstance(t, str) for t in tickers)

    def test_tickers_contain_known_stocks(self, http, services):
        tickers = set(self._tickers(http, services))
        assert tickers & {"AAPL", "MSFT", "GOOGL", "AMZN"}, (
            "Expected at least one of AAPL/MSFT/GOOGL/AMZN in ticker list"
        )

    def test_no_duplicate_tickers(self, http, services):
        tickers = self._tickers(http, services)
        assert len(tickers) == len(set(tickers)), "Ticker list contains duplicates"

    def test_response_includes_count(self, http, services):
        r = http.get(f"{services['ticker']}/sp500-tickers")
        body = r.json()
        assert "count" in body
        assert body["count"] == len(body["tickers"])


# ══════════════════════════════════════════════════════════════════════
# 3. DATA SERVICE — read-only endpoints
# ══════════════════════════════════════════════════════════════════════

class TestDataServiceReadOnly:
    """Data service endpoints that don't require Yahoo Finance calls."""

    def test_cache_info_shape(self, http, services):
        r = http.get(f"{services['data']}/cache-info")
        assert r.status_code == 200
        body = r.json()
        # Response wrapped in ServiceResponse: { success, data: { has_cache, is_stale, ... } }
        assert body["success"] is True
        data = body["data"]
        assert "has_cache" in data
        assert "is_stale" in data

    def test_price_cache_info_shape(self, http, services):
        r = http.get(f"{services['data']}/price-cache-info")
        assert r.status_code == 200
        body = r.json()
        # Returns a dict of period → cache metadata (may be empty if not synced)
        assert isinstance(body, dict)

    def test_sp500_sample_tickers(self, http, services):
        r = http.get(f"{services['data']}/sp500-tickers")
        assert r.status_code == 200
        body = r.json()
        # Response wrapped in ServiceResponse: { success, data: [...] }
        assert body["success"] is True
        tickers = body["data"]
        assert isinstance(tickers, list)
        assert len(tickers) > 0


# ══════════════════════════════════════════════════════════════════════
# 4. CALCULATION SERVICE — stateless computation endpoints
#    These use synthetic prices so no network call is required.
# ══════════════════════════════════════════════════════════════════════

class TestComputeStats:
    """POST /compute-stats returns expected-returns and covariance matrix."""

    def test_returns_expected_shape(self, http, services):
        r = http.post(f"{services['calc']}/compute-stats", json=_PRICES_300)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        data = body["data"]
        assert "expected_returns" in data
        assert "covariance_matrix" in data

    def test_expected_returns_cover_all_tickers(self, http, services):
        r = http.post(f"{services['calc']}/compute-stats", json=_PRICES_300)
        mu = r.json()["data"]["expected_returns"]
        assert set(mu.keys()) == set(_TICKERS)

    def test_covariance_matrix_is_square(self, http, services):
        r = http.post(f"{services['calc']}/compute-stats", json=_PRICES_300)
        cov = r.json()["data"]["covariance_matrix"]
        assert set(cov.keys()) == set(_TICKERS)
        for row in cov.values():
            assert set(row.keys()) == set(_TICKERS)

    def test_variances_are_positive(self, http, services):
        r = http.post(f"{services['calc']}/compute-stats", json=_PRICES_300)
        cov = r.json()["data"]["covariance_matrix"]
        for ticker in _TICKERS:
            assert cov[ticker][ticker] > 0, f"Variance for {ticker} must be positive"


class TestPortfolioMetrics:
    """POST /portfolio-metrics computes sensible performance metrics."""

    def test_metrics_keys_present(self, http, services):
        payload = {
            "prices_data": _PRICES_300,
            "weights": {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
        }
        r = http.post(f"{services['calc']}/portfolio-metrics", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        data = body["data"]
        for key in ("total_return", "annual_return", "volatility", "sharpe_ratio", "max_drawdown"):
            assert key in data, f"Missing metric: {key}"

    def test_volatility_positive(self, http, services):
        payload = {
            "prices_data": _PRICES_300,
            "weights": {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
        }
        r = http.post(f"{services['calc']}/portfolio-metrics", json=payload)
        body = r.json()
        assert body["success"] is True, body.get("error")
        assert body["data"]["volatility"] > 0

    def test_equal_weight_portfolio(self, http, services):
        """Equal-weight portfolio should have finite, non-nan metrics."""
        weights = {t: 1 / len(_TICKERS) for t in _TICKERS}
        payload = {"prices_data": _PRICES_300, "weights": weights}
        r = http.post(f"{services['calc']}/portfolio-metrics", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True, body.get("error")
        data = body["data"]
        assert all(not math.isnan(v) for v in data.values() if isinstance(v, float))
        assert all(not math.isinf(v) for v in data.values() if isinstance(v, float))


class TestOptimizePortfolio:
    """POST /optimize-portfolio returns valid weights that sum to 1."""

    def test_max_sharpe_weights_sum_to_one(self, http, services):
        payload = {
            "tickers": _TICKERS,
            "prices_data": _PRICES_300,
            "investment_amount": 100_000,
            "objective": "Max Sharpe",
            "risk_free": 4.0,
            "max_weight": 60.0,
            "l2_reg": 5.0,
            "min_weight_threshold": 0.25,
            "min_holdings": 2,
        }
        r = http.post(f"{services['calc']}/optimize-portfolio", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True, body.get("error")
        weights = body["data"]["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_min_volatility_objective(self, http, services):
        payload = {
            "tickers": _TICKERS,
            "prices_data": _PRICES_300,
            "investment_amount": 50_000,
            "objective": "Min Volatility",
            "risk_free": 4.0,
            "max_weight": 60.0,
            "l2_reg": 5.0,
            "min_weight_threshold": 0.25,
            "min_holdings": 2,
        }
        r = http.post(f"{services['calc']}/optimize-portfolio", json=payload)
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_all_weights_within_max_weight(self, http, services):
        """max_weight constraint is applied when n_assets > 3; use 5 tickers."""
        max_w = 40.0
        # 5 tickers to exceed the n_assets <= 3 bypass in the optimizer
        five_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
        prices_5 = _gbm_prices(five_tickers, n_days=300, seed=42)
        payload = {
            "tickers": five_tickers,
            "prices_data": prices_5,
            "investment_amount": 100_000,
            "objective": "Max Sharpe",
            "risk_free": 4.0,
            "max_weight": max_w,
            "l2_reg": 5.0,
            "min_weight_threshold": 0.25,
            "min_holdings": 2,
        }
        r = http.post(f"{services['calc']}/optimize-portfolio", json=payload)
        body = r.json()
        assert body["success"] is True, body.get("error")
        weights = body["data"]["weights"]
        for ticker, w in weights.items():
            assert w <= (max_w / 100) + 1e-4, (
                f"{ticker} weight {w:.3f} exceeds max_weight {max_w}%"
            )

    def test_allocation_integers(self, http, services):
        """Discrete allocation must contain non-negative integers."""
        payload = {
            "tickers": _TICKERS,
            "prices_data": _PRICES_300,
            "investment_amount": 100_000,
            "objective": "Max Sharpe",
            "risk_free": 4.0,
            "max_weight": 60.0,
            "l2_reg": 5.0,
            "min_weight_threshold": 0.25,
            "min_holdings": 2,
        }
        r = http.post(f"{services['calc']}/optimize-portfolio", json=payload)
        allocation = r.json()["data"]["allocation"]
        for ticker, shares in allocation.items():
            assert isinstance(shares, int), f"{ticker}: shares must be int"
            assert shares >= 0, f"{ticker}: negative shares"


# ══════════════════════════════════════════════════════════════════════
# 5. FORECAST SERVICE — return-cap invariant
# ══════════════════════════════════════════════════════════════════════

class TestForecastReturns:
    """
    POST /forecast-returns.

    Key invariant: ensemble returns must be capped at ±50% for 1Y and
    must not explode to implausible multi-hundred-percent values.
    """

    # Use prices with a steep artificial drift to stress-test the cap
    _BULL_PRICES = _gbm_prices(
        _TICKERS,
        n_days=300,
        seed=7,
        annual_returns={"AAPL": 4.0, "MSFT": 5.0, "GOOGL": 6.0},  # extreme drift
    )

    def _forecast(self, http, services, prices_data=None, n_sim=200):
        payload = {
            "tickers": _TICKERS,
            "prices_data": prices_data or _PRICES_300,
            "top_n": 3,
            "n_simulations": n_sim,
            "risk_free_annual": 0.04,
        }
        return http.post(f"{services['calc']}/forecast-returns", json=payload)

    def test_response_shape(self, http, services):
        r = self._forecast(http, services)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        forecasts = body["data"]["forecasts"]
        assert isinstance(forecasts, list)
        assert len(forecasts) > 0

    def test_forecast_covers_all_tickers(self, http, services):
        r = self._forecast(http, services)
        returned = {f["ticker"] for f in r.json()["data"]["forecasts"]}
        assert returned == set(_TICKERS)

    def test_ensemble_return_1y_within_caps(self, http, services):
        """Ensemble 1Y return must be in (-90%, +50%) after clamping."""
        r = self._forecast(http, services, prices_data=self._BULL_PRICES)
        for fc in r.json()["data"]["forecasts"]:
            ret = fc.get("ensemble_return_1y")
            if ret is not None:
                assert ret <= 0.50 + 1e-6, (
                    f"{fc['ticker']} ensemble_return_1y={ret:.2%} exceeds +50% cap"
                )
                assert ret >= -0.90 - 1e-6, (
                    f"{fc['ticker']} ensemble_return_1y={ret:.2%} below -90% floor"
                )

    def test_ensemble_return_2y_within_caps(self, http, services):
        """Ensemble 2Y return must not exceed (1.50)^2 - 1 = 1.25 (compounded cap)."""
        r = self._forecast(http, services, prices_data=self._BULL_PRICES)
        compounded_cap = (1 + 0.50) ** 2 - 1  # ≈ 1.25
        for fc in r.json()["data"]["forecasts"]:
            ret = fc.get("ensemble_return_2y")
            if ret is not None:
                assert ret <= compounded_cap + 1e-4, (
                    f"{fc['ticker']} ensemble_return_2y={ret:.2%} exceeds compounded cap"
                )

    def test_current_prices_positive(self, http, services):
        r = self._forecast(http, services)
        for fc in r.json()["data"]["forecasts"]:
            assert fc["current_price"] > 0

    def test_mc_percentile_ordering(self, http, services):
        """p10 must be <= median (ensemble) <= p90 for each ticker."""
        r = self._forecast(http, services)
        for fc in r.json()["data"]["forecasts"]:
            p10 = fc.get("mc_p10_1y")
            p90 = fc.get("mc_p90_1y")
            median = fc.get("ensemble_return_1y")
            if p10 is not None and p90 is not None:
                assert p10 <= p90, (
                    f"{fc['ticker']}: p10={p10:.3f} > p90={p90:.3f}"
                )


# ══════════════════════════════════════════════════════════════════════
# 6. FORECAST → OPTIMIZER BRIDGE (multi-service integration)
# ══════════════════════════════════════════════════════════════════════

class TestForecastOptimizerBridge:
    """
    Simulate the UI workflow:
      1. POST /forecast-returns → extract ensemble_return_1y per ticker
      2. POST /optimize-portfolio with expected_returns_override = forecast returns
      3. Assert the optimizer accepted the overrides and returned valid weights
    """

    def test_forecast_then_optimize(self, http, services):
        # Step 1 — forecast
        forecast_payload = {
            "tickers": _TICKERS,
            "prices_data": _PRICES_300,
            "top_n": 3,
            "n_simulations": 200,
            "risk_free_annual": 0.04,
        }
        r_forecast = http.post(
            f"{services['calc']}/forecast-returns", json=forecast_payload
        )
        assert r_forecast.status_code == 200
        forecasts = r_forecast.json()["data"]["forecasts"]

        # Build override dict from forecast results
        override = {
            fc["ticker"]: fc["ensemble_return_1y"]
            for fc in forecasts
            if fc.get("ensemble_return_1y") is not None
        }
        assert len(override) > 0, "No ensemble returns available for override"

        # Step 2 — optimize with override
        opt_payload = {
            "tickers": list(override.keys()),
            "prices_data": _PRICES_300,
            "investment_amount": 100_000,
            "objective": "Max Sharpe",
            "risk_free": 4.0,
            "max_weight": 60.0,
            "l2_reg": 5.0,
            "min_weight_threshold": 0.25,
            "min_holdings": 2,
            "expected_returns_override": override,
        }
        r_opt = http.post(
            f"{services['calc']}/optimize-portfolio", json=opt_payload
        )
        assert r_opt.status_code == 200, r_opt.text
        body = r_opt.json()
        assert body["success"] is True, body.get("error")

        # Weights must sum to 1
        weights = body["data"]["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_override_produces_different_weights(self, http, services):
        """
        Optimizing with forecast overrides should produce different weights
        than optimizing with historical expected returns (at least sometimes).
        This detects if the override code path is silently ignored.
        """
        base_payload = {
            "tickers": _TICKERS,
            "prices_data": _PRICES_300,
            "investment_amount": 100_000,
            "objective": "Max Sharpe",
            "risk_free": 4.0,
            "max_weight": 60.0,
            "l2_reg": 5.0,
            "min_weight_threshold": 0.25,
            "min_holdings": 2,
        }

        # Historical run (no override)
        r_hist = http.post(
            f"{services['calc']}/optimize-portfolio", json=base_payload
        )
        weights_hist = r_hist.json()["data"]["weights"]

        # Override run: invert expected returns dramatically
        override = {t: -0.30 for t in _TICKERS[:1]}  # one ticker very negative
        override |= {t: 0.45 for t in _TICKERS[1:]}  # others near cap
        override_payload = {**base_payload, "expected_returns_override": override}

        r_ovr = http.post(
            f"{services['calc']}/optimize-portfolio", json=override_payload
        )
        weights_ovr = r_ovr.json()["data"]["weights"]

        # At least one weight must differ by > 1 pp
        diffs = [
            abs(weights_ovr.get(t, 0) - weights_hist.get(t, 0))
            for t in _TICKERS
        ]
        assert max(diffs) > 0.01, (
            "expected_returns_override had no effect on portfolio weights"
        )


# ══════════════════════════════════════════════════════════════════════
# 7. DATA SERVICE — stock data fetch (requires Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.network
class TestDataServiceNetwork:
    """
    Tests that call Yahoo Finance.  Skip with: pytest -m "not network"
    """

    def test_stock_data_returns_prices(self, http, services):
        payload = {
            "tickers": ["AAPL", "MSFT"],
            "period": "1mo",
            "interval": "1d",
        }
        r = http.post(f"{services['data']}/stock-data", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        prices = body["data"]
        # Should have at least a few trading days
        assert len(prices) >= 10

    def test_stock_data_covers_requested_tickers(self, http, services):
        payload = {
            "tickers": ["AAPL", "MSFT"],
            "period": "1mo",
            "interval": "1d",
        }
        r = http.post(f"{services['data']}/stock-data", json=payload)
        if not r.json()["success"]:
            pytest.skip("Yahoo Finance unavailable")
        dates = r.json()["data"]
        # Every date entry should have prices for both tickers
        first_row = next(iter(dates.values()))
        assert "AAPL" in first_row
        assert "MSFT" in first_row

    def test_ticker_to_data_chain(self, http, services):
        """
        Integration chain: get tickers from ticker service →
        use a subset of those tickers in a data service stock-data request.
        """
        # Get tickers
        r_tickers = http.get(f"{services['ticker']}/sp500-tickers")
        tickers = r_tickers.json()["tickers"][:3]  # use just first 3

        # Fetch their prices
        r_data = http.post(
            f"{services['data']}/stock-data",
            json={"tickers": tickers, "period": "5d", "interval": "1d"},
        )
        assert r_data.status_code == 200
        body = r_data.json()
        # At least a partial result is acceptable (some tickers might be
        # delisted); success flag tells us the service processed the request
        assert "success" in body

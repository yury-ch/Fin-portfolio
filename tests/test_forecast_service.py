import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.calculation_service import ForecastService

try:
    from fastapi.testclient import TestClient
    from services.calculation_service import app
    HAS_TEST_CLIENT = True
except Exception:
    HAS_TEST_CLIENT = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """300 business days of GBM prices for 5 tickers + SPY."""
    np.random.seed(99)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    params = {
        "AAPL":  (150, 0.001,  0.018),
        "MSFT":  (250, 0.0008, 0.016),
        "GOOGL": (130, 0.0009, 0.017),
        "NVDA":  (400, 0.002,  0.030),
        "TSLA":  (200, 0.0005, 0.035),
        "SPY":   (440, 0.0004, 0.010),
    }
    data = {}
    for ticker, (p0, mu, sigma) in params.items():
        log_ret = np.random.normal(mu - 0.5 * sigma ** 2, sigma, 300)
        data[ticker] = p0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def fs():
    """Seeded ForecastService for deterministic tests."""
    return ForecastService(rng_seed=42)


# ---------------------------------------------------------------------------
# TestMonteCarlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:

    def test_mc_forecast_returns_expected_keys(self, fs, synthetic_prices):
        result = fs._mc_forecast(synthetic_prices["AAPL"], n_simulations=200)
        for key in ("mc_return_1y", "mc_return_2y", "mc_p10_1y", "mc_p90_1y",
                    "mc_p10_2y", "mc_p90_2y", "mc_paths_sample"):
            assert key in result, f"Missing key: {key}"

    def test_mc_forecast_p10_lt_median_lt_p90(self, fs, synthetic_prices):
        result = fs._mc_forecast(synthetic_prices["AAPL"], n_simulations=500)
        assert result["mc_p10_1y"] < result["mc_return_1y"] < result["mc_p90_1y"], (
            f"Expected p10 < median < p90, got "
            f"{result['mc_p10_1y']:.4f} / {result['mc_return_1y']:.4f} / {result['mc_p90_1y']:.4f}"
        )

    def test_mc_forecast_paths_shape(self, fs, synthetic_prices):
        result = fs._mc_forecast(synthetic_prices["AAPL"], n_simulations=200)
        paths = result["mc_paths_sample"]
        assert len(paths) == 10, f"Expected 10 fan paths, got {len(paths)}"
        assert all(len(p) == ForecastService._TRADING_DAYS_2Y for p in paths), (
            f"Each path should have {ForecastService._TRADING_DAYS_2Y} steps"
        )

    def test_mc_forecast_zero_volatility_returns_empty(self, fs):
        idx = pd.date_range("2023-01-01", periods=100, freq="B")
        flat = pd.Series(100.0, index=idx)
        result = fs._mc_forecast(flat, n_simulations=100)
        assert result == {}, "Constant price series should return empty dict"

    def test_mc_forecast_reproducible_with_seed(self, synthetic_prices):
        fs1 = ForecastService(rng_seed=7)
        fs2 = ForecastService(rng_seed=7)
        r1 = fs1._mc_forecast(synthetic_prices["MSFT"], n_simulations=200)
        r2 = fs2._mc_forecast(synthetic_prices["MSFT"], n_simulations=200)
        assert abs(r1["mc_return_1y"] - r2["mc_return_1y"]) < 1e-12, (
            "Same seed should produce identical mc_return_1y"
        )


# ---------------------------------------------------------------------------
# TestCAPM
# ---------------------------------------------------------------------------

class TestCAPM:

    def test_capm_positive_beta_for_spy_like_stock(self, fs, synthetic_prices):
        """Build a stock that IS SPY + small noise so beta must be close to 1."""
        np.random.seed(0)
        spy_ret = synthetic_prices["SPY"].pct_change().dropna().values
        noise = np.random.normal(0, 0.001, len(spy_ret))
        stock_ret = spy_ret + noise   # highly correlated with SPY
        result = fs._capm_forecast(stock_ret, spy_ret, risk_free_annual=0.04)
        assert "beta" in result
        assert result["beta"] > 0.5, f"Expected beta close to 1, got {result['beta']:.4f}"

    def test_capm_insufficient_spy_returns_empty(self, fs, synthetic_prices):
        stock_ret = synthetic_prices["AAPL"].pct_change().dropna().values[:50]
        spy_ret   = synthetic_prices["SPY"].pct_change().dropna().values[:10]  # < 30
        result = fs._capm_forecast(stock_ret, spy_ret, risk_free_annual=0.04)
        assert result == {}, "Fewer than 30 SPY observations should return empty dict"

    def test_capm_2y_is_compounded_1y(self, fs, synthetic_prices):
        stock_ret = synthetic_prices["MSFT"].pct_change().dropna().values
        spy_ret   = synthetic_prices["SPY"].pct_change().dropna().values
        result = fs._capm_forecast(stock_ret, spy_ret, risk_free_annual=0.04)
        expected_2y = (1 + result["capm_return_1y"]) ** 2 - 1
        assert abs(result["capm_return_2y"] - expected_2y) < 1e-10, (
            f"2Y CAPM should be compounded 1Y: expected {expected_2y:.8f}, "
            f"got {result['capm_return_2y']:.8f}"
        )

    def test_capm_formula_sml(self, fs, synthetic_prices):
        """Verify SML formula: capm_1y = rf + beta * (E_market - rf)."""
        spy_ret = synthetic_prices["SPY"].pct_change().dropna().values
        # Build stock = 1.5x SPY so beta ≈ 1.5
        stock_ret = spy_ret * 1.5
        rf = 0.04
        result = fs._capm_forecast(stock_ret, spy_ret, risk_free_annual=rf)
        beta = result["beta"]
        spy_cum = float(np.prod(1 + spy_ret))
        n_years = len(spy_ret) / 252.0
        e_market = spy_cum ** (1.0 / n_years) - 1
        expected = rf + beta * (e_market - rf)
        assert abs(result["capm_return_1y"] - expected) < 1e-10, (
            f"SML mismatch: expected {expected:.8f}, got {result['capm_return_1y']:.8f}"
        )


# ---------------------------------------------------------------------------
# TestTrend
# ---------------------------------------------------------------------------

class TestTrend:

    def test_trend_perfect_uptrend_recovers_slope(self, fs):
        """Log-linear fit on exact GBM path recovers the daily rate."""
        daily_rate = 0.0008
        T = 300
        idx = pd.date_range("2023-01-01", periods=T, freq="B")
        prices = pd.Series(100 * np.exp(daily_rate * np.arange(T)), index=idx)
        result = fs._trend_forecast(prices)
        expected_1y = float(np.exp(daily_rate * 252) - 1)
        assert abs(result["trend_return_1y"] - expected_1y) < 1e-6, (
            f"Expected trend_return_1y ≈ {expected_1y:.6f}, got {result['trend_return_1y']:.6f}"
        )

    def test_trend_flat_prices_near_zero(self, fs):
        idx = pd.date_range("2023-01-01", periods=100, freq="B")
        prices = pd.Series(150.0, index=idx)
        result = fs._trend_forecast(prices)
        assert abs(result["trend_return_1y"]) < 1e-6, (
            f"Flat prices should give trend_return_1y ≈ 0, got {result['trend_return_1y']:.8f}"
        )

    def test_trend_returns_expected_keys(self, fs, synthetic_prices):
        result = fs._trend_forecast(synthetic_prices["NVDA"])
        assert "trend_return_1y" in result
        assert "trend_return_2y" in result


# ---------------------------------------------------------------------------
# TestEnsemble
# ---------------------------------------------------------------------------

class TestEnsemble:

    def test_ensemble_all_valid(self):
        result = ForecastService._ensemble([0.10, 0.20, 0.30])
        assert abs(result - 0.20) < 1e-10

    def test_ensemble_partial_none(self):
        result = ForecastService._ensemble([0.10, None, 0.30])
        assert abs(result - 0.20) < 1e-10

    def test_ensemble_all_none(self):
        result = ForecastService._ensemble([None, None, None])
        assert result is None


# ---------------------------------------------------------------------------
# TestForecastIntegration
# ---------------------------------------------------------------------------

class TestForecastIntegration:

    def test_forecast_full_pipeline(self, fs, synthetic_prices):
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        results = fs.forecast(
            prices=synthetic_prices,
            tickers=tickers,
            n_simulations=200,
            risk_free_annual=0.04,
            spy_ticker="SPY",
        )
        assert len(results) == 5
        for r in results:
            assert r["ticker"] in tickers
            assert r["mc_return_1y"] is not None
            assert r["trend_return_1y"] is not None
            assert r["ensemble_return_1y"] is not None

    def test_forecast_missing_spy_capm_is_none(self, fs, synthetic_prices):
        prices_no_spy = synthetic_prices.drop(columns=["SPY"])
        results = fs.forecast(
            prices=prices_no_spy,
            tickers=["AAPL", "MSFT"],
            n_simulations=200,
            risk_free_annual=0.04,
            spy_ticker="SPY",
        )
        for r in results:
            assert r.get("beta") is None, "beta should be None when SPY absent"
            assert r.get("capm_return_1y") is None
            assert r["mc_return_1y"] is not None, "MC should still work without SPY"
            assert r["trend_return_1y"] is not None

    def test_forecast_insufficient_history_excluded(self, fs):
        dates = pd.date_range("2023-01-01", periods=30, freq="B")  # < 63 rows
        short_prices = pd.DataFrame({
            "SHORT": np.linspace(100, 110, 30),
            "SPY": np.linspace(400, 420, 30),
        }, index=dates)
        results = fs.forecast(
            prices=short_prices,
            tickers=["SHORT"],
            n_simulations=100,
            risk_free_annual=0.04,
            spy_ticker="SPY",
        )
        assert results == [], "Ticker with < 63 rows should be excluded"

    def test_forecast_unknown_ticker_excluded(self, fs, synthetic_prices):
        results = fs.forecast(
            prices=synthetic_prices,
            tickers=["AAPL", "UNKNOWN_XYZ"],
            n_simulations=100,
            risk_free_annual=0.04,
            spy_ticker="SPY",
        )
        tickers_in_result = [r["ticker"] for r in results]
        assert "UNKNOWN_XYZ" not in tickers_in_result
        assert "AAPL" in tickers_in_result


# ---------------------------------------------------------------------------
# TestForecastAPI
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TEST_CLIENT, reason="fastapi[testclient] not available")
class TestForecastAPI:

    @pytest.fixture
    def api_client(self):
        return TestClient(app)

    @pytest.fixture
    def valid_payload(self, synthetic_prices):
        prices_dict = {
            str(idx.date()): {col: float(val) for col, val in row.items()}
            for idx, row in synthetic_prices.iterrows()
        }
        return {
            "tickers": ["AAPL", "MSFT"],
            "prices_data": prices_dict,
            "n_simulations": 100,
            "risk_free_annual": 0.04,
            "spy_ticker": "SPY",
        }

    def test_forecast_endpoint_success(self, api_client, valid_payload):
        response = api_client.post("/forecast-returns", json=valid_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "forecasts" in data["data"]
        assert len(data["data"]["forecasts"]) == 2

    def test_forecast_endpoint_missing_params(self, api_client):
        response = api_client.post("/forecast-returns", json={})
        assert response.status_code == 422

    def test_forecast_endpoint_empty_tickers_returns_error(self, api_client):
        response = api_client.post("/forecast-returns", json={
            "tickers": [],
            "prices_data": {},
        })
        assert response.status_code in (400, 200)
        if response.status_code == 200:
            assert response.json()["success"] is False

    def test_health_endpoint_unaffected(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

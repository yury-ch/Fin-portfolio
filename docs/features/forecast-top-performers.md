# F-01 — Forecast: Top Performers (1Y / 2Y)

| Field | Value |
|---|---|
| ID | F-01 |
| Status | Shipped |
| Priority | P2 |
| Owner | — |
| Created | 2026-03-28 |

---

## 1. User Story

> As a portfolio analyst, I want to see 1-year and 2-year return projections for the top-ranked stocks so that I can compare forward-looking expectations against historical composite scores before making allocation decisions.

---

## 2. Acceptance Criteria

1. A **Forecast tab** appears in the Streamlit UI alongside the existing tabs.
2. The user can select the analysis lookback period, the number of top stocks to forecast (3–20), the annual risk-free rate, and the Monte Carlo simulation count.
3. Clicking **Run Forecast** fetches prices, calls the calculation service, and renders results without page reload.
4. The output shows forecasts from **three independent methods** — Monte Carlo GBM, CAPM, and log-linear trend — plus an **ensemble average**.
5. Results include **1-year and 2-year horizons**.
6. For each of the top 5 tickers, a **fan chart** (price path) shows 10 representative simulation paths and a p10–p90 confidence band.
7. If SPY price data is unavailable, CAPM fields degrade gracefully to `—` rather than crashing.
8. A **CSV download** of the forecast table is available.
9. All existing tests continue to pass after the feature is merged.

---

## 3. Algorithm Specifications

### 3.1 Monte Carlo — Geometric Brownian Motion (GBM)

Uses Ito's lemma to simulate log-price paths.

**Inputs:** historical daily prices for the ticker.

**Parameters:**
- `N` — number of simulations (user-selectable: 500 / 1000 / 2000 / 5000)
- `T_1y = 252` trading days, `T_2y = 504` trading days

**Estimation:**
```
mu_daily    = mean( log(P_t / P_{t-1}) )
sigma_daily = std ( log(P_t / P_{t-1}) )
drift       = mu_daily - 0.5 * sigma_daily^2   # Ito correction
```

**Simulation:**
```
Z_i ~ N(0, 1)  independently for i = 1..T
P_T = P_0 * exp( sum_{i=1}^{T} (drift + sigma_daily * Z_i) )
```

Repeat N times to obtain terminal price distribution.

**Outputs:**
```
mc_return_1y = median(P_{T=252}) / P_0 - 1
mc_p10_1y    = percentile(P_{T=252}, 10) / P_0 - 1
mc_p90_1y    = percentile(P_{T=252}, 90) / P_0 - 1
mc_return_2y = median(P_{T=504}) / P_0 - 1
mc_p10_2y    = percentile(P_{T=504}, 10) / P_0 - 1
mc_p90_2y    = percentile(P_{T=504}, 90) / P_0 - 1
```

**Fan chart:** 10 paths are selected at evenly-spaced quantiles of the 2Y terminal price distribution (not random selection), ensuring the fan representatively covers the full outcome range.

---

### 3.2 CAPM — Security Market Line

**Requires:** SPY price history in the price frame.

**Beta estimation (OLS):**
```
returns_stock = daily simple returns of the ticker
returns_spy   = daily simple returns of SPY (aligned by date)

cov_matrix = np.cov(returns_stock, returns_spy, ddof=1)
beta       = cov_matrix[0, 1] / cov_matrix[1, 1]
```

**Market premium:**
```
spy_cumulative = product(1 + returns_spy)
n_years        = len(returns_spy) / 252
E_market       = spy_cumulative ^ (1 / n_years) - 1   # geometric annualisation
```

**Expected returns:**
```
risk_free_annual  = user-supplied (default 4%)

capm_return_1y = risk_free_annual + beta * (E_market - risk_free_annual)
capm_return_2y = (1 + capm_return_1y)^2 - 1
```

**Degradation:** if SPY is absent or has fewer than 30 observations, `beta`, `capm_return_1y`, and `capm_return_2y` are set to `None`.

---

### 3.3 Log-Linear Trend Extrapolation

**Inputs:** historical daily prices for the ticker.

**OLS regression:**
```
y = log(P_t)
x = [0, 1, 2, ..., N-1]   # integer day index

[slope, intercept] = polyfit(x, y, deg=1)
daily_rate = slope          # continuously compounded daily growth rate
```

**Forecast:**
```
trend_return_1y = exp(daily_rate * 252) - 1
trend_return_2y = exp(daily_rate * 504) - 1
```

No confidence bands — this is a single-point extrapolation of the historical trend.

---

### 3.4 Ensemble

```
valid_1y = [v for v in [mc_return_1y, capm_return_1y, trend_return_1y] if v is not None]
ensemble_return_1y = mean(valid_1y)   # None if all are None

valid_2y = [v for v in [mc_return_2y, capm_return_2y, trend_return_2y] if v is not None]
ensemble_return_2y = mean(valid_2y)
```

The ensemble is an unweighted average. No method is given preferential weighting because each captures a structurally different view of future returns.

---

## 4. API Contract

### 4.1 Request Model — `ForecastRequest`

```python
class ForecastRequest(BaseModel):
    tickers: List[str]
    prices_data: Dict[str, Dict[str, float]]  # {date_str: {ticker: price}}
    top_n: int = 10
    n_simulations: int = 1000
    risk_free_annual: float = 0.04            # decimal (0.04 = 4%)
    spy_ticker: str = "SPY"
```

### 4.2 Response Models

```python
class TickerForecast(BaseModel):
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
    forecasts: List[TickerForecast]
    n_simulations: int
    risk_free_annual: float
    generated_at: datetime = datetime.now()
```

### 4.3 Endpoint

```
POST /forecast-returns
Host: localhost:8002
Content-Type: application/json

Request body: ForecastRequest
Response body: ServiceResponse{ success: bool, data: ForecastResult, error: str }
```

**Notes:**
- Server-side ranking is NOT performed. Callers pass the already-filtered top-N tickers.
- `mc_paths_sample` carries exactly 10 paths regardless of `n_simulations` — payload stays ≈400KB for 10 tickers.
- Timeout: 180 seconds (5000 simulations × 20 tickers is the worst case, ~30s on a typical laptop).

---

## 5. Files to Create / Modify

| File | Action | Description |
|---|---|---|
| `shared/models.py` | Modify | Add `ForecastRequest`, `TickerForecast`, `ForecastResult` at bottom of file |
| `services/calculation_service.py` | Modify | Add `ForecastService` class; add `forecast_service` singleton; add `/forecast-returns` endpoint; extend model imports |
| `services/presentation_service.py` | Modify | Add `ServiceClient.get_forecast()`; add `_render_forecast_results()` helper; extend `st.tabs()`; add Forecast tab block; add session state keys |
| `requirements-microservices.txt` | Modify | Add `plotly` (fan charts rendered by Streamlit) |
| `tests/test_forecast_service.py` | Create | 23 unit + integration + API tests |

No new service files are required. `ForecastService` follows the existing `CalculationService` pattern.

---

## 6. UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Tab: 🔮 Forecast                                               │
├─────────────────────────────────────────────────────────────────┤
│  [Lookback period ▼] [Top N ——●——] [RF rate %] [Simulations ▼] │
│  [🚀 Run Forecast ──────────────────────────────────────────]   │
├─────────────────────────────────────────────────────────────────┤
│  📊 Ensemble Forecast Summary                                   │
│  ┌──────┬───────┬────────┬────────┬──────────┬────────┬──────┐ │
│  │Ticker│ Price │Ens. 1Y │Ens. 2Y │MC 1Y p50 │MC p90  │Beta  │ │
│  └──────┴───────┴────────┴────────┴──────────┴────────┴──────┘ │
├─────────────────────────────────────────────────────────────────┤
│  📋 Method Comparison — 1-Year Forecast                        │
│  ┌──────┬──────────┬──────┬───────┬──────────┐                 │
│  │Ticker│MC (median)│CAPM  │Trend  │Ensemble  │                 │
│  └──────┴──────────┴──────┴───────┴──────────┘                 │
│  ▶ Method Comparison — 2-Year Forecast (expander)              │
├─────────────────────────────────────────────────────────────────┤
│  📈 Monte Carlo Fan Charts — Top 5 Tickers                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │  TICKER_1   │ │  TICKER_2   │ │  TICKER_3   │              │
│  │  fan chart  │ │  fan chart  │ │  fan chart  │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
│  ┌─────────────┐ ┌─────────────┐                               │
│  │  TICKER_4   │ │  TICKER_5   │                               │
│  └─────────────┘ └─────────────┘                               │
├─────────────────────────────────────────────────────────────────┤
│  [📥 Download Forecast Data (CSV)]                             │
└─────────────────────────────────────────────────────────────────┘
```

**Fan chart anatomy (per ticker):**
- 10 thin lines (low opacity) — representative simulated paths
- Filled band between path[0] and path[9] (p10–p90 of terminal distribution)
- Thick median line (path[4])
- Dashed horizontal at current price
- Dotted vertical at day 252 (1Y boundary)
- X-axis: trading days (0–504), Y-axis: price ($)

---

## 7. Implementation Plan

### Phase 1 — Core (no UI, fully testable)

**Step 1.1 — Pydantic models** (`shared/models.py`)
- Add `ForecastRequest`, `TickerForecast`, `ForecastResult`
- No other file changes needed yet
- Verify: `from shared.models import ForecastRequest` works in a Python shell

**Step 1.2 — `ForecastService` private methods** (`services/calculation_service.py`)
- Add the class with `_mc_forecast`, `_capm_forecast`, `_trend_forecast`, `_ensemble`
- Do NOT add the FastAPI endpoint yet
- Run: `pytest tests/test_forecast_service.py::TestMonteCarlo tests/test_forecast_service.py::TestCAPM tests/test_forecast_service.py::TestTrend tests/test_forecast_service.py::TestEnsemble -v`

**Step 1.3 — `ForecastService.forecast()` main method**
- Wires private methods together; handles SPY alignment and missing-ticker guard
- Run: `pytest tests/test_forecast_service.py::TestForecastIntegration -v`

### Phase 2 — FastAPI endpoint

**Step 2.1 — Singleton and imports** (`services/calculation_service.py`)
- Extend model import line: add `ForecastRequest, ForecastResult, TickerForecast`
- Add `forecast_service = ForecastService()` after `calculation_service = CalculationService()`

**Step 2.2 — `/forecast-returns` endpoint**
- Add POST handler following the exact pattern of `/optimize-portfolio`
- Run: `pytest tests/test_forecast_service.py::TestForecastAPI -v`

**Step 2.3 — Smoke test**
```bash
curl -s -X POST http://localhost:8002/forecast-returns \
  -H "Content-Type: application/json" \
  -d '{"tickers":["AAPL"],"prices_data":{}}' | python -m json.tool
```
Expected: `{"success": false, "error": "..."}`  (empty prices triggers guard)

### Phase 3 — Streamlit UI

**Step 3.1 — Dependency** (`requirements-microservices.txt`)
- Add `plotly` on its own line

**Step 3.2 — Session state** (`services/presentation_service.py`)
- Add `'forecast_result': None` and `'forecast_prices': None` to `_STATE_DEFAULTS`

**Step 3.3 — `ServiceClient.get_forecast()`**
- New method following the pattern of `ServiceClient.optimize_portfolio()`
- Adds `"SPY"` to the ticker list automatically
- Calls `POST http://localhost:8002/forecast-returns`

**Step 3.4 — `_render_forecast_results()` helper**
- Defined before the `client = ServiceClient()` bootstrap line
- Contains all table and chart rendering logic (keeps the tab block clean)

**Step 3.5 — Tab block**
- Extend `st.tabs(...)` to include `"🔮 Forecast"`
- Add the full `with tab_forecast:` block

### Phase 4 — Full regression

```bash
pytest tests/ -v --tb=short
```

All 83+ existing tests must remain green. The 23 new forecast tests must all pass.

---

## 8. Test Plan

File: `tests/test_forecast_service.py`

### TestMonteCarlo (5 tests)

| # | Name | Assertion |
|---|---|---|
| 1 | `test_mc_forecast_returns_expected_keys` | All output keys present |
| 2 | `test_mc_forecast_p10_lt_median_lt_p90` | `mc_p10_1y < mc_return_1y < mc_p90_1y` |
| 3 | `test_mc_forecast_paths_shape` | `mc_paths_sample` is 10 lists of length 504 |
| 4 | `test_mc_forecast_zero_volatility` | Constant prices → empty dict (degenerate guard) |
| 5 | `test_mc_forecast_reproducible` | Same seed → identical results |

### TestCAPM (4 tests)

| # | Name | Assertion |
|---|---|---|
| 6 | `test_capm_positive_beta` | Stock that closely tracks SPY produces beta > 0.8 |
| 7 | `test_capm_insufficient_spy` | < 30 SPY observations → empty dict |
| 8 | `test_capm_2y_is_compounded_1y` | `capm_return_2y ≈ (1 + capm_return_1y)^2 − 1` to 1e-10 |
| 9 | `test_capm_zero_market_premium` | E_market == rf → `capm_return_1y ≈ rf` for any beta |

### TestTrend (3 tests)

| # | Name | Assertion |
|---|---|---|
| 10 | `test_trend_perfect_uptrend` | Known slope recovered within 1e-6 |
| 11 | `test_trend_flat_prices` | `trend_return_1y ≈ 0` |
| 12 | `test_trend_returns_expected_keys` | Keys `trend_return_1y`, `trend_return_2y` present |

### TestEnsemble (3 tests)

| # | Name | Assertion |
|---|---|---|
| 13 | `test_ensemble_all_valid` | Returns exact mean |
| 14 | `test_ensemble_partial_none` | Ignores None, averages remainder |
| 15 | `test_ensemble_all_none` | Returns None |

### TestForecastIntegration (4 tests)

| # | Name | Assertion |
|---|---|---|
| 16 | `test_forecast_full_pipeline` | 5 tickers + SPY → all fields present in each result |
| 17 | `test_forecast_missing_spy` | CAPM fields all None, MC + trend still populated |
| 18 | `test_forecast_insufficient_history` | Ticker < 63 rows excluded from output |
| 19 | `test_forecast_top_n_respected` | len(result) <= len(tickers) |

### TestForecastAPI (4 tests)

| # | Name | Assertion |
|---|---|---|
| 20 | `test_forecast_endpoint_success` | POST with valid payload → 200 + `success=True` |
| 21 | `test_forecast_endpoint_missing_params` | Empty body → 422 Unprocessable |
| 22 | `test_forecast_endpoint_service_error` | Mocked exception → `success=False` with error string |
| 23 | `test_health_endpoint_unaffected` | GET `/health` still returns 200 after ForecastService added |

### Test fixture

```python
@pytest.fixture
def synthetic_prices():
    """300 business days of GBM prices for 5 tickers + SPY."""
    np.random.seed(99)
    dates = pd.date_range('2023-01-01', periods=300, freq='B')
    params = {
        'AAPL': (150, 0.001,  0.018),
        'MSFT': (250, 0.0008, 0.016),
        'GOOGL': (130, 0.0009, 0.017),
        'NVDA':  (400, 0.002,  0.030),
        'TSLA':  (200, 0.0005, 0.035),
        'SPY':   (440, 0.0004, 0.010),
    }
    data = {}
    for ticker, (p0, mu, sigma) in params.items():
        log_ret = np.random.normal(mu - 0.5 * sigma**2, sigma, 300)
        data[ticker] = p0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def forecast_service_seeded():
    return ForecastService(rng_seed=42)
```

---

## 9. Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| JSON payload too large (many tickers × 5000 sims) | `mc_paths_sample` always 10 paths, payload capped at ~1MB; `select_slider` caps sims at 5000 |
| SPY not in price cache (it's an ETF, not an S&P 500 constituent) | Presentation layer explicitly appends `"SPY"` to the price fetch request; `YahooPriceLoader` fetches it from yfinance as a fallback |
| Monte Carlo forecast is sensitive to lookback period | User controls lookback via the period selector; caveat shown in UI caption |
| GBM assumes constant drift and volatility | Stated limitation in UI caption: "statistical projections, not investment advice" |
| `plotly` not in existing requirements | Added to `requirements-microservices.txt` in Phase 3.1 |

---

## 10. Future Enhancements (out of scope for F-01)

- **Volatility regime detection** — use EWMA σ instead of historical σ for GBM drift in high-volatility periods
- **Factor model forecasts** — Fama-French 3-factor or 5-factor expected return estimates
- **Scenario analysis** — bull / base / bear scenarios with user-defined market return assumptions
- **Backtest validation** — compare 1Y-ago forecasts against realised returns to track model accuracy over time
- **12M momentum signal** — feed `Recent_12M_Return` (planned in TECHNICAL_DEBT item 17) into the ensemble weighting

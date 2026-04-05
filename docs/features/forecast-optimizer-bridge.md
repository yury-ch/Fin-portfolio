# F-02 — Forecast → Optimizer Bridge

| Field | Value |
|---|---|
| ID | F-02 |
| Status | Shipped |
| Priority | P1 |
| Shipped | 2026-04-01 |

---

## 1. User Story

> As a portfolio analyst, I want to feed the forecast ensemble return estimates directly into the portfolio optimizer so that allocation weights reflect forward-looking expectations rather than purely historical returns.

---

## 2. What It Does

After running a forecast, the user can click **"📤 Use Forecast Returns in Optimizer"** to transfer the `ensemble_return_1y` values for each ticker into the optimizer as expected returns. The optimizer then builds weights that maximise Sharpe (or target return) using the forecast-derived drift instead of the historical geometric mean.

---

## 3. User Flow

1. **Forecast tab** — run a forecast (any period, any N)
2. A **"📤 Use Forecast Returns in Optimizer"** button appears below the results table
3. Clicking it:
   - Stores `ensemble_return_1y` per ticker in `st.session_state["forecast_expected_returns"]`
   - Pre-populates the optimizer ticker list
   - Sets a one-shot flag to auto-select the new radio option on next render
4. **Portfolio Optimizer tab** — Stock Selection radio now shows **"Use Forecast Returns"** (only visible when a forecast exists)
5. Selecting it displays the tickers with a confirmation message
6. Click **🚀 Optimize Portfolio** — the optimizer uses forecast returns as `mu`

---

## 4. Algorithm Detail

### Expected Return Override

PyPortfolioOpt's `EfficientFrontier` accepts `mu` as a `pd.Series`. After computing the historical `mu` via `mean_historical_return(prices, compounding=True, frequency=252)`, the service overlays the caller-supplied values:

```python
if expected_returns_override:
    override = pd.Series(expected_returns_override)
    common = mu.index.intersection(override.index)
    if not common.empty:
        mu = mu.copy()
        mu[common] = override[common]
```

- Only tickers present in **both** the override dict and the price history are replaced
- Tickers in the optimizer but absent from the forecast fall back to historical `mu`
- The covariance matrix `S` is always computed from price history (unchanged)
- Units: annualised decimal (e.g. `0.18` = 18%)

---

## 5. Files Changed

| File | Change |
|---|---|
| `shared/models.py` | Added `expected_returns_override: Optional[Dict[str, float]] = None` to `PortfolioOptimizationRequest` |
| `services/calculation_service.py` | `optimize_portfolio()` accepts and applies `expected_returns_override`; endpoint passes it through |
| `services/presentation_service.py` | Session state keys `forecast_expected_returns`, `use_forecast_returns`; "Send to Optimizer" button; new radio option; `ServiceClient.optimize_portfolio` passes override |

---

## 6. Edge Cases

| Case | Behaviour |
|---|---|
| Forecast not yet run | "Use Forecast Returns" radio option is hidden |
| Ticker in optimizer not in forecast | Falls back to historical `mu` silently |
| `ensemble_return_1y` is None for some tickers | Filtered out before storing in session state |
| Stale forecast | Data persists in session state until next forecast run; no automatic expiry |

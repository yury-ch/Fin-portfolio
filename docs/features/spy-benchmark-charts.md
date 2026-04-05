# F-03 — SPY Benchmark Line in Fan Charts

| Field | Value |
|---|---|
| ID | F-03 |
| Status | Shipped |
| Priority | P2 |
| Shipped | 2026-04-01 |

---

## 1. User Story

> As a portfolio analyst, I want to see the S&P 500 (SPY) historical growth rate projected on the same chart as the Monte Carlo fan paths so that I can immediately judge whether a stock's simulated outcomes beat simply holding the index.

---

## 2. What It Does

Each Monte Carlo fan chart now shows an **orange dashed line** representing SPY's historical compound annual growth rate (CAGR) projected forward from the stock's current price. If simulated paths cluster below the orange line, the stock is not expected to outperform the index under historical SPY returns.

---

## 3. Visual Layout

```
Price ($)
  |         /-- MC p90
  |        / median path (blue)
  |  -----/-------------- SPY CAGR (orange dashed)
  | /-- MC p10
  |_________________________ Trading days
  0        252 (1Y)       504 (2Y)
```

- **Blue fan**: 10 MC paths spanning p10–p90 of the terminal distribution
- **Shaded band**: p10–p90 envelope
- **Grey dashed horizontal**: current price baseline
- **Grey dotted vertical**: 1-year mark
- **Orange dashed**: SPY CAGR compound-growth projection

---

## 4. Algorithm

SPY CAGR is computed **once** from the fetched price data before the chart loop, then scaled per chart:

```python
spy_ret = spy_series.pct_change().dropna().values
spy_cumulative = float(np.prod(1 + spy_ret))
n_years = len(spy_ret) / 252.0
spy_cagr = spy_cumulative ** (1.0 / n_years) - 1
spy_daily_growth = (1 + spy_cagr) ** (1 / 252) - 1

# Per chart (stock_start = fc["current_price"]):
spy_future = stock_start * (1 + spy_daily_growth) ** np.arange(n_days)
```

**Why a deterministic line rather than a MC simulation of SPY?**
The fan chart shows uncertainty for a specific stock. Running a separate MC for SPY would not capture correlation between the stock and the index, making the comparison misleading. A single CAGR line is an honest, interpretable benchmark: "this is what SPY has historically compounded at."

---

## 5. Files Changed

| File | Change |
|---|---|
| `services/presentation_service.py` | `_render_forecast_results` accepts `prices_df: Optional[pd.DataFrame]`; SPY CAGR pre-computed before loop; Plotly trace added per chart; legend enabled; caption updated; call site passes `st.session_state["forecast_prices"]` |

---

## 6. Edge Cases

| Case | Behaviour |
|---|---|
| SPY absent from fetched prices | `spy_daily_growth` stays `None`; overlay silently skipped; caption omits SPY note |
| SPY history < 30 days | Guard `len(spy_series) >= 30`; overlay skipped |
| Plotly not installed | Fallback `st.line_chart` path unchanged (no SPY overlay) |

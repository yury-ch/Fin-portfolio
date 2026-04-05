# F-06 — Forecast Return Caps (Drift Clamping)

| Field | Value |
|---|---|
| ID | F-06 |
| Status | Shipped |
| Priority | P1 (Bug Fix) |
| Shipped | 2026-04-01 |

---

## 1. Problem

The Monte Carlo, trend, and CAPM forecasts produced unrealistic results (e.g. STX: **413% 1Y, 2594% 2Y**) when a stock had a large recent run-up. All three models extrapolated exceptional historical performance indefinitely:

- **MC**: daily drift `μ` computed from 1Y history of a 400%-return stock drove the GBM median to 400%+
- **Trend**: log-linear slope from the same period projected `exp(0.006 × 252) − 1 ≈ 413%`
- **CAPM**: extreme beta with a high-returning market produced similar overestimates

---

## 2. Fix

A `_MAX_ANNUAL_RETURN = 0.50` (50%) and `_MIN_ANNUAL_RETURN = -0.90` (-90%) cap was added to `ForecastService`. The daily equivalent is used to clamp all three models:

```python
max_daily = np.log(1 + self._MAX_ANNUAL_RETURN) / self._TRADING_DAYS_1Y
min_daily = np.log(1 + self._MIN_ANNUAL_RETURN) / self._TRADING_DAYS_1Y
```

| Model | Clamp applied to |
|---|---|
| Monte Carlo | `mu_daily` before GBM drift computation |
| Trend | `daily_rate` from `np.polyfit` |
| CAPM | Final `capm_1y` return value |

**Effect on a normal stock (~10–15% history):** unchanged — well within bounds.
**Effect on STX (400% history):** 1Y capped at 50%, 2Y at ~125%.

---

## 3. Tuning

The constants are class-level and easy to adjust:

```python
class ForecastService:
    _MAX_ANNUAL_RETURN: float = 0.50   # upper bound — change here
    _MIN_ANNUAL_RETURN: float = -0.90  # lower bound — change here
```

50% was chosen as an aggressive-but-plausible upper bound for a 1-year stock forecast. Lower it to 40% for a more conservative tool; raise to 70% for more aggressive scenarios.

---

## 4. Files Changed

| File | Change |
|---|---|
| `services/calculation_service.py` | `ForecastService` class: added `_MAX_ANNUAL_RETURN`, `_MIN_ANNUAL_RETURN` constants; clamping applied in `_mc_forecast`, `_trend_forecast`, `_capm_forecast` |

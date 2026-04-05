# Feature Catalog

This directory contains specification and implementation documents for all shipped and planned features.

| # | Feature | Status | Shipped |
|---|---|---|---|
| F-01 | [Forecast — Top Performers (1Y / 2Y)](forecast-top-performers.md) | ✅ Shipped | 2026-03-28 |
| F-02 | [Forecast → Optimizer Bridge](forecast-optimizer-bridge.md) | ✅ Shipped | 2026-04-01 |
| F-03 | [SPY Benchmark Line in Fan Charts](spy-benchmark-charts.md) | ✅ Shipped | 2026-04-01 |
| F-04 | [12-Month Momentum Signal (12-1M)](momentum-12m-lookback.md) | ✅ Shipped | 2026-04-01 |
| F-05 | [Centralized Configuration (Env Vars)](centralized-config.md) | ✅ Shipped | 2026-04-01 |
| F-06 | [Forecast Return Caps (Drift Clamping)](forecast-return-caps.md) | ✅ Shipped | 2026-04-01 |
| F-07 | Portfolio Comparison (Historical vs Forecast) | ✅ Shipped | 2026-04-04 |
| F-09 | Fan Charts — Pagination / Slider | ✅ Shipped | Merged into F-01 delivery |
| F-10 | [End-to-End Test Suite](e2e-tests.md) | ✅ Shipped | 2026-04-04 |

---

## Planned

| # | Feature | Priority | Notes |
|---|---|---|---|
| F-08 | Backtesting Tab | ✅ Shipped | 2026-04-05 |
| F-11 | Forecast Confidence Indicator | P2 | Badge based on history length + volatility |
| F-12 | Prometheus Observability | P3 | `prometheus-fastapi-instrumentator` on all 3 services |
| F-13 | SQLite Metadata Store | P3 | Replace scattered JSON files in `sp500_data/` |

---

## Document Structure

Each feature document contains:
- User story
- What it does / user flow
- Algorithm / formula specification (where applicable)
- Files created / modified
- Edge cases handled

# F-10 — End-to-End Test Suite

**Status:** ✅ Shipped 2026-04-04
**Priority at creation:** P2

---

## User story

> As a developer, I want a single command that boots all three microservices and runs real HTTP tests against them, so I can catch integration regressions that unit tests (which mock everything) cannot see.

---

## What it does

`tests/e2e/` contains 33 tests across 7 classes that exercise the full live service stack via real HTTP calls — no mocking, no `TestClient`.

### Run commands

```bash
# All non-network tests (no Yahoo Finance calls required)
pytest tests/e2e/ -m "not network"

# Full suite including Yahoo Finance / Wikipedia calls
pytest tests/e2e/
```

Services are started automatically if not already running; if the ports are occupied the fixture reuses the live stack.

---

## Test classes

| Class | Tests | What it verifies |
|---|---|---|
| `TestServiceHealth` | 3 | All 3 services respond `status=healthy` on `/health` |
| `TestTickerService` | 5 | Response shape, no duplicates, known tickers present, `count` matches list length |
| `TestDataServiceReadOnly` | 3 | `/cache-info`, `/price-cache-info`, `/sp500-tickers` shapes — no network needed |
| `TestComputeStats` | 4 | Expected-returns and covariance-matrix keyed by ticker, variances positive |
| `TestPortfolioMetrics` | 3 | All metric keys present, volatility > 0, no NaN / Inf |
| `TestOptimizePortfolio` | 4 | Weights sum to 1, `max_weight` enforced (5-ticker case), integer allocation |
| `TestForecastReturns` | 6 | Return caps (±50%/1Y, compounded/2Y), p10 ≤ p90, all tickers covered |
| `TestForecastOptimizerBridge` | 2 | **Multi-service**: forecast returns → optimizer override → weights change |
| `TestDataServiceNetwork` | 3 | Stock data fetch, ticker → data chain — `@pytest.mark.network` |

---

## Architecture

### `tests/e2e/conftest.py`

Session-scoped `services` fixture:

1. Checks if each port (8000, 8001, 8002) is already in use.
2. If not, spawns the service as a subprocess (`python services/<name>.py`).
3. Polls `/health` every 1 second (timeout 45 s) before proceeding.
4. On session teardown, terminates any processes it launched.

Session-scoped `http` fixture: a single `httpx.Client(timeout=30)` shared across all tests.

### Synthetic price data

The `_gbm_prices()` helper generates GBM price paths (no Yahoo Finance) for the calculation-service tests. Parameters:

- `tickers` — list of ticker symbols
- `n_days` — number of trading days (default 300)
- `seed` — reproducibility seed
- `annual_returns` — per-ticker drift (default 10% p.a.)

The output format matches what every calc-service endpoint expects:
`{ "YYYY-MM-DD": { "AAPL": 153.2, "MSFT": 310.1 }, ... }`

---

## Service bug discovered and fixed

While writing the tests, a transpose bug was found in `calculation_service.py`:

**`/compute-stats` and `/portfolio-metrics`** used `pd.DataFrame(prices_data)` which, given input shape `{ date → { ticker → price } }`, produces a DataFrame with **tickers as the index and dates as columns** — the transpose of what PyPortfolioOpt expects. PyPortfolioOpt was therefore computing "expected returns" per date instead of per ticker, returning date strings as keys.

**Fix** (both endpoints):
```python
# Before
prices_df = pd.DataFrame(prices_data)

# After
prices_df = pd.DataFrame.from_dict(prices_data, orient='index')
prices_df.index = pd.to_datetime(prices_df.index)
```

`/optimize-portfolio` was already correct (`orient='index'` via `pd.DataFrame.from_dict`); the two secondary endpoints had never been exercised end-to-end before this test suite.

---

## Files created / modified

| File | Action |
|---|---|
| `tests/e2e/__init__.py` | Created |
| `tests/e2e/conftest.py` | Created — service-startup fixtures |
| `tests/e2e/test_e2e_flows.py` | Created — 33 tests |
| `pytest.ini` | Added `network` marker registration |
| `services/calculation_service.py` | Fixed `orient='index'` in `/compute-stats` and `/portfolio-metrics` |
| `docs/features/README.md` | Added F-10 row |
| `TECHNICAL_DEBT.md` | Added test-coverage update + bug-fix entry |

---

## `@pytest.mark.network` marker

Tests that call Yahoo Finance or Wikipedia are decorated `@pytest.mark.network`. They are deselected in CI by default:

```bash
pytest tests/e2e/ -m "not network"   # CI / offline
pytest tests/e2e/                    # full integration run
```

The marker is registered in `pytest.ini` to suppress the `PytestUnknownMarkWarning`.

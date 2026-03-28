# Technical Debt Report

This document outlines the key areas of technical debt in the S&P 500 Portfolio Optimizer project. Items are prioritized P1 (critical) → P3 (minor). Resolved items are marked ✅.

---

## Status Summary

| # | Item | Priority | Status |
|---|---|---|---|
| 1 | Code duplication — shared analysis logic | P1 | ✅ Resolved |
| 2 | Monolithic `app.py` still active | P1 | ✅ Completed |
| 3 | Competing architectures | P2 | ✅ Direction decided |
| 4 | Brittle service startup / no health checks | P2 | ✅ Resolved |
| 5 | No docker-compose.yml | P2 | ✅ Resolved |
| 6 | Inefficient Dockerfile / no `.dockerignore` | P3 | ✅ Resolved |
| 7 | Hardcoded configuration | P3 | ⏳ Open |
| 8 | Disorganized legacy data files | P3 | ✅ Partially resolved |
| 9 | Unpinned dependencies | P3 | ⏳ Open |
| 10 | Drawdown score inversion — worst stocks rank highest | **P1** | ✅ Resolved |
| 11 | `standardize_analysis_columns` drawdown corruption | **P1** | ✅ Resolved |
| 12 | Inconsistent risk-free rate (0% / 2% / param) | P2 | ✅ Resolved |
| 13 | Return + Sharpe double-counting in composite weights | P2 | ✅ Resolved |
| 14 | Min-max outlier collapse in composite scoring | P2 | ✅ Resolved |
| 15 | Arithmetic vs. log return inconsistency (screening vs. optimiser) | P3 | ✅ Resolved |
| 16 | Sample covariance unstable for large N — no shrinkage | P3 | ✅ Resolved |
| 17 | 3-month momentum suboptimal lookback | P3 | ⏳ Open |

---

## ✅ Resolved

### 1. Code Duplication — shared analysis logic (was P1)

`shared/analysis_engine.py` created with `standardize_analysis_columns`, `AnalysisEngine`, and `AnalysisResult`. `services/presentation_service.py` and `services/data_service.py` both import from this shared module. `app.py` still has its own copy of `standardize_analysis_columns`, but `app.py` is now deprecated (see item 2).

### 2. Monolithic `app.py` — deleted (was P1)

`app.py` (1,111 lines) has been removed. `run-monolith.sh` is now an error stub that prints a message directing users to `./run-microservices.sh` and exits with code 1. `stop-monolith.sh` was also deleted. No Python file imported `app.py`; documentation updated accordingly.

### 3. Competing architectures — direction decided (was P2)

The microservices stack (`services/presentation_service.py` + `ticker_service` + `data_service` + `calculation_service`) is the committed architecture. The monolith is deprecated. `docker-entrypoint.sh` and `run-microservices.sh` are the canonical entry points.

### 4. Brittle startup / no health checks (was P2)

`docker-entrypoint.sh` now implements `wait_for_healthy()` — polls each service's `/health` endpoint before starting the Streamlit UI, with configurable retries and timeout. Services fail fast with a clear error if dependencies are unavailable.

### 5. docker-compose.yml — added (was P2)

`docker-compose.yml` created with a single `fin-portfolio` service. Defines all four port mappings with environment-variable overrides (`TICKER_PORT`, `DATA_PORT`, `CALC_PORT`, `UI_PORT`), a named volume mount for `sp500_data/` so the price/analysis cache persists across container restarts, `restart: unless-stopped`, and a `healthcheck` polling `data_service /health`. One-command startup: `docker compose up`.

### 6. `.dockerignore` added / Dockerfile EXPOSE fixed (was P3)

`.dockerignore` created excluding `.venv/`, `sp500_data/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.git/`, and dev-only files (`requirements-test.txt`, `run_tests.py`, `test_*.py`). The `pip install` layer is now stable across source changes. `Dockerfile` EXPOSE updated from `8001 8002 8501` to `8000 8001 8002 8501` (ticker_service port was missing).

### 10. Drawdown score inversion — fixed (was P1)

`_append_scores` in `shared/analysis_engine.py` called `_safe_normalize(df['Max_Drawdown'], inverse=True)`. `Max_Drawdown` is already negative (−0.30 for a 30% loss), so `_safe_normalize` without inversion correctly maps the most negative value (worst) → 0.0 and the least negative (best) → 1.0. The `inverse=True` flag was flipping this, awarding Drawdown_Score = 1.0 to the stock with the largest peak-to-trough loss. Fixed by removing `inverse=True`. Regression tests added: `test_drawdown_score_direction_best_has_highest_score`, `test_drawdown_score_worst_ticker_scores_zero`, `test_drawdown_score_best_ticker_scores_one`.

### 11. `standardize_analysis_columns` drawdown guard — fixed (was P1)

The condition `df['Max_Drawdown'].min() > -1` was intended to detect legacy positive-percent data (e.g., `30` for a 30% loss) and convert it to decimal form (`−0.30`). The condition also fired for canonical decimal values like `−0.30` (since `−0.30 > -1` is True), transforming them to `+0.003`. Fixed by changing the guard to `df['Max_Drawdown'].max() > 1`, which only fires when at least one value is in positive-percent form. Regression tests added: `test_decimal_drawdown_not_corrupted`, `test_near_minus_one_drawdown_not_corrupted`, `test_zero_drawdown_not_corrupted`, `test_positive_percent_drawdown_is_converted`.

### 12. Inconsistent risk-free rate — unified (was P2)

Three locations used different `rf` values (0%, 2%, and a UI-supplied parameter). Created `shared/config.py` with `DEFAULT_RISK_FREE_RATE = 0.0`. Applied in `analysis_engine._analyze_series` (screening Sharpe) and `calculation_service.calculate_portfolio_metrics` (replacing hardcoded `0.02`). The portfolio optimizer retains its user-supplied `risk_free` parameter. Regression tests: `test_config_constant_exists_and_is_float`, `test_calculation_service_no_hardcoded_rf`.

### 13. Return + Sharpe double-counting — recalibrated (was P2)

Since Sharpe = Return / Volatility, the original weights (25% Return + 25% Sharpe + 20% Vol) produced an effective return factor weight of ~40% and vol weight of ~35%, exceeding stated intent. Recalibrated weights to 20% Return + 25% Sharpe + 15% Vol + 20% Drawdown + 20% Momentum, reducing the Return+Sharpe combined weight and restoring independent factor balance. Test: `test_composite_score_matches_new_weights`.

### 14. Min-max outlier collapse — replaced with percentile rank (was P2)

`Annual_Return` and `Recent_3M_Return` are fat-tailed: one extreme outlier (e.g., NVDA +180%) collapsed all other scores to near zero. Replaced `_safe_normalize` with `Series.rank(pct=True)` for these two metrics. Bounded metrics (Volatility, Sharpe_Ratio, Max_Drawdown) retain min-max. Regression tests: `test_outlier_does_not_collapse_other_return_scores`, `test_return_score_reflects_rank_order`, `test_momentum_outlier_does_not_collapse_other_momentum_scores`.

### 15. Arithmetic vs. log return inconsistency — fixed (was P3)

`calculation_service.compute_stats()` called `expected_returns.mean_historical_return(prices, frequency=252)` which computes an arithmetic mean. The screening engine (`analysis_engine._analyze_series`) uses a geometric mean (`exp(mean(log_returns) × 252) − 1`). For a 25%-vol stock, arithmetic overstates expected return by ~3.1 percentage points, systematically biasing the optimiser toward high-volatility names. Fixed by passing `compounding=True` to align both methods: `mean_historical_return(prices, compounding=True, frequency=252)`.

### 16. Sample covariance unstable for large N — replaced with Ledoit-Wolf (was P3)

`calculation_service.compute_stats()` used `risk_models.sample_cov(prices, frequency=252)`. For typical session sizes (20–100 tickers, 1–2 years of data) the sample covariance matrix has near-zero eigenvalues, causing the optimiser to assign extreme weight concentration. Fixed by replacing with `risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()`. Ledoit-Wolf shrinkage is analytically optimal and ships with PyPortfolioOpt. L2 regularisation (already in place for N > 3) is retained as a secondary constraint.

### 8. Disorganized legacy data files — partially resolved (was P3)

`sp500_data/*.parquet` added to `.gitignore` — generated cache files are no longer tracked. Per-period parquet files (`prices_1y_1d.parquet`, etc.) are the authoritative price cache. Legacy `sp500_analysis.parquet` and `metadata.parquet` still exist on disk but are no longer referenced by the microservices stack.

---

## ⏳ Open — Next Steps (by priority)

### P3 — Pin all dependencies

**Action:** Run `pip-compile requirements-microservices.txt` to generate a fully pinned lockfile. Currently only `numpy<2` is constrained; `streamlit`, `fastapi`, `pandas`, `yfinance` etc. all float to latest.

**Why:** Builds are not reproducible. A `yfinance` or `streamlit` major release could silently break the application.

---

### P3 — Centralized configuration

**Action:** Replace hardcoded service URLs (`localhost:8000/8001/8002`) and file paths in `presentation_service.py` and `data_service.py` with environment variables loaded via `pydantic BaseSettings`.

**Why:** Ports and hosts cannot be changed without editing source code. Required for any non-local deployment.

---

### P3 — 3-month momentum suboptimal lookback

**Location:** `analysis_engine.py:105–108`

The 63-day (≈3M) window is shorter than the academically validated 12-1 month signal (Jegadeesh & Titman). The trailing 1-month period exhibits short-term reversal; including it degrades signal quality. A 3M window also increases sensitivity to earnings-announcement noise.

**Improvement:** Add `Recent_12M_Return = prices.iloc[-1] / prices.iloc[-252] − 1` alongside the 3M metric and validate which lookback (or combination) produces higher quintile spread in backtests against the current universe.

---

## Architecture Recommendations

Forward-looking decisions that are not blocking but should inform future development.

---

### Storage: keep Parquet, add SQLite for metadata

**Current:** Parquet for price/analysis cache; JSON files for metadata, sync reports, and validation snapshots scattered across `sp500_data/`.

**Recommendation:**
- **Keep Parquet** for price and analysis caches — columnar format is optimal for wide time-series frames (500 tickers × N days). A relational DB would be slower for these read patterns.
- **Replace JSON metadata files** with a single SQLite database (`sp500_data/state.db`). This gives atomic writes, queryable validation history, and eliminates the need to glob for the latest `ticker_validation_*.json` file.
- **Consider DuckDB** if ad-hoc cross-period analytics are needed in the future — it can query parquet files directly with SQL, no ETL required.

---

### UI: Streamlit is appropriate now; React is the natural upgrade path

**Current:** Streamlit (`presentation_service.py`).

**Streamlit limitations to watch:**
- Every widget interaction reruns the entire script — performance degrades as the app grows
- No real URL routing (can't deep-link to a ticker or saved portfolio)
- No live data push (price streaming requires polling hacks)
- Poor multi-user concurrency (separate Python process per user)

**Recommendation:** Streamlit is the right choice for an internal analyst tool. If the app evolves toward multi-user access, user accounts, saved portfolios, or live price streaming — migrate the UI to React/Next.js calling the existing FastAPI services. The backend is already structured for this transition.

---

### Synchronous portfolio optimization is a latency risk

**Current:** `calculation_service.py` runs `scipy.optimize` synchronously inside a FastAPI request handler. For 500 tickers over 5 years this can take 5–30 seconds and blocks the entire service for other requests.

**Recommendation:** Move optimization to a background task. FastAPI's built-in `BackgroundTasks` is sufficient for single-user use; Celery + Redis for multi-user production. The UI would poll a `/job/{id}/status` endpoint instead of waiting on the HTTP response.

---

### Price data source resilience

**Current:** Yahoo Finance (`yfinance`) is the sole price provider. It has no SLA, applies rate limits without warning, and changes its API structure periodically.

**Recommendation:** `YahooPriceLoader` in `shared/price_loader.py` is already well-isolated behind a clear interface. Add a secondary provider (Polygon.io free tier or Alpha Vantage) as a fallback that activates when Yahoo returns empty data for a batch. No architectural change required — just an alternative implementation of the same interface.

---

### Observability

**Current:** Zero metrics, zero structured logging beyond print/logger calls, zero alerting. Silent failures in nightly cron jobs (price sync, validation) are only visible when the Data Health tab is checked manually.

**Recommendation:**
- Add `prometheus-fastapi-instrumentator` to the three FastAPI services (~1 hour effort) for request latency, error rates, and cache-hit visibility.
- Write sync job outcomes (success/failure, rows loaded, tickers failed) to the SQLite state DB (see Storage above) so the Data Health dashboard can show trend data, not just the last run.

---

### Test coverage gaps

**Current:** Tests exist for `ticker_provider` and `calculation_service`. The following are untested:

| Module | Risk | Status |
|---|---|---|
| `price_sync_service.py` delta logic | Bug silently corrupts the price cache | ✅ `test_price_sync_service.py` (8 tests) |
| `data_service.py` endpoints | API contract breakage goes undetected | ✅ `test_data_service.py` (24 tests) |
| `shared/price_loader.py` | Cache normalization edge cases | ✅ `test_price_cache.py` |
| `shared/analysis_engine.py` | Scoring formula regressions | ✅ `test_analysis_engine.py` |

**Bugs surfaced by tests:**
- `price_sync_service._delta_sync`: `if last_ts is None` guard is dead code — `pd.DatetimeIndex([]).max()` returns `NaT`, not `None`. Low severity.
- `data_service.POST /sp500-analysis`: broad `except Exception` swallows `HTTPException`, returning HTTP 200 `success=False` instead of the intended HTTP 503. Clients cannot distinguish "cache missing" from other errors. Fix: re-raise `HTTPException` before the generic except block.

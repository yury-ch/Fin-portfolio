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
| 5 | No docker-compose.yml | P2 | ⏳ Open |
| 6 | Inefficient Dockerfile / no `.dockerignore` | P3 | ⏳ Open |
| 7 | Hardcoded configuration | P3 | ⏳ Open |
| 8 | Disorganized legacy data files | P3 | ✅ Partially resolved |
| 9 | Unpinned dependencies | P3 | ⏳ Open |

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

### 8. Disorganized legacy data files — partially resolved (was P3)

`sp500_data/*.parquet` added to `.gitignore` — generated cache files are no longer tracked. Per-period parquet files (`prices_1y_1d.parquet`, etc.) are the authoritative price cache. Legacy `sp500_analysis.parquet` and `metadata.parquet` still exist on disk but are no longer referenced by the microservices stack.

---

## ⏳ Open — Next Steps (by priority)

### P2 — Add `docker-compose.yml`

**Action:** Create `docker-compose.yml` defining all four services with:
- `depends_on` + `healthcheck` directives (replacing the shell polling in `docker-entrypoint.sh`)
- Named volumes for `sp500_data/`
- Environment variable overrides for ports and paths

**Why:** Shell-based orchestration is fragile. `docker-compose` gives restart policies, log aggregation, and one-command startup (`docker compose up`).

---

### P3 — Add `.dockerignore`

**Action:** Create `.dockerignore` excluding `.venv`, `.git`, `sp500_data`, `__pycache__`, `*.pyc`, `.pytest_cache`.

**Why:** The current `Dockerfile` copies the full context including the virtual environment and all cached data files. This bloats the image and invalidates layer caching on every run.

---

### P3 — Pin all dependencies

**Action:** Run `pip-compile requirements-microservices.txt` to generate a fully pinned lockfile. Currently only `numpy<2` is constrained; `streamlit`, `fastapi`, `pandas`, `yfinance` etc. all float to latest.

**Why:** Builds are not reproducible. A `yfinance` or `streamlit` major release could silently break the application.

---

### P3 — Centralized configuration

**Action:** Replace hardcoded service URLs (`localhost:8000/8001/8002`) and file paths in `presentation_service.py` and `data_service.py` with environment variables loaded via `pydantic BaseSettings`.

**Why:** Ports and hosts cannot be changed without editing source code. Required for any non-local deployment.

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

| Module | Risk |
|---|---|
| `price_sync_service.py` delta logic | Bug silently corrupts the price cache |
| `data_service.py` endpoints | API contract breakage goes undetected |
| `shared/price_loader.py` | Cache normalization edge cases |
| `shared/analysis_engine.py` | Scoring formula regressions |

**Recommendation:** Prioritize tests for `price_sync_service._delta_sync()` — a merge bug there would produce wrong analysis results with no error message.

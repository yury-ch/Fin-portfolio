# Technical Debt Report

This document outlines the key areas of technical debt in the S&P 500 Portfolio Optimizer project. Items are prioritized P1 (critical) → P3 (minor). Resolved items are marked ✅.

---

## Status Summary

| # | Item | Priority | Status |
|---|---|---|---|
| 1 | Code duplication — shared analysis logic | P1 | ✅ Resolved |
| 2 | Monolithic `app.py` still active | P1 | ✅ Deprecated (not yet deleted) |
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

### 2. Monolithic `app.py` — deprecated (was P1)

`app.py` (1,111 lines) is now marked deprecated with a header warning directing users to the microservices stack. `run-monolith.sh` also prints a deprecation notice. The file has not been deleted yet — see next steps below.

### 3. Competing architectures — direction decided (was P2)

The microservices stack (`services/presentation_service.py` + `ticker_service` + `data_service` + `calculation_service`) is the committed architecture. The monolith is deprecated. `docker-entrypoint.sh` and `run-microservices.sh` are the canonical entry points.

### 4. Brittle startup / no health checks (was P2)

`docker-entrypoint.sh` now implements `wait_for_healthy()` — polls each service's `/health` endpoint before starting the Streamlit UI, with configurable retries and timeout. Services fail fast with a clear error if dependencies are unavailable.

### 8. Disorganized legacy data files — partially resolved (was P3)

`sp500_data/*.parquet` added to `.gitignore` — generated cache files are no longer tracked. Per-period parquet files (`prices_1y_1d.parquet`, etc.) are the authoritative price cache. Legacy `sp500_analysis.parquet` and `metadata.parquet` still exist on disk but are no longer referenced by the microservices stack.

---

## ⏳ Open — Next Steps (by priority)

### P1 — Delete or hollow out `app.py`

**Action:** Remove `app.py` entirely, or strip it to a thin redirect stub that prints the deprecation message and exits. The duplicated `standardize_analysis_columns` and all analysis/optimization logic inside it should go.

**Why now:** The deprecation notice does not prevent someone from running it and getting divergent results. It is the single largest remaining source of confusion.

---

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

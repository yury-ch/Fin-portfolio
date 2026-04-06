# Architecture & Technical Decisions

This document records the architecture of the S&P 500 Portfolio Optimizer and the reasoning behind key design decisions. It is intended for engineers joining the project, architects reviewing the design, or the future maintainer evaluating the next evolution step.

---

## 1. System Overview

The system is an internal analyst tool that:

1. Maintains a universe of S&P 500 constituents (sourced from Wikipedia)
2. Downloads and caches historical price data from Yahoo Finance
3. Computes per-ticker financial metrics (return, volatility, Sharpe, drawdown, momentum)
4. Runs mean-variance portfolio optimization against a user-selected ticker subset
5. Forecasts returns using an ensemble of Monte Carlo, CAPM, and trend models
6. Backtests forecast accuracy on historical rolling windows
7. Compares historical vs forward-looking optimised allocations
8. Presents results through an interactive web UI with a guided tab workflow

**Primary user:** a single analyst on a local machine or a small private server. The system is not designed for multi-user concurrency or public internet exposure.

---

## 2. Architecture: Decomposed Microservices

### 2.1 Service Map

```
┌─────────────────────────────────────────────────────────────┐
│                      Single Container                       │
│                                                             │
│  ┌─────────────────┐   ┌─────────────────┐                 │
│  │  ticker_service  │   │  data_service   │                 │
│  │    port 8000     │   │   port 8001     │                 │
│  │  FastAPI         │   │  FastAPI        │                 │
│  │                  │   │                 │                 │
│  │ S&P 500 universe │   │ Price cache     │                 │
│  │ Wikipedia fetch  │   │ Analysis cache  │                 │
│  │ CSV persistence  │   │ Parquet files   │                 │
│  └─────────────────┘   └─────────────────┘                 │
│                                                             │
│  ┌─────────────────┐   ┌─────────────────────────────────┐ │
│  │calc_service      │   │  presentation_service           │ │
│  │   port 8002      │   │     port 8501                   │ │
│  │  FastAPI         │   │  Streamlit                      │ │
│  │                  │   │                                 │ │
│  │ PyPortfolioOpt   │   │ Tabs (left → right workflow):   │ │
│  │ optimization     │   │  Data Health                    │ │
│  │ forecast engine  │   │  Analyzer                       │ │
│  │ (MC+CAPM+Trend)  │   │  Optimizer                      │ │
│  └─────────────────┘   │  Forecast                       │ │
│                         │  Backtest                       │ │
│                         │  Compare                        │ │
│                         │  Universe                       │ │
│                         └─────────────────────────────────┘ │
│                                                             │
│  docker-entrypoint.sh: starts services in dependency order  │
│  waits for /health on each before proceeding                │
└─────────────────────────────────────────────────────────────┘
                          │ volume
                    ./sp500_data/
```

### 2.2 Shared Library

All four services import from `shared/`:

| Module | Responsibility |
|---|---|
| `analysis_engine.py` | Canonical metrics computation and composite scoring |
| `price_loader.py` | Parquet cache read/write, normalization, Yahoo Finance wrapper |
| `ticker_provider.py` | Wikipedia scraper with CSV cache and embedded fallback |
| `models.py` | Pydantic v2 request/response models shared across services |
| `settings.py` | Centralised configuration via `pydantic BaseSettings` (env vars) |
| `config.py` | Algorithmic constants (risk-free rate, scoring weights, thresholds) |

### 2.3 Offline Jobs

Two data pipeline scripts run outside the request path, invoked manually or via cron:

| Script | Trigger | What it does |
|---|---|---|
| `price_sync_service.py` | Weekly cron / manual | Delta-syncs Yahoo Finance prices into per-period Parquet files |
| `analysis_sync_service.py` | After price sync | Reads price Parquet, runs `AnalysisEngine`, writes analysis Parquet |
| `ticker_validation_service.py` | Monthly cron | Compares cached constituents against Wikipedia; writes validation snapshot |

`data_service` will also auto-spawn `analysis_sync_service` in the background at startup if the analysis cache is missing or stale and a price cache is available.

---

## 3. Key Technical Decisions

### 3.1 Single Container, Multiple Processes

**Decision:** All four services run inside one Docker container via `docker-entrypoint.sh`, not as separate containers.

**Why:** Services communicate over `localhost` with configurable URLs. Running multiple processes in one container is a pragmatic tradeoff for a single-user tool.

**Trade-off accepted:** Crash isolation is weak — a fatal error in one service does not restart the others. Restart policies in `docker-compose.yml` mitigate this at the container level, not the process level.

**Resolved (F-05):** Service URLs are now centralised in `shared/settings.py` via `pydantic BaseSettings`, configurable through environment variables. The remaining step is splitting into separate Compose services with proper `depends_on`.

---

### 3.2 Parquet for Price and Analysis Caches

**Decision:** Historical price data and computed analysis results are stored as columnar Parquet files, not in a relational database.

**Why:** The price frame is wide (≈500 tickers × 1,260 rows for 5 years). Columnar format gives fast column-slice reads (single-ticker retrieval), compact on-disk size, and zero-overhead serialisation via `pandas.read_parquet`. A relational database would require a schema migration every time the ticker universe changes and would be slower for this read pattern.

**What is stored as JSON:** Metadata (sync timestamps, row counts, ticker counts) and ticker validation snapshots. These are small and human-readable; structured queries are not required.

**Future consideration:** Replace JSON metadata files with a single SQLite database (`sp500_data/state.db`). This gives atomic writes, queryable history for the Data Health dashboard, and eliminates glob-for-latest-file patterns. DuckDB is the natural upgrade if ad-hoc cross-period analytics are needed (it queries Parquet directly with SQL).

---

### 3.3 Incremental (Delta) Price Sync

**Decision:** `price_sync_service` performs delta syncs rather than full re-downloads on each run.

**How it works:**
1. Load existing Parquet cache.
2. Compute `start = last_cached_date − buffer_days` (default: 5 days overlap).
3. Download only the new window from Yahoo Finance.
4. Concatenate, deduplicate on index with `keep="last"` (new data wins on conflicts), sort, trim to the retention horizon.

**Why delta:** A full 5-year download of 500 tickers takes 20–30 minutes and hits Yahoo rate limits. An incremental sync of the last week takes under 2 minutes.

**Known bug:** `if last_ts is None` guard in `_delta_sync` is dead code — `pd.DatetimeIndex([]).max()` returns `NaT`, not `None`. Harmless at runtime; documented in TECHNICAL_DEBT.md.

---

### 3.4 Analysis Separated from the Request Path

**Decision:** Portfolio analysis (metrics, scoring) is computed offline by `analysis_sync_service`, not on-demand inside a FastAPI request.

**Why:** Computing annualised return, Sharpe, drawdown, and momentum scores for 500 tickers takes 5–15 seconds. Running this synchronously on every "Analyze Stocks" click would make the UI unusable. Pre-computing and caching results reduces the `POST /sp500-analysis` response time to a Parquet read (<100 ms).

**Trade-off:** Analysis results are stale until the next sync. Acceptable for weekly-cadence portfolio review; not acceptable for real-time trading.

---

### 3.5 Synchronous Portfolio Optimization

**Decision:** `calculation_service` runs `scipy.optimize` (via PyPortfolioOpt's `EfficientFrontier`) synchronously inside a FastAPI request handler.

**Current behaviour:** For 20–50 selected tickers, optimization completes in under 2 seconds. For 500 tickers, it can take 5–30 seconds and blocks all other requests to the service during that time.

**Accepted risk:** Acceptable for single-user, infrequent optimization calls. Not acceptable for multi-user or high-frequency use.

**Evolution path:** Move optimization to a background task. FastAPI's `BackgroundTasks` suffices for single-user; Celery + Redis for production. The UI polls a `/job/{id}/status` endpoint instead of waiting on the HTTP response.

---

### 3.6 Yahoo Finance as Sole Price Source

**Decision:** `YahooPriceLoader` in `shared/price_loader.py` wraps `yfinance` exclusively.

**Why:** Zero cost, no API key required, covers all S&P 500 tickers. Sufficient for a weekly-sync internal tool.

**Risk:** Yahoo Finance has no SLA, applies undocumented rate limits, and periodically changes its response structure. The `yfinance` library absorbs some of this churn but not all.

**Mitigation already in place:** `YahooPriceLoader` is isolated behind a clear interface with no callers outside `shared/`. A secondary provider (Polygon.io free tier, Alpha Vantage) can be added as a fallback implementation without touching any service code.

---

### 3.7 Streamlit for the UI

**Decision:** `presentation_service.py` uses Streamlit.

**Why:** Streamlit is appropriate for a single-analyst internal tool. It provides interactive widgets, dataframe display, and charts with minimal code and no frontend build pipeline.

**Known limitations:**
- Every widget interaction reruns the entire Python script (mitigated with `st.session_state`)
- No URL routing — cannot deep-link to a specific ticker or saved portfolio
- No live data push — price streaming requires polling
- Poor multi-user concurrency — separate Python process per user session

**Evolution path:** If the tool evolves toward multi-user access, saved portfolios, or live streaming, migrate the UI to React/Next.js calling the existing FastAPI services. The backend is already structured for this transition.

---

### 3.8 Monolith → Microservices Migration

**History:** The original `app.py` (1,111 lines) was a monolithic Streamlit application combining data fetching, analysis, optimization, and UI in a single file. It was deleted after the microservices stack reached feature parity.

**What the migration fixed:**
- Analysis logic was duplicated across `app.py`, `data_service.py`, and `presentation_service.py`. Consolidated into `shared/analysis_engine.py`.
- Service startup was fragile — no health checks, no dependency ordering. `docker-entrypoint.sh` now implements `wait_for_healthy()` with configurable retries.
- The UI recomputed analysis on every widget interaction. Separated into offline sync + cached read.

**Subsequent improvements:** Service URLs were centralised via `shared/settings.py` (F-05). Forecast, backtest, and compare tabs were added to close the analysis loop (F-02, F-03).

---

## 4. Data Flow

### 4.1 First-Time Setup

```
Wikipedia ──fetch──► ticker_provider ──► sp500_constituents.csv
                                                │
Yahoo Finance ──batch download──► price_sync_service
                                                │
                              sp500_data/price_cache/*.parquet
                                                │
                              analysis_sync_service
                                                │
                              sp500_data/sp500_analysis_*.parquet
```

### 4.2 Ongoing (Weekly)

```
cron (Monday 23:00) ──► run-price-sync.sh ──► delta sync → price Parquet
cron (Monday 23:05) ──► run-analysis-sync.sh ──► analysis Parquet
cron (1st of month) ──► ticker_validation_service ──► validation snapshot JSON
```

### 4.3 Request Path (UI interaction)

```
Browser ──► presentation_service (Streamlit :8501)
                │
                ├── GET /sp500-tickers ──► ticker_service (:8000)
                │                               └── sp500_constituents.csv
                │
                ├── POST /sp500-analysis ──► data_service (:8001)
                │                               └── sp500_analysis_*.parquet
                │
                ├── POST /stock-data ──► data_service (:8001)
                │                           └── price_cache/*.parquet
                │
                ├── POST /optimize-portfolio ──► calculation_service (:8002)
                │                                    └── PyPortfolioOpt
                │
                ├── POST /forecast-returns ──► calculation_service (:8002)
                │                                  └── MC + CAPM + Trend ensemble
                │
                └── POST /backtest ──► calculation_service (:8002)
                                           └── Rolling-window forecast validation
```

---

## 5. Storage Layout

```
sp500_data/
  sp500_constituents.csv          ← ticker universe (Wikipedia cache)
  sp500_analysis_1y.parquet       ← pre-computed analysis, 1-year window
  sp500_analysis_2y.parquet
  sp500_analysis_3y.parquet
  sp500_analysis_5y.parquet
  price_cache/
    prices_master.parquet         ← 5y/1d master price cache (all tickers)
    prices_1y_1d.parquet          ← trimmed slices for faster reads
    prices_2y_1d.parquet
    prices_3y_1d.parquet
    metadata.json                 ← sync timestamps, row counts per period
    sync_report.json              ← last run outcome (failed tickers, etc.)
  validation/
    ticker_validation_YYYYMMDDTHHMMSSZ.json  ← monthly diff snapshots
```

**Gitignore policy:** `sp500_data/*.parquet` and `sp500_data/price_cache/` and `sp500_data/validation/` are excluded from version control. Generated caches are not source artifacts.

---

## 6. Deployment

### Local (primary)
```bash
./run-microservices.sh     # starts all 4 processes, waits for health
```

### Docker (portable)
```bash
docker compose up          # builds image, mounts sp500_data/ as volume
```

The `docker-compose.yml` mounts `./sp500_data` as a host volume so price and analysis caches survive container restarts. Port defaults can be overridden via environment variables (`TICKER_PORT`, `DATA_PORT`, `CALC_PORT`, `UI_PORT`).

### Cron (data refresh)
See `ops/monthly_cron_sample.txt`. The reference schedule runs price+analysis sync weekly (Monday 23:00 CET) and ticker validation monthly (1st of month 06:30 CET).

---

## 7. Test Coverage

| Module | Test file | Coverage focus |
|---|---|---|
| `shared/analysis_engine.py` | `tests/test_analysis_engine.py` | Metrics computation, short-series skip |
| `shared/price_loader.py` | `tests/test_price_cache.py` | Cache roundtrip, stale detection, trim |
| `shared/ticker_provider.py` | `tests/test_ticker_provider.py` | Wikipedia fetch, network fallback |
| `services/ticker_service.py` | `tests/test_ticker_service.py` | API endpoints |
| `services/calculation_service.py` | `tests/test_calculation_service.py` | Optimization, metrics, API |
| `services/analysis_sync_service.py` | `tests/test_analysis_sync_service.py` | Slice loading |
| `services/price_sync_service.py` | `tests/test_price_sync_service.py` | Delta sync logic (8 cases) |
| `services/data_service.py` | `tests/test_data_service.py` | All endpoints (24 cases) |

Run with: `pytest` (configured via `pytest.ini`).

**Known gaps:** `presentation_service.py` (Streamlit UI) has no automated tests. Streamlit's execution model makes unit testing impractical; end-to-end browser testing (Playwright) is the appropriate tool if coverage is required.

---

## 8. Known Issues and Next Steps

See [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) for the full prioritised backlog (T-01 through T-12).

**Resolved since initial architecture:**
- ~~Hardcoded `localhost:800x` URLs~~ → centralised in `shared/settings.py` (F-05)
- ~~Monolithic `app.py`~~ → deleted after microservices reached feature parity

**Current P1 items:**
| Item | Impact |
|---|---|
| T-01: Dockerfile floating dependencies | Builds not reproducible |
| T-02: FastAPI missing return type annotations | Docs/validation incomplete |
| T-03: Silent exception swallowing in data_service | Clients cannot distinguish error types |

# S&P 500 Portfolio Optimizer

A portfolio optimization tool for S&P 500 stocks. Analyzes the full S&P 500 universe and builds optimal portfolios based on return, Sharpe ratio, volatility, drawdown, and momentum scoring.

## Architecture

The project uses a **microservices architecture**.

```
services/
  ticker_service.py       — port 8000  Wikipedia S&P 500 constituent loader
  data_service.py         — port 8001  Price cache + analysis cache (FastAPI)
  calculation_service.py  — port 8002  Portfolio optimization (FastAPI)
  presentation_service.py — port 8501  Streamlit UI

shared/
  analysis_engine.py      — canonical analysis + scoring logic
  price_loader.py         — Parquet-based price cache
  ticker_provider.py      — Wikipedia ticker fetcher with fallback
  models.py               — Pydantic request/response models

sp500_data/               — local data store (parquet + csv)
```

See [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) for full architectural context.

## How to Run

### Microservices (primary)

```bash
./run-microservices.sh
```

Open **http://localhost:8501**.

**First-time setup** — seed the data cache once before using the UI:

```bash
./run-price-sync.sh       # download 5y price history → sp500_data/
./run-analysis-sync.sh    # compute analysis parquet files
```

After the initial sync, `data_service` will auto-seed the analysis cache on future restarts if it detects a stale or missing cache (provided the price cache exists).

### Docker

```bash
docker build -t fin-portfolio .
docker run -p 8501:8501 -p 8000:8000 -p 8001:8001 -p 8002:8002 fin-portfolio
```

Open **http://localhost:8501**. The container entrypoint waits for each service to pass its `/health` check before starting the next one.

## Service Endpoints

| Service | URL | Purpose |
|---|---|---|
| Web UI | http://localhost:8501 | Streamlit portfolio optimizer |
| Ticker API | http://localhost:8000/docs | S&P 500 constituent management |
| Data API | http://localhost:8001/docs | Price + analysis cache |
| Calculation API | http://localhost:8002/docs | Portfolio optimization |

## Key Changes

### UI Improvements (`services/presentation_service.py`)

**Critical UX fixes:**
- Tab order corrected — "🔍 S&P 500 Stock Analyzer" is now the first (default) tab, matching the natural workflow (analyze → optimize)
- Portfolio Optimizer now requires an explicit **🚀 Optimize Portfolio** button click; results persist in session state and are never re-computed silently on widget changes
- Deprecated `st.experimental_rerun()` replaced with `st.rerun()`
- Container reorder trick removed; sections render in visual top-to-bottom order

**Flow & feedback:**
- Stock Analyzer no longer auto-fires an API call on first visit — shows a prompt to click "Analyze Stocks" instead
- "Use Top 20 for Portfolio Optimization" button now shows step-by-step navigation instructions pointing to the optimizer tab
- Cache status table fetched once per session; manual **🔄 Refresh Status** button added
- Expected performance metrics replaced with `st.metric` cards (Expected Return / Volatility / Sharpe) in a 3-column layout

**Polish & consistency:**
- All Optimizer subheaders have consistent emojis (⚙️ 🔒 🎯 ⚖️ 📈 💰 📉)
- Both download buttons styled with `📥` icon and rendered side-by-side
- `st.markdown("---")` replaced with `st.divider()`
- Price Preview moved into a collapsed `st.expander`
- `page_icon="📈"` added to `set_page_config`

---

### Technical Debt Resolution

#### P1 — Code Duplication Eliminated
- `standardize_analysis_columns()` consolidated into `shared/analysis_engine.py` — previously duplicated in `data_service.py` and `presentation_service.py`
- Dead `compute_sp500_analysis()` method removed from `data_service.py` (the service is cache-first; `shared/analysis_engine.py` is the canonical implementation)

#### P1 — Monolith Removed
- `app.py` deleted; `run-monolith.sh` replaced with an error stub directing users to microservices

#### P2 — Service Reliability
- `presentation_service.py` startup retries each backend health check up to 10× with a 2 s backoff and shows a Streamlit spinner instead of failing immediately
- `docker-entrypoint.sh` waits for each service `/health` endpoint before starting the next, aborting cleanly if any service fails to come up
- `data_service.py` auto-seeds the analysis cache on startup via FastAPI `lifespan` if the cache is stale/missing and a price cache is available

# S&P 500 Portfolio Optimizer

A portfolio optimization tool for S&P 500 stocks. Screens the full universe, builds optimal portfolios, forecasts returns with Monte Carlo / CAPM / trend ensemble, backtests forecast accuracy, and compares historical vs forward-looking allocations.

## Quick Start

```bash
# 1. Seed the data cache (first time only)
./run-price-sync.sh          # download 5y price history → sp500_data/
./run-analysis-sync.sh       # compute analysis parquet files

# 2. Start services
./run-microservices.sh       # or: docker compose up

# 3. Open the UI
open http://localhost:8501
```

After the initial sync, `data_service` auto-seeds the analysis cache on future restarts if it detects a stale or missing cache.

## Architecture

```
services/
  ticker_service.py       — port 8000  S&P 500 constituent loader (Wikipedia + CSV)
  data_service.py         — port 8001  Price cache + analysis cache (Parquet)
  calculation_service.py  — port 8002  Portfolio optimization + forecast engine
  presentation_service.py — port 8501  Streamlit UI

shared/
  analysis_engine.py      — canonical analysis + composite scoring
  price_loader.py         — Parquet-based price cache manager
  ticker_provider.py      — Wikipedia ticker fetcher with CSV fallback
  models.py               — Pydantic v2 request/response models
  settings.py             — centralized configuration (env vars)
  config.py               — algorithmic constants

sp500_data/               — local data store (Parquet + CSV + JSON metadata)
```

## Workflow

The UI tabs guide you left to right through a complete analysis:

```
📋 Data Health → 🔍 Analyzer → 📊 Optimizer → 🔮 Forecast → 📅 Backtest → 🔀 Compare → 🔄 Universe
```

1. **Data Health** — verify price and analysis caches are fresh
2. **Analyzer** — score and rank all S&P 500 stocks by composite metric
3. **Optimizer** — build a mean-variance optimal portfolio from selected stocks
4. **Forecast** — run ensemble forecast (MC + CAPM + Trend) on portfolio tickers
5. **Backtest** — validate forecast accuracy on historical rolling windows
6. **Compare** — side-by-side historical vs forecast-optimised allocations

"Use Portfolio Tickers" flows the same stock set through all downstream tabs automatically.

## Service Endpoints

| Service | URL | Purpose |
|---|---|---|
| Web UI | http://localhost:8501 | Streamlit portfolio optimizer |
| Ticker API | http://localhost:8000/docs | S&P 500 constituent management |
| Data API | http://localhost:8001/docs | Price + analysis cache |
| Calculation API | http://localhost:8002/docs | Portfolio optimization + forecast |

## Docker

```bash
docker compose up              # build and start
docker compose up -d           # detached mode
docker compose down            # stop and remove
```

`sp500_data/` is mounted as a volume — caches persist across container restarts. Override ports via environment variables (e.g. `UI_PORT=9501 docker compose up`).

## Documentation

| Document | What it covers |
|---|---|
| [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | Plain-language guide to every algorithm |
| [Feature Catalog](docs/features/README.md) | Shipped and planned features |
| [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) | Open tech debt and architecture recommendations |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and data flow |
| [CLAUDE.md](CLAUDE.md) | AI assistant conventions and guardrails |

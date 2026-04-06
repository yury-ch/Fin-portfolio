# Services — Microservice Layer

## Service Inventory

| Service | File | Port | Role | Depends on |
|---|---|---|---|---|
| Ticker | `ticker_service.py` | 8000 | S&P 500 constituent list | Wikipedia (external) |
| Data | `data_service.py` | 8001 | Price cache + analysis cache (Parquet) | Ticker service, Yahoo Finance |
| Calculation | `calculation_service.py` | 8002 | Optimization + forecast engine | — |
| Presentation | `presentation_service.py` | 8501 | Streamlit UI | Data + Calculation services |

## Import Pattern

Every service file starts with:
```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
This resolves `shared.*` imports. Do not remove it. Import from `shared.models`, `shared.settings`, `shared.config`, `shared.analysis_engine` — never duplicate logic from shared.

## FastAPI Conventions

- Each service creates `app = FastAPI(title="...")` at module level
- Every service has `GET /health` returning `{"status": "healthy"}`
- Error handling: raise `HTTPException` with informative `detail`, never swallow exceptions silently
- Complex responses use the `{"success": bool, "data": ..., "error": ...}` wrapper pattern
- Service URLs come from `shared.settings.settings` — never hardcode `localhost:XXXX`

## presentation_service.py (Streamlit)

- **Largest file** (~2000+ lines) — UI only, no business logic
- All data comes from HTTP calls to backend services via `ServiceClient` class
- Service URLs from `shared.settings.settings`
- Tab rendering follows pattern: helper functions like `_render_forecast_results()`, `_render_backtest_results()`
- Charts use **Plotly** (`plotly.graph_objects`), with `st.line_chart` fallback if Plotly unavailable
- Session state keys defined in `_STATE_DEFAULTS` dict — add new keys there
- Tab order: Data Health → Analyzer → Optimizer → Forecast → Backtest → Compare → Universe
- `_pipeline_status()` helper shows workflow completion at top of each workflow tab

## calculation_service.py

- Contains `CalculationService` (optimization) and `ForecastService` (MC/CAPM/trend)
- `ForecastService` uses seeded `np.random.default_rng` — respect the seed for reproducibility
- `_MAX_ANNUAL_RETURN = 0.50` and `_MIN_ANNUAL_RETURN = -0.90` are critical guardrails — do not change
- `optimize_portfolio` accepts many params; weight pruning (`enforce_min_holdings`) uses thresholds in percent, not decimal
- `compute_stats()` clips mu to `[-0.90, 1.50]` — prevents corporate-action spikes

## data_service.py

- Manages Parquet cache in `sp500_data/`
- Analysis files per period: `sp500_analysis_{period}.parquet`
- `ANALYSIS_PERIODS = ["1y", "2y", "3y", "5y"]` — never change without updating all consumers
- Cache staleness threshold: `PRICE_CACHE_MAX_AGE_HOURS = (7 * 24) + 4`
- `DataService` has `WikipediaTickerProvider` and `PriceCacheManager`

## Adding a New Service

1. Create `services/new_service.py` following the FastAPI pattern
2. Add `GET /health` endpoint
3. Add port to `shared/settings.py` and `.env.example`
4. Add to `docker-compose.yml` and `run-microservices.sh` startup sequence
5. Add to `tests/e2e/conftest.py` `_SERVICE_DEFS`

# fin-portfolio

S&P 500 portfolio analyzer and optimizer — 4 Python microservices (FastAPI + Streamlit).

## CRITICAL: Path Escaping

The project path contains `!` (`!deeplearning-ai-cources`). **Always double-quote paths** in Bash.
For pytest, use the heredoc pattern to avoid history expansion:
```bash
python3 << 'PYEOF'
import subprocess
subprocess.run(["python", "-m", "pytest", "tests/", "-q", "--tb=short"],
               cwd="/Users/chakatouski_yury/Dev_projects/ChatGPT-UseCases/!deeplearning-ai-cources/Claude Code Assistant/fin-portfolio")
PYEOF
```

## Architecture

| Service | File | Port | Role |
|---|---|---|---|
| Ticker | `services/ticker_service.py` | 8000 | S&P 500 constituent list (Wikipedia + CSV cache) |
| Data | `services/data_service.py` | 8001 | Price download (yfinance), analysis cache (Parquet) |
| Calculation | `services/calculation_service.py` | 8002 | Portfolio optimization (PyPortfolioOpt), forecast (MC/CAPM/Trend) |
| Presentation | `services/presentation_service.py` | 8501 | Streamlit UI — calls the other three via HTTP |

- **Shared library**: `shared/` — models, analysis_engine, settings, config, price_loader, ticker_provider
- **Data cache**: `sp500_data/` — Parquet for prices/analysis, CSV for ticker lists, JSON for metadata
- **Boot order**: ticker → data → calculation → presentation
- **Start**: `./run-microservices.sh` or `docker compose up`

## Code Conventions

- **Pydantic v2**: Use `.model_dump()`, **never** `.dict()`. Models inherit `pydantic.BaseModel`.
- **Settings**: `from shared.settings import settings` — module-level singleton, reads `.env` via `pydantic_settings.BaseSettings`.
- **Returns**: Always geometric (log-return) mean: `exp(mean_log * 252) - 1`. Never arithmetic mean for annualized returns.
- **Covariance**: Always Ledoit-Wolf shrinkage via `CovarianceShrinkage(...).ledoit_wolf()`. Never raw sample covariance.
- **Data format**: Parquet for time-series, JSON for metadata. Prices DataFrame has `DatetimeIndex` rows, ticker-string columns.
- **prices_data in API requests**: Dict orientation is `{"YYYY-MM-DD": {"AAPL": 153.2, ...}}` — date strings as outer keys, ticker→price as inner dict.
- **Risk-free rate**: `shared/config.py` defines `DEFAULT_RISK_FREE_RATE = 0.0` for screening. The optimizer accepts a user-supplied rate via request parameter.

## Numerical Guardrails — DO NOT CHANGE

| Guardrail | Value | Location |
|---|---|---|
| Optimizer mu clip | `[-0.90, 1.50]` | `calculation_service.py` `compute_stats()` |
| Forecast max annual return | `0.50` (50%) | `ForecastService._MAX_ANNUAL_RETURN` |
| Forecast min annual return | `-0.90` | `ForecastService._MIN_ANNUAL_RETURN` |
| Ensemble 2Y clip | `[MIN - 0.10, (1+MAX)² - 1]` | `ForecastService.forecast_ticker()` |
| Drawdown | Always negative `[-1, 0]` | `analysis_engine.py` |
| `standardize_analysis_columns` | `pct / 100`, negates drawdown | `shared/analysis_engine.py` |

## Composite Scoring Weights

| Measure | Weight |
|---|---|
| Sharpe Ratio | 25% |
| Annual Return | 20% |
| Max Drawdown | 20% |
| Momentum (50% 3M + 50% 12-1M) | 20% |
| Volatility | 15% |

Do not change these without updating both `analysis_engine.py` and all test assertions.

## Testing

- **Run**: `python -m pytest tests/ -q --tb=short` (from project root)
- **Skip network tests**: `python -m pytest tests/ -q -m "not network"`
- **E2E tests**: `python -m pytest tests/e2e/ -v` (services must be running)
- **Anaconda pytest**: `/Users/chakatouski_yury/anaconda3/bin/pytest`
- Mark any test hitting Yahoo Finance or Wikipedia with `@pytest.mark.network`
- Fixtures use `np.random.seed(42)` for reproducibility

## Streamlit Tab Workflow

Tabs flow left to right — each feeds the next:

```
📋 Data Health → 🔍 Analyzer → 📊 Optimizer → 🔮 Forecast → 📅 Backtest → 🔀 Compare → 🔄 Universe
```

- Optimizer stores `optimization_input_tickers` (full input list) and `optimization_tickers` (non-zero-weight after pruning)
- Downstream tabs (Forecast, Backtest) offer **Use Portfolio Tickers** defaulting to `optimization_input_tickers`
- Pipeline status bar at top of workflow tabs shows completion state

## Common Mistakes to Avoid

1. Using `.dict()` instead of `.model_dump()` on Pydantic models
2. Forgetting to quote paths containing `!` in shell commands
3. Using arithmetic mean for annualized returns (must be geometric)
4. Using sample covariance instead of Ledoit-Wolf shrinkage
5. Making drawdown positive — it is always negative
6. Double-negating drawdown in `standardize_analysis_columns`
7. Changing composite score weights without updating tests
8. Passing prices as dict-of-list instead of dict-of-dict (date-keyed)
9. Not clipping mu or forecast returns to their guardrail bounds
10. Forgetting `@pytest.mark.network` on tests that make real HTTP calls
11. Running services out of boot order (ticker must be up before data)

## Slash Commands

| Command | What it does |
|---|---|
| `/test` | Run full test suite, report pass/fail counts |
| `/fix` | Run tests, diagnose failures, auto-fix, re-run |
| `/status` | Check service health and cache freshness |
| `/sync` | Run price sync + analysis sync pipeline |
| `/logs` | Tail/search structured logs from microservices |
| `/debt` | Show prioritised open technical debt |
| `/features` | Show shipped/planned feature catalog |
| `/validate` | Pre-commit check: tests + convention grep |
| `/review` | Review git diff against project conventions |

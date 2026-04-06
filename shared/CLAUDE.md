# Shared Library — Handle With Care

Changes here ripple to **every service**. Always run the full test suite after modifying any file in `shared/`.

## Module Map

| Module | Role | Key detail |
|---|---|---|
| `models.py` | Pydantic v2 request/response models | Add new fields as `Optional` with defaults to preserve backward compat |
| `analysis_engine.py` | `AnalysisEngine` (screening metrics) + `standardize_analysis_columns()` | Single source of truth for column renaming |
| `settings.py` | `Settings(BaseSettings)` singleton | `from shared.settings import settings` — never instantiate `Settings()` elsewhere |
| `config.py` | Algorithmic constants | Currently just `DEFAULT_RISK_FREE_RATE = 0.0` — changing it changes all Sharpe ratios globally |
| `price_loader.py` | `PriceCacheManager` + `YahooPriceLoader` | Manages Parquet cache under `sp500_data/price_cache/` |
| `ticker_provider.py` | `WikipediaTickerProvider` with CSV fallback | `DEFAULT_SP500_SAMPLE` is 100-ticker hardcoded fallback |

## Column Name Contract

Raw analysis produces **lowercase snake_case**:
`ticker`, `total_return_pct`, `sharpe_ratio`, `volatility_pct`, `max_drawdown_pct`, `current_price`, `composite_score`

`standardize_analysis_columns()` maps to **PascalCase** with unit conversion:
`Ticker`, `Annual_Return`, `Sharpe_Ratio`, `Volatility`, `Max_Drawdown`, `Current_Price`, `Composite_Score`

Conversion rules:
- Percentage columns divided by 100 (e.g., `45.2` → `0.452`)
- Drawdown negated: `Max_Drawdown = -abs(pct) / 100` — always negative
- `_safe_normalize` returns `0.5` for zero-range series (not 0, not 1)
- Volatility and drawdown are inverse-scored (lower is better)

Presentation layer and tests expect **PascalCase**. Never rename without a global search.

## Rules

1. **Never add a model field without a default value** — breaks existing API callers
2. **Never change `standardize_analysis_columns` mapping** without updating all test assertions
3. **`settings` is a module-level singleton** — do not call `Settings()` elsewhere
4. **`DEFAULT_RISK_FREE_RATE = 0.0`** — changing it changes all Sharpe ratios in screening and portfolio metrics
5. **Composite score weights** are in `analysis_engine.py` — Return 20%, Volatility 15%, Sharpe 25%, Drawdown 20%, Momentum 20%
6. **`pd.read_html` requires `io.StringIO`** — use `pd.read_html(io.StringIO(text))`, not `pd.read_html(text)` (pandas FutureWarning)

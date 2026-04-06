# Testing Conventions

## Running Tests

```bash
# Unit tests (no services needed)
python -m pytest tests/ -q --tb=short

# Skip network-dependent tests (Yahoo Finance, Wikipedia)
python -m pytest tests/ -q -m "not network"

# E2E tests (services must be running on ports 8000-8002)
python -m pytest tests/e2e/ -v
```

Always run from project root. The `!` in the project path requires quoting — see root CLAUDE.md for the heredoc workaround.

## Test File Naming

| Type | Location | Pattern |
|---|---|---|
| Unit tests | `tests/test_{service}.py` | One file per service/module |
| E2E tests | `tests/e2e/test_e2e_flows.py` | All cross-service flows in one file |
| Path setup | `tests/conftest.py` | Adds project root to `sys.path` |
| E2E lifecycle | `tests/e2e/conftest.py` | Service subprocess management |

## Fixture Patterns

- `sample_prices`: reproducible price DataFrame — `np.random.seed(42)`, 100 days, 3 tickers
- E2E uses `_gbm_prices()` helper for synthetic GBM test data with known properties
- Unit tests: `TestClient` from Starlette/FastAPI for HTTP endpoint testing
- E2E tests: `httpx.Client` for real HTTP against running services

## Assertion Conventions

- `pytest.approx()` for all float comparisons (weights, returns, ratios)
- Weights must sum to `pytest.approx(1.0)`
- Drawdown must be `<= 0`
- Forecast returns must be within `[-0.90, 0.50]`
- Composite scores must be in `[0, 1]`
- Use absolute tolerance for percentage comparisons: `pytest.approx(0.5, abs=0.01)`

## Adding New Tests

- New service method → add to `test_{service}.py`, use class-based grouping
- New cross-service feature → add to `tests/e2e/test_e2e_flows.py`
- Mark network-dependent tests: `@pytest.mark.network`
- Never depend on real market data for assertions — use synthetic prices with known properties

## Known Gotchas

1. `conftest.py` adds project root to `sys.path` — if imports fail, check this first
2. E2E conftest skips launching a service if its port is already in use — check for stale processes with `lsof -i :8000`
3. Tests import `from services.calculation_service import CalculationService, app` — the module-level `app` must exist
4. Ensemble return cap test: MC median can slightly exceed drift-based cap due to `drift = mu - 0.5*sigma²` convexity — ensemble output is clipped post-hoc

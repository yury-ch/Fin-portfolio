# Repository Guidelines

## Project Structure & Module Organization
This repo delivers the S&P 500 Portfolio Optimizer as both a Streamlit monolith (`app.py`) and a FastAPI microservice stack. Service code lives under `services/` (data, calculation, presentation) and shares contracts in `shared/models.py`. Cached parquet artifacts reside in `sp500_data/`—clear locally when debugging but never commit generated data. Wrapper scripts (`start-microservices.py`, `run-monolith.sh`, `run-microservices.sh`, `stop-monolith.sh`) orchestrate local workflows, while docs such as `MICROSERVICES.md` and `TEST_DOCUMENTATION.md` describe architecture and QA expectations.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create and enter the repo-standard virtual environment.
- `pip install -r requirements.txt` for the monolith; use `requirements-microservices.txt` when touching FastAPI services and `requirements-test.txt` before running pytest.
- `streamlit run app.py`: launch the legacy UI for quick demos.
- `python start-microservices.py` then `streamlit run services/presentation_service.py`: boot the microservices (data on 8001, calculation on 8002) plus the Streamlit UI.
- `python run_tests.py`: execute the curated pytest targets (`TestCalculationService`, `TestEdgeCases`, optional API tests); call `python -m pytest test_calculation_service.py -k <pattern> -v` for focused debugging.

## Coding Style & Naming Conventions
Use Python 3.10+, 4-space indentation, and descriptive docstrings. Favor type hints and Pydantic models defined in `shared/models.py` for request/response payloads. Classes (e.g., `DataService`) are PascalCase, modules stay snake_case, and logging goes through the standard `logging` module with contextual messages. Keep UI code inside `services/presentation_service.py`, grouping Streamlit sections with readable headings.

## Testing Guidelines
Pytest backs all business logic; mirror the existing fixture structure in `test_calculation_service.py` to generate deterministic price data and to patch external clients such as PyPortfolioOpt, yfinance, or FastAPI’s TestClient. Extend `run_tests.py` whenever you add modules so contributors keep a single entry point. Add regression cases for every bug fix, keep skipped tests temporary, and document coverage gaps in `TEST_DOCUMENTATION.md`.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit prefixes (`feat:`, `fix:`, `docs:`). Keep subject lines under 72 characters, explain the reasoning in the body, and separate unrelated changes. Pull requests should link issues, call out whether they affect the monolith, the microservices, or both, and paste `python run_tests.py` output (or justify skips). Include UI screenshots or terminal captures when behavior changes and note follow-up tasks explicitly.

## Security & Configuration Notes
Avoid hardcoding credentials—yfinance currently needs none, but future sources might. Load secrets with `os.getenv`, and document new ports or dependencies in `MICROSERVICES.md`. Review `sp500_data/` before pushing to keep bulky or sensitive parquet files out of the repo, and sanitize any cached analytics that could leak proprietary data.

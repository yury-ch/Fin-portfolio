# F-05 — Centralized Configuration (Environment Variables)

| Field | Value |
|---|---|
| ID | F-05 |
| Status | Shipped |
| Priority | P3 (TD-07) |
| Shipped | 2026-04-01 |

---

## 1. Problem

Service URLs (`http://localhost:8001` etc.) and the data directory path were hardcoded in source files. Changing a port required editing Python source code. This blocked non-local deployments and made Docker port remapping fragile.

---

## 2. Solution

A `shared/settings.py` module exposes a `Settings(BaseSettings)` singleton loaded from environment variables (or a `.env` file). All services import from it.

---

## 3. Configuration Reference

| Env Var | Default | Used By |
|---|---|---|
| `DATA_SERVICE_URL` | `http://localhost:8001` | `presentation_service.py` |
| `CALC_SERVICE_URL` | `http://localhost:8002` | `presentation_service.py` |
| `TICKER_SERVICE_URL` | `http://localhost:8000` | `presentation_service.py` |
| `SP500_DATA_DIR` | `<project_root>/sp500_data` | `data_service.py` |
| `DATA_PORT` | `8001` | `docker-compose.yml` port mapping |
| `CALC_PORT` | `8002` | `docker-compose.yml` port mapping |
| `TICKER_PORT` | `8000` | `docker-compose.yml` port mapping |
| `UI_PORT` | `8501` | `docker-compose.yml` port mapping |

All defaults match the previous hardcoded values — **no behaviour change on existing deployments**.

---

## 4. Usage

**Override a port (local):**
```bash
DATA_SERVICE_URL=http://localhost:9001 streamlit run services/presentation_service.py
```

**Using a `.env` file:**
```bash
cp .env.example .env
# edit .env
./run-microservices.sh
```

**Docker Compose with custom ports:**
```bash
DATA_PORT=9001 CALC_PORT=9002 docker compose up
```

---

## 5. Files Changed

| File | Change |
|---|---|
| `shared/settings.py` | **New** — `pydantic BaseSettings` singleton with env var support and plain-`os.environ` fallback |
| `services/presentation_service.py` | `DATA_SERVICE_URL`, `CALCULATION_SERVICE_URL`, `TICKER_SERVICE_URL` now read from `settings.*` |
| `services/data_service.py` | `DATA_DIR` now reads from `settings.sp500_data_dir` |
| `docker-compose.yml` | `environment` block passes `DATA_SERVICE_URL`, `CALC_SERVICE_URL`, `TICKER_SERVICE_URL`, `SP500_DATA_DIR` |
| `requirements-microservices.txt` | Added `pydantic-settings>=2.0`, `python-dotenv>=1.0` |
| `.env.example` | **New** — template for local overrides |
| `.gitignore` | `.env` added |

---

## 6. Implementation Notes

- `shared/settings.py` includes a `try/except ImportError` fallback to plain `os.environ` reads for environments where `pydantic-settings` is not yet installed
- Pydantic v2 coerces `SP500_DATA_DIR` string to a `pathlib.Path` object automatically
- The `.env` file path is resolved relative to the process CWD at startup; all services are launched from the project root, so resolution is consistent

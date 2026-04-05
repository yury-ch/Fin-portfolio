# shared/settings.py
# Centralised application settings loaded from environment variables.
# Override any value by setting the env var or adding it to a .env file
# at the project root.  All defaults match the existing hardcoded values
# so behaviour is unchanged on existing deployments.

from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

        # ── Service URLs (used by presentation_service.py) ────────────────
        data_service_url: str = "http://localhost:8001"
        calc_service_url: str = "http://localhost:8002"
        ticker_service_url: str = "http://localhost:8000"

        # ── Data directory (used by data_service.py) ──────────────────────
        sp500_data_dir: Path = Path(__file__).resolve().parents[1] / "sp500_data"

        # ── Service ports ─────────────────────────────────────────────────
        ticker_port: int = 8000
        data_port: int = 8001
        calc_port: int = 8002
        ui_port: int = 8501

except ImportError:
    # Fallback for environments where pydantic-settings is not yet installed.
    # Reads the same env vars via os.environ with identical defaults.
    import os

    class Settings:  # type: ignore[no-redef]
        data_service_url: str = os.environ.get("DATA_SERVICE_URL", "http://localhost:8001")
        calc_service_url: str = os.environ.get("CALC_SERVICE_URL", "http://localhost:8002")
        ticker_service_url: str = os.environ.get("TICKER_SERVICE_URL", "http://localhost:8000")
        sp500_data_dir: Path = Path(
            os.environ.get("SP500_DATA_DIR", str(Path(__file__).resolve().parents[1] / "sp500_data"))
        )
        ticker_port: int = int(os.environ.get("TICKER_PORT", "8000"))
        data_port: int = int(os.environ.get("DATA_PORT", "8001"))
        calc_port: int = int(os.environ.get("CALC_PORT", "8002"))
        ui_port: int = int(os.environ.get("UI_PORT", "8501"))


# Module-level singleton — all importers share one instance.
settings = Settings()

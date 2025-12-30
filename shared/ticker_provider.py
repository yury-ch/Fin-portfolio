import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "sp500_data"
DEFAULT_CACHE_FILE = DATA_DIR / "sp500_constituents.csv"
LEGACY_CACHE_FILE = DATA_DIR / "nasdaq_sp500_constituents.csv"

DEFAULT_SP500_SAMPLE = [
    # Top 50 (Mega caps)
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","UNH","XOM",
    "JPM","JNJ","V","PG","HD","CVX","MA","ABBV","PFE","KO",
    "AVGO","COST","PEP","TMO","WMT","MRK","DIS","ADBE","NFLX","CRM",
    "BAC","ACN","LLY","ORCL","WFC","VZ","CMCSA","CSCO","ABT","DHR",
    "NKE","TXN","PM","BMY","UNP","QCOM","RTX","HON","INTC","T",
    
    # Next 50 (Large caps)
    "AMAT","SPGI","CAT","INTU","ISRG","NOW","LOW","GS","MS","AMD",
    "AMGN","BKNG","TJX","BLK","AXP","SYK","VRTX","PLD","GILD","MDLZ",
    "SBUX","TMUS","CVS","CI","LRCX","CB","MO","PYPL","MMC","SO",
    "ZTS","SCHW","FIS","DUK","BSX","CL","ITW","EQIX","AON","CSX",
    "ADI","NOC","MU","SHW","ICE","KLAC","APD","USB","CME","REGN",
    "EMR","PNC","EOG","FCX","GD","NSC","TGT","HUM","COP","PSA"
]


class WikipediaTickerProvider:
    """Fetches and caches the official S&P 500 constituents from Wikipedia."""

    def __init__(self, cache_path: Optional[Path] = None, fallback: Optional[List[str]] = None):
        self.cache_path = cache_path or DEFAULT_CACHE_FILE
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.legacy_cache_path = LEGACY_CACHE_FILE if cache_path is None else None
        self.fallback = list(fallback) if fallback is not None else []

    def fetch_remote_dataframe(self) -> pd.DataFrame:
        """Download and parse the S&P 500 constituents table from Wikipedia."""
        headers = {
            "User-Agent": "S&P-Portfolio-Optimizer/1.0 (+https://github.com/)",
            "Accept": "text/html,application/xhtml+xml",
        }
        response = requests.get(WIKIPEDIA_SP500_URL, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        for table in tables:
            columns = [str(col).strip().lower() for col in table.columns]
            if any(col in ("symbol", "ticker") for col in columns):
                return table
        raise ValueError("Wikipedia response did not contain a constituents table")

    def save_cache(self, df: pd.DataFrame) -> None:
        df.to_csv(self.cache_path, index=False)

    def load_cache(self) -> Optional[pd.DataFrame]:
        paths = [self.cache_path]
        if self.legacy_cache_path and self.legacy_cache_path != self.cache_path:
            paths.append(self.legacy_cache_path)
        for path in paths:
            if path.exists():
                try:
                    return pd.read_csv(path)
                except Exception as exc:
                    logger.warning(f"Failed to read cached constituents: {exc}")
        return None

    @staticmethod
    def normalize_ticker(value: str) -> str:
        value = value.upper().strip()
        return value.replace(".", "-")

    def extract_tickers(self, df: Optional[pd.DataFrame]) -> List[str]:
        if df is None or df.empty:
            return []
        for column in ("Symbol", "symbol", "Ticker", "ticker"):
            if column in df.columns:
                series = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .map(self.normalize_ticker)
                )
                tickers = [ticker for ticker in series if ticker]
                if tickers:
                    # Preserve order while deduplicating
                    return list(dict.fromkeys(tickers))
        return []

    def get_cached_constituents(self) -> List[str]:
        cached = self.extract_tickers(self.load_cache())
        return cached or []

    def get_fallback_constituents(self) -> List[str]:
        fallback = list(self.fallback)
        if fallback:
            self.persist_fallback()
        return fallback

    def get_remote_constituents(self) -> List[str]:
        df = self.fetch_remote_dataframe()
        tickers = self.extract_tickers(df)
        if not tickers:
            raise ValueError("Wikipedia response did not include tickers")
        self.save_cache(df)
        return tickers

    def persist_fallback(self) -> None:
        """Persist fallback tickers so downstream consumers see a file."""
        if not self.fallback:
            return
        try:
            df = pd.DataFrame({"Symbol": list(self.fallback)})
            self.save_cache(df)
        except Exception as exc:
            logger.warning(f"Failed to persist fallback tickers: {exc}")

    def get_constituents(self, force_refresh: bool = False) -> List[str]:
        """Return the current constituents, falling back to cache or defaults."""
        if not force_refresh:
            cached = self.get_cached_constituents()
            if cached:
                return cached
        try:
            return self.get_remote_constituents()
        except Exception as exc:
            logger.warning(f"Failed to fetch Wikipedia constituents: {exc}")
        cached = self.get_cached_constituents()
        if cached:
            return cached
        return self.get_fallback_constituents()

    def get_cache_info(self) -> dict:
        """Return cache metadata for diagnostics."""
        active_path = self.cache_path
        if not active_path.exists() and self.legacy_cache_path and self.legacy_cache_path.exists():
            active_path = self.legacy_cache_path
        info = {
            "path": str(active_path),
            "exists": active_path.exists()
        }
        if info["exists"]:
            stat = active_path.stat()
            info["last_modified"] = datetime.fromtimestamp(stat.st_mtime)
            info["size_bytes"] = stat.st_size
        return info

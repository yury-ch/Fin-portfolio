import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "sp500_data"
PRICE_CACHE_DIR = DATA_DIR / "price_cache"


def chunked(items: Iterable[str], size: int) -> List[List[str]]:
    """Yield lists of size `size` from `items`."""
    bucket: List[str] = []
    for item in items:
        bucket.append(item)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


@dataclass
class PriceCacheMetadata:
    """Metadata stored for each price cache entry."""

    period: str
    interval: str
    last_synced: datetime
    num_tickers: int
    rows: int
    data_through: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "period": self.period,
            "interval": self.interval,
            "last_synced": self.last_synced.isoformat(),
            "num_tickers": self.num_tickers,
            "rows": self.rows,
            "data_through": self.data_through.isoformat() if self.data_through else None,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "PriceCacheMetadata":
        synced = payload.get("last_synced")
        through = payload.get("data_through")
        synced_dt = pd.to_datetime(synced) if synced else None
        through_dt = pd.to_datetime(through) if through else None
        if isinstance(synced_dt, pd.Timestamp):
            synced_dt = synced_dt.to_pydatetime()
        if isinstance(through_dt, pd.Timestamp):
            through_dt = through_dt.to_pydatetime()
        return cls(
            period=payload["period"],
            interval=payload["interval"],
            last_synced=synced_dt or datetime.min.replace(tzinfo=timezone.utc),
            num_tickers=payload.get("num_tickers", 0),
            rows=payload.get("rows", 0),
            data_through=through_dt,
        )


class PriceCacheManager:
    """Manage persisted parquet files that store Yahoo Finance price history."""

    def __init__(self, cache_dir: Optional[Path] = None, metadata_filename: str = "metadata.json"):
        self.cache_dir = cache_dir or PRICE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / metadata_filename
        self._metadata: Dict[str, Dict] = self._load_metadata()

    def cache_path(self, period: str, interval: str) -> Path:
        safe_period = period.replace("/", "_")
        safe_interval = interval.replace("/", "_")
        return self.cache_dir / f"prices_{safe_period}_{safe_interval}.parquet"

    @staticmethod
    def _key(period: str, interval: str) -> str:
        return f"{period}:{interval}"

    def _load_metadata(self) -> Dict[str, Dict]:
        if not self.metadata_path.exists():
            return {}
        try:
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.warning("Failed to read price cache metadata: %s", exc)
            return {}

    def _write_metadata(self) -> None:
        try:
            with self.metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(self._metadata, handle, indent=2)
        except Exception as exc:
            logger.warning("Unable to persist price cache metadata: %s", exc)

    @staticmethod
    def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        cleaned = df.copy()
        cleaned = cleaned.sort_index()
        cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
        cleaned.replace([np.inf, -np.inf], pd.NA, inplace=True)
        cleaned.dropna(axis=1, how="all", inplace=True)
        cleaned = cleaned.ffill().bfill()
        cleaned.dropna(axis=1, how="all", inplace=True)
        cleaned.dropna(axis=0, how="any", inplace=True)
        return cleaned

    def save_prices(
        self,
        df: pd.DataFrame,
        period: str,
        interval: str,
        source: str = "yahoo",
    ) -> PriceCacheMetadata:
        path = self.cache_path(period, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.normalize_frame(df)
        if df is None or df.empty:
            raise ValueError("Cannot save empty price DataFrame.")
        df.to_parquet(path)
        metadata = PriceCacheMetadata(
            period=period,
            interval=interval,
            last_synced=datetime.now(timezone.utc),
            num_tickers=len(df.columns),
            rows=len(df.index),
            data_through=df.index.max().to_pydatetime() if not df.empty else None,
        )
        record = metadata.to_dict()
        record["source"] = source
        self._metadata[self._key(period, interval)] = record
        self._write_metadata()
        return metadata

    def load_prices(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        max_age_hours: Optional[int] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[PriceCacheMetadata]]:
        key = self._key(period, interval)
        raw_meta = self._metadata.get(key)
        if not raw_meta:
            return None, None
        metadata = PriceCacheMetadata.from_dict(raw_meta)
        if max_age_hours is not None:
            limit = timedelta(hours=max_age_hours)
            now = datetime.now(timezone.utc)
            if metadata.last_synced.tzinfo is None:
                metadata.last_synced = metadata.last_synced.replace(tzinfo=timezone.utc)
            if now - metadata.last_synced > limit:
                logger.info(
                    "Price cache for %s/%s is stale (age %.2f hours).",
                    period,
                    interval,
                    (now - metadata.last_synced).total_seconds() / 3600.0,
                )
                return None, metadata
        path = self.cache_path(period, interval)
        if not path.exists():
            return None, metadata
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            df = self.normalize_frame(df)
        except Exception as exc:
            logger.error("Failed to read cached price parquet %s: %s", path, exc)
            return None, metadata
        available = [ticker for ticker in tickers if ticker in df.columns]
        if not available:
            return None, metadata
        subset = df[available].copy()
        subset = self.normalize_frame(subset)
        return subset, metadata

    def cache_summary(self) -> Dict[str, Dict]:
        """Return a dict describing all cached periods."""
        summary: Dict[str, Dict] = {}
        for key, meta in self._metadata.items():
            summary[key] = meta
        return summary


class YahooPriceLoader:
    """Wrapper around yfinance that handles batch downloads."""

    def __init__(self, batch_size: int = 5):
        self.batch_size = max(1, batch_size)

    def fetch_prices(self, tickers: List[str], period: str, interval: str) -> pd.DataFrame:
        successful_data: Dict[str, pd.Series] = {}
        failed_tickers: List[str] = []

        for batch in chunked(tickers, self.batch_size):
            try:
                if len(batch) == 1:
                    ticker = batch[0]
                    logger.info("Downloading data for %s", ticker)
                    data = yf.download(
                        ticker,
                        period=period,
                        interval=interval,
                        auto_adjust=True,
                        prepost=False,
                        threads=False,
                        progress=False,
                    )
                    series = self._extract_close_series(data)
                    if series is not None:
                        successful_data[ticker] = series
                    else:
                        failed_tickers.append(ticker)
                else:
                    logger.info("Downloading batch of %d tickers", len(batch))
                    data = yf.download(
                        " ".join(batch),
                        period=period,
                        interval=interval,
                        group_by="ticker",
                        auto_adjust=True,
                        prepost=False,
                        threads=False,
                        progress=False,
                    )
                    if data.empty:
                        failed_tickers.extend(batch)
                    else:
                        for ticker in batch:
                            try:
                                ticker_data = data[ticker]
                                if ticker_data is not None and "Close" in ticker_data.columns:
                                    successful_data[ticker] = ticker_data["Close"].dropna()
                                else:
                                    failed_tickers.append(ticker)
                            except Exception:
                                failed_tickers.append(ticker)
            except Exception as exc:
                logger.error("Error downloading batch %s: %s", batch, exc)
                failed_tickers.extend(batch)
            finally:
                time.sleep(0.2)

        if not successful_data:
            logger.error("Yahoo download produced no data; returning empty DataFrame")
            return pd.DataFrame()

        prices_df = pd.DataFrame(successful_data)
        prices_df.dropna(how="all", inplace=True)

        if failed_tickers:
            logger.warning("Failed to download data for %d tickers", len(failed_tickers))

        return prices_df

    @staticmethod
    def _extract_close_series(data: pd.DataFrame) -> Optional[pd.Series]:
        if data is None or data.empty:
            return None
        if hasattr(data.columns, "levels"):
            candidates = [col for col in data.columns if "close" in str(col).lower()]
            if candidates:
                series = data[candidates[0]].dropna()
                if isinstance(series, pd.DataFrame):
                    return series.iloc[:, 0]
                return series
        if "Close" in data.columns:
            return data["Close"].dropna()
        if len(data.columns) >= 4:
            for col in data.columns:
                if "close" in str(col).lower():
                    return data[col].dropna()
            return data.iloc[:, 3].dropna()
        return None

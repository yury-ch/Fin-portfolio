"""Incremental Yahoo Finance price synchronizer."""

import argparse
import asyncio
import logging
import os
import sys
from datetime import timedelta
from typing import List, Sequence

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.price_loader import PriceCacheManager, YahooPriceLoader
from shared.ticker_provider import WikipediaTickerProvider, DEFAULT_SP500_SAMPLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("price_sync_service")

# The master cache always stores MASTER_PERIOD/MASTER_INTERVAL from PriceCacheManager
DEFAULT_PERIODS = ["1y", "2y", "3y", "5y"]


class PriceSyncService:
    """Load S&P 500 price data into parquet asynchronously."""

    def __init__(
        self,
        periods: Sequence[str],
        interval: str,
        batch_size: int,
        max_workers: int,
        initial_period: str,
        delta_buffer_days: int,
        force_refresh_tickers: bool = False,
    ):
        self.periods = periods or DEFAULT_PERIODS
        self.interval = interval
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.initial_period = initial_period
        self.delta_buffer_days = max(1, delta_buffer_days)
        self.force_refresh_tickers = force_refresh_tickers
        self.cache_manager = PriceCacheManager()
        self.price_loader = YahooPriceLoader(batch_size=batch_size)
        self.ticker_provider = WikipediaTickerProvider(fallback=list(DEFAULT_SP500_SAMPLE))

    def load_tickers(self) -> List[str]:
        tickers = self.ticker_provider.get_constituents(force_refresh=self.force_refresh_tickers)
        if not tickers:
            logger.warning("Falling back to embedded ticker sample (%d symbols)", len(DEFAULT_SP500_SAMPLE))
            tickers = list(DEFAULT_SP500_SAMPLE)
        return tickers

    def _initial_sync(self, tickers: List[str], period: str) -> pd.DataFrame:
        logger.info("Initial backfill for %s/%s using %s window", period, self.interval, self.initial_period)
        df = self.price_loader.fetch_prices(tickers, self.initial_period, self.interval)
        if df.empty:
            raise RuntimeError(f"Initial Yahoo backfill returned no rows for {period}")
        trimmed = self.cache_manager.trim_history(df, period)
        return trimmed

    def _delta_sync(self, tickers: List[str], period: str, existing: pd.DataFrame) -> pd.DataFrame:
        last_ts = existing.index.max()
        if last_ts is None:
            return self._initial_sync(tickers, period)
        start = (last_ts - timedelta(days=self.delta_buffer_days)).to_pydatetime()
        logger.info(
            "Delta sync for %s/%s from %s (buffer %d days)",
            period,
            self.interval,
            start,
            self.delta_buffer_days,
        )
        delta_df = self.price_loader.fetch_prices(tickers, None, self.interval, start=pd.Timestamp(start))
        if delta_df.empty:
            logger.warning("No delta data returned for %s; keeping existing cache", period)
            merged = existing
        else:
            combined = pd.concat([existing, delta_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            merged = combined.sort_index()
        trimmed = self.cache_manager.trim_history(merged, period)
        return trimmed

    def _sync_period(self, tickers: List[str], period: str):
        df, _ = self.cache_manager.load_full(period, self.interval)
        if df is None or df.empty:
            refreshed = self._initial_sync(tickers, period)
        else:
            refreshed = self._delta_sync(tickers, period, df)
        if not isinstance(refreshed.index, pd.DatetimeIndex):
            refreshed.index = pd.to_datetime(refreshed.index)
        metadata = self.cache_manager.save_prices(refreshed, period, self.interval)
        logger.info(
            "Sync complete for %s/%s (%d rows, %d tickers through %s)",
            period,
            self.interval,
            getattr(metadata, "rows", 0),
            len(refreshed.columns),
            metadata.data_through.isoformat() if metadata.data_through else "unknown",
        )
        return metadata

    async def run(self):
        tickers = self.load_tickers()
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(self.max_workers)

        async def wrapper(period: str):
            async with semaphore:
                return await loop.run_in_executor(None, self._sync_period, tickers, period)

        tasks = [wrapper(period) for period in self.periods]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [res for res in results if isinstance(res, Exception)]
        if failures:
            for failure in failures:
                logger.error("Sync failure: %s", failure)
            raise RuntimeError(f"{len(failures)} sync tasks failed")
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yahoo price synchronizer")
    parser.add_argument(
        "--periods",
        nargs="+",
        default=DEFAULT_PERIODS,
        help="Retention horizons to refresh (default: %(default)s)",
    )
    parser.add_argument("--interval", default="1d", help="Yahoo interval (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=5, help="Tickers per Yahoo request (default: %(default)s)")
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Concurrent periods to sync in parallel (default: %(default)s)",
    )
    parser.add_argument(
        "--initial-period",
        default="5y",
        help="Period used for the very first backfill when no cache exists (default: %(default)s)",
    )
    parser.add_argument(
        "--delta-buffer-days",
        type=int,
        default=5,
        help="Extra overlap (days) when fetching deltas to guard against data gaps (default: %(default)s)",
    )
    parser.add_argument(
        "--force-refresh-tickers",
        action="store_true",
        help="Force refresh the ticker universe from Wikipedia before sync",
    )
    return parser.parse_args()


async def run_from_cli():
    args = parse_args()
    service = PriceSyncService(
        periods=args.periods,
        interval=args.interval,
        batch_size=args.batch_size,
        max_workers=args.workers,
        initial_period=args.initial_period,
        delta_buffer_days=args.delta_buffer_days,
        force_refresh_tickers=args.force_refresh_tickers,
    )
    await service.run()


if __name__ == "__main__":
    asyncio.run(run_from_cli())

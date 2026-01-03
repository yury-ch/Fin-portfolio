"""Standalone Yahoo Finance price synchronizer."""

import argparse
import asyncio
import logging
import os
import sys
from typing import List, Sequence

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.price_loader import PriceCacheManager, YahooPriceLoader
from shared.ticker_provider import WikipediaTickerProvider, DEFAULT_SP500_SAMPLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("price_sync_service")

DEFAULT_PERIODS = ["1y", "2y", "3y"]


class PriceSyncService:
    """Load S&P 500 price data into parquet asynchronously."""

    def __init__(
        self,
        periods: Sequence[str],
        interval: str,
        batch_size: int,
        max_workers: int,
        force_refresh_tickers: bool = False,
    ):
        self.periods = periods or DEFAULT_PERIODS
        self.interval = interval
        self.batch_size = batch_size
        self.max_workers = max_workers
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

    def _sync_period(self, tickers: List[str], period: str):
        logger.info("Starting Yahoo sync for %d tickers (%s / %s)", len(tickers), period, self.interval)
        df = self.price_loader.fetch_prices(tickers, period, self.interval)
        if df.empty:
            raise RuntimeError(f"Yahoo download returned no rows for {period}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        metadata = self.cache_manager.save_prices(df, period, self.interval)
        logger.info(
            "Sync complete for %s/%s (%d rows, %d tickers through %s)",
            period,
            self.interval,
            metadata.rows,
            metadata.num_tickers,
            metadata.data_through.isoformat() if metadata.data_through else "unknown",
        )
        return metadata

    async def run(self) -> List:
        tickers = self.load_tickers()
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = []

        async def wrapper(period: str):
            async with semaphore:
                return await loop.run_in_executor(None, self._sync_period, tickers, period)

        for period in self.periods:
            tasks.append(wrapper(period))
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
        help="Yahoo periods to sync (default: %(default)s)",
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
        force_refresh_tickers=args.force_refresh_tickers,
    )
    await service.run()


if __name__ == "__main__":
    asyncio.run(run_from_cli())

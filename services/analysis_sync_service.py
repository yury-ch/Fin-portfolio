"""Standalone analyzer that converts cached prices into parquet analysis files."""

import argparse
import asyncio
import logging
import os
import sys
from typing import List, Sequence

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.analysis_engine import AnalysisEngine
from shared.price_loader import MASTER_PERIOD, PriceCacheManager
from shared.ticker_provider import DEFAULT_SP500_SAMPLE, WikipediaTickerProvider
from services.data_service import DataService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analysis_sync_service")

DEFAULT_PERIODS = ["1y", "2y", "3y", "5y"]


class AnalysisSyncService:
    """Compute analysis parquet files using cached Yahoo price data."""

    def __init__(
        self,
        periods: Sequence[str],
        interval: str,
        max_workers: int,
        force_refresh_tickers: bool = False,
    ):
        self.periods = periods or DEFAULT_PERIODS
        self.interval = interval
        self.max_workers = max_workers
        self.force_refresh_tickers = force_refresh_tickers
        self.price_cache = PriceCacheManager()
        self.analysis_engine = AnalysisEngine()
        self.ticker_provider = WikipediaTickerProvider(fallback=list(DEFAULT_SP500_SAMPLE))
        self.data_service = DataService()

    def load_tickers(self) -> List[str]:
        tickers = self.ticker_provider.get_constituents(force_refresh=self.force_refresh_tickers)
        if not tickers:
            logger.warning("Falling back to default ticker sample (%d symbols)", len(DEFAULT_SP500_SAMPLE))
            tickers = list(DEFAULT_SP500_SAMPLE)
        return tickers

    def _load_slice(self, tickers: List[str], period: str) -> pd.DataFrame:
        master_df, _ = self.price_cache.load_full(MASTER_PERIOD, self.interval)
        if master_df is None or master_df.empty:
            raise RuntimeError(f"No master price cache available. Run price sync for {MASTER_PERIOD}.")
        available = list(master_df.columns)
        missing = len(set(tickers) - set(available))
        if missing > 0:
            logger.warning("Master price cache missing %d tickers for analysis slice %s", missing, period)
        slice_df = self.price_cache.trim_history(master_df[tickers], period)
        if slice_df.empty:
            raise RuntimeError(f"No data in master cache for {period}. Ensure price sync covered this window.")
        return slice_df

    def _analyze_period(self, tickers: List[str], period: str):
        logger.info("Starting analysis for %s/%s from master %s", period, self.interval, MASTER_PERIOD)
        price_frame = self._load_slice(tickers, period)
        if price_frame is None or price_frame.empty:
            raise RuntimeError(f"No cached prices available for {period}/{self.interval}. Run price sync first.")

        df, latest_ts = self.analysis_engine.analyze_prices(price_frame)
        if df.empty:
            raise RuntimeError(f"Analysis produced no rows for {period}. Ensure price cache has enough history.")

        metadata = self.data_service.save_analysis_data(df, period, latest_ts)
        logger.info(
            "Analysis cache saved for %s/%s (%d stocks, data through %s)",
            period,
            self.interval,
            metadata.get("num_stocks", 0),
            metadata.get("data_through"),
        )
        return metadata

    async def run(self):
        tickers = self.load_tickers()
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(self.max_workers)

        async def wrapper(period: str):
            async with semaphore:
                return await loop.run_in_executor(None, self._analyze_period, tickers, period)

        tasks = [wrapper(period) for period in self.periods]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [res for res in results if isinstance(res, Exception)]
        if failures:
            for failure in failures:
                logger.error("Analysis sync failed: %s", failure)
            raise RuntimeError(f"{len(failures)} analysis tasks failed")
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis cache synchronizer")
    parser.add_argument(
        "--periods",
        nargs="+",
        default=DEFAULT_PERIODS,
        help="Analysis horizons to refresh (default: %(default)s)",
    )
    parser.add_argument("--interval", default="1d", help="Price cache interval to use (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent periods to process (default: %(default)s)")
    parser.add_argument(
        "--force-refresh-tickers",
        action="store_true",
        help="Refresh ticker universe from Wikipedia before running",
    )
    return parser.parse_args()


async def run_from_cli():
    args = parse_args()
    service = AnalysisSyncService(
        periods=args.periods,
        interval=args.interval,
        max_workers=args.workers,
        force_refresh_tickers=args.force_refresh_tickers,
    )
    await service.run()


if __name__ == "__main__":
    asyncio.run(run_from_cli())

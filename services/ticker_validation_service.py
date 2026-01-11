"""Monthly validator comparing cached S&P constituents with Wikipedia."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.ticker_provider import DEFAULT_SP500_SAMPLE, WikipediaTickerProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ticker_validation_service")

VALIDATION_DIR = Path(__file__).resolve().parents[1] / "sp500_data" / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def load_cached_tickers(provider: WikipediaTickerProvider) -> List[str]:
    cached = provider.get_cached_constituents()
    if cached:
        return cached
    logger.warning("No cached constituents found; falling back to default sample")
    return list(DEFAULT_SP500_SAMPLE)


def compare_lists(current: List[str], remote: List[str]) -> Dict:
    current_set = set(current)
    remote_set = set(remote)
    missing = sorted(current_set - remote_set)
    added = sorted(remote_set - current_set)
    equal = current_set == remote_set
    return {
        "match": equal,
        "missing_from_remote": missing,
        "new_in_remote": added,
        "cached_count": len(current),
        "remote_count": len(remote),
    }


def save_report(report: Dict) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = VALIDATION_DIR / f"ticker_validation_{timestamp}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return path


def run_validation(force_refresh_remote: bool = False) -> Dict:
    provider = WikipediaTickerProvider()
    cached = load_cached_tickers(provider)
    try:
        remote = provider.get_remote_constituents() if force_refresh_remote else provider.get_remote_constituents()
    except Exception as exc:
        logger.error("Failed to download Wikipedia constituents: %s", exc)
        raise
    summary = compare_lists(cached, remote)
    summary.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_path": provider.get_cache_info().get("path"),
        }
    )
    report_path = save_report(summary)
    logger.info(
        "Validation completed (%s). Cached=%d Remote=%d Diff=%d/%d Report=%s",
        "match" if summary["match"] else "differs",
        summary["cached_count"],
        summary["remote_count"],
        len(summary["missing_from_remote"]),
        len(summary["new_in_remote"]),
        report_path,
    )
    if not summary["match"]:
        logger.warning("Ticker sets differ. Review report at %s", report_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cached S&P 500 tickers against Wikipedia")
    parser.add_argument(
        "--force-refresh-remote",
        action="store_true",
        help="Force a fresh download from Wikipedia even if cached data exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_validation(force_refresh_remote=args.force_refresh_remote)


if __name__ == "__main__":
    main()

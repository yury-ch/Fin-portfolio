"""Tests for PriceSyncService._delta_sync() — the highest-risk untested path.

A bug in _delta_sync() would silently corrupt the price cache:
wrong merge → wrong analysis results → wrong portfolio recommendations.
"""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.price_sync_service import PriceSyncService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(**kwargs) -> PriceSyncService:
    defaults = dict(
        periods=["1y"],
        interval="1d",
        batch_size=5,
        max_workers=1,
        initial_period="5y",
        delta_buffer_days=5,
    )
    defaults.update(kwargs)
    svc = PriceSyncService(**defaults)
    # Replace cache_manager with a lightweight mock that doesn't touch disk
    svc.cache_manager = MagicMock()
    svc.price_loader = MagicMock()
    return svc


def _daily_prices(start: str, periods: int, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"AAPL": [base + i for i in range(periods)]}, index=idx)


# ---------------------------------------------------------------------------
# _delta_sync tests
# ---------------------------------------------------------------------------

class TestDeltaSync:

    def test_new_rows_are_appended(self):
        """Delta rows that fall after the existing cache end date must be kept."""
        svc = _make_service()

        existing = _daily_prices("2024-01-01", 10)          # Jan 1-10
        delta = _daily_prices("2024-01-08", 7)              # Jan 8-14 (overlaps last 3 days)

        svc.price_loader.fetch_prices.return_value = delta
        svc.cache_manager.trim_history.side_effect = lambda df, _period: df

        result = svc._delta_sync(["AAPL"], "1y", existing)

        # Result must include dates from both existing and delta
        assert result.index.max() == delta.index.max()
        assert result.index.min() == existing.index.min()

    def test_no_duplicate_index_entries(self):
        """Overlapping rows must be deduplicated — no duplicate timestamps."""
        svc = _make_service()

        existing = _daily_prices("2024-01-01", 10)          # 10 rows
        delta = _daily_prices("2024-01-08", 5)              # 5 rows, 3 overlap

        svc.price_loader.fetch_prices.return_value = delta
        svc.cache_manager.trim_history.side_effect = lambda df, _period: df

        result = svc._delta_sync(["AAPL"], "1y", existing)

        assert result.index.is_unique, "Merged index must not have duplicates"

    def test_delta_values_win_on_overlap(self):
        """When a date exists in both existing and delta, delta (keep='last') wins."""
        svc = _make_service()

        existing = _daily_prices("2024-01-01", 10, base=100.0)
        # Override the overlapping day with a different value
        delta = _daily_prices("2024-01-08", 5, base=999.0)

        svc.price_loader.fetch_prices.return_value = delta
        svc.cache_manager.trim_history.side_effect = lambda df, _period: df

        result = svc._delta_sync(["AAPL"], "1y", existing)

        # Jan 8 should have delta value (999.x), not the original (107.0)
        jan8 = pd.Timestamp("2024-01-08")
        assert result.loc[jan8, "AAPL"] >= 999.0

    def test_empty_delta_preserves_existing_cache(self):
        """If Yahoo returns nothing, the existing cache must be returned unchanged."""
        svc = _make_service()

        existing = _daily_prices("2024-01-01", 10)
        svc.price_loader.fetch_prices.return_value = pd.DataFrame()  # empty
        svc.cache_manager.trim_history.side_effect = lambda df, _period: df

        result = svc._delta_sync(["AAPL"], "1y", existing)

        pd.testing.assert_frame_equal(result, existing)

    def test_delta_start_respects_buffer_days(self):
        """fetch_prices must be called with start = last_ts - delta_buffer_days."""
        svc = _make_service(delta_buffer_days=7)

        existing = _daily_prices("2024-01-01", 20)          # last date = Jan 20
        last_ts = existing.index.max()
        expected_start = last_ts - timedelta(days=7)

        svc.price_loader.fetch_prices.return_value = _daily_prices("2024-01-14", 5)
        svc.cache_manager.trim_history.side_effect = lambda df, _period: df

        svc._delta_sync(["AAPL"], "1y", existing)

        call_kwargs = svc.price_loader.fetch_prices.call_args
        actual_start = call_kwargs.kwargs.get("start") or call_kwargs.args[3]
        assert pd.Timestamp(actual_start).normalize() == pd.Timestamp(expected_start).normalize()

    def test_empty_existing_index_returns_empty_without_crash(self):
        """Empty existing cache with NaT max — delta returns empty (no crash).

        Note: pd.DatetimeIndex([]).max() returns NaT, not None, so the
        `if last_ts is None` guard in _delta_sync is dead code. The method
        proceeds to call fetch_prices; if that also returns empty, the existing
        (empty) frame comes back unchanged.
        """
        svc = _make_service()

        empty = pd.DataFrame({"AAPL": pd.Series([], dtype=float)})
        empty.index = pd.DatetimeIndex([])

        svc.price_loader.fetch_prices.return_value = pd.DataFrame()  # also empty
        svc.cache_manager.trim_history.side_effect = lambda df, _period: df

        result = svc._delta_sync(["AAPL"], "1y", empty)

        assert result.empty  # no crash, returns empty frame

    def test_result_is_sorted_by_index(self):
        """Merged result must be sorted ascending by date."""
        svc = _make_service()

        existing = _daily_prices("2024-01-01", 10)
        delta = _daily_prices("2024-01-06", 10)

        svc.price_loader.fetch_prices.return_value = delta
        svc.cache_manager.trim_history.side_effect = lambda df, _period: df

        result = svc._delta_sync(["AAPL"], "1y", existing)

        assert result.index.is_monotonic_increasing

    def test_trim_history_is_called_with_correct_period(self):
        """trim_history must be called with the period string passed to _delta_sync."""
        svc = _make_service()

        existing = _daily_prices("2024-01-01", 10)
        delta = _daily_prices("2024-01-08", 5)

        svc.price_loader.fetch_prices.return_value = delta
        svc.cache_manager.trim_history.side_effect = lambda df, period: df

        svc._delta_sync(["AAPL"], "2y", existing)

        svc.cache_manager.trim_history.assert_called_once()
        _df_arg, period_arg = svc.cache_manager.trim_history.call_args.args
        assert period_arg == "2y"

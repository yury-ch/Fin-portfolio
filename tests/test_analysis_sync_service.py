import pandas as pd
import pytest

from services.analysis_sync_service import AnalysisSyncService


def make_master_prices() -> pd.DataFrame:
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    idx = pd.date_range(end=end, periods=800, freq="D")
    return pd.DataFrame(
        {
            "AAPL": range(len(idx)),
            "MSFT": range(1000, 1000 + len(idx)),
        },
        index=idx,
    )


def test_load_slice_skips_missing_tickers_without_keyerror():
    service = AnalysisSyncService(periods=["1y"], interval="1d", max_workers=1)
    master_df = make_master_prices()
    service.price_cache.load_full = lambda period, interval: (master_df, None)

    result = service._load_slice(["AAPL", "CVNA"], "1y")

    assert not result.empty
    assert "AAPL" in result.columns
    assert "CVNA" not in result.columns


def test_load_slice_raises_when_no_requested_tickers_are_cached():
    service = AnalysisSyncService(periods=["1y"], interval="1d", max_workers=1)
    master_df = make_master_prices()
    service.price_cache.load_full = lambda period, interval: (master_df, None)

    with pytest.raises(RuntimeError, match="does not contain any requested tickers"):
        service._load_slice(["CVNA"], "1y")

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from shared.price_loader import PriceCacheManager


def test_price_cache_roundtrip(tmp_path):
    manager = PriceCacheManager(cache_dir=tmp_path)
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"AAPL": range(5), "MSFT": range(5, 10)}, index=idx)

    metadata = manager.save_prices(df, "1y", "1d")
    assert metadata.num_tickers == 2
    cached, loaded_meta = manager.load_prices(["AAPL", "MSFT"], "1y", "1d", max_age_hours=100)

    assert cached is not None
    assert list(cached.columns) == ["AAPL", "MSFT"]
    assert loaded_meta is not None
    assert loaded_meta.period == "1y"
    assert not cached.isna().any().any()


def test_price_cache_stale_detection(tmp_path):
    manager = PriceCacheManager(cache_dir=tmp_path)
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"AAPL": range(5)}, index=idx)
    manager.save_prices(df, "1y", "1d")

    metadata_path = manager.metadata_path
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["1y:1d"]["last_synced"] = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle)

    reloaded = PriceCacheManager(cache_dir=tmp_path)
    cached, _ = reloaded.load_prices(["AAPL"], "1y", "1d", max_age_hours=24)
    assert cached is None


def test_trim_history(tmp_path):
    manager = PriceCacheManager(cache_dir=tmp_path)
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    idx = pd.date_range(end=end, periods=800, freq="D")
    df = pd.DataFrame({"AAPL": range(len(idx))}, index=idx)
    trimmed = manager.trim_history(df, "1y")
    assert len(trimmed) > 10
    assert (trimmed.index.max() - trimmed.index.min()).days <= 370

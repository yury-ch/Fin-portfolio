"""Endpoint contract tests for data_service.py.

Uses FastAPI TestClient with the real DataService but mocked I/O
(no disk parquet files, no Yahoo network calls).
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    from starlette.testclient import TestClient
except ImportError:
    from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analysis_df(tickers=("AAPL", "MSFT")):
    """Minimal analysis DataFrame as produced by AnalysisEngine."""
    return pd.DataFrame({
        "Ticker": list(tickers),
        "Annual_Return": [0.15, 0.12],
        "Volatility": [0.20, 0.18],
        "Sharpe_Ratio": [0.75, 0.67],
        "Max_Drawdown": [-0.10, -0.08],
        "Recent_3M_Return": [0.05, 0.04],
        "Composite_Score": [0.70, 0.65],
        "Current_Price": [180.0, 320.0],
    })


def _fresh_metadata(period="1y"):
    return {
        "last_updated": datetime.now(),
        "period": period,
        "num_stocks": 2,
        "version": "1.0",
        "data_through": datetime.now(),
    }


def _stale_metadata(period="1y"):
    meta = _fresh_metadata(period)
    meta["last_updated"] = datetime.now() - timedelta(days=30)
    return meta


# ---------------------------------------------------------------------------
# Fixture: TestClient with lifespan startup suppressed
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Return a TestClient with auto-seed startup hook disabled."""
    with patch("services.data_service._trigger_analysis_sync_if_needed"):
        from services.data_service import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_returns_healthy(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# /sp500-tickers
# ---------------------------------------------------------------------------

class TestSp500Tickers:

    def test_returns_list(self, client):
        with patch("services.data_service.data_service.get_sp500_universe",
                   return_value=["AAPL", "MSFT", "GOOGL"]):
            r = client.get("/sp500-tickers")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert "AAPL" in body["data"]

    def test_returns_non_empty_list(self, client):
        r = client.get("/sp500-tickers")
        assert r.status_code == 200
        assert isinstance(r.json()["data"], list)
        assert len(r.json()["data"]) > 0


# ---------------------------------------------------------------------------
# /cache-info
# ---------------------------------------------------------------------------

class TestCacheInfo:

    def test_no_cache_reports_has_cache_false(self, client):
        with patch("services.data_service.data_service.get_cached_analysis",
                   return_value=(None, None)):
            r = client.get("/cache-info")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["data"]["has_cache"] is False
        assert body["data"]["is_stale"] is True

    def test_fresh_cache_reports_not_stale(self, client):
        with patch("services.data_service.data_service.get_cached_analysis",
                   return_value=(_analysis_df(), _fresh_metadata())):
            r = client.get("/cache-info")
        assert r.status_code == 200
        body = r.json()
        assert body["data"]["has_cache"] is True
        assert body["data"]["is_stale"] is False

    def test_stale_cache_reports_is_stale(self, client):
        with patch("services.data_service.data_service.get_cached_analysis",
                   return_value=(_analysis_df(), _stale_metadata())):
            r = client.get("/cache-info")
        assert r.status_code == 200
        body = r.json()
        assert body["data"]["has_cache"] is True
        assert body["data"]["is_stale"] is True


# ---------------------------------------------------------------------------
# /price-cache-info
# ---------------------------------------------------------------------------

class TestPriceCacheInfo:

    def test_returns_summary_dict(self, client):
        fake_summary = {
            "1y:1d": {
                "period": "1y",
                "interval": "1d",
                "last_synced": datetime.now().isoformat(),
                "num_tickers": 500,
                "rows": 252,
                "data_through": datetime.now().isoformat(),
            }
        }
        with patch("services.data_service.data_service.price_cache") as mock_cache:
            mock_cache.cache_summary.return_value = fake_summary
            r = client.get("/price-cache-info")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert "1y:1d" in body["data"]
        assert body["data"]["1y:1d"]["num_tickers"] == 500

    def test_empty_cache_returns_empty_dict(self, client):
        with patch("services.data_service.data_service.price_cache") as mock_cache:
            mock_cache.cache_summary.return_value = {}
            r = client.get("/price-cache-info")
        assert r.status_code == 200
        assert r.json()["data"] == {}


# ---------------------------------------------------------------------------
# DELETE /cache
# ---------------------------------------------------------------------------

class TestClearCache:

    def test_clears_all_returns_count(self, client):
        with patch("services.data_service.data_service.clear_cache", return_value=4):
            r = client.delete("/cache")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["data"]["files_removed"] == 4
        assert body["data"]["period"] is None

    def test_clears_specific_period(self, client):
        with patch("services.data_service.data_service.clear_cache", return_value=2) as mock_clear:
            r = client.delete("/cache?period=1y")
        assert r.status_code == 200
        assert r.json()["data"]["period"] == "1y"
        mock_clear.assert_called_once_with("1y")


# ---------------------------------------------------------------------------
# POST /stock-data
# ---------------------------------------------------------------------------

class TestStockData:

    def test_returns_price_dict(self, client):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        fake_prices = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=idx)
        with patch("services.data_service.data_service.load_prices",
                   return_value=fake_prices):
            r = client.post("/stock-data", json={"tickers": ["AAPL"], "period": "1y", "interval": "1d"})
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert isinstance(body["data"], dict)

    def test_returns_success_false_on_exception(self, client):
        with patch("services.data_service.data_service.load_prices",
                   side_effect=RuntimeError("yahoo down")):
            r = client.post("/stock-data", json={"tickers": ["AAPL"], "period": "1y", "interval": "1d"})
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert "yahoo down" in body["error"]


# ---------------------------------------------------------------------------
# POST /sp500-analysis
# ---------------------------------------------------------------------------

class TestSp500Analysis:

    def test_returns_records_when_cache_is_fresh(self, client):
        with patch("services.data_service.data_service.analyze_sp500_stocks",
                   return_value=(_analysis_df(), _fresh_metadata())):
            r = client.post("/sp500-analysis",
                            json={"tickers": ["AAPL", "MSFT"], "period": "1y"})
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert len(body["data"]["records"]) == 2
        assert body["data"]["metadata"] is not None

    def test_http_exception_swallowed_as_success_false(self, client):
        """BUG: broad `except Exception` in /sp500-analysis catches HTTPException,
        returning HTTP 200 success=False instead of the intended HTTP 503.
        This test documents the current (buggy) behavior.
        Fix: re-raise HTTPException before the generic except, or use
        `except Exception as e: if isinstance(e, HTTPException): raise`.
        """
        from fastapi import HTTPException
        with patch("services.data_service.data_service.analyze_sp500_stocks",
                   side_effect=HTTPException(status_code=503, detail="cache missing")):
            r = client.post("/sp500-analysis",
                            json={"tickers": ["AAPL"], "period": "1y"})
        assert r.status_code == 503

    def test_returns_success_false_on_generic_exception(self, client):
        with patch("services.data_service.data_service.analyze_sp500_stocks",
                   side_effect=RuntimeError("unexpected")):
            r = client.post("/sp500-analysis",
                            json={"tickers": ["AAPL"], "period": "1y"})
        assert r.status_code == 200
        assert r.json()["success"] is False


# ---------------------------------------------------------------------------
# GET /ticker-validation
# ---------------------------------------------------------------------------

class TestTickerValidation:

    def test_no_validation_dir_returns_null_data(self, client, tmp_path):
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/ticker-validation")
        assert r.status_code == 200
        assert r.json()["data"] is None

    def test_returns_latest_snapshot(self, client, tmp_path):
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        snapshot = {"match": True, "cached_count": 503, "remote_count": 503,
                    "timestamp": "2024-01-15T10:00:00+00:00"}
        (val_dir / "ticker_validation_20240115T100000Z.json").write_text(
            json.dumps(snapshot))
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/ticker-validation")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["data"]["match"] is True
        assert body["data"]["cached_count"] == 503


# ---------------------------------------------------------------------------
# GET /sync-report
# ---------------------------------------------------------------------------

class TestSyncReport:

    def test_missing_report_returns_null_data(self, client, tmp_path):
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/sync-report")
        assert r.status_code == 200
        assert r.json()["data"] is None

    def test_returns_report_contents(self, client, tmp_path):
        cache_dir = tmp_path / "price_cache"
        cache_dir.mkdir()
        report = {"run_timestamp": "2024-01-15T10:00:00+00:00",
                  "tickers_attempted": 503, "tickers_failed_any_period": 2}
        (cache_dir / "sync_report.json").write_text(json.dumps(report))
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/sync-report")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["data"]["tickers_attempted"] == 503


# ---------------------------------------------------------------------------
# GET /ticker-changes
# ---------------------------------------------------------------------------

class TestTickerChanges:

    def _write_snapshot(self, val_dir, filename, added, removed, ts):
        snap = {
            "match": not (added or removed),
            "new_in_remote": added,
            "missing_from_remote": removed,
            "timestamp": ts,
            "cached_count": 500,
            "remote_count": 500 + len(added) - len(removed),
        }
        (val_dir / filename).write_text(json.dumps(snap))

    def test_no_snapshots_returns_empty_change_log(self, client, tmp_path):
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/ticker-changes")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["data"]["changes"] == []
        assert body["data"]["total_runs"] == 0

    def test_change_log_contains_added_and_removed(self, client, tmp_path):
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        self._write_snapshot(val_dir, "ticker_validation_20240101T000000Z.json",
                             added=["CRH"], removed=["SOLS"],
                             ts="2024-01-01T00:00:00+00:00")
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/ticker-changes")
        body = r.json()
        assert body["success"] is True
        actions = {e["ticker"]: e["action"] for e in body["data"]["changes"]}
        assert actions["CRH"] == "added"
        assert actions["SOLS"] == "removed"

    def test_deduplication_keeps_earliest_event(self, client, tmp_path):
        """Same ticker appearing in two snapshots → only first occurrence kept."""
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        self._write_snapshot(val_dir, "ticker_validation_20240101T000000Z.json",
                             added=["CRH"], removed=[],
                             ts="2024-01-01T00:00:00+00:00")
        self._write_snapshot(val_dir, "ticker_validation_20240115T000000Z.json",
                             added=["CRH"], removed=[],
                             ts="2024-01-15T00:00:00+00:00")
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/ticker-changes")
        body = r.json()
        crh_entries = [e for e in body["data"]["changes"] if e["ticker"] == "CRH"]
        assert len(crh_entries) == 1
        assert crh_entries[0]["timestamp"] == "2024-01-01T00:00:00+00:00"

    def test_total_runs_reflects_snapshot_count(self, client, tmp_path):
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        for i in range(1, 4):
            self._write_snapshot(
                val_dir, f"ticker_validation_2024010{i}T000000Z.json",
                added=[], removed=[], ts=f"2024-01-0{i}T00:00:00+00:00")
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/ticker-changes")
        assert r.json()["data"]["total_runs"] == 3

    def test_pending_block_reflects_latest_snapshot(self, client, tmp_path):
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        self._write_snapshot(val_dir, "ticker_validation_20240101T000000Z.json",
                             added=[], removed=[], ts="2024-01-01T00:00:00+00:00")
        self._write_snapshot(val_dir, "ticker_validation_20240115T000000Z.json",
                             added=["CVNA", "FIX"], removed=["SOLS"],
                             ts="2024-01-15T00:00:00+00:00")
        with patch("services.data_service.DATA_DIR", tmp_path):
            r = client.get("/ticker-changes")
        pending = r.json()["data"]["pending"]
        assert "CVNA" in pending["added"]
        assert "SOLS" in pending["removed"]
        assert pending["match"] is False

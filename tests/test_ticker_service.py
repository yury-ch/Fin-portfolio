from datetime import datetime
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from services import ticker_service


class FakeProvider:
    def __init__(self):
        self.remote_tickers = ["AAA", "BBB", "CCC"]
        self.cache_tickers = ["CACHED"]
        self.fallback_tickers = ["FALLBACK"]
        self.cache_info: Dict[str, object] = {
            "path": "/tmp/cache.csv",
            "exists": True,
            "last_modified": datetime(2024, 1, 1, 12, 0),
            "size_bytes": 128,
        }
        self.fail_remote = False
        self.remote_calls = 0
        self.cache_calls = 0
        self.fallback_calls = 0

    def get_remote_constituents(self):
        self.remote_calls += 1
        if self.fail_remote:
            raise RuntimeError("network down")
        return list(self.remote_tickers)

    def get_cached_constituents(self):
        self.cache_calls += 1
        return list(self.cache_tickers)

    def get_fallback_constituents(self):
        self.fallback_calls += 1
        return list(self.fallback_tickers)

    def get_cache_info(self):
        return self.cache_info


@pytest.fixture()
def client():
    provider = FakeProvider()
    ticker_service.app.dependency_overrides[ticker_service.get_provider] = lambda: provider
    with TestClient(ticker_service.app) as test_client:
        yield test_client, provider
    ticker_service.app.dependency_overrides.clear()


def test_get_tickers_prefers_wikipedia(client):
    client_app, provider = client
    response = client_app.get("/sp500-tickers")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == len(provider.remote_tickers)
    assert data["tickers"] == provider.remote_tickers
    assert data["source"] == "wikipedia"
    assert provider.remote_calls == 1
    assert provider.cache_calls == 0


def test_falls_back_to_cache_when_allowed(client):
    client_app, provider = client
    provider.fail_remote = True
    response = client_app.get("/sp500-tickers")
    assert response.status_code == 200
    data = response.json()
    assert data["tickers"] == provider.cache_tickers
    assert data["source"] == "cache"
    assert provider.remote_calls == 1
    assert provider.cache_calls == 1


def test_disallow_fallback_returns_error(client):
    client_app, provider = client
    provider.fail_remote = True
    response = client_app.get("/sp500-tickers", params={"allow_fallback": "false"})
    assert response.status_code == 503
    assert provider.remote_calls == 1
    assert provider.cache_calls == 0


def test_refresh_endpoint_requires_wikipedia(client):
    client_app, provider = client
    response = client_app.post("/refresh")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == len(provider.remote_tickers)
    assert payload["source"] == "wikipedia"
    assert provider.remote_calls == 1

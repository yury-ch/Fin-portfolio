from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from shared.ticker_provider import WikipediaTickerProvider

HTML_SAMPLE = """
<table>
    <thead>
        <tr><th>Symbol</th><th>Name</th></tr>
    </thead>
    <tbody>
        <tr><td>AAPL</td><td>Apple Inc.</td></tr>
        <tr><td>BRK.B</td><td>Berkshire Hathaway Inc.</td></tr>
    </tbody>
</table>
"""


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "constituents.csv"


def mock_response(text: str) -> Mock:
    response = Mock()
    response.status_code = 200
    response.text = text
    response.raise_for_status = Mock()
    return response


def test_fetch_from_wikipedia_success(cache_path: Path):
    provider = WikipediaTickerProvider(cache_path=cache_path, fallback=["MSFT"])
    with patch("requests.get", return_value=mock_response(HTML_SAMPLE)):
        tickers = provider.get_constituents(force_refresh=True)
    assert tickers == ["AAPL", "BRK-B"]
    assert cache_path.exists()
    cached = pd.read_csv(cache_path)
    assert list(cached["Symbol"]) == ["AAPL", "BRK.B"]


def test_falls_back_to_cache_when_network_fails(cache_path: Path):
    cache_path.write_text("Symbol,Name\nAAPL,Apple Inc.\nBRK.B,Berkshire Hathaway Inc.\n")
    provider = WikipediaTickerProvider(cache_path=cache_path, fallback=["MSFT"])
    with patch("requests.get", side_effect=Exception("network down")):
        tickers = provider.get_constituents(force_refresh=True)
    assert tickers == ["AAPL", "BRK-B"]


def test_uses_fallback_when_no_cache(cache_path: Path):
    provider = WikipediaTickerProvider(cache_path=cache_path, fallback=["MSFT", "GOOGL"])
    with patch("requests.get", side_effect=Exception("network down")):
        tickers = provider.get_constituents(force_refresh=True)
    assert tickers == ["MSFT", "GOOGL"]
    assert cache_path.exists()
    cached = pd.read_csv(cache_path)
    assert list(cached["Symbol"]) == ["MSFT", "GOOGL"]

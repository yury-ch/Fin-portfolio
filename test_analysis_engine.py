import pandas as pd

from shared.analysis_engine import AnalysisEngine


def make_price_series():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    prices = 100 + (pd.Series(range(200), index=idx) * 0.5)
    return pd.DataFrame({"AAPL": prices, "MSFT": prices * 1.1})


def test_analysis_engine_generates_metrics():
    engine = AnalysisEngine()
    frame = make_price_series()
    df, latest = engine.analyze_prices(frame)

    assert not df.empty
    assert set(df.columns) >= {"Ticker", "Annual_Return", "Volatility", "Composite_Score"}
    assert latest is not None


def test_analysis_engine_skips_short_series():
    engine = AnalysisEngine()
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    frame = pd.DataFrame({"AAPL": pd.Series(range(10), index=idx)})
    df, latest = engine.analyze_prices(frame)
    assert df.empty
    assert latest is None

import numpy as np
import pandas as pd
import pytest

from shared.analysis_engine import AnalysisEngine, standardize_analysis_columns


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


# ---------------------------------------------------------------------------
# P1 regression: drawdown score direction
# ---------------------------------------------------------------------------

def _make_drawdown_frame():
    """Three tickers with different max drawdowns; all other metrics similar."""
    idx = pd.date_range("2023-01-01", periods=200, freq="D")
    base = 100.0 + np.arange(200) * 0.5

    a = base.copy()
    a[50:60] = a[50:60] * 0.97   # ~3% dip — best

    b = base.copy()
    b[50:60] = b[50:60] * 0.85   # ~15% dip

    c = base.copy()
    c[50:60] = c[50:60] * 0.65   # ~35% dip — worst

    return pd.DataFrame(
        {"A_SMALL_DD": a, "B_MED_DD": b, "C_LARGE_DD": c},
        index=idx,
    )


def test_drawdown_score_direction_best_has_highest_score():
    """Ticker with smallest drawdown must receive the highest Drawdown_Score."""
    engine = AnalysisEngine()
    df, _ = engine.analyze_prices(_make_drawdown_frame())
    assert not df.empty
    scores = df.set_index("Ticker")["Drawdown_Score"]
    assert scores["A_SMALL_DD"] > scores["B_MED_DD"] > scores["C_LARGE_DD"], (
        f"Expected A > B > C but got A={scores['A_SMALL_DD']:.3f} "
        f"B={scores['B_MED_DD']:.3f} C={scores['C_LARGE_DD']:.3f}"
    )


def test_drawdown_score_worst_ticker_scores_zero():
    """The ticker with the largest drawdown must receive Drawdown_Score = 0.0."""
    engine = AnalysisEngine()
    df, _ = engine.analyze_prices(_make_drawdown_frame())
    scores = df.set_index("Ticker")["Drawdown_Score"]
    assert abs(scores["C_LARGE_DD"] - 0.0) < 1e-9, (
        f"Worst-drawdown ticker should score 0.0, got {scores['C_LARGE_DD']:.6f}"
    )


def test_drawdown_score_best_ticker_scores_one():
    """The ticker with the smallest drawdown must receive Drawdown_Score = 1.0."""
    engine = AnalysisEngine()
    df, _ = engine.analyze_prices(_make_drawdown_frame())
    scores = df.set_index("Ticker")["Drawdown_Score"]
    assert abs(scores["A_SMALL_DD"] - 1.0) < 1e-9, (
        f"Best-drawdown ticker should score 1.0, got {scores['A_SMALL_DD']:.6f}"
    )


# ---------------------------------------------------------------------------
# P1 regression: standardize_analysis_columns drawdown guard
# ---------------------------------------------------------------------------

def test_decimal_drawdown_not_corrupted():
    """Decimal-form Max_Drawdown (e.g., -0.30) must pass through unchanged."""
    df = pd.DataFrame({"Ticker": ["AAPL"], "Max_Drawdown": [-0.30]})
    result = standardize_analysis_columns(df)
    assert abs(result["Max_Drawdown"].iloc[0] - (-0.30)) < 1e-9, (
        f"Decimal -0.30 was corrupted to {result['Max_Drawdown'].iloc[0]}"
    )


def test_near_minus_one_drawdown_not_corrupted():
    """A drawdown of -0.95 (95% loss) must not trigger the conversion."""
    df = pd.DataFrame({"Ticker": ["X"], "Max_Drawdown": [-0.95]})
    result = standardize_analysis_columns(df)
    assert abs(result["Max_Drawdown"].iloc[0] - (-0.95)) < 1e-9, (
        f"Decimal -0.95 was corrupted to {result['Max_Drawdown'].iloc[0]}"
    )


def test_zero_drawdown_not_corrupted():
    """Max_Drawdown of 0.0 (no drawdown) must pass through unchanged."""
    df = pd.DataFrame({"Ticker": ["X"], "Max_Drawdown": [0.0]})
    result = standardize_analysis_columns(df)
    assert result["Max_Drawdown"].iloc[0] == 0.0


def test_positive_percent_drawdown_is_converted():
    """Legacy positive-percent drawdown (30 for 30% loss) must convert to -0.30."""
    df = pd.DataFrame({
        "Ticker": ["AAPL"],
        "Max_Drawdown": [30.0],
        "Annual_Return": [15.0],
        "Volatility": [18.0],
    })
    result = standardize_analysis_columns(df)
    assert abs(result["Max_Drawdown"].iloc[0] - (-0.30)) < 1e-9, (
        f"Legacy 30.0 should convert to -0.30, got {result['Max_Drawdown'].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# P2 regression: DEFAULT_RISK_FREE_RATE constant
# ---------------------------------------------------------------------------

def test_config_constant_exists_and_is_float():
    from shared.config import DEFAULT_RISK_FREE_RATE
    assert isinstance(DEFAULT_RISK_FREE_RATE, float)


def test_config_constant_value():
    from shared.config import DEFAULT_RISK_FREE_RATE
    assert DEFAULT_RISK_FREE_RATE == 0.0


def test_analysis_engine_imports_constant():
    import inspect
    import shared.analysis_engine as ae_module
    source = inspect.getsource(ae_module)
    assert "DEFAULT_RISK_FREE_RATE" in source


def test_calculation_service_no_hardcoded_rf():
    """calculation_service must not hard-code 0.02 for risk-free rate."""
    import inspect
    import services.calculation_service as cs_module
    source = inspect.getsource(cs_module)
    assert "0.02" not in source, (
        "calculation_service.py must not hard-code 0.02; use DEFAULT_RISK_FREE_RATE"
    )


# ---------------------------------------------------------------------------
# P2 regression: composite weight recalibration
# ---------------------------------------------------------------------------

def test_composite_score_matches_new_weights():
    """Back-calculate composite from component scores; must match 0.20/0.15/0.25/0.20/0.20."""
    engine = AnalysisEngine()
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    prices = pd.DataFrame(
        {"T1": 100 + np.arange(200) * 0.3, "T2": 100 + np.arange(200) * 0.5},
        index=idx,
    )
    df, _ = engine.analyze_prices(prices)
    for _, row in df.iterrows():
        expected = (
            0.20 * row["Return_Score"]
            + 0.15 * row["Vol_Score"]
            + 0.25 * row["Sharpe_Score"]
            + 0.20 * row["Drawdown_Score"]
            + 0.20 * row["Momentum_Score"]
        )
        assert abs(expected - row["Composite_Score"]) < 1e-9, (
            f"Composite_Score {row['Composite_Score']:.6f} != weighted sum {expected:.6f}. "
            "Weights must be 0.20/0.15/0.25/0.20/0.20."
        )


def test_high_momentum_stock_has_higher_momentum_score():
    """Ticker with recent price surge must receive higher Momentum_Score.

    Boost only the last 20 prices (not all 63) so that prices[-63] is unchanged
    and the 3M momentum ratio prices[-1]/prices[-63] is genuinely higher for HIGH.
    """
    engine = AnalysisEngine()
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    base = 100 + np.arange(200, dtype=float) * 0.5
    high_mo = base.copy()
    high_mo[-20:] = high_mo[-20:] * 1.30   # surge in final 20 days
    low_mo = base.copy()
    df, _ = engine.analyze_prices(pd.DataFrame({"HIGH": high_mo, "LOW": low_mo}, index=idx))
    scores = df.set_index("Ticker")["Momentum_Score"]
    assert scores["HIGH"] > scores["LOW"], (
        f"HIGH Momentum_Score {scores['HIGH']:.3f} should exceed LOW {scores['LOW']:.3f}"
    )


# ---------------------------------------------------------------------------
# P2 regression: percentile rank for fat-tailed metrics
# ---------------------------------------------------------------------------

def test_outlier_does_not_collapse_other_return_scores():
    """An extreme outlier must not compress non-outlier Return_Scores to near zero."""
    rng = np.random.RandomState(42)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    moderate_a = pd.Series(100 * (1 + rng.normal(0.0003, 0.01, 200)).cumprod(), index=idx)
    moderate_b = pd.Series(100 * (1 + rng.normal(0.0004, 0.01, 200)).cumprod(), index=idx)
    moderate_c = pd.Series(100 * (1 + rng.normal(0.0005, 0.01, 200)).cumprod(), index=idx)
    outlier    = pd.Series(100 * (1 + rng.normal(0.003,  0.01, 200)).cumprod(), index=idx)
    df, _ = AnalysisEngine().analyze_prices(pd.DataFrame({
        "MOD_A": moderate_a, "MOD_B": moderate_b, "MOD_C": moderate_c, "OUTLIER": outlier,
    }, index=idx))
    scores = df.set_index("Ticker")["Return_Score"]
    non_outlier = [scores["MOD_A"], scores["MOD_B"], scores["MOD_C"]]
    assert all(s > 0.15 for s in non_outlier), (
        f"Non-outlier Return_Scores collapsed: {non_outlier}. "
        "Percentile rank should keep them well above 0."
    )


def test_return_score_reflects_rank_order():
    """Higher-return ticker must have strictly higher Return_Score."""
    rng = np.random.RandomState(7)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    t_low  = pd.Series(100 * (1 + rng.normal(0.0002, 0.01, 200)).cumprod(), index=idx)
    t_high = pd.Series(100 * (1 + rng.normal(0.0008, 0.01, 200)).cumprod(), index=idx)
    df, _ = AnalysisEngine().analyze_prices(pd.DataFrame({"LOW": t_low, "HIGH": t_high}, index=idx))
    scores = df.set_index("Ticker")["Return_Score"]
    assert scores["HIGH"] > scores["LOW"]


def test_momentum_outlier_does_not_collapse_other_momentum_scores():
    """An extreme momentum outlier must not collapse NORM Momentum_Score to near zero."""
    engine = AnalysisEngine()
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    base = 100 + np.arange(200, dtype=float) * 0.3
    normal_mo = base.copy()
    extreme_mo = base.copy()
    extreme_mo[-63:] = extreme_mo[-63:] * 5.0   # +400% in last 3M
    df, _ = engine.analyze_prices(pd.DataFrame({"NORM": normal_mo, "EXTREME": extreme_mo}, index=idx))
    scores = df.set_index("Ticker")["Momentum_Score"]
    assert scores["NORM"] > 0.3, (
        f"NORM Momentum_Score {scores['NORM']:.3f} too low — outlier may be collapsing it."
    )

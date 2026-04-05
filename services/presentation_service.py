import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# Allow `shared` imports when running via `streamlit run services/presentation_service.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.analysis_engine import standardize_analysis_columns  # noqa: E402
from shared.settings import settings  # noqa: E402

DATA_SERVICE_URL = settings.data_service_url
CALCULATION_SERVICE_URL = settings.calc_service_url
TICKER_SERVICE_URL = settings.ticker_service_url
ANALYSIS_PERIODS = ["1y", "2y", "3y", "5y"]
DEFAULT_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","JPM","V"
]


def format_timestamp(value: Optional[str]) -> str:
    if not value or value == "Unknown":
        return "Unknown"
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return str(value)
    return ts.strftime("%Y-%m-%d %H:%M")


def _render_optimization_results(
    result: dict,
    prices_df: pd.DataFrame,
    investment: float,
    risk_free: float,
) -> None:
    """Render all portfolio optimization output sections."""
    weights_series = pd.Series(result['weights']).sort_values(ascending=False)

    with st.expander("📊 Price Preview", expanded=False):
        st.dataframe(prices_df.tail(), use_container_width=True, hide_index=True)

    st.subheader("📈 Expected Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Return", f"{result['expected_annual_return'] * 100:.2f}%")
    m2.metric("Expected Volatility", f"{result['annual_volatility'] * 100:.2f}%")
    m3.metric(f"Sharpe (rf={risk_free:.2f}%)", f"{result['sharpe_ratio']:.2f}")

    st.subheader("⚖️ Optimal Weights")
    weights_display = pd.DataFrame({
        "Ticker": weights_series.index,
        "Weight": (weights_series * 100).round(2).astype(str) + '%'
    })
    st.dataframe(weights_display, use_container_width=True, hide_index=True)

    latest_prices = prices_df[weights_series.index].iloc[-1]
    alloc_usd = weights_series * investment
    allocation_df = pd.DataFrame({
        "Ticker": weights_series.index,
        "Weight": (weights_series * 100).round(2),
        "Last Price": latest_prices.round(2),
        "Allocation (USD)": alloc_usd.round(2)
    })
    st.subheader("💰 Dollar Allocation")
    st.dataframe(allocation_df, use_container_width=True, hide_index=True)

    st.subheader("📉 Historical Performance")
    normalized = prices_df[weights_series.index] / prices_df[weights_series.index].iloc[0] * 100
    try:
        import plotly.graph_objects as _go
        fig = _go.Figure()
        for col in normalized.columns:
            fig.add_trace(_go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Indexed price (start = 100)",
            height=360, margin=dict(l=30, r=10, t=20, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(normalized)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📥 Download Weights (CSV)",
            data=weights_series.to_csv().encode("utf-8"),
            file_name="weights.csv",
            mime="text/csv",
        )
    with dl2:
        st.download_button(
            "📥 Download Full Allocation (CSV)",
            data=allocation_df.to_csv(index=False).encode("utf-8"),
            file_name="allocation.csv",
            mime="text/csv",
        )


def _render_backtest_results(bt: dict) -> None:
    """Render backtest output: summary metrics, scatter, per-window table, per-ticker bar chart."""
    # ── Summary ───────────────────────────────────────────────────────────────
    st.subheader("📊 Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall MAE",      f"{bt['overall_mae'] * 100:.1f} pp")
    m2.metric("Directional Accuracy", f"{bt['overall_hit_rate'] * 100:.1f}%")
    m3.metric("Windows",          str(bt['n_windows']))
    m4.metric("Tickers evaluated", str(bt['n_tickers']))
    st.caption(
        "MAE = mean absolute error between predicted and actual 1-year return (percentage points). "
        "Directional accuracy = % of forecasts that correctly predicted the sign of the return."
    )

    # ── Per-window table ──────────────────────────────────────────────────────
    st.subheader("🗓️ Per-Window Results")
    window_rows = []
    for w in bt["windows"]:
        window_rows.append({
            "Train start":  w["window_start"],
            "Train end":    w["window_end"],
            "Test end":     w["test_end"],
            "Tickers":      w["n_tickers"],
            "MAE (pp)":     round(w["mae"] * 100, 1),
            "Hit rate (%)": round(w["hit_rate"] * 100, 1),
        })
    st.dataframe(pd.DataFrame(window_rows), use_container_width=True, hide_index=True)

    # ── Scatter: predicted vs actual ──────────────────────────────────────────
    st.subheader("🎯 Predicted vs Actual Returns")
    scatter_rows = []
    for i, w in enumerate(bt["windows"]):
        label = f"W{i+1}: {w['window_end'][:7]}→{w['test_end'][:7]}"
        common = set(w["predicted"]) & set(w["actuals"])
        for t in sorted(common):
            scatter_rows.append({
                "Ticker":    t,
                "Window":    label,
                "Predicted": round(w["predicted"][t] * 100, 1),
                "Actual":    round(w["actuals"][t] * 100, 1),
            })
    if scatter_rows:
        scatter_df = pd.DataFrame(scatter_rows)
        try:
            import plotly.graph_objects as _go
            windows_seen = scatter_df["Window"].unique().tolist()
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
            fig = _go.Figure()
            # Perfect-prediction diagonal
            all_vals = scatter_df[["Predicted", "Actual"]].values.flatten()
            lo, hi = float(all_vals.min()) - 5, float(all_vals.max()) + 5
            fig.add_trace(_go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(dash="dash", color="grey", width=1),
                name="Perfect forecast", showlegend=True,
            ))
            for j, win in enumerate(windows_seen):
                sub = scatter_df[scatter_df["Window"] == win]
                fig.add_trace(_go.Scatter(
                    x=sub["Predicted"], y=sub["Actual"],
                    mode="markers+text", text=sub["Ticker"],
                    textposition="top center", textfont=dict(size=9),
                    marker=dict(size=8, color=colors[j % len(colors)]),
                    name=win,
                ))
            fig.update_layout(
                xaxis_title="Predicted 1Y return (%)",
                yaxis_title="Actual 1Y return (%)",
                height=420, margin=dict(l=30, r=10, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(scatter_df, use_container_width=True, hide_index=True)

    # ── Per-ticker hit rate ───────────────────────────────────────────────────
    st.subheader("📈 Per-Ticker Directional Accuracy")
    ticker_hr = bt.get("per_ticker_hit_rate", {})
    ticker_mae = bt.get("per_ticker_mae", {})
    if ticker_hr:
        ticker_df = pd.DataFrame({
            "Ticker":       list(ticker_hr.keys()),
            "Hit rate (%)": [round(v * 100, 1) for v in ticker_hr.values()],
            "MAE (pp)":     [round(ticker_mae.get(t, 0) * 100, 1) for t in ticker_hr.keys()],
        }).sort_values("Hit rate (%)", ascending=False).reset_index(drop=True)
        try:
            import plotly.graph_objects as _go
            bar_colors = ["#00CC96" if v >= 50 else "#EF553B" for v in ticker_df["Hit rate (%)"]]
            fig2 = _go.Figure(_go.Bar(
                x=ticker_df["Ticker"], y=ticker_df["Hit rate (%)"],
                marker_color=bar_colors, text=ticker_df["Hit rate (%)"],
                textposition="outside",
            ))
            fig2.add_hline(y=50, line_dash="dash", line_color="grey", annotation_text="50% baseline")
            fig2.update_layout(
                yaxis_title="Hit rate (%)", yaxis=dict(range=[0, 105]),
                height=320, margin=dict(l=30, r=10, t=20, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            pass
        st.dataframe(ticker_df, use_container_width=True, hide_index=True)

    # ── Download ───────────────────────────────────────────────────────────────
    if scatter_rows:
        st.download_button(
            "📥 Download Results (CSV)",
            data=pd.DataFrame(scatter_rows).to_csv(index=False).encode("utf-8"),
            file_name="backtest_results.csv",
            mime="text/csv",
        )


def _render_comparison(
    hist_result: dict,
    fc_result: dict,
    prices_df: pd.DataFrame,
    investment: float,
    risk_free: float,
) -> None:
    """Render side-by-side historical vs forecast-optimised portfolio comparison."""
    # ── Metrics ───────────────────────────────────────────────────────────────
    st.subheader("📈 Performance Comparison")
    col_h, col_f = st.columns(2)
    with col_h:
        st.markdown("**📊 Historical**")
        st.metric("Expected Return",    f"{hist_result['expected_annual_return'] * 100:.2f}%")
        st.metric("Expected Volatility", f"{hist_result['annual_volatility'] * 100:.2f}%")
        st.metric(f"Sharpe (rf={risk_free:.2f}%)", f"{hist_result['sharpe_ratio']:.2f}")
    with col_f:
        ret_delta    = (fc_result['expected_annual_return'] - hist_result['expected_annual_return']) * 100
        vol_delta    = (fc_result['annual_volatility']      - hist_result['annual_volatility'])      * 100
        sharpe_delta =  fc_result['sharpe_ratio']           - hist_result['sharpe_ratio']
        st.markdown("**🔮 Forecast-Optimised**")
        st.metric("Expected Return",     f"{fc_result['expected_annual_return'] * 100:.2f}%",
                  delta=f"{ret_delta:+.2f}%")
        st.metric("Expected Volatility", f"{fc_result['annual_volatility'] * 100:.2f}%",
                  delta=f"{vol_delta:+.2f}%", delta_color="inverse")
        st.metric(f"Sharpe (rf={risk_free:.2f}%)", f"{fc_result['sharpe_ratio']:.2f}",
                  delta=f"{sharpe_delta:+.2f}")

    # ── Weight shift table ─────────────────────────────────────────────────────
    st.subheader("⚖️ Weight Shifts")
    hist_w = pd.Series(hist_result['weights'])
    fc_w   = pd.Series(fc_result['weights'])
    all_tickers = hist_w.index.union(fc_w.index)
    shift_df = pd.DataFrame({
        "Ticker":         all_tickers,
        "Historical (%)": (hist_w.reindex(all_tickers, fill_value=0) * 100).round(2).values,
        "Forecast (%)":   (fc_w.reindex(all_tickers,   fill_value=0) * 100).round(2).values,
    })
    shift_df["Δ (pp)"] = (shift_df["Forecast (%)"] - shift_df["Historical (%)"]).round(2)
    shift_df = shift_df.sort_values("Δ (pp)", ascending=False).reset_index(drop=True)
    st.dataframe(shift_df, use_container_width=True, hide_index=True)

    # ── Bar chart ──────────────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as _go
        fig = _go.Figure([
            _go.Bar(x=shift_df["Ticker"], y=shift_df["Historical (%)"], name="Historical"),
            _go.Bar(x=shift_df["Ticker"], y=shift_df["Forecast (%)"],   name="Forecast"),
        ])
        fig.update_layout(
            barmode="group",
            xaxis_title="Ticker", yaxis_title="Weight (%)",
            height=360, margin=dict(l=30, r=10, t=20, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    # ── Download ───────────────────────────────────────────────────────────────
    st.download_button(
        "📥 Download Comparison (CSV)",
        data=shift_df.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_comparison.csv",
        mime="text/csv",
    )


class ServiceClient:
    """Thin wrapper around the FastAPI services."""

    def check_service_health(self, service_url: str) -> bool:
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_sp500_tickers(self) -> List[str]:
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/sp500-tickers", timeout=10)
            payload = response.json()
            if payload.get("success"):
                return payload.get("data", [])
        except Exception as exc:
            st.error(f"Failed to fetch S&P tickers: {exc}")
        return []

    def get_cache_info(self) -> Dict[str, dict]:
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/cache-info", timeout=10)
            payload = response.json()
            if payload.get("success"):
                return payload.get("data", {}).get("periods", {}) or {}
        except Exception as exc:
            st.warning(f"Unable to retrieve cache summary: {exc}")
        return {}

    def clear_cache(self, period: Optional[str]) -> bool:
        try:
            params = {"period": period} if period else {}
            response = requests.delete(f"{DATA_SERVICE_URL}/cache", params=params, timeout=30)
            payload = response.json()
            return payload.get("success", False)
        except Exception as exc:
            st.error(f"Cache clear failed: {exc}")
            return False

    def get_sp500_analysis(self, tickers: List[str], period: str, force_refresh: bool) -> Tuple[pd.DataFrame, dict]:
        try:
            payload = {
                "tickers": tickers,
                "period": period,
                "force_refresh": force_refresh
            }
            response = requests.post(f"{DATA_SERVICE_URL}/sp500-analysis", json=payload, timeout=600)
            data = response.json()
            if not response.ok:
                raise RuntimeError(data.get("detail") or data.get("error") or f"HTTP {response.status_code}")
            if not data.get("success"):
                raise RuntimeError(data.get("error") or "Unknown analysis error")
            content = data.get("data", {})
            df = pd.DataFrame(content.get("records", []))
            df = standardize_analysis_columns(df)
            metadata = content.get("metadata", {}) or {}
            return df, metadata
        except Exception as exc:
            st.error(f"Analysis request failed: {exc}")
            return pd.DataFrame(), {}

    def get_stock_data(self, tickers: List[str], period: str, interval: str) -> pd.DataFrame:
        try:
            payload = {"tickers": tickers, "period": period, "interval": interval}
            response = requests.post(f"{DATA_SERVICE_URL}/stock-data", json=payload, timeout=180)
            data = response.json()
            if not data.get("success"):
                raise RuntimeError(data.get("error", "Unknown data error"))
            raw = data.get("data", {})
            frame = pd.DataFrame.from_dict(raw, orient="index")
            frame.index = pd.to_datetime(frame.index)
            frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
            frame = frame.ffill().dropna(how="all")
            return frame
        except Exception as exc:
            st.error(f"Price download failed: {exc}")
            return pd.DataFrame()

    def get_price_cache_info(self) -> dict:
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/price-cache-info", timeout=10)
            payload = response.json()
            if payload.get("success"):
                return payload.get("data", {})
        except Exception as exc:
            st.warning(f"Unable to retrieve price cache info: {exc}")
        return {}

    def get_ticker_validation(self) -> Optional[dict]:
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/ticker-validation", timeout=10)
            payload = response.json()
            if payload.get("success"):
                return payload.get("data")
        except Exception as exc:
            st.warning(f"Unable to retrieve ticker validation: {exc}")
        return None

    def get_ticker_health(self) -> dict:
        try:
            response = requests.get(f"{TICKER_SERVICE_URL}/health", timeout=5)
            return response.json()
        except Exception:
            return {}

    def get_sync_report(self) -> Optional[dict]:
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/sync-report", timeout=10)
            payload = response.json()
            if payload.get("success"):
                return payload.get("data")
        except Exception as exc:
            st.warning(f"Unable to retrieve sync report: {exc}")
        return None

    def get_ticker_changes(self) -> Optional[dict]:
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/ticker-changes", timeout=10)
            payload = response.json()
            if payload.get("success"):
                return payload.get("data")
        except Exception as exc:
            st.warning(f"Unable to retrieve ticker changes: {exc}")
        return None

    def optimize_portfolio(
        self,
        prices: pd.DataFrame,
        tickers: List[str],
        investment: float,
        objective: str,
        risk_free: float,
        target_return: float,
        max_weight: float,
        l2_reg: float,
        min_weight_threshold: float,
        min_holdings: int,
        expected_returns_override: Optional[Dict[str, float]] = None,
    ) -> Optional[dict]:
        try:
            payload = {
                "tickers": tickers,
                "prices_data": {
                    str(idx): {col: float(val) for col, val in row.items()}
                    for idx, row in prices.round(6).iterrows()
                },
                "investment_amount": investment,
                "objective": objective,
                "risk_free": risk_free,
                "target_return": target_return,
                "max_weight": max_weight,
                "l2_reg": l2_reg,
                "min_weight_threshold": min_weight_threshold,
                "min_holdings": min_holdings,
                "expected_returns_override": expected_returns_override,
            }
            response = requests.post(f"{CALCULATION_SERVICE_URL}/optimize-portfolio", json=payload, timeout=120)
            data = response.json()
            if not data.get("success"):
                raise RuntimeError(data.get("error", "Calculation service error"))
            return data.get("data")
        except Exception as exc:
            st.error(f"Optimization call failed: {exc}")
            return None

    def get_forecast(
        self,
        tickers: List[str],
        prices: pd.DataFrame,
        risk_free_pct: float = 4.0,
        n_simulations: int = 1000,
    ) -> Optional[dict]:
        """Call /forecast-returns on the calculation service."""
        try:
            payload = {
                "tickers": tickers,
                "prices_data": {
                    str(idx): {col: float(val) for col, val in row.items()}
                    for idx, row in prices.round(6).iterrows()
                },
                "top_n": len(tickers),
                "n_simulations": n_simulations,
                "risk_free_annual": risk_free_pct / 100.0,
                "spy_ticker": "SPY",
            }
            response = requests.post(
                f"{CALCULATION_SERVICE_URL}/forecast-returns",
                json=payload,
                timeout=180,
            )
            data = response.json()
            if not data.get("success"):
                raise RuntimeError(data.get("error", "Forecast service error"))
            return data.get("data")
        except Exception as exc:
            st.error(f"Forecast call failed: {exc}")
            return None

    def get_backtest(
        self,
        tickers: List[str],
        prices: pd.DataFrame,
        n_windows: int = 3,
        n_simulations: int = 500,
        risk_free_pct: float = 4.0,
    ) -> Optional[dict]:
        """Call /backtest on the calculation service."""
        try:
            payload = {
                "tickers": tickers,
                "prices_data": {
                    str(idx): {col: float(val) for col, val in row.items()}
                    for idx, row in prices.round(6).iterrows()
                },
                "n_windows":        n_windows,
                "train_days":       252,
                "test_days":        252,
                "n_simulations":    n_simulations,
                "risk_free_annual": risk_free_pct / 100.0,
                "spy_ticker":       "SPY",
            }
            response = requests.post(
                f"{CALCULATION_SERVICE_URL}/backtest",
                json=payload,
                timeout=300,
            )
            data = response.json()
            if not data.get("success"):
                raise RuntimeError(data.get("error", "Backtest service error"))
            return data.get("data")
        except Exception as exc:
            st.error(f"Backtest call failed: {exc}")
            return None


# ── App bootstrap ────────────────────────────────────────────────────────────

client = ServiceClient()

st.set_page_config(page_title="S&P Portfolio Optimizer", layout="wide", page_icon="📈")
st.title("📈 S&P 500 Portfolio Optimizer")
st.caption("Build an optimal S&P portfolio given a risk–return preference, investment amount, and horizon.")

# Session state initialisation
_STATE_DEFAULTS = {
    'analysis_cache': {},
    'cache_metadata': {},
    'recommended_tickers': ",".join(DEFAULT_TICKERS),
    'sp500_tickers': [],
    'optimization_result': None,
    'optimization_prices': None,
    'health_data': None,
    'universe_changes': None,
    'forecast_result': None,
    'forecast_prices': None,
    'forecast_expected_returns': None,  # Dict[str, float] populated after a forecast run
    'use_forecast_returns': False,       # one-shot flag: auto-select "Use Forecast Returns" radio
    'optimization_last_params': None,   # Dict: snapshot of optimizer settings at last run
    'optimization_input_tickers': [],    # List[str]: full ticker list fed into the optimizer (before weight pruning)
    'optimization_tickers': [],         # List[str]: non-zero-weight tickers after optimizer pruning
    'forecast_tickers': [],             # List[str]: tickers from last forecast run
    'comparison_results': None,         # Dict: {'historical', 'forecast', 'prices', 'investment', 'risk_free'}
    'backtest_result': None,            # Dict: backtest response from /backtest endpoint
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Service health checks ────────────────────────────────────────────────────

_HEALTH_RETRIES = 10
_HEALTH_BACKOFF = 2.0  # seconds between retries


def _wait_for_service(url: str, name: str, start_cmd: str) -> bool:
    """Poll a service /health endpoint, showing a spinner while waiting."""
    if client.check_service_health(url):
        return True
    with st.spinner(f"Waiting for {name} to start…"):
        for _ in range(_HEALTH_RETRIES - 1):
            time.sleep(_HEALTH_BACKOFF)
            if client.check_service_health(url):
                return True
    st.error(
        f"⚠️ {name} did not become healthy after {_HEALTH_RETRIES} attempts. "
        f"Start it with: `{start_cmd}`"
    )
    return False


if not _wait_for_service(DATA_SERVICE_URL, "Data service (port 8001)", "python services/data_service.py"):
    st.stop()
if not _wait_for_service(CALCULATION_SERVICE_URL, "Calculation service (port 8002)", "python services/calculation_service.py"):
    st.stop()

if not st.session_state['sp500_tickers']:
    fetched = client.get_sp500_tickers()
    st.session_state['sp500_tickers'] = fetched if fetched else DEFAULT_TICKERS

# ── Shared helpers ───────────────────────────────────────────────────────────

def _status_badge(age_hours: Optional[float]) -> str:
    if age_hours is None:
        return "⚫ Missing"
    if age_hours < 24:
        return "🟢 Fresh"
    if age_hours < 168:
        return "🟡 Recent"
    return "🔴 Stale"


def refresh_cache_summary() -> None:
    st.session_state['cache_metadata'].update(client.get_cache_info())


def get_analysis(period: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, dict]:
    cache_entry = st.session_state['analysis_cache'].get(period)
    if cache_entry and not force_refresh:
        return cache_entry['df'], cache_entry.get('metadata', {})
    df, metadata = client.get_sp500_analysis(st.session_state['sp500_tickers'], period, force_refresh)
    if not df.empty:
        st.session_state['analysis_cache'][period] = {'df': df, 'metadata': metadata}
        if metadata:
            st.session_state['cache_metadata'][period] = metadata
    return df, metadata


# ── Pipeline status helper ───────────────────────────────────────────────────

def _pipeline_status() -> None:
    """Compact one-line pipeline progress bar shown at the top of workflow tabs."""
    input_tickers = st.session_state.get('optimization_input_tickers', [])
    opt_tickers = st.session_state.get('optimization_tickers', [])
    fc_tickers = st.session_state.get('forecast_tickers', [])
    bt_result = st.session_state.get('backtest_result')
    parts = []
    if input_tickers:
        pruned = f" → {len(opt_tickers)} after pruning" if opt_tickers and len(opt_tickers) != len(input_tickers) else ""
        parts.append(f"📊 Portfolio: {len(input_tickers)} stocks{pruned} ✅")
    else:
        parts.append("📊 Portfolio: _not run_")
    if fc_tickers:
        parts.append(f"🔮 Forecast: {len(fc_tickers)} stocks ✅")
    else:
        parts.append("🔮 Forecast: _not run_")
    if bt_result:
        n_win = bt_result.get('n_windows', '?')
        parts.append(f"📅 Backtest: {n_win} windows ✅")
    else:
        parts.append("📅 Backtest: _not run_")
    st.caption("**Pipeline:** " + " | ".join(parts))


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_health, tab_analyzer, tab_optimizer, tab_forecast, tab_backtest, tab_compare, tab_universe = st.tabs([
    "📋 Data Health",
    "🔍 S&P 500 Analyzer",
    "📊 Optimizer",
    "🔮 Forecast",
    "📅 Backtest",
    "🔀 Compare",
    "🔄 Universe",
])

# ── Tab: Stock Analyzer ──────────────────────────────────────────────────────

with tab_analyzer:
    st.header("🔍 S&P 500 Stock Analyzer")
    universe_size = len(st.session_state['sp500_tickers'])
    st.caption(f"Analyze the full S&P 500 universe ({universe_size} tickers) via the data microservice.")

    # Load cache summary once per session; user can manually refresh
    if not st.session_state['cache_metadata']:
        refresh_cache_summary()

    period_options = ANALYSIS_PERIODS
    cache_summary = st.session_state['cache_metadata']

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    with ctrl1:
        analysis_period = st.selectbox("Analysis period", period_options, index=0, key="analysis_period")
    with ctrl2:
        analyze_button = st.button("🚀 Analyze Stocks", type="primary", use_container_width=True)
    with ctrl3:
        force_refresh = st.button("🔄 Force Refresh")
    with ctrl4:
        if st.button("🔄 Refresh", key="refresh_analyzer"):
            refresh_cache_summary()
            st.rerun()

    status_rows = []
    for period in period_options:
        meta = cache_summary.get(period)
        if meta:
            status_rows.append({
                "Period": period,
                "Last Analysis": format_timestamp(meta.get("last_updated")),
                "Prices Through": format_timestamp(meta.get("data_through")),
                "Stocks": meta.get("num_stocks", "—")
            })
        else:
            status_rows.append({"Period": period, "Last Analysis": "Not run", "Prices Through": "—", "Stocks": "—"})
    st.table(pd.DataFrame(status_rows).set_index("Period"))

    analysis_df = st.session_state['analysis_cache'].get(analysis_period, {}).get('df')
    analysis_meta = st.session_state['analysis_cache'].get(analysis_period, {}).get('metadata')

    if analyze_button or force_refresh:
        with st.spinner("Requesting analysis from data service..."):
            analysis_df, analysis_meta = get_analysis(analysis_period, force_refresh=force_refresh)

    if analysis_df is not None and not analysis_df.empty:
        st.subheader("🏆 Top 20 Stock Recommendations")
        st.caption("Composite score blends return, Sharpe, low volatility/drawdown, and recent momentum.")
        if analysis_meta:
            ts = format_timestamp(analysis_meta.get("last_updated"))
            thru = format_timestamp(analysis_meta.get("data_through"))
            st.caption(f"Last analysis: {ts} | Prices through: {thru}")

        base_cols = ['Ticker', 'Annual_Return', 'Sharpe_Ratio', 'Volatility',
                     'Max_Drawdown', 'Recent_3M_Return', 'Current_Price', 'Composite_Score']
        has_12m = 'Recent_12M_Return' in analysis_df.columns
        display_cols = base_cols[:6] + (['Recent_12M_Return'] if has_12m else []) + base_cols[6:]
        display_df = analysis_df[display_cols].head(20).copy()
        display_df['Annual_Return'] = (display_df['Annual_Return'] * 100).round(2).astype(str) + '%'
        display_df['Volatility'] = (display_df['Volatility'] * 100).round(2).astype(str) + '%'
        display_df['Max_Drawdown'] = (display_df['Max_Drawdown'] * 100).round(2).astype(str) + '%'
        display_df['Recent_3M_Return'] = (display_df['Recent_3M_Return'] * 100).round(2).astype(str) + '%'
        if has_12m:
            display_df['Recent_12M_Return'] = (display_df['Recent_12M_Return'] * 100).round(2).astype(str) + '%'
        display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].round(2)
        display_df['Current_Price'] = '$' + display_df['Current_Price'].round(2).astype(str)
        display_df['Composite_Score'] = display_df['Composite_Score'].round(3)
        col_labels = ['Ticker', 'Annual Return', 'Sharpe Ratio', 'Volatility',
                      'Max Drawdown', '3M Return'] + (['12M Return'] if has_12m else []) + ['Price', 'Score']
        display_df.columns = col_labels
        st.dataframe(display_df, use_container_width=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            if st.button("📈 Use Top 20 for Portfolio Optimization", key="use_top20"):
                st.session_state['recommended_tickers'] = ",".join(analysis_df['Ticker'].head(20).tolist())
                st.info(
                    "Top 20 tickers ready. Switch to the **📊 Portfolio Optimizer** tab, "
                    "choose **Manual Entry**, then click **🚀 Optimize Portfolio**."
                )
        with col_b:
            csv_data = analysis_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Analysis",
                data=csv_data,
                file_name=f"sp500_analysis_{analysis_period}.csv",
                mime="text/csv"
            )

        with st.expander("🗄️ Data Management", expanded=False):
            if st.button("🧹 Clear Cached Analysis for this Period"):
                if client.clear_cache(analysis_period):
                    st.session_state['analysis_cache'].pop(analysis_period, None)
                    st.session_state['cache_metadata'].pop(analysis_period, None)
                    st.success("Cache cleared.")
                else:
                    st.error("Failed to clear cache.")
    else:
        st.info("Click **🚀 Analyze Stocks** to load analysis for the selected period.")

# ── Tab: Portfolio Optimizer ─────────────────────────────────────────────────

with tab_optimizer:
    st.header("📊 Portfolio Optimizer")
    st.caption("Adjust settings, choose a universe, and let the services build an allocation for you.")
    _pipeline_status()

    # ── ⚙️ Analysis Settings ─────────────────────────────────────────────────
    st.subheader("⚙️ Analysis Settings")
    settings_col1, settings_col2, settings_col3 = st.columns(3)
    with settings_col1:
        horizon = st.selectbox(
            "Analysis period / lookback",
            ["1y", "2y", "3y", "5y"],
            index=0,
            key="micro_horizon"
        )
        interval = st.selectbox(
            "Price frequency",
            ["1d", "1wk", "1mo"],
            index=0,
            key="micro_interval"
        )
    with settings_col2:
        objective = st.selectbox(
            "Optimization objective",
            ["Max Sharpe", "Min Volatility (target return)", "Target Return"],
            index=0,
            key="micro_objective"
        )
        target_return = st.number_input(
            "Target annual return (%)",
            value=10.0,
            step=0.5,
            key="micro_target_return"
        )
    with settings_col3:
        risk_free = st.number_input(
            "Risk-free rate (annual, %)",
            value=4.0,
            step=0.25,
            key="micro_risk_free"
        )
        investment = st.number_input(
            "Total investment (USD)",
            value=250_000,
            step=10_000,
            key="micro_investment"
        )
    st.caption("Tip: Monthly data is usually smoother for 5–10 year horizons.")

    # ── 🔒 Portfolio Constraints ──────────────────────────────────────────────
    st.subheader("🔒 Portfolio Constraints")
    constraint_cols = st.columns(4)
    with constraint_cols[0]:
        max_weight = st.slider(
            "Max weight per stock (%)",
            min_value=5,
            max_value=50,
            value=30,
            step=1,
            key="micro_max_weight"
        )
    with constraint_cols[1]:
        l2_reg = st.slider(
            "L2 regularization",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="micro_l2_reg"
        )
    with constraint_cols[2]:
        min_weight_threshold = st.slider(
            "Prune weights below (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.25,
            step=0.05,
            key="micro_min_weight_threshold"
        )
    with constraint_cols[3]:
        min_holdings = st.number_input(
            "Minimum holdings",
            min_value=2,
            max_value=60,
            value=3,
            step=1,
            key="micro_min_holdings"
        )

    # ── 🎯 Stock Selection ────────────────────────────────────────────────────
    st.subheader("🎯 Stock Selection")
    has_forecast = bool(st.session_state.get("forecast_expected_returns"))
    radio_options = ["Manual Entry", "Use Top Performers from Analysis", "Custom from Analysis"]
    if has_forecast:
        radio_options.append("Use Forecast Returns")
    default_radio = radio_options.index("Use Forecast Returns") if st.session_state.get("use_forecast_returns") and has_forecast else 0
    stock_selection = st.radio(
        "Choose how to feed tickers into the optimizer",
        radio_options,
        index=default_radio,
        horizontal=True,
    )
    # Clear the one-shot flag once the tab has rendered
    st.session_state["use_forecast_returns"] = False

    tickers: List[str] = []

    if stock_selection == "Manual Entry":
        default_value = st.session_state.get('recommended_tickers', ",".join(DEFAULT_TICKERS))
        tickers_str = st.text_area("Tickers (comma-separated)", value=default_value)
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        if st.button("🔄 Clear Recommendations"):
            st.session_state['recommended_tickers'] = ",".join(DEFAULT_TICKERS)
            st.rerun()
    elif stock_selection == "Use Top Performers from Analysis":
        num_top_stocks = st.slider("Number of top stocks", min_value=5, max_value=25, value=10, step=1)
        cached_df, metadata = get_analysis(horizon, force_refresh=False)
        if cached_df is not None and not cached_df.empty:
            tickers = cached_df.head(num_top_stocks)['Ticker'].tolist()
            if metadata:
                freshness = format_timestamp(metadata.get("last_updated"))
                st.info(f"Using cached analysis from {freshness}.")
            st.session_state['recommended_tickers'] = ",".join(tickers)
        else:
            st.warning("No cached analysis available for this horizon. Falling back to defaults.")
            tickers = DEFAULT_TICKERS
    elif stock_selection == "Use Forecast Returns":
        er_override = st.session_state.get("forecast_expected_returns") or {}
        tickers = list(er_override.keys())
        if tickers:
            st.info(
                f"Using {len(tickers)} tickers from last forecast. "
                "Ensemble 1Y return estimates will replace historical expected returns."
            )
            st.session_state['recommended_tickers'] = ",".join(tickers)
        else:
            st.warning("No forecast results found. Run a forecast first.")
            tickers = DEFAULT_TICKERS
    else:
        custom_analysis_df, _ = get_analysis(horizon, force_refresh=False)
        if custom_analysis_df is not None and not custom_analysis_df.empty:
            options = custom_analysis_df['Ticker'].tolist()
            default_choices = options[: min(10, len(options))]
            selected = st.multiselect(
                "Select stocks from analysis (top 20 shown)",
                options=options[:20],
                default=default_choices
            )
            tickers = selected if selected else default_choices
            st.session_state['recommended_tickers'] = ",".join(tickers)
        else:
            st.warning("Analysis data unavailable; reverting to defaults.")
            tickers = DEFAULT_TICKERS

    if not tickers:
        st.warning("No tickers provided. Using defaults.")
        tickers = DEFAULT_TICKERS

    st.caption(f"Optimizing with {len(tickers)} tickers.")

    # ── Run ───────────────────────────────────────────────────────────────────
    st.divider()
    run_button = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)

    if run_button:
        # Clear previous result so stale output isn't shown on error
        st.session_state['optimization_result'] = None
        st.session_state['optimization_prices'] = None

        with st.spinner("Fetching price data…"):
            prices_df = client.get_stock_data(tickers, horizon, interval)
        if prices_df.empty:
            st.error("Insufficient price data returned from the data service.")
        else:
            # Store input tickers (before pruning) so downstream tabs can use the same universe
            st.session_state['optimization_input_tickers'] = tickers
            st.session_state['optimization_prices'] = prices_df
            er_override = (
                st.session_state.get("forecast_expected_returns")
                if stock_selection == "Use Forecast Returns"
                else None
            )
            with st.spinner("Optimizing portfolio…"):
                result = client.optimize_portfolio(
                    prices=prices_df,
                    tickers=tickers,
                    investment=investment,
                    objective=objective,
                    risk_free=risk_free,
                    target_return=target_return,
                    max_weight=max_weight,
                    l2_reg=l2_reg,
                    min_weight_threshold=min_weight_threshold,
                    min_holdings=min_holdings,
                    expected_returns_override=er_override,
                )
            if not result:
                st.error("Optimization failed. Adjust constraints or verify services are running.")
            else:
                st.session_state['optimization_result'] = result
                st.session_state['optimization_tickers'] = [
                    t for t, w in result.get('weights', {}).items() if w > 0
                ]
                st.session_state['optimization_last_params'] = {
                    'investment':          investment,
                    'objective':           objective,
                    'risk_free':           risk_free,
                    'target_return':       target_return,
                    'max_weight':          max_weight,
                    'l2_reg':              l2_reg,
                    'min_weight_threshold': min_weight_threshold,
                    'min_holdings':        min_holdings,
                }

    # ── Results (from session state) ──────────────────────────────────────────
    if st.session_state['optimization_result'] and st.session_state['optimization_prices'] is not None:
        _render_optimization_results(
            st.session_state['optimization_result'],
            st.session_state['optimization_prices'],
            investment,
            risk_free,
        )
        opt_t = st.session_state.get('optimization_tickers', [])
        if opt_t:
            st.caption(
                f"Portfolio locked ({len(opt_t)} stocks). "
                "➡️ Next: **🔮 Forecast** tab — select **Use Portfolio Tickers** to forecast these stocks."
            )

# ── Tab: Data Health ──────────────────────────────────────────────────────────

with tab_health:
    st.header("📋 Data Health Dashboard")
    st.caption("Overview of price cache freshness, analysis cache status, and S&P 500 ticker universe health.")

    # ── Section A: Load / refresh data ───────────────────────────────────────
    if st.button("🔄 Refresh", key="refresh_health") or not st.session_state['health_data']:
        with st.spinner("Loading health data…"):
            st.session_state['health_data'] = {
                'price_cache': client.get_price_cache_info(),
                'analysis_cache': client.get_cache_info(),
                'ticker_validation': client.get_ticker_validation(),
                'ticker_health': client.get_ticker_health(),
                'sync_report': client.get_sync_report(),
            }

    hd = st.session_state['health_data'] or {}
    price_cache: dict = hd.get('price_cache') or {}
    analysis_cache: dict = hd.get('analysis_cache') or {}
    ticker_validation: Optional[dict] = hd.get('ticker_validation')
    ticker_health: dict = hd.get('ticker_health') or {}
    sync_report: Optional[dict] = hd.get('sync_report')

    # ── Section B: Top-level metrics ─────────────────────────────────────────
    MASTER_KEY = "5y:1d"
    master_price = price_cache.get(MASTER_KEY, {})

    # Price metrics
    price_last_synced_str = master_price.get("last_synced")
    price_data_through_str = master_price.get("data_through")
    price_num_tickers = master_price.get("num_tickers")

    price_last_synced_dt = pd.to_datetime(price_last_synced_str) if price_last_synced_str else None
    price_age_hours = (
        (datetime.now(timezone.utc) - price_last_synced_dt.to_pydatetime()).total_seconds() / 3600
        if price_last_synced_dt is not None else None
    )
    price_age_days = price_age_hours / 24 if price_age_hours is not None else None

    # Analysis metrics (most recent period)
    analysis_last_updated_dt: Optional[datetime] = None
    analysis_data_through_str: Optional[str] = None
    analysis_num_stocks: Optional[int] = None
    for _p in ["5y", "3y", "2y", "1y"]:
        _meta = analysis_cache.get(_p)
        if _meta:
            _lu = _meta.get("last_updated")
            if _lu:
                _dt = pd.to_datetime(_lu)
                if _dt.tzinfo is None:
                    _dt = _dt.tz_localize("UTC")
                _dt_py = _dt.to_pydatetime()
                if analysis_last_updated_dt is None or _dt_py > analysis_last_updated_dt:
                    analysis_last_updated_dt = _dt_py
                    analysis_data_through_str = _meta.get("data_through")
                    analysis_num_stocks = _meta.get("num_stocks")

    analysis_age_hours = (
        (datetime.now(timezone.utc) - analysis_last_updated_dt).total_seconds() / 3600
        if analysis_last_updated_dt else None
    )
    analysis_age_days = analysis_age_hours / 24 if analysis_age_hours is not None else None

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        label = format_timestamp(price_last_synced_str) if price_last_synced_str else "—"
        delta = f"-{price_age_days:.0f}d" if price_age_days is not None else None
        st.metric("Prices Last Synced", label, delta=delta, delta_color="inverse")
    with m2:
        label2 = format_timestamp(analysis_last_updated_dt.isoformat() if analysis_last_updated_dt else None)
        delta2 = f"-{analysis_age_days:.0f}d" if analysis_age_days is not None else None
        st.metric("Analysis Last Run", label2, delta=delta2, delta_color="inverse")
    with m3:
        st.metric("Universe Size", f"{price_num_tickers} tickers" if price_num_tickers else "—")
    with m4:
        thru_label = format_timestamp(price_data_through_str) if price_data_through_str else "—"
        st.metric("Data Through", thru_label)

    # ── Section C: Staleness alerts ───────────────────────────────────────────
    if price_age_hours is None:
        st.error("Price cache is missing. Run: `./run-price-sync.sh`")
    elif price_age_hours > 168:
        st.error(
            f"Price cache is {price_age_days:.0f} days old. "
            "Run: `./run-price-sync.sh`"
        )
    elif price_age_hours > 24:
        st.warning(f"Price cache is {price_age_hours:.0f}h old — consider refreshing.")

    if analysis_age_hours is None:
        st.error("Analysis cache is missing. Run: `./run-analysis-sync.sh`")
    elif analysis_age_hours > 168:
        st.error(
            f"Analysis cache is {analysis_age_days:.0f} days old. "
            "Run: `./run-analysis-sync.sh`"
        )

    st.divider()

    # ── Section D: Price Cache table ──────────────────────────────────────────
    st.subheader("📥 Price Cache")
    if price_cache:
        price_rows = []
        for key, meta in sorted(price_cache.items()):
            synced_str = meta.get("last_synced")
            synced_dt = pd.to_datetime(synced_str) if synced_str else None
            if synced_dt is not None:
                synced_dt_py = synced_dt.to_pydatetime()
                if synced_dt_py.tzinfo is None:
                    synced_dt_py = synced_dt_py.replace(tzinfo=timezone.utc)
                age_h = (datetime.now(timezone.utc) - synced_dt_py).total_seconds() / 3600
            else:
                age_h = None
            price_rows.append({
                "Period:Interval": key,
                "Last Synced": format_timestamp(synced_str),
                "Data Through": format_timestamp(meta.get("data_through")),
                "Tickers": meta.get("num_tickers", "—"),
                "Trading Days": meta.get("rows", "—"),
                "Status": _status_badge(age_h),
            })
        st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No price cache metadata available.")

    # ── Section E: Analysis Cache table ───────────────────────────────────────
    st.subheader("📊 Analysis Cache")
    if analysis_cache:
        analysis_rows = []
        for period in ["1y", "2y", "3y", "5y"]:
            meta = analysis_cache.get(period)
            if not meta:
                analysis_rows.append({
                    "Period": period, "Last Run": "—", "Data Through": "—",
                    "Stocks": "—", "Age (h)": "—", "Status": "⚫ Missing",
                })
                continue
            lu_str = meta.get("last_updated")
            lu_dt = pd.to_datetime(lu_str) if lu_str else None
            if lu_dt is not None:
                lu_dt_py = lu_dt.to_pydatetime()
                if lu_dt_py.tzinfo is None:
                    lu_dt_py = lu_dt_py.replace(tzinfo=timezone.utc)
                age_h = (datetime.now(timezone.utc) - lu_dt_py).total_seconds() / 3600
            else:
                age_h = None
            analysis_rows.append({
                "Period": period,
                "Last Run": format_timestamp(lu_str),
                "Data Through": format_timestamp(meta.get("data_through")),
                "Stocks": meta.get("num_stocks", "—"),
                "Age (h)": f"{age_h:.0f}" if age_h is not None else "—",
                "Status": _status_badge(age_h),
            })
        st.dataframe(pd.DataFrame(analysis_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No analysis cache metadata available.")

    # ── Section F: Data Lag ───────────────────────────────────────────────────
    st.subheader("⏱ Data Lag")
    lag_rows = []
    for period in ["1y", "2y", "3y", "5y"]:
        key = f"{period}:1d"
        p_through = price_cache.get(key, {}).get("data_through") or price_cache.get(MASTER_KEY, {}).get("data_through")
        a_meta = analysis_cache.get(period, {})
        a_through = a_meta.get("data_through") if a_meta else None
        p_dt = pd.to_datetime(p_through) if p_through else None
        a_dt = pd.to_datetime(a_through) if a_through else None
        if p_dt is not None and a_dt is not None:
            lag_days = (p_dt - a_dt).days
            lag_str = f"{lag_days}d"
        else:
            lag_str = "—"
        lag_rows.append({
            "Period": period,
            "Price Through": format_timestamp(p_through),
            "Analysis Through": format_timestamp(a_through),
            "Lag (days)": lag_str,
        })
    st.dataframe(pd.DataFrame(lag_rows), use_container_width=True, hide_index=True)

    # ── Section G: Ticker Universe ────────────────────────────────────────────
    st.subheader("🏷️ Ticker Universe")

    cache_info_h = ticker_health.get("cache", {}) if ticker_health else {}
    last_modified = cache_info_h.get("last_modified") or cache_info_h.get("last_updated")
    size_bytes = cache_info_h.get("size_bytes")

    if ticker_validation:
        run_ts = ticker_validation.get("run_timestamp") or ticker_validation.get("timestamp")
        matched = ticker_validation.get("universe_matched", False)
        cached_count = ticker_validation.get("cached_count")
        remote_count = ticker_validation.get("remote_count")
        missing = ticker_validation.get("missing_from_remote") or []
        new_in_remote = ticker_validation.get("new_in_remote") or []
    else:
        run_ts = matched = cached_count = remote_count = None
        missing = new_in_remote = []

    mu1, mu2, mu3, mu4 = st.columns(4)
    mu1.metric("File Last Modified", format_timestamp(last_modified) if last_modified else "—")
    mu2.metric("File Size", f"{size_bytes / 1024:.1f} KB" if size_bytes else "—")
    mu3.metric("Last Validation Run", format_timestamp(run_ts) if run_ts else "—")
    mu4.metric("Universe Matched", "✅ Yes" if matched else ("❌ No" if ticker_validation else "—"))
    if cached_count is not None and remote_count is not None:
        st.caption(f"Cached tickers: {cached_count} | Remote tickers: {remote_count}")

    if ticker_validation:
        if missing:
            st.warning(f"**Missing from remote (possibly delisted):** {', '.join(missing)}")
            st.dataframe(pd.DataFrame({"Delisted / Missing": missing}), use_container_width=True, hide_index=True)
        if new_in_remote:
            st.info(f"**New in remote (pending add):** {', '.join(new_in_remote)}")
            st.dataframe(pd.DataFrame({"Pending Add": new_in_remote}), use_container_width=True, hide_index=True)
        if not missing and not new_in_remote:
            st.info("Ticker universe is in sync with the remote S&P 500 list.")
    else:
        st.info("No ticker validation data found. Run price sync to generate a validation report.")

    st.caption("To refresh universe and prices: run `./run-price-sync.sh`")

    st.divider()

    # ── Section H: Last Price Sync Report ─────────────────────────────────────
    st.subheader("📋 Last Price Sync Report")

    if sync_report:
        run_ts = sync_report.get("run_timestamp")
        attempted = sync_report.get("tickers_attempted", 0)
        ok_all = sync_report.get("tickers_ok_all_periods", 0)
        failed_any = sync_report.get("tickers_failed_any_period", 0)
        failed_all_n = sync_report.get("tickers_failed_all_periods", 0)

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Sync Run", format_timestamp(run_ts) if run_ts else "—")
        sm2.metric("Tickers Attempted", attempted)
        sm3.metric("OK in All Periods", ok_all)
        sm4.metric("Failed in Any Period", failed_any, delta=f"+{failed_any}" if failed_any else None, delta_color="inverse")

        # Per-period breakdown table
        periods_data = sync_report.get("periods", {})
        if periods_data:
            period_rows = []
            for p in ["1y", "2y", "3y", "5y"]:
                ps = periods_data.get(p, {})
                if ps:
                    period_rows.append({
                        "Period": p,
                        "Attempted": ps.get("tickers_attempted", "—"),
                        "Loaded": ps.get("tickers_loaded", "—"),
                        "Failed": ps.get("tickers_failed", "—"),
                        "Rows": ps.get("rows", "—"),
                        "Data Through": format_timestamp(ps.get("data_through")),
                        "Sync Time": format_timestamp(ps.get("last_synced")),
                    })
            if period_rows:
                st.dataframe(pd.DataFrame(period_rows), use_container_width=True, hide_index=True)

        # Failed tickers detail
        failed_any_list: list = sync_report.get("failed_any_period") or []
        failed_all_list: list = sync_report.get("failed_all_periods") or []

        if failed_all_list:
            st.error(f"{failed_all_n} ticker(s) failed in ALL periods — not available in any cache.")
            st.dataframe(
                pd.DataFrame({"Ticker": failed_all_list}),
                use_container_width=True, hide_index=True,
            )

        if failed_any_list:
            remaining = [t for t in failed_any_list if t not in set(failed_all_list)]
            if remaining or failed_all_list:
                st.warning(f"{failed_any} ticker(s) failed to load in at least one period.")
            if remaining:
                # Build per-ticker "failed in which periods" column
                ticker_period_rows = []
                for ticker in failed_any_list:
                    bad_periods = [
                        p for p in ["1y", "2y", "3y", "5y"]
                        if ticker in (periods_data.get(p, {}).get("failed_list") or [])
                    ]
                    ticker_period_rows.append({
                        "Ticker": ticker,
                        "Failed in Periods": ", ".join(bad_periods),
                    })
                st.dataframe(
                    pd.DataFrame(ticker_period_rows),
                    use_container_width=True, hide_index=True,
                )

        if not failed_any_list:
            st.success("All tickers loaded successfully in all periods.")
    else:
        st.info("No sync report found. Run `./run-price-sync.sh` to generate one.")

# ── Tab: Universe Changes ─────────────────────────────────────────────────────

with tab_universe:
    st.header("🔄 S&P 500 Universe Changes")
    st.caption(
        "History of additions and removals detected by comparing the local constituent "
        "cache against the live Wikipedia S&P 500 list."
    )

    client = ServiceClient()

    if st.button("🔄 Refresh", key="refresh_universe"):
        with st.spinner("Loading change history..."):
            st.session_state['universe_changes'] = client.get_ticker_changes()

    if st.session_state['universe_changes'] is None:
        with st.spinner("Loading change history..."):
            st.session_state['universe_changes'] = client.get_ticker_changes()

    changes_data = st.session_state['universe_changes'] or {}
    changes_log: list = changes_data.get("changes") or []
    total_runs: int = changes_data.get("total_runs", 0)
    pending: dict = changes_data.get("pending") or {}

    # ── Section A: Summary metrics ────────────────────────────────────────────
    added_count = sum(1 for c in changes_log if c.get("action") == "added")
    removed_count = sum(1 for c in changes_log if c.get("action") == "removed")

    ma1, ma2, ma3, ma4 = st.columns(4)
    ma1.metric("Total Changes", len(changes_log))
    ma2.metric("Added to S&P 500", added_count)
    ma3.metric("Removed from S&P 500", removed_count)
    ma4.metric("Validation Runs", total_runs)

    st.divider()

    # ── Section B: Pending changes ────────────────────────────────────────────
    if pending:
        if not pending.get("match", True):
            st.warning("⚠️ Local cache is out of sync with Wikipedia.")
            pb1, pb2 = st.columns(2)
            with pb1:
                pending_added = pending.get("added") or []
                st.markdown(f"**➕ Pending Additions ({len(pending_added)})**")
                if pending_added:
                    for t in pending_added:
                        st.markdown(f"- `{t}`")
                else:
                    st.markdown("_None_")
            with pb2:
                pending_removed = pending.get("removed") or []
                st.markdown(f"**➖ Pending Removals ({len(pending_removed)})**")
                if pending_removed:
                    for t in pending_removed:
                        st.markdown(f"- `{t}`")
                else:
                    st.markdown("_None_")
            st.caption("Run `./run-price-sync.sh --force-refresh-tickers` to apply changes.")
        else:
            st.success("✅ Local cache matches Wikipedia — no pending changes.")

    st.divider()

    # ── Section C: Change history table ──────────────────────────────────────
    st.subheader("📋 Change History")

    if changes_log:
        action_filter = st.radio(
            "Filter by action",
            ["All", "➕ Added", "➖ Removed"],
            horizontal=True,
            key="universe_filter",
        )

        filtered = changes_log
        if action_filter == "➕ Added":
            filtered = [c for c in changes_log if c.get("action") == "added"]
        elif action_filter == "➖ Removed":
            filtered = [c for c in changes_log if c.get("action") == "removed"]

        table_rows = [
            {
                "Detected On": format_timestamp(c.get("timestamp")),
                "Ticker": c.get("ticker", "—"),
                "Action": "➕ Added" if c.get("action") == "added" else "➖ Removed",
                "Source File": c.get("source_file", "—"),
            }
            for c in filtered
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No changes detected yet. "
            "Run `python services/ticker_validation_service.py` to populate history."
        )

# ── Tab: Forecast ─────────────────────────────────────────────────────────────

def _render_forecast_results(
    forecasts: list,
    fc_meta: dict,
    prices_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render the full Forecast tab output."""
    import numpy as np

    try:
        import plotly.graph_objects as go
        has_plotly = True
    except ImportError:
        has_plotly = False

    # ── Pre-compute SPY benchmark (compound-growth line, normalized to start=1.0)
    # Computed once here; scaled to each stock's current_price inside the chart loop.
    spy_daily_growth: Optional[float] = None
    if prices_df is not None and "SPY" in prices_df.columns:
        spy_series = prices_df["SPY"].dropna()
        if len(spy_series) >= 30:
            spy_ret = spy_series.pct_change().dropna().values
            spy_cumulative = float(np.prod(1 + spy_ret))
            n_years = len(spy_ret) / 252.0
            spy_cagr = spy_cumulative ** (1.0 / n_years) - 1
            spy_daily_growth = (1 + spy_cagr) ** (1 / 252) - 1

    # ── Ensemble summary ──────────────────────────────────────────────────────
    st.subheader("📊 Ensemble Forecast Summary")
    st.caption(
        f"Based on {fc_meta.get('n_simulations', '?'):,} simulations per ticker  |  "
        f"Risk-free rate: {fc_meta.get('risk_free_annual', 0) * 100:.2f}%"
    )

    def _pct(val):
        return f"{val * 100:.1f}%" if val is not None else "—"

    summary_rows = [
        {
            "Ticker": fc["ticker"],
            "Price": f"${fc['current_price']:.2f}",
            "Ens. 1Y": _pct(fc.get("ensemble_return_1y")),
            "Ens. 2Y": _pct(fc.get("ensemble_return_2y")),
            "MC 1Y (p50)": _pct(fc.get("mc_return_1y")),
            "MC p90": _pct(fc.get("mc_p90_1y")),
            "MC p10": _pct(fc.get("mc_p10_1y")),
            "CAPM 1Y": _pct(fc.get("capm_return_1y")),
            "Beta": f"{fc['beta']:.2f}" if fc.get("beta") is not None else "—",
            "Trend 1Y": _pct(fc.get("trend_return_1y")),
        }
        for fc in forecasts
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Method comparison 1Y ──────────────────────────────────────────────────
    st.subheader("📋 Method Comparison — 1-Year Forecast")
    st.dataframe(pd.DataFrame([
        {
            "Ticker": fc["ticker"],
            "MC (median)": _pct(fc.get("mc_return_1y")),
            "CAPM": _pct(fc.get("capm_return_1y")),
            "Trend": _pct(fc.get("trend_return_1y")),
            "Ensemble": _pct(fc.get("ensemble_return_1y")),
        }
        for fc in forecasts
    ]), use_container_width=True, hide_index=True)

    # ── Method comparison 2Y ──────────────────────────────────────────────────
    with st.expander("📋 Method Comparison — 2-Year Forecast", expanded=False):
        st.dataframe(pd.DataFrame([
            {
                "Ticker": fc["ticker"],
                "MC (median)": _pct(fc.get("mc_return_2y")),
                "CAPM": _pct(fc.get("capm_return_2y")),
                "Trend": _pct(fc.get("trend_return_2y")),
                "Ensemble": _pct(fc.get("ensemble_return_2y")),
            }
            for fc in forecasts
        ]), use_container_width=True, hide_index=True)

    # ── Fan charts ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Fan Charts")
    max_fan = min(len(forecasts), 20)
    fan_n = st.slider("Number of tickers to show", min_value=1, max_value=max_fan,
                      value=min(5, max_fan), key="fc_fan_n")
    st.caption(
        "10 representative paths spanning the p10–p90 range of simulated outcomes. "
        "Shaded band = p10 to p90. Grey dashed = current price. Dotted vertical = 1Y mark. "
        + ("Orange dashed = SPY historical CAGR projection." if spy_daily_growth is not None else "")
    )

    fan_tickers = forecasts[:fan_n]
    fan_cols = st.columns(min(len(fan_tickers), 3))

    for i, fc in enumerate(fan_tickers):
        paths = fc.get("mc_paths_sample")
        if not paths:
            continue
        col = fan_cols[i % 3]
        with col:
            if has_plotly:
                paths_arr = np.array(paths)
                n_days = paths_arr.shape[1]
                x_days = list(range(n_days))
                fig = go.Figure()
                for path in paths_arr:
                    fig.add_trace(go.Scatter(
                        x=x_days, y=path.tolist(), mode="lines",
                        line=dict(width=0.8, color="steelblue"), opacity=0.35, showlegend=False,
                    ))
                fig.add_trace(go.Scatter(
                    x=x_days + x_days[::-1],
                    y=paths_arr[9].tolist() + paths_arr[0].tolist()[::-1],
                    fill="toself", fillcolor="rgba(70,130,180,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=x_days, y=paths_arr[4].tolist(), mode="lines",
                    line=dict(width=2.5, color="steelblue"), showlegend=False,
                ))
                # SPY benchmark overlay
                if spy_daily_growth is not None:
                    spy_future = fc["current_price"] * (1 + spy_daily_growth) ** np.arange(n_days)
                    fig.add_trace(go.Scatter(
                        x=x_days, y=spy_future.tolist(), mode="lines",
                        name="SPY CAGR", line=dict(width=1.8, color="orange", dash="dash"),
                        showlegend=True,
                    ))
                fig.add_hline(
                    y=fc["current_price"], line_dash="dash", line_color="grey",
                    annotation_text=f"Now: ${fc['current_price']:.0f}",
                )
                fig.add_vline(x=252, line_dash="dot", line_color="grey", annotation_text="1Y")
                fig.update_layout(
                    title=fc["ticker"], xaxis_title="Trading days", yaxis_title="Price ($)",
                    height=320, margin=dict(l=30, r=10, t=40, b=30),
                    showlegend=spy_daily_growth is not None,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(size=9)),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: simple line chart of median path
                import numpy as np
                paths_arr = np.array(paths)
                st.line_chart(
                    pd.DataFrame({"price": paths_arr[4]},
                                 index=range(paths_arr.shape[1])),
                    use_container_width=True,
                )
                st.caption(f"{fc['ticker']} — install plotly for full fan chart")

    # ── Download ──────────────────────────────────────────────────────────────
    flat = [{k: v for k, v in fc.items() if k != "mc_paths_sample"} for fc in forecasts]
    st.download_button(
        "📥 Download Forecast Data (CSV)",
        data=pd.DataFrame(flat).to_csv(index=False).encode("utf-8"),
        file_name="forecast_results.csv",
        mime="text/csv",
    )


with tab_forecast:
    st.header("🔮 Return Forecast")
    st.caption(
        "Ensemble forecast blending Monte Carlo GBM, CAPM, and log-linear trend regression. "
        "Statistical projections only — not investment advice."
    )
    _pipeline_status()

    # ── Ticker selection ───────────────────────────────────────────────────────
    st.subheader("🎯 Ticker Selection")
    _has_opt_tickers = bool(st.session_state.get('optimization_tickers'))
    fc_ticker_options = ["Top N Performers", "Manual Entry"]
    if _has_opt_tickers:
        fc_ticker_options.insert(0, "Use Portfolio Tickers")
    fc_default_mode = 0 if _has_opt_tickers else 0
    fc_stock_mode = st.radio(
        "Which stocks to forecast",
        fc_ticker_options,
        index=fc_default_mode,
        horizontal=True,
        key="fc_stock_mode",
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        forecast_period = st.selectbox("Lookback period", ANALYSIS_PERIODS, index=0, key="fc_period")
    with fc2:
        forecast_rf = st.number_input("Risk-free rate (annual %)", value=4.0, step=0.25, key="fc_rf")
    with fc3:
        forecast_simulations = st.select_slider(
            "Simulations", options=[500, 1000, 2000, 5000], value=1000, key="fc_sims"
        )

    if fc_stock_mode == "Use Portfolio Tickers":
        fc_tickers_to_run = st.session_state['optimization_tickers']
        st.caption(f"Using {len(fc_tickers_to_run)} tickers from the optimized portfolio.")
    elif fc_stock_mode == "Manual Entry":
        fc_manual = st.text_input(
            "Tickers (comma-separated)",
            value=",".join(DEFAULT_TICKERS[:10]),
            key="fc_manual_tickers",
        )
        fc_tickers_to_run = [t.strip().upper() for t in fc_manual.split(",") if t.strip()]
    else:
        forecast_top_n = st.slider("Top N stocks", min_value=3, max_value=20, value=10, key="fc_top_n")
        fc_tickers_to_run = None  # resolved after analysis load

    if st.button("🚀 Run Forecast", type="primary", use_container_width=True, key="fc_run"):
        st.session_state["forecast_result"] = None
        st.session_state["forecast_prices"] = None

        if fc_stock_mode == "Top N Performers":
            with st.spinner("Loading top performers…"):
                analysis_df, _ = get_analysis(forecast_period, force_refresh=False)
            if analysis_df is None or analysis_df.empty:
                st.error("No analysis data available. Run the S&P 500 Analyzer first.")
                fc_tickers_to_run = None
            else:
                fc_tickers_to_run = analysis_df["Ticker"].head(forecast_top_n).tolist()

        if fc_tickers_to_run:
            top_tickers = fc_tickers_to_run
            price_tickers = list(dict.fromkeys(top_tickers + ["SPY"]))

            with st.spinner(f"Fetching prices for {len(price_tickers)} tickers…"):
                prices_df = client.get_stock_data(price_tickers, forecast_period, "1d")

            if prices_df.empty:
                st.error("No price data returned.")
            else:
                st.session_state["forecast_prices"] = prices_df
                with st.spinner(
                    f"Running {forecast_simulations:,} Monte Carlo simulations per ticker…"
                ):
                    result = client.get_forecast(
                        tickers=top_tickers,
                        prices=prices_df,
                        risk_free_pct=forecast_rf,
                        n_simulations=forecast_simulations,
                    )
                if result:
                    st.session_state["forecast_result"] = result
                    # Store ensemble returns so the Optimizer tab can use them
                    er_map = {
                        fc["ticker"]: fc["ensemble_return_1y"]
                        for fc in result.get("forecasts", [])
                        if fc.get("ensemble_return_1y") is not None
                    }
                    st.session_state["forecast_expected_returns"] = er_map or None
                    st.session_state["forecast_tickers"] = list(er_map.keys())
                else:
                    st.error("Forecast computation failed.")

    fc_result = st.session_state.get("forecast_result")
    if fc_result:
        forecasts_list = fc_result.get("forecasts", [])
        if forecasts_list:
            _render_forecast_results(
                forecasts_list, fc_result,
                prices_df=st.session_state.get("forecast_prices"),
            )
            er_override = st.session_state.get("forecast_expected_returns")
            if er_override:
                st.divider()
                ticker_list = list(er_override.keys())
                st.caption(
                    f"Forecast complete for {len(ticker_list)} tickers. "
                    "➡️ Next: **📅 Backtest** tab — select **Use Portfolio Tickers** to validate these forecasts."
                )
                if st.button(
                    "📤 Use Forecast Returns in Optimizer",
                    type="secondary",
                    key="fc_send_to_optimizer",
                    use_container_width=True,
                ):
                    st.session_state["recommended_tickers"] = ",".join(ticker_list)
                    st.session_state["use_forecast_returns"] = True
                    st.success(
                        "Done! Switch to the **📊 Optimizer** tab, select "
                        "**Use Forecast Returns** in Stock Selection, then click Optimize."
                    )
        else:
            st.warning("No forecasts returned.")

# ── Tab: Compare ──────────────────────────────────────────────────────────────

with tab_compare:
    st.header("🔀 Portfolio Comparison")
    st.caption("Side-by-side: historical expected returns vs ensemble forecast returns, same tickers and constraints.")
    _pipeline_status()

    _prices    = st.session_state.get('optimization_prices')
    _forecast  = st.session_state.get('forecast_expected_returns')
    _params    = st.session_state.get('optimization_last_params') or {}

    if _prices is None:
        st.info("Run the **Portfolio Optimizer** first to provide price data for comparison.")
    elif not _forecast:
        st.info("Run a **Forecast** first — ensemble return estimates are needed for the forecast-optimised portfolio.")
    else:
        _common_tickers = [t for t in _forecast.keys() if t in _prices.columns]
        if len(_common_tickers) < 2:
            st.warning(
                "Price data covers fewer than 2 forecast tickers. "
                "Re-run the optimizer using **Use Forecast Returns** so prices match the forecast universe."
            )
        else:
            st.info(
                f"Ready to compare **{len(_common_tickers)} tickers** using the same constraints as the last optimizer run."
            )
            if st.button("🔀 Run Comparison", type="primary", use_container_width=True,
                         key="compare_run"):
                st.session_state['comparison_results'] = None
                _compare_prices = _prices[_common_tickers]
                _inv  = _params.get('investment',          250_000)
                _obj  = _params.get('objective',           'Max Sharpe')
                _rf   = _params.get('risk_free',           4.0)
                _tr   = _params.get('target_return',       10.0)
                _mw   = _params.get('max_weight',          30)
                _l2   = _params.get('l2_reg',              5.0)
                _mwt  = _params.get('min_weight_threshold', 0.25)
                _mh   = _params.get('min_holdings',        3)

                with st.spinner("Running historical optimisation…"):
                    _hist = client.optimize_portfolio(
                        prices=_compare_prices, tickers=_common_tickers,
                        investment=_inv, objective=_obj, risk_free=_rf,
                        target_return=_tr, max_weight=_mw, l2_reg=_l2,
                        min_weight_threshold=_mwt, min_holdings=_mh,
                        expected_returns_override=None,
                    )
                with st.spinner("Running forecast-optimised portfolio…"):
                    _fc = client.optimize_portfolio(
                        prices=_compare_prices, tickers=_common_tickers,
                        investment=_inv, objective=_obj, risk_free=_rf,
                        target_return=_tr, max_weight=_mw, l2_reg=_l2,
                        min_weight_threshold=_mwt, min_holdings=_mh,
                        expected_returns_override={t: _forecast[t] for t in _common_tickers},
                    )
                if _hist and _fc:
                    st.session_state['comparison_results'] = {
                        'historical': _hist, 'forecast': _fc,
                        'prices': _compare_prices, 'investment': _inv, 'risk_free': _rf,
                    }
                else:
                    st.error("One or both optimisations failed. Check service logs and constraints.")

    if st.session_state.get('comparison_results'):
        _cr = st.session_state['comparison_results']
        _render_comparison(
            _cr['historical'], _cr['forecast'],
            _cr['prices'], _cr['investment'], _cr['risk_free'],
        )

# ── Tab: Backtest ─────────────────────────────────────────────────────────────

with tab_backtest:
    st.header("📅 Forecast Backtester")
    st.caption(
        "Validates the ensemble forecast model on historical data. "
        "Splits price history into rolling train/test windows, runs the forecast on each training window, "
        "then compares predicted vs actual 1-year returns."
    )
    _pipeline_status()

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_horizon = st.selectbox(
            "Price history (data fetch)",
            ["3y", "5y"],
            index=0,
            key="bt_horizon",
        )
        bt_interval = "1d"  # daily data required for reliable GBM fits
    with bt_col2:
        bt_n_windows = st.slider(
            "Number of rolling windows",
            min_value=2, max_value=4, value=3, step=1,
            key="bt_n_windows",
        )
        bt_n_sims = st.select_slider(
            "Monte Carlo simulations per window",
            options=[200, 500, 1000],
            value=500,
            key="bt_n_sims",
        )
    with bt_col3:
        bt_rf = st.number_input(
            "Risk-free rate (annual, %)",
            value=4.0, step=0.25,
            key="bt_risk_free",
        )

    # ── Ticker selection ───────────────────────────────────────────────────────
    st.subheader("🎯 Ticker Selection")
    _has_opt = bool(st.session_state.get('optimization_tickers'))
    _has_fc = bool(st.session_state.get("forecast_expected_returns"))
    bt_ticker_options = ["Manual Entry", "Use Top Performers from Analysis"]
    if _has_opt:
        bt_ticker_options.insert(0, "Use Portfolio Tickers")
    if _has_fc and "Use Forecast Tickers" not in bt_ticker_options:
        bt_ticker_options.append("Use Forecast Tickers")
    bt_stock_mode = st.radio(
        "Tickers to backtest",
        bt_ticker_options,
        index=0,
        horizontal=True,
        key="bt_stock_mode",
    )

    if bt_stock_mode == "Use Portfolio Tickers":
        bt_tickers = st.session_state['optimization_tickers']
        st.caption(f"Using {len(bt_tickers)} tickers from the optimized portfolio.")
    elif bt_stock_mode == "Manual Entry":
        bt_tickers_raw = st.text_input(
            "Tickers (comma-separated)",
            value=",".join(DEFAULT_TICKERS[:10]),
            key="bt_tickers_manual",
        )
        bt_tickers = [t.strip().upper() for t in bt_tickers_raw.split(",") if t.strip()]
    elif bt_stock_mode == "Use Forecast Tickers" and _has_fc:
        bt_tickers = list(st.session_state["forecast_expected_returns"].keys())
        st.caption(f"Using {len(bt_tickers)} tickers from last forecast.")
    else:
        _bt_analysis = st.session_state.get('analysis_cache', {})
        _bt_period = list(_bt_analysis.keys())[0] if _bt_analysis else None
        if _bt_period:
            _bt_entry = _bt_analysis[_bt_period]
            _bt_df = _bt_entry.get('df') if isinstance(_bt_entry, dict) else _bt_entry
            bt_tickers = _bt_df['Ticker'].head(15).tolist() if _bt_df is not None and not _bt_df.empty else DEFAULT_TICKERS[:10]
        else:
            st.caption("No analysis cached — using defaults.")
            bt_tickers = DEFAULT_TICKERS[:10]

    # include SPY for CAPM beta
    if "SPY" not in bt_tickers:
        bt_tickers_fetch = bt_tickers + ["SPY"]
    else:
        bt_tickers_fetch = bt_tickers

    st.caption(f"Backtesting {len(bt_tickers)} tickers across {bt_n_windows} rolling 1-year windows.")

    # ── Run ────────────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔁 Run Backtest", type="primary", use_container_width=True, key="bt_run"):
        st.session_state['backtest_result'] = None
        with st.spinner(f"Fetching {bt_horizon} of price data…"):
            bt_prices = client.get_stock_data(bt_tickers_fetch, bt_horizon, bt_interval)
        if bt_prices is None or bt_prices.empty:
            st.error("No price data returned. Check that the data service is running.")
        else:
            bt_stock_tickers = [t for t in bt_tickers if t in bt_prices.columns]
            if len(bt_stock_tickers) < 2:
                st.error("Fewer than 2 tickers had price data. Adjust your selection.")
            else:
                with st.spinner(
                    f"Running {bt_n_windows} backtest windows "
                    f"({bt_n_sims} MC sims each) — this may take a minute…"
                ):
                    bt_result = client.get_backtest(
                        tickers=bt_stock_tickers,
                        prices=bt_prices,
                        n_windows=bt_n_windows,
                        n_simulations=bt_n_sims,
                        risk_free_pct=bt_rf,
                    )
                if bt_result:
                    st.session_state['backtest_result'] = bt_result
                else:
                    st.error("Backtest failed. Check that the calculation service is running and the data horizon is long enough.")

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.get('backtest_result'):
        _render_backtest_results(st.session_state['backtest_result'])
        st.caption(
            "Backtest complete. "
            "➡️ Next: **🔀 Compare** tab — see how forecast weights differ from historical weights."
        )

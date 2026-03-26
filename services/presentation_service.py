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

DATA_SERVICE_URL = "http://localhost:8001"
CALCULATION_SERVICE_URL = "http://localhost:8002"
TICKER_SERVICE_URL = "http://localhost:8000"
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
        st.dataframe(prices_df.tail(), use_container_width=True)

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

    st.subheader("📉 Historical Performance")
    normalized = prices_df[weights_series.index] / prices_df[weights_series.index].iloc[0] * 100
    st.line_chart(normalized)


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
            if not data.get("success"):
                raise RuntimeError(data.get("error", "Unknown analysis error"))
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
            }
            response = requests.post(f"{CALCULATION_SERVICE_URL}/optimize-portfolio", json=payload, timeout=120)
            data = response.json()
            if not data.get("success"):
                raise RuntimeError(data.get("error", "Calculation service error"))
            return data.get("data")
        except Exception as exc:
            st.error(f"Optimization call failed: {exc}")
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


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_analyzer, tab_optimizer, tab_health, tab_universe = st.tabs([
    "🔍 S&P 500 Stock Analyzer",
    "📊 Portfolio Optimizer",
    "📋 Data Health",
    "🔄 Universe Changes",
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
        analysis_period = st.selectbox("Analysis period", period_options, index=0, key="analysis_period_micro")
    with ctrl2:
        analyze_button = st.button("🚀 Analyze Stocks", type="primary")
    with ctrl3:
        force_refresh = st.button("🔄 Force Refresh")
    with ctrl4:
        if st.button("🔄 Refresh Status"):
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

        display_cols = ['Ticker', 'Annual_Return', 'Sharpe_Ratio', 'Volatility',
                        'Max_Drawdown', 'Recent_3M_Return', 'Current_Price', 'Composite_Score']
        display_df = analysis_df[display_cols].head(20).copy()
        display_df['Annual_Return'] = (display_df['Annual_Return'] * 100).round(2).astype(str) + '%'
        display_df['Volatility'] = (display_df['Volatility'] * 100).round(2).astype(str) + '%'
        display_df['Max_Drawdown'] = (display_df['Max_Drawdown'] * 100).round(2).astype(str) + '%'
        display_df['Recent_3M_Return'] = (display_df['Recent_3M_Return'] * 100).round(2).astype(str) + '%'
        display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].round(2)
        display_df['Current_Price'] = '$' + display_df['Current_Price'].round(2).astype(str)
        display_df['Composite_Score'] = display_df['Composite_Score'].round(3)
        display_df.columns = ['Ticker', 'Annual Return', 'Sharpe Ratio', 'Volatility',
                              'Max Drawdown', '3M Return', 'Price', 'Score']
        st.dataframe(display_df, use_container_width=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            if st.button("📈 Use Top 20 for Portfolio Optimization", key="use_top20_micro"):
                st.session_state['recommended_tickers'] = ",".join(analysis_df['Ticker'].head(20).tolist())
                st.success(
                    "✅ Top 20 tickers ready. Switch to the **📊 Portfolio Optimizer** tab, "
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
    stock_selection = st.radio(
        "Choose how to feed tickers into the optimizer",
        ["Manual Entry", "Use Top Performers from Analysis", "Custom from Analysis"],
        horizontal=True
    )

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
                st.success(f"Using cached analysis from {freshness}.")
            st.session_state['recommended_tickers'] = ",".join(tickers)
        else:
            st.warning("No cached analysis available for this horizon. Falling back to defaults.")
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

    st.info(f"Optimizing with {len(tickers)} tickers.")

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
            st.session_state['optimization_prices'] = prices_df
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
                )
            if not result:
                st.error("Optimization failed. Adjust constraints or verify services are running.")
            else:
                st.session_state['optimization_result'] = result

    # ── Results (from session state) ──────────────────────────────────────────
    if st.session_state['optimization_result'] and st.session_state['optimization_prices'] is not None:
        _render_optimization_results(
            st.session_state['optimization_result'],
            st.session_state['optimization_prices'],
            investment,
            risk_free,
        )

# ── Tab: Data Health ──────────────────────────────────────────────────────────

with tab_health:
    st.header("📋 Data Health Dashboard")
    st.caption("Overview of price cache freshness, analysis cache status, and S&P 500 ticker universe health.")

    # ── Section A: Load / refresh data ───────────────────────────────────────
    if st.button("🔄 Refresh Dashboard") or not st.session_state['health_data']:
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
        st.error("⚠️ Price cache is missing. Run: `./run-price-sync.sh`")
    elif price_age_hours > 168:
        st.error(
            f"⚠️ Price cache is {price_age_days:.0f} days old. "
            "Run: `./run-price-sync.sh`"
        )
    elif price_age_hours > 24:
        st.warning(f"Price cache is {price_age_hours:.0f}h old — consider refreshing.")

    if analysis_age_hours is None:
        st.error("⚠️ Analysis cache is missing. Run: `./run-analysis-sync.sh`")
    elif analysis_age_hours > 168:
        st.error(
            f"⚠️ Analysis cache is {analysis_age_days:.0f} days old. "
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

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Constituent file:** `sp500_data/sp500_constituents.csv`")
        if last_modified:
            st.markdown(f"**Last modified:** {format_timestamp(last_modified)}")
        if size_bytes:
            st.markdown(f"**File size:** {size_bytes / 1024:.1f} KB")

    if ticker_validation:
        run_ts = ticker_validation.get("run_timestamp") or ticker_validation.get("timestamp")
        matched = ticker_validation.get("universe_matched", False)
        cached_count = ticker_validation.get("cached_count")
        remote_count = ticker_validation.get("remote_count")
        missing = ticker_validation.get("missing_from_remote") or []
        new_in_remote = ticker_validation.get("new_in_remote") or []

        with col_b:
            if run_ts:
                st.markdown(f"**Last validation run:** {format_timestamp(run_ts)}")
            match_icon = "✅ Yes" if matched else "❌ No"
            st.markdown(f"**Universe matched:** {match_icon}")
            if cached_count is not None and remote_count is not None:
                st.markdown(f"**Cached count:** {cached_count} | **Remote count:** {remote_count}")

        if missing:
            st.warning(f"**Missing from remote (possibly delisted):** {', '.join(missing)}")
            st.dataframe(pd.DataFrame({"Delisted / Missing": missing}), use_container_width=True, hide_index=True)
        if new_in_remote:
            st.info(f"**New in remote (pending add):** {', '.join(new_in_remote)}")
            st.dataframe(pd.DataFrame({"Pending Add": new_in_remote}), use_container_width=True, hide_index=True)
        if not missing and not new_in_remote:
            st.success("Ticker universe is in sync with the remote S&P 500 list.")
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
            st.error(f"⚠️ {failed_all_n} ticker(s) failed in ALL periods — not available in any cache.")
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

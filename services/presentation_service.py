import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# Allow `shared` imports when running via `streamlit run services/presentation_service.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_SERVICE_URL = "http://localhost:8001"
CALCULATION_SERVICE_URL = "http://localhost:8002"
ANALYSIS_PERIODS = ["1y", "2y", "3y", "5y"]
DEFAULT_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","JPM","V"
]


def standardize_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the monolith's analyzer schema."""
    if df is None or df.empty:
        return df
    column_mapping = {
        'ticker': 'Ticker',
        'total_return_pct': 'Annual_Return',
        'sharpe_ratio': 'Sharpe_Ratio',
        'volatility_pct': 'Volatility',
        'max_drawdown_pct': 'Max_Drawdown',
        'current_price': 'Current_Price',
        'composite_score': 'Composite_Score'
    }
    df = df.copy()
    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
    if 'Recent_3M_Return' not in df.columns:
        df['Recent_3M_Return'] = 0.0
    if 'Annual_Return' in df.columns and df['Annual_Return'].max() > 1:
        df['Annual_Return'] = df['Annual_Return'] / 100.0
    if 'Volatility' in df.columns and df['Volatility'].max() > 1:
        df['Volatility'] = df['Volatility'] / 100.0
    if 'Max_Drawdown' in df.columns and df['Max_Drawdown'].min() > -1:
        df['Max_Drawdown'] = -df['Max_Drawdown'] / 100.0
    return df


def format_timestamp(value: Optional[str]) -> str:
    if not value or value == "Unknown":
        return "Unknown"
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return str(value)
    return ts.strftime("%Y-%m-%d %H:%M")


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


client = ServiceClient()

st.set_page_config(page_title="S&P Portfolio Optimizer", layout="wide")
st.title("📈 S&P 500 Portfolio Optimizer")
st.caption("Build an optimal S&P portfolio given a risk–return preference, investment amount, and horizon.")

if 'analysis_cache' not in st.session_state:
    st.session_state['analysis_cache'] = {}
if 'cache_metadata' not in st.session_state:
    st.session_state['cache_metadata'] = {}
if 'recommended_tickers' not in st.session_state:
    st.session_state['recommended_tickers'] = ",".join(DEFAULT_TICKERS)
if 'sp500_tickers' not in st.session_state:
    st.session_state['sp500_tickers'] = []

if not client.check_service_health(DATA_SERVICE_URL):
    st.error("⚠️ Data service is unavailable. Start it with `python services/data_service.py`.")
    st.stop()
if not client.check_service_health(CALCULATION_SERVICE_URL):
    st.error("⚠️ Calculation service is unavailable. Start it with `python services/calculation_service.py`.")
    st.stop()

if not st.session_state['sp500_tickers']:
    fetched = client.get_sp500_tickers()
    st.session_state['sp500_tickers'] = fetched if fetched else DEFAULT_TICKERS

tab1, tab2 = st.tabs(["📊 Portfolio Optimizer", "🔍 S&P 500 Stock Analyzer"])


def refresh_cache_summary():
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


with tab2:
    st.header("🔍 S&P 500 Stock Analyzer")
    universe_size = len(st.session_state['sp500_tickers'])
    st.caption(f"Analyze the full S&P 500 universe ({universe_size} tickers) via the data microservice.")
    refresh_cache_summary()

    period_options = ANALYSIS_PERIODS
    cache_summary = st.session_state['cache_metadata']

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analysis_period = st.selectbox("Analysis period", period_options, index=0, key="analysis_period_micro")
    with col2:
        analyze_button = st.button("🚀 Analyze Stocks", type="primary")
    with col3:
        force_refresh = st.button("🔄 Force Refresh")

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

    analysis_df, analysis_meta = st.session_state['analysis_cache'].get(analysis_period, {}).get('df'), \
        st.session_state['analysis_cache'].get(analysis_period, {}).get('metadata')

    if analyze_button or force_refresh or analysis_df is None:
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
                st.success("Top 20 tickers copied to optimizer tab.")
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
        st.info("No analysis data available. Click 'Analyze Stocks' to start the computation.")


with tab1:
    st.header("📊 Portfolio Optimizer")
    st.caption("Adjust settings, choose a universe, and let the services build an allocation for you.")
    selection_container = st.container()
    analysis_container = st.container()
    constraint_container = st.container()

    with analysis_container:
        st.subheader("Analysis Settings")
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

    with constraint_container:
        st.subheader("Portfolio Constraints")
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

    with selection_container:
        st.subheader("Stock Selection Method")
        stock_selection = st.radio(
            "Choose how to feed tickers into the optimizer",
            ["Manual Entry", "Use Top Performers from Analysis", "Custom from Analysis"],
            horizontal=True
        )

        tickers: List[str] = []
        custom_analysis_df: Optional[pd.DataFrame] = None

        if stock_selection == "Manual Entry":
            default_value = st.session_state.get('recommended_tickers', ",".join(DEFAULT_TICKERS))
            tickers_str = st.text_area("Tickers (comma-separated)", value=default_value)
            tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            if st.button("🔄 Clear Recommendations"):
                st.session_state['recommended_tickers'] = ",".join(DEFAULT_TICKERS)
                st.experimental_rerun()
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
        st.markdown("---")

    prices_df = client.get_stock_data(tickers, horizon, interval)
    if prices_df.empty:
        st.error("Insufficient price data returned from the data service.")
        st.stop()

    st.subheader("Price Preview")
    st.dataframe(prices_df.tail(), use_container_width=True)

    with st.spinner("Optimizing portfolio..."):
        optimization_result = client.optimize_portfolio(
            prices=prices_df,
            tickers=tickers,
            investment=investment,
            objective=objective,
            risk_free=risk_free,
            target_return=target_return,
            max_weight=max_weight,
            l2_reg=l2_reg,
            min_weight_threshold=min_weight_threshold,
            min_holdings=min_holdings
        )

    if not optimization_result:
        st.error("Optimization failed. Adjust constraints or verify services are running.")
        st.stop()

    st.subheader("Optimal Weights")
    weights_series = pd.Series(optimization_result['weights']).sort_values(ascending=False)
    weights_display = pd.DataFrame({
        "Ticker": weights_series.index,
        "Weight": (weights_series * 100).round(2).astype(str) + '%'
    })
    st.dataframe(weights_display, use_container_width=True, hide_index=True)

    perf = {
        "Expected Return (annualized)": f"{optimization_result['expected_annual_return']*100:.2f}%",
        "Expected Volatility (annualized)": f"{optimization_result['annual_volatility']*100:.2f}%",
        f"Sharpe (rf={risk_free:.2f}%)": f"{optimization_result['sharpe_ratio']:.2f}"
    }
    st.subheader("Performance (Expected)")
    st.table(pd.DataFrame(perf, index=["Metrics"]))

    latest_prices = prices_df[weights_series.index].iloc[-1]
    alloc_usd = weights_series * investment
    allocation_df = pd.DataFrame({
        "Ticker": weights_series.index,
        "Weight": (weights_series * 100).round(2),
        "Last Price": latest_prices.round(2),
        "Allocation (USD)": alloc_usd.round(2)
    })
    st.subheader("Dollar Allocation")
    st.dataframe(allocation_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Weights (CSV)",
        data=weights_series.to_csv().encode("utf-8"),
        file_name="weights.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download Full Allocation (CSV)",
        data=allocation_df.to_csv(index=False).encode("utf-8"),
        file_name="allocation.csv",
        mime="text/csv"
    )

    st.subheader("Historical Performance (Normalized)")
    normalized = prices_df[weights_series.index] / prices_df[weights_series.index].iloc[0] * 100
    st.line_chart(normalized)

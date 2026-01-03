import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for per-ticker metrics."""

    Ticker: str
    Annual_Return: float
    Volatility: float
    Sharpe_Ratio: float
    Max_Drawdown: float
    Recent_3M_Return: float
    Current_Price: float


class AnalysisEngine:
    """Compute derived metrics from cached price data."""

    def __init__(self, min_history: int = 60):
        self.min_history = min_history

    def analyze_prices(self, price_frame: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
        """Return an analysis dataframe and the latest timestamp."""
        if price_frame is None or price_frame.empty:
            return pd.DataFrame(), None
        results: List[Dict[str, float]] = []
        latest_ts: Optional[pd.Timestamp] = None
        if not isinstance(price_frame.index, pd.DatetimeIndex):
            price_frame = price_frame.copy()
            price_frame.index = pd.to_datetime(price_frame.index)

        for ticker in price_frame.columns:
            series = price_frame[ticker].dropna()
            if len(series) < self.min_history:
                continue
            metrics = self._analyze_series(ticker, series)
            if metrics:
                results.append(metrics)
                series_ts = series.index.max()
                if latest_ts is None or (isinstance(series_ts, pd.Timestamp) and series_ts > latest_ts):
                    latest_ts = series_ts

        if not results:
            return pd.DataFrame(), latest_ts
        df = pd.DataFrame(results)
        df = self._append_scores(df)
        return df, latest_ts

    def _analyze_series(self, ticker: str, prices: pd.Series) -> Optional[Dict[str, float]]:
        prices = prices.sort_index()
        returns = prices.pct_change().dropna()
        if returns.empty or len(returns) < self.min_history:
            return None
        daily_log = np.log(prices).diff().dropna()
        if daily_log.empty:
            return None
        mean_log = daily_log.mean()
        ann_return = float(np.exp(mean_log * 252) - 1)
        ann_vol = float(returns.std() * np.sqrt(252))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
        cumulative = (1 + returns).cumprod()
        if cumulative.empty:
            return None
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        if len(prices) >= 63:
            recent_return = float((prices.iloc[-1] / prices.iloc[-63]) - 1)
        else:
            recent_return = 0.0
        current_price = float(prices.iloc[-1])
        return AnalysisResult(
            Ticker=ticker,
            Annual_Return=ann_return,
            Volatility=ann_vol,
            Sharpe_Ratio=sharpe,
            Max_Drawdown=max_drawdown,
            Recent_3M_Return=recent_return,
            Current_Price=current_price,
        ).__dict__

    @staticmethod
    def _safe_normalize(series: pd.Series, inverse: bool = False) -> pd.Series:
        range_val = series.max() - series.min()
        if range_val == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        normalized = (series - series.min()) / range_val
        return 1 - normalized if inverse else normalized

    def _append_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mirror the Streamlit composite scoring."""
        df['Return_Score'] = self._safe_normalize(df['Annual_Return'])
        df['Vol_Score'] = self._safe_normalize(df['Volatility'], inverse=True)
        df['Sharpe_Score'] = self._safe_normalize(df['Sharpe_Ratio'])
        df['Drawdown_Score'] = self._safe_normalize(df['Max_Drawdown'], inverse=True)
        df['Momentum_Score'] = self._safe_normalize(df['Recent_3M_Return'])
        df['Composite_Score'] = (
            0.25 * df['Return_Score'] +
            0.20 * df['Vol_Score'] +
            0.25 * df['Sharpe_Score'] +
            0.15 * df['Drawdown_Score'] +
            0.15 * df['Momentum_Score']
        )
        df = df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        return df

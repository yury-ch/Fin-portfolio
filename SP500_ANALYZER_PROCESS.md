# S&P 500 Stock Analyzer - Process Documentation

## Overview

The S&P 500 Stock Analyzer is a comprehensive system that evaluates and ranks stocks from a curated list of 100 top S&P 500 companies to identify the best performing stocks based on multiple financial metrics.

## Stock Universe

The analyzer processes **100 carefully selected S&P 500 stocks**, divided into:

### Top 50 Mega-Cap Stocks
```
AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, BRK-B, UNH, XOM,
JPM, JNJ, V, PG, HD, CVX, MA, ABBV, PFE, KO,
AVGO, COST, PEP, TMO, WMT, MRK, DIS, ADBE, NFLX, CRM,
BAC, ACN, LLY, ORCL, WFC, VZ, CMCSA, CSCO, ABT, DHR,
NKE, TXN, PM, BMY, UNP, QCOM, RTX, HON, INTC, T
```

### Next 50 Large-Cap Stocks
```
AMAT, SPGI, CAT, INTU, ISRG, NOW, LOW, GS, MS, AMD,
AMGN, BKNG, TJX, BLK, AXP, SYK, VRTX, PLD, GILD, MDLZ,
SBUX, TMUS, CVS, CI, LRCX, CB, MO, PYPL, MMC, SO,
ZTS, SCHW, FIS, DUK, BSX, CL, ITW, EQIX, AON, CSX,
ADI, NOC, MU, SHW, ICE, KLAC, APD, USB, CME, REGN,
EMR, PNC, EOG, FCX, GD, NSC, TGT, HUM, COP, PSA
```

## Analysis Process

### 1. Data Collection (`app.py:320-330`)

For each stock, the system:
- Downloads historical price data from Yahoo Finance using `yfinance`
- Supports different time periods: 6M, 1Y, 2Y, 3Y
- Uses daily interval data
- Extracts Adjusted Close prices (or Close if Adj Close unavailable)
- Requires minimum 60 days of data for reliable analysis

### 2. Metric Calculations (`app.py:333-361`)

The analyzer calculates **5 key financial metrics** for each stock:

#### A. **Annualized Return** (`app.py:334`)
```python
ann_return = float((1 + returns.mean()) ** 252 - 1)
```
- Converts daily returns to annualized return
- Uses 252 trading days per year
- Higher values are better

#### B. **Annualized Volatility** (`app.py:335`)
```python
ann_vol = float(returns.std() * np.sqrt(252))
```
- Measures price stability
- Annualized standard deviation of returns
- Lower values are better (less risky)

#### C. **Sharpe Ratio** (`app.py:336`)
```python
sharpe = float(ann_return / ann_vol if ann_vol > 0 else 0)
```
- Risk-adjusted return metric
- Return per unit of risk
- Higher values are better

#### D. **Maximum Drawdown** (`app.py:338-342`)
```python
cumulative = (1 + returns).cumprod()
rolling_max = cumulative.expanding().max()
drawdown = (cumulative - rolling_max) / rolling_max
max_drawdown = float(drawdown.min())
```
- Largest peak-to-trough decline
- Measures worst-case scenario risk
- Lower absolute values are better (less severe drawdowns)

#### E. **3-Month Recent Momentum** (`app.py:344-348`)
```python
recent_return = float((prices.iloc[-1] / prices.iloc[-63]) - 1)
```
- Short-term performance indicator
- Compares current price to price 63 days ago (~3 months)
- Higher values indicate recent outperformance

### 3. Scoring System (`app.py:378-400`)

#### Score Normalization
Each metric is normalized to a 0-1 scale using:

```python
def safe_normalize(series, inverse=False):
    range_val = series.max() - series.min()
    if range_val == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    normalized = (series - series.min()) / range_val
    return 1 - normalized if inverse else normalized
```

- **Direct metrics** (Return, Sharpe, Momentum): Higher raw values = higher scores
- **Inverse metrics** (Volatility, Max Drawdown): Lower raw values = higher scores

#### Individual Scores
- `Return_Score` = normalized Annual Return
- `Vol_Score` = inverse normalized Volatility  
- `Sharpe_Score` = normalized Sharpe Ratio
- `Drawdown_Score` = inverse normalized Max Drawdown
- `Momentum_Score` = normalized Recent 3M Return

#### Composite Score Formula (`app.py:394-400`)
```python
Composite_Score = (
    0.25 * Return_Score +      # 25% weight - Long-term performance
    0.20 * Vol_Score +         # 20% weight - Risk management  
    0.25 * Sharpe_Score +      # 25% weight - Risk-adjusted returns
    0.15 * Drawdown_Score +    # 15% weight - Downside protection
    0.15 * Momentum_Score      # 15% weight - Recent momentum
)
```

**Weighting Rationale:**
- **Return & Sharpe (50%)**: Primary focus on performance and risk-adjusted returns
- **Volatility (20%)**: Significant weight on stability
- **Drawdown & Momentum (30%)**: Balanced consideration of risk protection and recent trends

### 4. Ranking and Top 10 Selection (`app.py:402-403`, `app.py:650`)

```python
df = df.sort_values('Composite_Score', ascending=False)
top_10 = analysis_results.head(10)
```

Stocks are ranked by **Composite Score** in descending order, and the **top 10 highest-scoring stocks** are selected as recommendations.

## Caching System

### Data Persistence (`app.py:222-283`)
- **Analysis Cache**: `sp500_data/sp500_analysis.parquet`
- **Metadata Cache**: `sp500_data/metadata.parquet`
- **Cache Validity**: 24 hours
- **Force Refresh**: Available to bypass cache

### Cache Benefits
- **Performance**: Avoids re-downloading 100 stocks repeatedly
- **Rate Limiting**: Prevents API throttling from Yahoo Finance
- **User Experience**: Near-instant results for subsequent analyses

## User Interface Features

### Analysis Tab (`app.py:611-737`)

1. **Period Selection**: 6M, 1Y, 2Y, 3Y analysis periods
2. **Analyze Button**: Triggers fresh analysis or loads cached data  
3. **Top 10 Display**: Formatted table showing:
   - Ticker symbol
   - Annual Return (%)
   - Sharpe Ratio
   - Volatility (%)
   - Max Drawdown (%)
   - 3-Month Return (%)
   - Current Price ($)
   - Composite Score

4. **Action Buttons**:
   - **"Use Top 10 for Portfolio Optimization"**: Transfers top stocks to optimizer
   - **"Download Full Analysis"**: CSV export of all 100 stocks

5. **Data Management**:
   - Cache size indicator
   - Clear cache functionality
   - Analysis summary statistics

### Integration with Portfolio Optimizer

The analyzer seamlessly integrates with the portfolio optimizer through three selection modes:

1. **"Use Top Performers from Analysis"** (`app.py:89-130`)
   - Automatically uses top N stocks from analysis
   - Configurable slider (5-25 stocks)
   - Real-time validation of cache freshness

2. **"Custom from Analysis"** (`app.py:131-150`)
   - Manual selection from analyzed stocks
   - Displays analysis scores for selected stocks
   - Validation against available analysis data

3. **"Manual Entry"** (`app.py:72-88`)
   - Free-form ticker entry
   - Default recommendations from analyzer
   - Fallback option when no analysis available

## Key Algorithm Strengths

### 1. **Multi-Factor Approach**
Combines return, risk, efficiency, and momentum metrics for holistic evaluation

### 2. **Risk-Adjusted Focus**
Emphasizes Sharpe ratio and drawdown protection alongside raw returns

### 3. **Balanced Weighting**
Thoughtful allocation across different performance dimensions

### 4. **Scalable Architecture**
Efficient processing of 100 stocks with progress tracking

### 5. **Robust Data Handling**
Error handling, rate limiting, and data quality checks

### 6. **User-Centric Design**  
Caching, progress indicators, and seamless integration with optimization workflow

## Technical Implementation

### Error Handling (`app.py:363-365`)
```python
except Exception as e:
    st.warning(f"Failed to analyze {ticker}: {str(e)}")
    continue
```
- Individual stock failures don't stop the entire analysis
- Warning messages for transparency
- Continues processing remaining stocks

### Rate Limiting (`app.py:368`)
```python
time.sleep(0.2)  # Rate limiting for larger dataset
```
- 0.2-second delay between API calls
- Prevents Yahoo Finance rate limiting
- Balances speed vs. reliability

### Progress Tracking (`app.py:315-317`, `app.py:367`)
```python
progress_bar = st.progress(0)
status_text = st.empty()
# ... 
progress_bar.progress((i + 1) / len(tickers))
```
- Real-time progress bar
- Current stock being analyzed
- Completion percentage

This comprehensive system provides users with data-driven stock recommendations based on quantitative analysis of historical performance, risk metrics, and recent momentum across 100 major S&P 500 companies.

---

# Portfolio Optimization - Optimal Weights Algorithm

## Overview

After stock selection (either from the S&P 500 analyzer or manual entry), the system uses **Modern Portfolio Theory (MPT)** with the **PyPortfolioOpt** library to calculate optimal portfolio weights that maximize risk-adjusted returns.

## Algorithm Process

### 1. **Statistical Analysis** (`app.py:539-542`)

The system first computes two fundamental statistical measures:

```python
def compute_stats(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    mu = expected_returns.mean_historical_return(prices)     # annualized expected returns
    S = risk_models.sample_cov(prices)                       # annualized covariance matrix
    return mu, S
```

#### A. **Expected Returns (μ)** 
- **Method**: Mean Historical Return
- **Calculation**: Average historical returns, annualized (252 trading days)
- **Formula**: `μ_i = (∑ daily_returns_i / n_days) × 252`
- **Purpose**: Estimates future expected return for each asset

#### B. **Covariance Matrix (Σ)**
- **Method**: Sample Covariance
- **Calculation**: Historical covariance between all asset pairs, annualized
- **Purpose**: Measures how assets move together (correlation × volatilities)
- **Matrix Size**: N×N where N = number of assets

### 2. **Efficient Frontier Setup** (`app.py:545-558`)

The system creates an **EfficientFrontier** object with constraints:

#### Dynamic Constraint Adjustment
```python
n_assets = len(prices.columns)
if n_assets <= 3:
    # Small portfolios: Relaxed constraints
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    l2 = 0  # Disable L2 regularization
else:
    # Larger portfolios: Apply weight limits
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w / 100.0))
    if l2 and l2 > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=l2)
```

#### Weight Constraints
- **Lower Bound**: 0 (no short selling)
- **Upper Bound**: 
  - Small portfolios (≤3 assets): 100% (no limit)
  - Larger portfolios: User-defined max weight (default 20%)

#### L2 Regularization (Optional)
- **Purpose**: Prevents over-concentration in few assets
- **Effect**: Encourages more diversified portfolios
- **Parameter**: γ (gamma) controls regularization strength

### 3. **Optimization Objectives** (`app.py:560-572`)

The system supports three optimization strategies:

#### A. **Maximum Sharpe Ratio** (Primary)
```python
if obj == "Max Sharpe":
    ef.max_sharpe(risk_free_rate=rf / 100.0)
```
- **Goal**: Maximize return per unit of risk
- **Formula**: Sharpe = (Return - Risk_Free_Rate) / Volatility  
- **Most Common**: Balances return and risk optimally

#### B. **Target Return with Minimum Volatility**
```python
elif obj == "Min Volatility (target return)":
    ef.efficient_return(target_return=tgt / 100.0)
```
- **Goal**: Achieve specific return with lowest possible risk
- **Constraint**: Portfolio must achieve user-specified target return
- **Use Case**: Conservative investors with return requirements

#### C. **Efficient Return** (Fallback)
```python
else:  # "Target Return" 
    ef.efficient_return(target_return=tgt / 100.0)
```
- **Same as above**: Target return approach
- **Alternative naming**: For UI clarity

### 4. **Fallback Mechanisms** (`app.py:573-580`)

The algorithm includes robust error handling:

#### Level 1 Fallback: Minimum Volatility
```python
except:
    try:
        ef.min_volatility()
```
- **Trigger**: When target return is unachievable
- **Goal**: Find lowest risk portfolio regardless of return

#### Level 2 Fallback: Equal Weights
```python
except:
    equal_weights = {asset: 1/n_assets for asset in prices.columns}
    ret, vol, sharpe = ef.portfolio_performance(weights=equal_weights, risk_free_rate=rf / 100.0)
    return pd.Series(equal_weights), {"expected_return": ret, "volatility": vol, "sharpe": sharpe}
```
- **Last Resort**: When optimization fails completely
- **Method**: Simple equal weighting (1/N for each asset)
- **Guaranteed**: Always produces valid portfolio

### 5. **Weight Cleaning & Performance Calculation** (`app.py:582-584`)

```python
cleaned = ef.clean_weights(cutoff=0)
ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=rf / 100.0, verbose=False)
return pd.Series(cleaned), {"expected_return": ret, "volatility": vol, "sharpe": sharpe}
```

#### Weight Cleaning
- **Cutoff**: Removes weights below threshold (0 = keeps all non-zero)
- **Purpose**: Eliminates tiny allocations for practical implementation

#### Performance Metrics
- **Expected Return**: Annualized portfolio return
- **Volatility**: Annualized portfolio standard deviation
- **Sharpe Ratio**: Risk-adjusted return measure

### 6. **Post-Optimization Processing** (`app.py:587-595`)

#### Minimum Holdings Enforcement
```python
def enforce_min_holdings(weights: pd.Series, min_n: int, prune_below_pct: float) -> pd.Series:
    if weights.empty:
        return weights
    keep = weights[weights >= (prune_below_pct / 100.0)].sort_values(ascending=False)
    if len(keep) >= min_n:
        return keep / keep.sum()  # Normalize to sum = 1
    top = weights.sort_values(ascending=False).head(min_n)
    return top / top.sum()  # Force minimum number of holdings
```

#### Two-Stage Filtering
1. **Threshold Filter**: Remove weights below minimum threshold (default 1%)
2. **Minimum Holdings**: Ensure at least N assets (default 3) in final portfolio
3. **Re-normalization**: Weights always sum to 1.0

## Mathematical Foundation

### Modern Portfolio Theory Optimization

The optimization solves this mathematical problem:

#### For Maximum Sharpe Ratio:
**Maximize**: `(w^T μ - rf) / √(w^T Σ w)`

Where:
- **w**: Portfolio weights vector (what we're solving for)
- **μ**: Expected returns vector  
- **Σ**: Covariance matrix
- **rf**: Risk-free rate

#### Subject to constraints:
- **∑ wi = 1** (weights sum to 100%)
- **0 ≤ wi ≤ wmax** (no short selling, maximum position size)
- **Optional**: L2 regularization penalty

### Portfolio Performance Formulas

#### Expected Portfolio Return:
**E[Rp] = w^T μ**

#### Portfolio Volatility:
**σp = √(w^T Σ w)**

#### Portfolio Sharpe Ratio:
**Sharpe = (E[Rp] - rf) / σp**

## Key Algorithm Strengths

### 1. **Mathematically Rigorous**
- Based on Nobel Prize-winning Modern Portfolio Theory
- Quadratic optimization with linear constraints
- Globally optimal solutions (convex optimization)

### 2. **Practical Constraints**
- Position size limits prevent over-concentration
- Minimum holdings ensure diversification
- L2 regularization reduces extreme positions

### 3. **Robust Error Handling**
- Multiple fallback strategies
- Handles singular matrices and infeasible constraints
- Always produces valid portfolio

### 4. **Flexible Objectives**
- Risk-return trade-off optimization (Max Sharpe)
- Risk minimization for target returns
- Adaptable to different investor preferences

### 5. **Professional Implementation**
- Uses industry-standard PyPortfolioOpt library
- Handles edge cases (small portfolios, numerical issues)
- Performance attribution and reporting

This optimization process transforms the selected stocks from the S&P 500 analyzer into a mathematically optimal portfolio that balances expected returns with risk, subject to practical investment constraints.

---

# Data Storage Architecture

## Storage Overview

The system uses a **hybrid storage approach** that balances performance with data freshness:

### 1. **Persistent Cache Storage** 💾
**Location**: `sp500_data/` directory
- **Analysis Results**: `sp500_analysis.parquet` (24KB)
  - Pre-calculated metrics for 100 S&P 500 stocks
  - 24-hour cache validity
  - Avoids 5-7 minute re-analysis process
  
- **Cache Metadata**: `metadata.parquet` (4KB)  
  - Timestamps, analysis periods, version info
  - Cache validation data

### 2. **In-Memory Caching** 🧠
**Mechanism**: Streamlit's `@st.cache_data` decorator
- **Price Data**: Session-level caching of Yahoo Finance downloads
- **User State**: Form inputs and selections in session state
- **Duration**: Until page refresh or session timeout

### 3. **External Data Sources** 🌐
**Yahoo Finance API**: Primary source for all historical price data
- Always fetched fresh for portfolio optimization
- Rate-limited (handled with batching and delays)
- No long-term storage of raw price data

## Data Flow Patterns

### Analysis Data (Cached)
1. **Cache Check**: Load existing analysis if < 24 hours old
2. **Fresh Analysis**: Download → Calculate → Save to Parquet
3. **Retrieval**: Fast Parquet read (~50ms vs 5-7 minutes)

### Price Data (Session Cache)  
1. **Memory Check**: Use cached data if parameters match
2. **API Download**: Fetch from Yahoo Finance if cache miss
3. **Processing**: Clean and format for optimization
4. **No Persistence**: Data not saved to disk

## Key Benefits

### ✅ **Performance Optimization**
- **Analysis caching** eliminates repeated expensive computations
- **Session caching** reduces redundant API calls during UI interactions
- **Parquet format** provides fast, compressed storage

### ✅ **Data Freshness Balance**  
- **Analysis results** cached for efficiency (24-hour TTL)
- **Price data** always current for portfolio optimization
- **Force refresh** available to bypass cache when needed

### ✅ **Resource Efficiency**
- **28KB total** for complete analysis cache
- **Rate limiting** respects Yahoo Finance API constraints  
- **Memory management** via Streamlit's automatic cache cleanup

## Technical Implementation

### Cache Management (`app.py:222-283`)
```python
def save_analysis_data(df: pd.DataFrame, period: str):
    df.to_parquet(ANALYSIS_FILE, index=False)  # Main data
    metadata.to_parquet(METADATA_FILE, index=False)  # Cache info

def is_data_stale(metadata: dict, max_age_hours: int = 24) -> bool:
    age = datetime.now() - metadata['last_updated']
    return age > timedelta(hours=max_age_hours)
```

### Price Loading (`app.py:413-526`) 
```python
@st.cache_data(show_spinner=True)  # Automatic session caching
def load_prices(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    # Batched downloads with rate limiting
    # Cache key: hash(tickers, period, interval)
```

## Storage Locations

- **Analysis Cache**: `sp500_data/sp500_analysis.parquet`
- **Metadata**: `sp500_data/metadata.parquet`  
- **Price Data**: In-memory only (not persisted)
- **User State**: Streamlit session state (temporary)

This architecture ensures **fast user experience** while maintaining **data accuracy** and **API efficiency**.

**📋 Complete technical documentation**: `DATA_STORAGE_ARCHITECTURE.md`
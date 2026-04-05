# S&P 500 Portfolio Optimizer — User Guide

This guide walks through every screen of the tool, explains what each number means, and shows the recommended workflows for screening stocks, forecasting returns, and building an optimised portfolio.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [First-Time Data Setup](#2-first-time-data-setup)
3. [Tab Overview](#3-tab-overview)
4. [Tab 1 — S&P 500 Stock Analyzer](#4-tab-1--sp500-stock-analyzer)
5. [Tab 2 — Forecast](#5-tab-2--forecast)
6. [Tab 3 — Portfolio Optimizer](#6-tab-3--portfolio-optimizer)
7. [Tab 4 — Data Health](#7-tab-4--data-health)
8. [Tab 5 — Ticker Universe](#8-tab-5--ticker-universe)
9. [End-to-End Workflow](#9-end-to-end-workflow)
10. [Understanding the Scores](#10-understanding-the-scores)
11. [Understanding the Forecast](#11-understanding-the-forecast)
12. [Tips and Gotchas](#12-tips-and-gotchas)

---

## 1. Quick Start

```bash
# Start the stack
./run-microservices.sh

# Open the UI
open http://localhost:8501
```

Or with Docker:

```bash
docker compose up -d
open http://localhost:8501
```

If this is your first time, go to [First-Time Data Setup](#2-first-time-data-setup) before opening the UI.

---

## 2. First-Time Data Setup

The tool analyses historical price data stored locally. You need to seed the cache once before the UI is useful.

### Step 1 — Download prices

```bash
./run-price-sync.sh
```

Downloads 5 years of daily close prices for all S&P 500 constituents from Yahoo Finance.
Takes **5–15 minutes** depending on network speed. Progress is printed to the terminal.

On completion you will see a line like:
```
✅  Price sync complete — 503 tickers loaded, data through 2026-04-04
```

### Step 2 — Run analysis

```bash
./run-analysis-sync.sh
```

Computes Sharpe ratios, drawdowns, volatility, momentum scores, and composite rankings for four time periods (1Y, 2Y, 3Y, 5Y). Writes the results to `sp500_data/`.
Takes **2–5 minutes**.

### Step 3 — Start the UI

```bash
./run-microservices.sh
# open http://localhost:8501
```

### Keeping data fresh

The analysis cache is considered **stale after 7 days**. Re-run both sync scripts weekly, or set up a cron job:

```bash
# Example cron — every Sunday at 06:00
0 6 * * 0 cd /path/to/fin-portfolio && ./run-price-sync.sh && ./run-analysis-sync.sh
```

---

## 3. Tab Overview

The UI has five tabs accessed from the top navigation bar.

| Tab | What it does |
|---|---|
| 🔍 S&P 500 Stock Analyzer | Screen all 500 stocks; sort by any metric; export to CSV |
| 📈 Forecast | 1Y / 2Y return forecasts using an ensemble of three models |
| ⚙️ Portfolio Optimizer | Build an efficient-frontier portfolio from any stock selection |
| 🏥 Data Health | Check cache freshness; see when data was last synced |
| 🌍 Ticker Universe | View S&P 500 constituents; see recent additions / removals |

Recommended order: **Analyzer → Forecast → Optimizer**.

---

## 4. Tab 1 — S&P 500 Stock Analyzer

### Purpose

Screen and rank all S&P 500 stocks on six financial metrics, then select a subset to carry into the Optimizer or Forecast tab.

### Controls

| Control | What it does |
|---|---|
| **Analysis Period** | Choose 1Y, 2Y, 3Y, or 5Y lookback. Longer periods smooth out noise but may include pre-COVID or rate-cycle regimes that are no longer relevant. |
| **🚀 Analyze Stocks** | Fetches pre-computed analysis from the cache. Does nothing if already loaded; use **🔄 Refresh** to force a reload. |
| **Sort column** | Click any column header to re-sort in place. |
| **Use Top 20 for Portfolio Optimization** | Copies the top 20 ranked stocks to the Optimizer's ticker selection. |

### Columns explained

| Column | Definition |
|---|---|
| **Composite_Score** | Weighted average of the five component scores below. Range 0–1; higher is better. |
| **Annual_Return** | Compound annualised return (CAGR) over the selected period. |
| **Sharpe_Ratio** | Risk-adjusted return. `(Annual_Return − Risk_Free_Rate) / Annual_Volatility`. Uses a 0% risk-free rate for screening. |
| **Max_Drawdown** | Largest peak-to-trough decline during the period, expressed as a negative decimal (e.g. `−0.35` = 35% loss). Less negative is better. |
| **Recent_3M_Return** | Simple return over the last 63 trading days (~3 calendar months). |
| **Recent_12M_Return** | Return over the last 252 trading days, skipping the most recent month (Jegadeesh & Titman 12-1M momentum signal). |
| **Volatility** | Annualised standard deviation of daily log returns. Lower is calmer. |

### Composite score weights

| Component | Weight | Why |
|---|---|---|
| Sharpe_Ratio | 25% | Best single risk-adjusted measure |
| Annual_Return | 20% | Raw return matters, but less than Sharpe |
| Max_Drawdown | 20% | Downside risk is penalised independently |
| Momentum_Score | 20% | Blended 3M + 12M momentum signal |
| Volatility | 15% | Lower vol rewarded, inversely scored |

> **Scoring method:** Sharpe, Return, and Momentum use percentile rank (robust to outliers like NVDA). Drawdown and Volatility use min-max normalisation.

### Typical use

1. Select **1Y** for recent performance or **3Y** to include a full rate cycle.
2. Click **🚀 Analyze Stocks**.
3. Sort by **Composite_Score** descending to see the top stocks.
4. Review the top 20–30 manually — look for a mix of sectors.
5. Click **Use Top 20 for Portfolio Optimization** when ready to proceed.

---

## 5. Tab 2 — Forecast

### Purpose

Generate 1-year and 2-year return forecasts for selected stocks using a three-model ensemble, visualised as fan charts (Monte Carlo simulation paths).

### Controls

| Control | What it does |
|---|---|
| **Tickers** | Comma-separated list, or populated automatically from the Analyzer's top performers. |
| **Number of simulations** | Monte Carlo paths per ticker. 500 is fast; 2000 is smoother. |
| **🚀 Run Forecast** | Runs all three models and displays results. |
| **Fan charts — number of tickers** | Slider to show N tickers' fan charts on screen simultaneously. |
| **📤 Use Forecast Returns in Optimizer** | Sends the ensemble 1Y returns to the Optimizer as expected-return overrides. |

### Results tables

**Forecast Summary** — one row per ticker:

| Column | Definition |
|---|---|
| **ensemble_return_1y** | Median of the three models' 1Y forecasts. This is the headline number. |
| **ensemble_return_2y** | Compounded 2Y ensemble forecast. |
| **mc_p10_1y / mc_p90_1y** | Monte Carlo 10th and 90th percentile 1Y outcomes — the uncertainty band. |
| **beta** | Systematic risk relative to SPY. Beta > 1 = more volatile than the market. |
| **capm_return_1y** | CAPM-implied expected return based on beta and market premium. |
| **trend_return_1y** | Log-linear regression extrapolation of the historical price trend. |

**Fan Charts** — one chart per ticker showing:
- Grey paths: individual Monte Carlo simulations
- Blue line: median simulated path
- Orange dashed line: SPY deterministic CAGR projection (benchmark)

### Return caps

Forecast returns are clamped to **+50% / −90% per year** to prevent unrealistic extrapolations (e.g. a stock that ran +400% in one year will not be forecast to do so again).

---

## 6. Tab 3 — Portfolio Optimizer

### Purpose

Build a mean-variance optimal portfolio from a selected set of stocks using Modern Portfolio Theory (Markowitz efficient frontier).

### Controls

| Control | What it does |
|---|---|
| **Tickers** | Stocks to include. Populated from the Analyzer or typed manually. |
| **Investment Amount ($)** | Total dollar amount to allocate. Used to compute discrete share counts. |
| **Objective** | Optimisation target (see below). |
| **Risk-Free Rate (%)** | Annual rate for Sharpe calculation. Default 4% (current ~T-bill rate). |
| **Max Weight per Stock (%)** | Hard upper bound on any single position. Set to 25–33% to force diversification. |
| **L2 Regularisation** | Penalises concentrated weights. Higher values → more equal weights. Has no effect for portfolios of ≤ 3 stocks. |
| **Min Weight Threshold (%)** | Positions below this fraction are dropped and the remaining weights are renormalized. Removes rounding noise. |
| **Minimum Holdings** | Portfolio must contain at least this many stocks. If optimisation concentrates below this, smallest positions are kept. |
| **Expected Returns Source** | Choose **Historical** (from price history) or **Use Forecast Returns** (from the Forecast tab via the bridge button). |
| **🚀 Optimize Portfolio** | Runs the optimisation. Results persist until you click again. |

### Objectives

| Objective | Best for |
|---|---|
| **Max Sharpe** | Best risk-adjusted return. The classic efficient-frontier tangency portfolio. |
| **Min Volatility** | Lowest-variance portfolio. Suitable if you want to minimise drawdown. |
| **Min Volatility (target return)** | Find the lowest-volatility portfolio that hits a specific return target. |

### Results

**Allocation table** — shares to buy:

| Column | Definition |
|---|---|
| **Weight** | Fraction of portfolio. Should sum to 1.0. |
| **Shares** | Whole shares to purchase given the investment amount. |
| **Est. Cost** | `Shares × Latest Price`. |

**Performance metrics** (three `st.metric` cards):

| Metric | Definition |
|---|---|
| **Expected Return** | Annualised return predicted by the optimiser. |
| **Volatility** | Annualised standard deviation of the optimised portfolio. |
| **Sharpe Ratio** | `(Expected Return − Risk_Free_Rate) / Volatility`. |

**Historical Performance** chart — shows the normalised price history for each stock in the portfolio, overlaid.

**Leftover Cash** — the unallocated amount after buying whole shares.

### Forecast → Optimizer Bridge

1. Run a forecast on your selected stocks.
2. Click **📤 Use Forecast Returns in Optimizer** in the Forecast tab.
3. Switch to the Optimizer tab — a green info box confirms the overrides are loaded.
4. Select **Use Forecast Returns** in the Expected Returns Source radio.
5. Click **🚀 Optimize Portfolio**.

The portfolio weights will now reflect the forward-looking ensemble forecasts rather than historical returns. This often shifts allocation toward stocks with improving momentum and away from past winners that may be mean-reverting.

---

## 7. Tab 4 — Data Health

### Purpose

Inspect the freshness of all cached data. Use this tab to diagnose stale analysis or missing price history before running the Analyzer or Optimizer.

### What you see

**Price Cache** table — one row per period (1Y / 2Y / 3Y / 5Y):

| Column | Definition |
|---|---|
| **Period** | Lookback window for the price series. |
| **Last Synced** | When the cache file was last written to disk. |
| **Data Through** | Latest date present in the price data. Should be within 1–3 trading days of today. |
| **Tickers** | Number of tickers loaded for this period. |
| **Age** | How old the cache is in hours. Yellow if > 24 h; red if > 168 h (7 days). |

**Analysis Cache** table — one row per period:

| Column | Definition |
|---|---|
| **Fresh / Stale** | Fresh = computed within the last 7 days. Stale = older; re-run `./run-analysis-sync.sh`. |
| **Last Updated** | Timestamp of the last analysis run. |
| **Stocks** | Number of stocks in the analysis snapshot. |

### Status indicators

- **✅ Fresh** — data is current; the Analyzer will use this cache.
- **⚠️ Stale** — data is older than 7 days; results may not reflect recent market moves.
- **❌ Missing** — no cache found; the Analyzer will show an error. Run `./run-analysis-sync.sh`.

---

## 8. Tab 5 — Ticker Universe

### Purpose

See which stocks are currently in the S&P 500 index and track constituent changes over time.

### What you see

**Current Universe** — the full list of S&P 500 tickers fetched from Wikipedia on last sync. Shows total count.

**Recent Changes** — additions and removals detected between consecutive validation snapshots:

| Column | Definition |
|---|---|
| **Date** | When the change was detected. |
| **Added** | Tickers that appeared in the index since the previous snapshot. |
| **Removed** | Tickers that left the index. |

This is useful for understanding why a stock might have appeared or disappeared from your analysis results.

### Refreshing the universe

Click **🔄 Refresh** to fetch the latest S&P 500 constituents from Wikipedia immediately. This updates the ticker cache used by the price sync scripts.

---

## 9. End-to-End Workflow

### Workflow A — Screen and Optimise (Historical Returns)

```
1. Analyzer tab
   ├── Select period (e.g. 2Y)
   ├── Click 🚀 Analyze Stocks
   ├── Sort by Composite_Score
   └── Click "Use Top 20 for Portfolio Optimization"

2. Optimizer tab
   ├── Confirm tickers are populated
   ├── Set Investment Amount
   ├── Set Max Weight = 25%  (forces at least 4 positions)
   ├── Set Objective = Max Sharpe
   └── Click 🚀 Optimize Portfolio

3. Review
   ├── Check Expected Return, Volatility, Sharpe cards
   ├── Review allocation table — sensible sector mix?
   └── Export to CSV with 📥 Download
```

### Workflow B — Forecast-Driven Optimisation

```
1. Analyzer tab
   ├── Analyze Stocks (1Y period)
   └── Click "Use Top 20 for Portfolio Optimization"

2. Forecast tab
   ├── Confirm tickers
   ├── Set simulations = 1000
   ├── Click 🚀 Run Forecast
   ├── Review fan charts — which stocks have wide uncertainty bands?
   └── Click 📤 Use Forecast Returns in Optimizer

3. Optimizer tab
   ├── Set Expected Returns Source = Use Forecast Returns
   ├── Set Max Weight = 30%
   └── Click 🚀 Optimize Portfolio

4. Compare
   ├── Note which positions increased vs. Workflow A
   └── Stocks with higher ensemble_return_1y should have higher weights
```

### Workflow C — Conservative Income Portfolio

```
1. Analyzer tab
   ├── Select 3Y (full rate cycle)
   └── Analyze Stocks

2. Sort by Sharpe_Ratio (high) and filter for low Volatility visually

3. Pick 8–12 low-volatility names manually

4. Optimizer tab
   ├── Paste chosen tickers
   ├── Objective = Min Volatility
   ├── Max Weight = 20%
   └── Optimize
```

---

## 10. Understanding the Scores

### Why percentile rank instead of min-max?

The S&P 500 always contains a few extreme outliers (e.g. NVDA with +200% annual return). Min-max normalisation collapses all other stocks to near zero, making it appear that only the outlier is worth considering. Percentile rank distributes scores evenly — the top 1% of performers always score near 1.0, the bottom 1% near 0.0, regardless of outlier magnitude.

### Why is my favourite stock ranked low?

Common reasons:
- **High volatility**: a 40% return with 60% volatility has a lower Sharpe than a 20% return with 15% volatility.
- **Large drawdown**: a stock that dropped 50% during the period is penalised even if it recovered.
- **Weak momentum**: recent 3M and 12M returns are below the S&P median.

### Why do scores change when I switch periods?

Each period uses only the prices from that window. A stock that performed well over 5 years may have had a poor 1Y if it corrected recently. The 2Y or 3Y period often provides the best balance between recency and statistical stability.

---

## 11. Understanding the Forecast

### Three models

| Model | How it works | Best when |
|---|---|---|
| **Monte Carlo GBM** | Simulates 1000 price paths using the stock's historical drift and volatility. Returns the median simulated outcome. | Volatility is stable and mean-reverting |
| **CAPM** | Uses `β × (Market Premium)` to project forward return based on systematic risk. | Stock follows the market closely (β ≈ 1) |
| **Trend Regression** | Fits a log-linear trend to historical prices and extrapolates. | Strong momentum names with consistent uptrends |

### Ensemble

The headline `ensemble_return_1y` is the **median of all three models** when all are available. This reduces the impact of any single model's assumptions.

### What the uncertainty band means

The `mc_p10` to `mc_p90` range is the Monte Carlo 10th–90th percentile. A **narrow band** means the model is confident (low historical volatility). A **wide band** means outcomes are highly uncertain — treat the forecast as directional guidance only.

### Why is the 2Y forecast not simply 2× the 1Y?

Returns compound. A 30% 1Y forecast compounds to `1.30² − 1 = 69%` over 2Y under constant drift. In practice the 2Y figure is also capped at `(1 + 50%)² − 1 = 125%` to prevent compounding of extreme 1Y caps.

### When to distrust a forecast

- The stock has fewer than 1 year of price history (CAPM and trend have insufficient data)
- The stock had a one-time event in its history (spin-off, merger, re-listing) causing an artificial price jump
- `mc_p90_1y − mc_p10_1y > 1.5` (very wide band — model uncertainty exceeds signal)

---

## 12. Tips and Gotchas

**"Analysis request failed: Not Found"**
Another process is using port 8001. Run `lsof -i :8001` to identify it, then kill it and restart: `./run-microservices.sh`.

**Stale cache warning even after re-sync**
Click **🔄 Refresh** on the Data Health tab. The UI caches health status per session; a refresh forces a new API call.

**Optimizer returns only 1–2 stocks**
Lower the `Min Weight Threshold` setting or increase the number of input tickers. With very few tickers and strong concentration by the optimizer, the min-holdings pruning kicks in.

**Fan charts are very wide / unrealistic**
High-volatility stocks naturally produce wide fans. Try a longer price history (2Y+) to get a more stable volatility estimate, or check the `mc_p10/p90` column — if the p90 is capped at 50% the drift clamping is active.

**Expected Return in Optimizer looks too high**
If you used **Use Forecast Returns**, check the `ensemble_return_1y` values in the Forecast tab. Returns above 40% are within the allowed cap but represent optimistic scenarios; consider running with **Historical** returns for a more conservative allocation.

**Port conflict on startup**
Services use ports 8000, 8001, 8002, 8501. Check for conflicts with `lsof -i :8000` etc. Override ports in Docker with environment variables: `UI_PORT=9501 docker compose up`.

**First price sync is slow**
Yahoo Finance rate-limits bulk downloads. The sync script uses a small delay between batches. 15 minutes for 503 tickers is normal on a residential connection.

**How to reset everything**
```bash
rm -rf sp500_data/
./run-price-sync.sh
./run-analysis-sync.sh
```

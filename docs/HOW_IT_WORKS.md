# How It Works — Plain-Language Algorithm Guide

This document explains every algorithm the tool uses in plain language, with no formulas or code. It answers the question: *"What is the software actually doing when I click a button?"*

---

## Table of Contents

1. [How stocks are scored (the Analyzer)](#1-how-stocks-are-scored-the-analyzer)
2. [How the composite ranking is built](#2-how-the-composite-ranking-is-built)
3. [How the forecast works](#3-how-the-forecast-works)
4. [How the portfolio optimizer works](#4-how-the-portfolio-optimizer-works)
5. [How prices are kept up to date](#5-how-prices-are-kept-up-to-date)
6. [How the backtester works](#6-how-the-backtester-works)
7. [How the comparison works](#7-how-the-comparison-works)
8. [The end-to-end workflow](#8-the-end-to-end-workflow)

---

## 1. How Stocks Are Scored (the Analyzer)

When you click **Analyze Stocks**, the tool looks at the daily closing price of every S&P 500 stock over the chosen time period (1, 2, 3, or 5 years) and computes five independent measures of quality. Think of it as five different judges scoring the same contestant.

---

### Annual Return — "How much money did it make?"

The tool calculates what percentage your investment would have grown by per year if you had bought the stock at the start of the period and held it.

> *Example: A stock that grew from $100 to $161 over 3 years has an annual return of about 17%, because 17% compounded for three years gets you to 161.*

The tool uses the **geometric (compound) average**, not a simple average. This is the correct way to measure investment growth because it accounts for compounding — earning returns on top of earlier returns.

---

### Volatility — "How bumpy was the ride?"

Volatility measures how wildly the stock's price bounced around from day to day. A stock that moves 3% every day is much more volatile than one that moves 0.3% per day, even if both ended up at the same place after a year.

> *Think of two roads from London to Edinburgh. Both get you there in the same time, but one goes in a straight line and the other zigzags over mountains. Volatility measures the zigzag.*

Higher volatility is worse. The tool rewards steadier stocks.

---

### Sharpe Ratio — "Was the return worth the risk?"

The Sharpe ratio is the central measure of investment quality. It asks: *how much return did the stock deliver per unit of risk taken?*

A stock that returned 20% but was extremely volatile deserves less credit than one that returned 15% very smoothly. The Sharpe ratio captures exactly this trade-off.

> *Imagine two delivery drivers. One delivers 20 packages a day but crashes the van every week. The other delivers 15 packages a day with a perfect record. The Sharpe ratio picks the safer driver.*

A Sharpe ratio above 1.0 is generally considered good. Above 2.0 is excellent.

---

### Maximum Drawdown — "What was the worst loss?"

Maximum drawdown records the single steepest fall the stock experienced during the period — from its highest point to its lowest before recovering.

> *If a stock reached $200 and then fell to $110 before bouncing back, its max drawdown is −45%. That −45% is the worst the investor felt during that period.*

The number is always negative (since it measures a loss). A drawdown of −10% is much better than −50%. The tool rewards stocks that hold their value during downturns.

---

### Momentum — "Is it going in the right direction recently?"

Momentum measures whether a stock has been trending upward recently. It uses two windows:

- **3-Month Momentum**: how much the stock rose or fell over the last ~3 months.
- **12-Month Momentum**: how much the stock rose over the past year, **excluding the most recent month**. This specific approach — looking at months 2 through 12, skipping the most recent — is a well-known signal from academic research (Jegadeesh & Titman, 1993). It captures the phenomenon that stocks trending up over the past year tend to keep trending up for a few more months.

> *Think of it like a cyclist. A rider who has been accelerating for the past 11 months is more likely to be moving fast next month than one who has been decelerating, even if both are at the same speed right now.*

---

## 2. How the Composite Ranking Is Built

After computing the five measures above, the tool needs to combine them into a single ranking. This is a two-step process.

---

### Step 1 — Make scores comparable (normalisation)

Each measure uses different units: returns are percentages, Sharpe ratios are pure numbers, drawdown is a negative fraction. Before combining them, the tool converts every measure to a 0–1 scale.

**For Annual Return and Momentum**, the tool uses **percentile ranking**:

> *Each stock is compared only to other stocks in the batch. A score of 0.90 means "this stock's return was higher than 90% of other stocks analysed."*

This approach is deliberately chosen to be **outlier-resistant**. If one stock like NVIDIA returns +200% while most others return 10–15%, a simple rescaling would make every other stock appear nearly identical (all crowded near zero). Percentile ranking avoids this: NVIDIA gets the top score, the second-best stock gets the second-highest score, and so on, regardless of how extreme the outlier is.

**For Volatility, Sharpe Ratio, and Drawdown**, the tool uses **min-max rescaling**:

> *The worst stock gets 0.0, the best gets 1.0, and everyone else is placed proportionally between them.*

These metrics don't have fat-tail problems, so a direct rescaling is appropriate.

---

### Step 2 — Weighted combination

The five normalised scores are blended with fixed weights:

| Measure | Weight | Why this weight |
|---|---|---|
| Sharpe Ratio | **25%** | The most information-rich single number — captures both return and risk in one figure |
| Annual Return | **20%** | Raw return matters, but counted separately from Sharpe to avoid double-counting |
| Max Drawdown | **20%** | Downside protection is independently important — some investors can't tolerate a 50% loss |
| Momentum | **20%** | Trend-following is a well-documented predictor of near-term performance |
| Volatility | **15%** | Already partially captured by Sharpe; gets a smaller independent weight |

> *The final Composite Score is like a judge at a diving competition giving separate scores for form, height, and difficulty, then combining them. Each judge has a different voting weight.*

**Important:** the scores are relative. A score of 0.80 doesn't mean "this is a good stock in absolute terms." It means "this stock scored better than 80% of its peers in the current batch." If you re-run the analysis with a different set of stocks, every score changes.

---

## 3. How the Forecast Works

When you click **Run Forecast**, the tool runs three separate forecasting models on each stock and then combines their outputs. Using multiple models rather than one is a deliberate choice — no single model is reliably best, so combining them reduces the chance of being very wrong.

---

### Model 1 — Monte Carlo Simulation

**The idea:** The tool runs the stock's recent price history thousands of times as if it were starting fresh, with random noise added each day, to see the range of outcomes.

**How it works in plain language:**

1. The tool measures how much the stock moved each day on average (its "drift") and how wildly it bounced around (its volatility).
2. It then simulates 1,000 imaginary futures. Each future starts at today's price and moves day by day, with a random nudge added each day — the direction and size of the nudge is drawn randomly but calibrated to match the stock's historical behaviour.
3. After running all 1,000 futures out to one year, the tool looks at the distribution of outcomes:
   - The **median** simulated price is the central forecast.
   - The range from the 10th percentile to the 90th percentile forms the uncertainty band shown in the fan chart.

> *Imagine rolling a weighted die 1,000 times to simulate a random walk. Sometimes you get unlucky early and fall behind; sometimes you get a run of lucky rolls. The median outcome after 1,000 games gives you the central expectation. The fan chart shows you the spread of all 1,000 game endings.*

**Limitation:** This model assumes the future will look like the recent past. It does not know about upcoming earnings, economic recessions, or interest rate changes.

---

### Model 2 — CAPM (Capital Asset Pricing Model)

**The idea:** Every stock moves partly because the whole market moves, and partly for its own reasons. CAPM uses only the market-related component to estimate expected return.

**How it works in plain language:**

1. The tool computes the stock's **beta** — a number that measures how much the stock tends to move when the S&P 500 moves. A beta of 1.5 means "when the market rises 10%, this stock tends to rise 15%."
2. It then applies a simple rule: if the market is expected to return about 7–8% per year above the risk-free rate, a stock with beta of 1.5 should return about 10–12% above the risk-free rate.

> *Beta is like a gearing ratio on a bicycle. A stock with beta 2.0 goes twice as fast as the market uphill — and twice as fast downhill.*

**Limitation:** CAPM assumes the relationship between the stock and the market will remain stable. High-beta stocks are not always high-reward; they are simply more sensitive to market swings.

---

### Model 3 — Trend Regression

**The idea:** If a stock has been growing at a steady pace for the past few years, the simplest assumption is that it will continue at roughly that pace.

**How it works in plain language:**

1. The tool draws the best-fit line through the stock's price history on a logarithmic scale (so that 10% growth per year looks like a straight line, regardless of the price level).
2. It extends that line one and two years into the future to get a forecast price.

> *Think of a runner who has been maintaining a 5-minute kilometre pace for the last year. The trend model predicts they'll finish the next 10 km in 50 minutes — a reasonable extrapolation if nothing changes.*

**Limitation:** Trend extrapolation works until it doesn't. A stock that has been growing 40% per year will not keep doing so indefinitely. This is the model most likely to produce optimistic forecasts for recent high-performers.

---

### The Ensemble — Combining All Three

After running all three models, the tool takes the **median** of their three forecasts as the headline prediction. The median is used rather than the average because it is less sensitive to one model producing an extreme outlier.

> *Three doctors independently examine the same patient and each suggests a different diagnosis. The ensemble approach asks: which diagnosis do at least two of the three agree on?*

**Return caps:** All forecasts are clamped to a maximum of +50% per year and a minimum of −90% per year. This prevents absurd extrapolations — a stock that gained +400% last year is not realistically expected to do the same next year.

---

## 4. How the Portfolio Optimizer Works

The optimizer answers a specific question: *given a set of stocks and a fixed amount of money, what fraction of the money should go into each stock to get the best possible risk-adjusted return?*

---

### The core idea — the Efficient Frontier

In the 1950s, Harry Markowitz showed that for any set of assets, there is a curve of optimal portfolios — the **Efficient Frontier**. Every portfolio on this curve has one defining property: you cannot get more return without also taking more risk, and you cannot reduce risk without also reducing return. Any portfolio not on this curve is leaving something on the table.

> *Imagine a map of all possible portfolios as a cloud of points. Each point represents a different way to split your money between the stocks. The Efficient Frontier is the upper-left edge of that cloud — the portfolios that are strictly best.*

The optimizer finds specific points on this frontier depending on your goal.

---

### Objective 1 — Max Sharpe (the default)

This finds the portfolio that maximises the Sharpe ratio — the best return per unit of risk, using the risk-free rate (e.g. 4% from T-bills) as the baseline. Mathematically, it finds the portfolio where the line from the risk-free rate is tangent to the Efficient Frontier.

> *Think of it as finding the steepest hill on a mountain range. You want the portfolio that climbs the most for each step you take.*

---

### Objective 2 — Min Volatility

This finds the least-risky portfolio that can be built from the given stocks. It is appropriate if your primary goal is capital preservation rather than maximum growth.

---

### Expected returns and covariance

The optimizer needs two inputs:

1. **Expected returns** — how much each stock is expected to earn per year. The tool estimates this from historical prices using a geometric average (the same calculation as in the Analyzer). Alternatively, you can supply the ensemble **forecast returns** from the Forecast tab.

2. **Covariance matrix** — how much the stocks move together. If two stocks tend to rise and fall at the same time, owning both doesn't reduce your risk much. The tool uses Ledoit-Wolf shrinkage to estimate this matrix — a technique that is more stable than a simple calculation when the number of stocks is large relative to the amount of history available.

> *The covariance matrix is like measuring how synchronised a group of swimmers are. If they all jump and dive at exactly the same moment, watching one tells you everything about all of them. If they are independent, adding more swimmers genuinely reduces the team's average splash.*

---

### Weight constraints

Left unconstrained, the optimizer almost always concentrates heavily in one or two stocks (the ones with the highest Sharpe). In practice this is undesirable. The tool applies three guardrails:

| Guardrail | What it does |
|---|---|
| **Max Weight per Stock** | No single stock can exceed this percentage. Set to 25–33% to force at least 3–4 positions. |
| **L2 Regularisation** | A mathematical penalty that pushes weights toward equal distribution. Think of it as a tiebreaker that says "if two portfolios have similar Sharpe ratios, pick the more diversified one." |
| **Min Holdings** | If the optimizer produces fewer holdings than this number, the tool keeps the next-largest positions to meet the minimum. |

---

### From weights to shares

Once the optimal weights are found (e.g. "40% AAPL, 35% MSFT, 25% GOOGL"), the tool translates them into actual share quantities based on today's prices and the investment amount you entered. It uses a **greedy rounding** algorithm:

1. Calculate the ideal (fractional) number of shares for each stock.
2. Buy the nearest whole number of shares that does not exceed the budget.
3. Report any leftover cash.

> *It's like splitting a restaurant bill. The table owes exactly $94.37 split four ways. You round each person's share to the nearest dollar and count what's left over.*

---

### Forecast → Optimizer Bridge

When you use the **📤 Use Forecast Returns in Optimizer** button, the optimizer replaces its historical expected-return estimates with the ensemble forecasts from the Forecast tab. Everything else (covariance, constraints, solver) stays the same.

This means the portfolio is built for *where the stocks are expected to go*, not just where they have been. Stocks with improving momentum and high forecast returns receive higher weights; stocks with declining trends receive lower weights.

---

## 5. How Prices Are Kept Up to Date

The tool stores all price data in a local cache (a folder called `sp500_data/`). Every price sync follows a careful incremental update process.

---

### Why incremental?

Downloading 5 years of daily prices for 500 stocks from scratch every time would take 15–30 minutes and hammer Yahoo Finance's rate limits. Instead, the tool only downloads the dates it doesn't already have.

---

### The incremental update process

1. **Find the latest date** already in the cache.
2. **Go back 5 days** from that date as a safety buffer (to catch late corrections and handle weekends/holidays).
3. **Download only the new data** from that point forward.
4. **Merge** the new data with the existing cache, with new data winning if there are any overlapping dates (to allow price corrections to propagate).
5. **Remove old data** that is now outside the retention window (e.g. for a 1-year cache, remove anything older than 1 year plus a small buffer).

> *Think of updating a newspaper archive. Rather than printing every edition ever published, you just add today's paper and throw out anything older than your retention limit.*

**Safety guarantee:** If Yahoo Finance returns nothing (outage, rate limit), the existing cache is returned unchanged. A failed download never corrupts the data.

---

## 6. How the Backtester Works

The backtester answers a critical question before you commit real money: *"How accurate were the forecast models when we test them against periods that have already happened?"*

---

### The rolling-window method

The tool takes the full price history you provide (e.g. 3 or 5 years) and slices it into a sequence of overlapping **training + test** windows. For each window:

1. **Training period** — the tool sees only the first year of that window's data, exactly as if it were standing at that point in the past with no knowledge of what came next.
2. It runs the ensemble forecast (Monte Carlo + CAPM + Trend) using only the training data.
3. **Test period** — the tool looks at what actually happened in the next 12 months and compares the predicted return to the actual return for each stock.

> *Think of it as a cooking competition where the judge tasted the dish, locked in a score, and only then revealed the final course. The score is honest because it was set before seeing the outcome.*

With 3 rolling windows and 3 years of history, the tool gets 3 independent "prediction → reality" comparisons per stock, giving a statistically meaningful sample.

---

### What the backtester measures

| Metric | What it means |
|---|---|
| **MAE (Mean Absolute Error)** | On average, how many percentage points was the forecast off from reality? An MAE of 8 pp means the model was typically within 8 percentage points of the actual return. |
| **Directional Accuracy (Hit Rate)** | What fraction of forecasts correctly predicted the direction of the return — up or down? A hit rate above 50% means the model beats a coin flip. |

Both metrics are reported per window (to spot if one time period was harder to predict), per ticker (to spot which stocks are easier or harder to forecast), and overall.

---

### What the backtester does not prove

Backtesting shows you how the model *has* performed. It cannot guarantee future performance. It is most useful for:

- **Disqualifying bad models** — if the hit rate is near 50% (coin flip) across many windows, the forecast has no predictive value for this set of stocks.
- **Calibrating confidence** — a consistently low MAE gives you more reason to weight forecast returns heavily in the optimizer.

---

## 7. How the Comparison Works

The **Compare** tab runs the optimizer twice using the same tickers and the same constraints, and places the two results side by side:

| Run | Expected returns used |
|---|---|
| **Historical** | Derived from the last 1–5 years of actual prices (geometric average) |
| **Forecast-optimised** | Replaced with the ensemble 1-year forecasts from the Forecast tab |

The difference in weights — which stocks the optimizer promotes or demotes — tells you how much the market's recent price history diverges from where the forecast models think these stocks are heading.

> *If the historical optimizer overweights a high-momentum stock but the forecast optimizer underweights it, the models may be signalling mean-reversion risk — the stock ran up faster than fundamentals justify.*

The **Δ (pp)** column in the weight-shift table shows the exact change in allocation for each stock. A large positive delta means the forecast tilts more money into that stock; a large negative delta means the forecast pulls money out.

---

## 8. The End-to-End Workflow

The tabs are ordered left to right to guide you through a complete investment analysis from raw data to a validated portfolio. A **Pipeline status bar** at the top of each workflow tab shows how far you have progressed.

```
📋 Data Health → 🔍 Analyzer → 📊 Optimizer → 🔮 Forecast → 📅 Backtest → 🔀 Compare
```

---

### Step 1 — Check data freshness (📋 Data Health)

Before doing any analysis, confirm that:
- Price cache is up to date (ideally synced within the last day).
- Analysis cache has been run for at least the 1Y period.
- No tickers are listed as failed in the last sync.

If data is stale, run `./run-price-sync.sh` and `./run-analysis-sync.sh` before continuing.

---

### Step 2 — Screen the universe (🔍 S&P 500 Analyzer)

Click **🚀 Analyze Stocks** to score and rank the full S&P 500. The top 20 by composite score are shown. This list is your candidate universe — stocks worth looking at more closely.

> You do not have to use all 20. The Optimizer and Forecast tabs let you narrow further.

---

### Step 3 — Build the optimal portfolio (📊 Optimizer)

Select **Use Top Performers from Analysis**, choose how many stocks to include (typically 10–20), and click **🚀 Optimize Portfolio**.

The optimizer builds the mathematically best allocation from that candidate list given your risk tolerance and constraints. After it runs, the **Pipeline status bar** updates:
`📊 Portfolio: 15 stocks ✅`

The optimizer stores the exact list of stocks with non-zero weight. All downstream tabs will offer **Use Portfolio Tickers** as their default source, keeping the full analysis consistent.

---

### Step 4 — Forecast those exact stocks (🔮 Forecast)

Switch to the Forecast tab. Because the optimizer has run, the ticker selection radio defaults to **Use Portfolio Tickers**. Click **🚀 Run Forecast**.

The three forecast models (Monte Carlo, CAPM, Trend) run on the same 15 (or however many) stocks you just optimized. After it runs: `🔮 Forecast: 15 stocks ✅`

> The fan charts show you the range of simulated futures for each stock. The ensemble return estimate is the headline single number per stock.

---

### Step 5 — Validate the forecast model (📅 Backtest)

Switch to the Backtest tab. Select **Use Portfolio Tickers** again and click **🔁 Run Backtest**.

The backtester tests the same forecast models against past data for the same stocks. It tells you:
- **Are the models accurate for this stock set?** (MAE)
- **Do they correctly predict direction?** (Hit rate)

After it runs: `📅 Backtest: 3 windows ✅`

If hit rate is well above 50% and MAE is acceptably small, you can trust the forecast returns more. If the models perform poorly on this stock set, treat forecast-optimized weights with more scepticism.

---

### Step 6 — Compare historical vs forecast allocation (🔀 Compare)

Click **🔀 Run Comparison**. The tool re-runs the optimizer twice (historical and forecast) on the same stocks and constraints, and shows:

- The change in expected return and Sharpe if you use forecast weights.
- Which stocks the forecast tilts toward and away from.
- A grouped bar chart for visual weight comparison.

This is the final sanity check. If the forecast-optimized portfolio shifts weight toward stocks the backtest validated as well-predicted, you have a coherent, evidence-backed allocation.

---

### The pipeline in one sentence

> Screen the universe (Analyzer) → lock in a portfolio (Optimizer) → forecast those exact stocks (Forecast) → validate the models historically (Backtest) → compare historical vs forward-looking allocations (Compare).

Every tab flows into the next. At no point do you need to manually copy tickers between tabs — the **Use Portfolio Tickers** option does it automatically.

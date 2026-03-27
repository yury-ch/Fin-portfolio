# Financial & Quantitative Review of Algorithms

**Reviewer perspective:** Senior financial advisor / quantitative analyst
**Scope:** `shared/analysis_engine.py`, `services/calculation_service.py`
**Reference:** `ALGORITHMS.md` for pipeline diagrams

---

## Executive Summary

The screening and optimisation pipeline is structurally sound and follows accepted quant-finance conventions in broad strokes. However, five issues — one a confirmed code defect that actively corrupts ranking — require attention before results can be trusted for real allocation decisions. Three are methodological and affect signal quality. Two relate to internal consistency.

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 1 | Drawdown score is inverted — worst stocks rank highest on this metric | **Critical** | Code defect |
| 2 | Drawdown column normalisation silently corrupts decimal-form values | **Critical** | Code defect |
| 3 | Risk-free rate = 0 in screening Sharpe; hardcoded 2% in metrics; parameterised in optimiser — three inconsistent assumptions | **Major** | Internal inconsistency |
| 4 | Return + Sharpe carry implicit ~50% combined weight; double-penalises volatility | **Major** | Methodology |
| 5 | Min-max normalisation is outlier-sensitive; scores collapse around a single dominant stock | **Major** | Methodology |
| 6 | `mean_historical_return` (arithmetic) used in optimiser vs. log-return method in screening — divergent return estimates | **Moderate** | Internal inconsistency |
| 7 | Sample covariance matrix — unstable for large N; no shrinkage | **Moderate** | Model risk |
| 8 | 3-month momentum only — suboptimal lookback, no reversal filter | **Minor** | Methodology |
| 9 | Composite score is a relative ranking, not an absolute signal | **Minor** | Disclosure / usage |

---

## Finding 1 — Drawdown Score Inversion (Critical Bug)

**Location:** `shared/analysis_engine.py:133`

```python
df['Drawdown_Score'] = self._safe_normalize(df['Max_Drawdown'], inverse=True)
```

**The defect:**
`Max_Drawdown` values are negative decimals. A drawdown of −30% is stored as `−0.30`; a drawdown of −5% is stored as `−0.05`. Smaller absolute value is better.

Trace through `_safe_normalize` with `inverse=True` for a three-ticker example:

| Ticker | Max_Drawdown | Without inverse | With inverse=True (current) |
|--------|-------------|-----------------|------------------------------|
| A | −0.05 (best) | 1.00 | **0.00** |
| B | −0.15 | 0.60 | **0.40** |
| C | −0.30 (worst) | 0.00 | **1.00** |

The `inverse=True` flag reverses the already-correct direction of the normalised values. The worst drawdown in the batch is awarded the highest Drawdown_Score (1.0). Stocks that lost 30–40% peak-to-trough receive a higher score on this metric than stocks that barely moved.

Because Drawdown carries a 15% composite weight, the ranking of any ticker near the top or bottom of the universe can change materially.

**Correct code:**

```python
df['Drawdown_Score'] = self._safe_normalize(df['Max_Drawdown'])  # no inverse
```

Verification: with `inverse=False`, `norm(−0.30) = 0.0` (worst → 0) and `norm(−0.05) = 1.0` (best → 1). This is the intended behaviour.

---

## Finding 2 — `standardize_analysis_columns` Drawdown Transformation Edge Case (Critical Bug)

**Location:** `shared/analysis_engine.py:34`

```python
if 'Max_Drawdown' in df.columns and df['Max_Drawdown'].min() > -1:
    df['Max_Drawdown'] = -df['Max_Drawdown'] / 100.0
```

**Intent:** Convert a positive-percent legacy value (e.g., `30` meaning 30% loss) to the canonical decimal form (`−0.30`).

**The defect:** The guard condition `min() > -1` fires for **both** cases:
- Positive percent form: `30 > -1` → True → transforms to `−0.30` ✓
- Already-correct decimal form: `−0.30 > -1` → True → transforms to `+0.003` ✗

When `Max_Drawdown` is already stored as `−0.30` (which is the current write path from `_analyze_series`), the function corrupts it to `0.003`. A 30% drawdown becomes effectively zero.

| Input form | Example value | Condition `> -1` | Result |
|-----------|--------------|------------------|--------|
| Positive percent (legacy) | `30.0` | True | `−0.30` ✓ |
| Negative percent (legacy) | `−30.0` | False | unchanged (not converted) ✗ |
| Decimal negative (current) | `−0.30` | **True** | `0.003` **✗ corrupted** |

The correct guard should distinguish percent from decimal by magnitude, e.g., `min() < -1` for negative-percent legacy data or `max() > 1` for positive-percent data:

```python
# Correct: only convert if values appear to be in percentage scale (|value| >> 1)
if 'Max_Drawdown' in df.columns and df['Max_Drawdown'].max() > 1:
    df['Max_Drawdown'] = -df['Max_Drawdown'] / 100.0
```

---

## Finding 3 — Inconsistent Risk-Free Rate Assumptions (Major)

The pipeline uses three different risk-free rate values with no justification:

| Location | `rf` used | Value |
|----------|----------|-------|
| `_analyze_series` (screening Sharpe) | Hardcoded | 0% |
| `calculate_portfolio_metrics` | Hardcoded | 2% |
| `optimize_portfolio` (Max Sharpe) | Parameterised | User input / default |

**Financial impact:**
With U.S. Federal Funds Rate at ~4.5–5.25% during 2024–2026, using `rf = 0` in screening overstates every stock's Sharpe ratio by roughly `rf / vol`. For a stock with 20% annual volatility, the inflation is `0.05 / 0.20 = +0.25` Sharpe units. This causes high-return, high-volatility stocks to appear risk-adjusted attractive when they are not.

The optimiser uses a calibrated `risk_free_rate` for the efficient frontier, meaning the screening ranking and the optimiser operate on different return assumptions. A stock selected by screening as "high Sharpe" may rank much lower in the optimiser's actual objective.

**Recommendation:** Define a single `risk_free_rate` constant (or config parameter) and use it uniformly across `_analyze_series`, `calculate_portfolio_metrics`, and the optimiser default.

---

## Finding 4 — Return + Sharpe Weight Double-Counting (Major)

The composite score formula:

```
Composite = 0.25 × Return_Score + 0.20 × Vol_Score + 0.25 × Sharpe_Score
          + 0.15 × Drawdown_Score + 0.15 × Momentum_Score
```

Sharpe is defined as `Annual_Return / Annual_Volatility` (with `rf = 0`). It is therefore almost entirely a function of return and volatility — the same two quantities already represented by `Return_Score` and `Vol_Score`.

**Effective factor exposure:**

| Factor | Explicit weight | Implicit Sharpe contribution | Effective weight |
|--------|----------------|------------------------------|-----------------|
| Return | 25% | ~+15% from Sharpe | ~40% |
| Volatility | 20% | ~−15% from Sharpe | ~35% |
| Drawdown | 15% | small overlap | ~15% |
| Momentum | 15% | none | ~15% |

Two-thirds of the composite weight is effectively redundant, rewarding return and penalising volatility twice. A low-return, low-volatility stock and a high-return, high-volatility stock may tie in Sharpe but will score very differently on Return and Vol individually — which is the desired discrimination. The issue is not the presence of all three but the weight calibration: Sharpe already captures the return/risk tradeoff. Adding independent Return and Vol scores on top inflates the weight on the return dimension beyond intent.

**Practical consequence:** The current weights favour high-growth, high-volatility names more than the stated 25% return weight implies. A pure-return chaser effectively scores on 40% of the composite.

---

## Finding 5 — Min-Max Outlier Sensitivity (Major)

Min-max normalisation:

```
norm(x) = (x − min) / (max − min)
```

When one ticker has an extreme value — e.g., NVIDIA with +180% annual return in 2024 — the entire `Return_Score` distribution collapses. If the universe is [−10%, +20%, +25%, +30%, +180%], the normalised scores are approximately [0, 0.16, 0.19, 0.22, 1.0]. The top four tickers are indistinguishable on this metric despite a 40% range among them.

This is a well-known weakness of min-max scaling in the presence of fat-tailed distributions (equities). The batch-relative, cross-run instability is correctly documented in `ALGORITHMS.md §2`, but the outlier compression effect is not.

**Alternatives to consider:**

| Method | Outlier robustness | Cross-run stability | Notes |
|--------|-------------------|---------------------|-------|
| Min-max (current) | Low | Low | Simple but fragile |
| Percentile rank | High | Low | Maps to [0, 1] via rank; outliers don't compress others |
| Z-score (clipped) | Moderate | Moderate | Clip at ±3σ before normalising |
| Winsorised min-max | High | Low | Cap at 5th/95th percentile before rescaling |

Percentile rank is the most common approach in quantitative equity screeners for exactly this reason.

---

## Finding 6 — Arithmetic vs. Log Return Inconsistency in Optimiser (Moderate)

**Location:** `calculation_service.py:48`

```python
mu = expected_returns.mean_historical_return(prices, frequency=252)
```

PyPortfolioOpt's `mean_historical_return` computes the **arithmetic** mean of simple returns, annualised by multiplying by 252. The screening pipeline (`_analyze_series`) uses the **geometric** (log-return) method: `exp(mean(log_returns) × 252) − 1`.

For a stock with daily returns drawn from a distribution with mean `μ_d` and variance `σ²_d`:

- Arithmetic annualised ≈ `μ_d × 252`
- Geometric annualised ≈ `(μ_d − σ²_d / 2) × 252`

The arithmetic estimate is always ≥ the geometric by `σ² × 252 / 2`. For a stock with 25% annual volatility, the arithmetic estimate overstates expected return by roughly `(0.25)² / 2 ≈ 3.1 percentage points` per year.

**Consequence:** The optimiser's expected-return inputs are systematically higher than the screening metrics. A stock selected for a "20% expected return" in screening feeds into an optimiser that sees it as a ~23% expected return, shifting weights toward high-volatility names.

PyPortfolioOpt supports geometric mean: `expected_returns.mean_historical_return(prices, compounding=True)`. Using this would align both layers of the pipeline.

---

## Finding 7 — Sample Covariance Instability for Large N (Moderate)

**Location:** `calculation_service.py:51`

```python
S = risk_models.sample_cov(prices, frequency=252)
```

Sample covariance is unbiased but inefficient for typical equity portfolios. For N tickers and T days of history:

- S&P 500 analysis typically has N in the range 50–500
- With 1–2 years of daily data, T ≈ 252–504
- When N approaches T, the sample covariance matrix is near-singular; eigenvalues shrink toward zero and the optimizer assigns extreme weights

**Recommended alternatives:**

| Method | PyPortfolioOpt API | When |
|--------|-------------------|------|
| Ledoit-Wolf shrinkage | `risk_models.CovarianceShrinkage(prices).ledoit_wolf()` | Default upgrade; always better than sample for N > 20 |
| Oracle approximating shrinkage | `.oracle_approximating()` | Large N (100+) |
| Constant correlation | `.constant_correlation()` | Interpretable; good for presentation |

For the current portfolio sizes (user selects a subset for optimisation), sample covariance is tolerable but L2 regularisation (already implemented) partially compensates for this. Still, for any run with N > 30–40, shrinkage is strictly preferable.

---

## Finding 8 — 3-Month Momentum Lookback (Minor)

The momentum metric uses a 63-trading-day (≈3-month) lookback (`prices.iloc[-1] / prices.iloc[-63] − 1`).

Academic literature on cross-sectional momentum (Jegadeesh & Titman 1993; Carhart 1997) identifies the strongest signal in the **12-1 month** window — that is, performance over the past 12 months excluding the most recent month. The reasons are:

- The trailing 1-month period exhibits **short-term reversal** (microstructure effects); including it degrades signal quality
- 3-month momentum captures an earlier phase of the price move but with lower information ratio and higher turnover

A 3-month window has higher sensitivity to earnings-announcement effects and sector rotations, which can appear as momentum but are actually mean-reverting events.

**Recommended extension:** Add a `Recent_12M_Return` metric (`prices.iloc[-1] / prices.iloc[-252] − 1`) and use it in place of or alongside the 3M metric. The composite weight allocation and lookback period should be validated against backtested quintile returns on the target universe.

---

## Finding 9 — Composite Score Is Relative, Not Absolute (Minor / Disclosure)

This is correctly noted in `ALGORITHMS.md §2` but deserves emphasis for operational use:

- Scores are **not stable across runs**. Adding or removing even one ticker shifts every score.
- A score of 0.75 today does not mean the same stock will score 0.75 tomorrow.
- The score is a **rank within the current batch**, not an intrinsic quality signal.

Operational implications:
1. Do not store or compare composite scores across runs without also storing the universe composition at the time.
2. Do not set threshold rules like "only buy stocks with Composite_Score > 0.60" — the threshold has no consistent meaning across sessions.
3. Alerting or monitoring systems should compare rank position, not raw score.

---

## Summary of Recommendations

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | Remove `inverse=True` from Drawdown_Score normalisation | Trivial | Critical — fixes active scoring inversion |
| 2 | Fix `standardize_analysis_columns` drawdown guard condition | Trivial | Critical — prevents decimal-form corruption |
| 3 | Define single shared `RISK_FREE_RATE` constant; use in all three locations | Low | Major — removes Sharpe inconsistency |
| 4 | Recalibrate composite weights or collapse Return+Sharpe to one | Low | Major — eliminates hidden double-counting |
| 5 | Replace min-max with percentile rank for outlier-heavy metrics | Low | Major — improves score distribution |
| 6 | Switch to `mean_historical_return(prices, compounding=True)` | Trivial | Moderate — aligns screening and optimiser |
| 7 | Switch to `CovarianceShrinkage().ledoit_wolf()` for N > 20 | Low | Moderate — reduces optimiser instability |
| 8 | Add 12-1M momentum metric alongside 3M | Low | Minor — better signal quality |
| 9 | Document relative-score limitation in UI | Trivial | Minor — prevents misuse |

Findings 1 and 2 are code defects producing incorrect results today. They should be addressed before any production use of the ranking output. Findings 3–5 are methodological improvements that will materially improve signal quality. Findings 6–9 represent incremental refinements appropriate for a second-pass improvement cycle.

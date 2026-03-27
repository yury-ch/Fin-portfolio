# Algorithms & Scoring Reference

All logic described here lives in `shared/analysis_engine.py` and `services/calculation_service.py`.

---

## 1. Per-Ticker Metrics Pipeline

Each ticker goes through a single pipeline. Minimum 60 trading days of price history required; tickers with fewer are silently skipped.

![Metrics Pipeline](docs/diagrams/01-metrics-pipeline.svg)

```mermaid
flowchart TD
    A["Price series\nP1, P2, ..., Pn\n(daily closes)"]

    A --> B["Simple daily returns\nrt = Pt/Pt-1 - 1"]
    A --> C["Log daily returns\nlt = ln(Pt) - ln(Pt-1)"]
    A --> D["Cumulative return series\nCt = product(1 + rt)"]

    B --> V["Annual Volatility\nstd(r) x sqrt(252)"]
    C --> R["Annual Return\nexp(mean(l) x 252) - 1"]
    R --> SH["Sharpe Ratio\nS = Return / Vol\nrisk-free = 0"]
    V --> SH

    D --> RX["Rolling max\nMt = max(C1...Ct)"]
    RX --> DD["Drawdown series\nDt = (Ct - Mt) / Mt"]
    DD --> MD["Max Drawdown\nMDD = min(D)\nnegative value"]

    A --> MO["3-Month Momentum\nM3 = P_last / P(-63) - 1\n0 if fewer than 63 bars"]

    A --> CP["Current Price\nP_last"]

    R & V & SH & MD & MO & CP --> OUT["AnalysisResult\nper ticker"]
```

### Formulas

| Metric | Formula | Notes |
|---|---|---|
| **Annual Return** | `exp(mean(log_returns) × 252) − 1` | Log-return method; more accurate than arithmetic mean for long series |
| **Annual Volatility** | `std(simple_returns) × √252` | Simple returns used for vol — consistent with PyPortfolioOpt convention |
| **Sharpe Ratio** | `Annual_Return / Annual_Volatility` | Risk-free rate = 0 at this stage; applied in optimization separately |
| **Max Drawdown** | `min((Cₜ − rolling_max(C)) / rolling_max(C))` | Always ≤ 0; −0.30 means a 30% peak-to-trough loss |
| **3-Month Momentum** | `P_last / P_{−63} − 1` | 63 trading days ≈ 3 calendar months |

---

## 2. Composite Scoring

![Composite Scoring](docs/diagrams/02-composite-scoring.svg)

After all tickers are analysed, scores are computed **across the batch** — not per ticker in isolation. This makes every score relative to the current universe.

```mermaid
flowchart TD
    subgraph INPUT["Raw metrics  (N tickers x 5 values)"]
        direction LR
        AR["Annual_Return"]
        VO["Volatility"]
        SR["Sharpe_Ratio"]
        DR["Max_Drawdown"]
        MO["Recent_3M_Return"]
    end

    subgraph NORM["Step 1 — Normalise to 0..1"]
        direction LR
        NR["Return_Score\nrank(pct=True)\n↑ outlier-robust rank"]
        NV["Vol_Score\n1 − norm(Volatility)\n↓ lower = better"]
        NS["Sharpe_Score\nnorm(Sharpe_Ratio)\n↑ higher = better"]
        ND["Drawdown_Score\nnorm(Max_Drawdown)\n↑ less negative = better"]
        NM["Momentum_Score\nrank(pct=True)\n↑ outlier-robust rank"]
    end

    subgraph WEIGHT["Step 2 — Weighted Sum"]
        direction LR
        W["Composite_Score =\n  0.20 x Return_Score\n+ 0.15 x Vol_Score\n+ 0.25 x Sharpe_Score\n+ 0.20 x Drawdown_Score\n+ 0.20 x Momentum_Score"]
    end

    AR --> NR
    VO --> NV
    SR --> NS
    DR --> ND
    MO --> NM

    NR & NV & NS & ND & NM --> W

    W --> RANK["Sort descending\nRank 1 = best composite"]
```

### Weight Rationale

| Component | Weight | Direction | Reasoning |
|---|---|---|---|
| Sharpe Ratio | **25 %** | ↑ | Primary risk-adjusted signal; unchanged — it is the most information-dense single metric |
| Annual Return | **20 %** | ↑ | Raw return objective; reduced from 25% to limit double-counting with Sharpe |
| Max Drawdown | **20 %** | ↑ | Tail-risk protection; increased to give independent tail-risk factor more voice |
| 3M Momentum | **20 %** | ↑ | Recency signal; fully independent of Sharpe — increased weight reflects this |
| Volatility | **15 %** | ↓ | Penalise pure volatility; reduced from 20% because Sharpe already incorporates vol |

### Normalisation Detail

Two normalisation methods are used depending on the metric's tail behaviour:

**Percentile rank** — `Series.rank(pct=True)` — applied to **Annual_Return** and **3M_Momentum**:
- Assigns each ticker its rank position as a fraction of the batch size (0 < score ≤ 1)
- Outlier-robust: one extreme value (e.g., NVDA +180% annual return) does not compress all other scores to near zero
- Direction is implicit: higher rank = higher score

**Min-max normalisation** — applied to **Volatility**, **Sharpe_Ratio**, and **Max_Drawdown**:
```
norm(x) = (x − min(x)) / (max(x) − min(x))
```
- If all values in a metric are identical (`range = 0`), every ticker receives `0.5`
- Inverse for Volatility: `1 − norm(x)` so that lower volatility scores higher
- Max_Drawdown is already negative (e.g., −0.30 for a 30% loss); `norm` without inversion correctly maps the least negative value (best) → 1.0 and the most negative (worst) → 0.0
- Scores are **not** stable across runs — adding or removing tickers from the batch shifts every score

---

## 3. Delta Price Sync Algorithm

`PriceSyncService._delta_sync()` — the highest-risk function in the pipeline. A merge bug here produces wrong analysis results silently.

![Delta Sync](docs/diagrams/03-delta-sync.svg)

```mermaid
flowchart TD
    START(["_delta_sync(tickers, period, existing_df)"])

    START --> CHECK{"existing_df\nindex empty?"}
    CHECK -- "NaT last_ts\n(rare edge case)" --> FETCH2["fetch_prices(start=NaT)\nreturns empty"]
    FETCH2 --> EMPTY["return existing_df unchanged\n(empty)"]

    CHECK -- "has dates" --> LAST["last_ts = existing_df.index.max()"]
    LAST --> START2["start = last_ts - delta_buffer_days\n(default: 5 days overlap)"]
    START2 --> FETCH["fetch_prices(\n  tickers,\n  period=None,\n  interval,\n  start=start\n)"]

    FETCH --> NODATA{"delta_df empty?"}
    NODATA -- yes --> KEEP["merged = existing_df\n(Yahoo returned nothing -\n keep what we have)"]
    NODATA -- no --> CONCAT["concat([existing_df, delta_df])"]

    CONCAT --> DEDUP["drop_duplicates(keep=last)\nnew data wins on overlap"]
    DEDUP --> SORT["sort_index() ascending"]
    SORT --> KEEP

    KEEP --> TRIM["trim_history(merged, period)\nkeep rows >= now - horizon\nif trimmed empty: keep tail(1)"]
    TRIM --> DONE(["return trimmed_df"])

    style DEDUP fill:#fef3c7,stroke:#d97706
    style KEEP fill:#dcfce7,stroke:#16a34a
```

### Key Design Choices

| Choice | Why |
|---|---|
| **5-day overlap buffer** | Guards against late-arriving data, weekend/holiday gaps, and Yahoo's inconsistent close dates |
| **`keep='last'` on duplicates** | New delta data wins over cached data for overlapping dates — ensures corrections and corporate actions propagate |
| **Return existing on empty delta** | Silent Yahoo outages do not corrupt the cache |
| **Trim after merge** | The combined frame may be larger than the period horizon; trim ensures consistent retention windows |

---

## 4. Portfolio Optimisation Pipeline

![Portfolio Optimisation](docs/diagrams/04-portfolio-optimisation.svg)

```mermaid
flowchart TD
    INPUT["prices_df\nN tickers x T days\nfrom price cache"]

    INPUT --> STATS["compute_stats()\nmu = mean_historical_return(prices)\nS = sample_cov(prices)\n(PyPortfolioOpt)"]

    STATS --> BOUNDS{"N assets <= 3?"}

    BOUNDS -- yes --> EF1["EfficientFrontier(mu, S)\nweight_bounds = (0, 1)\nno L2 regularisation"]
    BOUNDS -- no --> EF2["EfficientFrontier(mu, S)\nweight_bounds = (0, max_weight/100)\nadd_objective(L2_reg, gamma=l2_reg)"]

    EF1 & EF2 --> OBJ{"objective"}

    OBJ -- "Max Sharpe" --> MS["ef.max_sharpe(\n  risk_free_rate=risk_free/100\n)"]
    OBJ -- "Min Vol\n(target return)" --> MR["ef.efficient_return(\n  target_return=target_return/100\n)"]

    MS & MR --> FAIL{"optimisation\nfailed?"}
    FAIL -- yes --> FALLBACK["ef.min_volatility()\n(fallback)"]
    FAIL -- no --> CLEAN

    FALLBACK --> CLEAN["raw_weights = ef.clean_weights(cutoff=0)"]

    CLEAN --> PRUNE["enforce_min_holdings(\n  weights,\n  min_n=min_holdings,\n  prune_below_pct=min_weight_threshold\n)"]

    PRUNE --> ALLOC["DiscreteAllocation(\n  final_weights,\n  latest_prices,\n  total_portfolio_value=investment_amount\n).greedy_portfolio()"]

    ALLOC --> PERF["ef.portfolio_performance()\nreturns expected_return, volatility, sharpe"]

    PERF --> OUT["OptimizationResult\nweights  allocation  leftover_cash\nexpected_annual_return\nannual_volatility  sharpe_ratio"]
```

### enforce_min_holdings Detail

![Enforce Min Holdings](docs/diagrams/05-enforce-min-holdings.svg)

```mermaid
flowchart TD
    W["raw_weights (may include\nnear-zero allocations)"]

    W --> THRESH["keep = weights >= prune_below_pct / 100\n(threshold in percent;\ne.g. 0.25% = 0.0025)"]

    THRESH --> ENOUGH{"len(keep) >= min_n?"}

    ENOUGH -- yes --> NORM1["return keep / keep.sum()\n(renormalise to 1.0)"]
    ENOUGH -- no --> TOP["top = weights.sort_values().head(min_n)\n(guarantee minimum holdings\neven if below threshold)"]
    TOP --> NORM2["return top / top.sum()"]
```

### Objective Comparison

| Objective | Solver call | When to use |
|---|---|---|
| **Max Sharpe** | `ef.max_sharpe(risk_free_rate)` | Default — maximises return per unit of risk |
| **Min Volatility (target return)** | `ef.efficient_return(target_return)` | When a specific return target is required with minimum risk |
| **Fallback** | `ef.min_volatility()` | Auto-triggered when primary optimisation fails (infeasible constraints) |

### L2 Regularisation

When `l2_reg > 0`, an L2 penalty `γ × ‖w‖²` is added to the objective. This pushes the solver toward more equal-weight solutions, reducing concentration in a single asset. L2 is disabled automatically when N ≤ 3 (too few assets for regularisation to be meaningful).

---

## 5. Column Normalisation (Cache Compatibility)

`standardize_analysis_columns()` is applied every time an analysis Parquet is loaded. It bridges the legacy `snake_case` schema (written by older sync jobs) and the current `Title_Case` schema expected by the UI.

```mermaid
flowchart LR
    subgraph OLD["Legacy column names"]
        direction TB
        C1["ticker"]
        C2["total_return_pct"]
        C3["sharpe_ratio"]
        C4["volatility_pct"]
        C5["max_drawdown_pct"]
        C6["current_price"]
        C7["composite_score"]
    end

    subgraph NEW["Canonical column names"]
        direction TB
        D1["Ticker"]
        D2["Annual_Return  ÷100 if >1"]
        D3["Sharpe_Ratio"]
        D4["Volatility  ÷100 if >1"]
        D5["Max_Drawdown  negate /100 if > -1"]
        D6["Current_Price"]
        D7["Composite_Score"]
        D8["Recent_3M_Return  added as 0.0 if missing"]
    end

    C1 --> D1
    C2 --> D2
    C3 --> D3
    C4 --> D4
    C5 --> D5
    C6 --> D6
    C7 --> D7
```

**Unit guards:** If `Annual_Return > 1` the value is divided by 100 (percentage stored as `15.3` → normalised to `0.153`). Same for Volatility. Max_Drawdown is negated and divided by 100 if it appears as a positive percentage.

---

## 6. Algorithm Interaction Summary

![Algorithm Flow](docs/diagrams/06-algorithm-flow.svg)

```mermaid
flowchart LR
    PS["price_sync_service\nDelta Sync\n§3"]
    AE["analysis_engine\nMetrics + Scoring\n§1 & §2"]
    CS["calculation_service\nOptimisation\n§4"]
    UI["presentation_service\nUI"]

    PS -->|"per-period\nParquet"| AE
    AE -->|"analysis\nParquet"| UI
    UI -->|"selected tickers\n+ price slice"| CS
    CS -->|"weights\nallocation\nperformance"| UI
```

The pipeline is strictly unidirectional:

1. **Price sync** produces raw OHLC data → Parquet
2. **Analysis engine** consumes price Parquet → analysis Parquet (metrics + composite score)
3. **UI** reads analysis Parquet to let the analyst select tickers
4. **Optimiser** consumes the selected price slice → portfolio weights and share allocation

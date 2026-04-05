# F-04 — 12-Month Momentum Signal (12-1M)

| Field | Value |
|---|---|
| ID | F-04 |
| Status | Shipped |
| Priority | P3 (TD-17) |
| Shipped | 2026-04-01 |

---

## 1. Background

The original scoring engine used a **3-month (63-day) momentum window** as the sole momentum signal. Academic research (Jegadeesh & Titman, 1993) shows the strongest momentum signal is the **12-1 month** window — returns measured over the prior 12 months, *excluding* the most recent month. The trailing 1-month period exhibits short-term **mean reversion** that degrades signal quality when included.

---

## 2. What Changed

### New metric: `Recent_12M_Return`

Computed in `AnalysisEngine._analyze_series`:

```python
# 12-1M: return from day -273 to day -22 (skips trailing ~1 month)
if len(prices) >= 273:
    recent_12m_return = float((prices.iloc[-22] / prices.iloc[-273]) - 1)
elif len(prices) >= 252:
    # Fallback: full 12M without skip (< 1Y + 1M history)
    recent_12m_return = float((prices.iloc[-1] / prices.iloc[-252]) - 1)
else:
    recent_12m_return = 0.0
```

| Condition | Behaviour |
|---|---|
| ≥ 273 trading days (~13M) | Full 12-1M signal — skips trailing 21 days |
| 252–272 days (~12M) | Plain 12M without skip (approximation) |
| < 252 days | 0.0 (no signal; stock excluded from momentum ranking) |

### Scoring update: blended momentum

`_append_scores` now produces three momentum columns:

| Column | Definition |
|---|---|
| `Momentum_3M_Score` | Percentile rank of `Recent_3M_Return` |
| `Momentum_12M_Score` | Percentile rank of `Recent_12M_Return` |
| `Momentum_Score` | `0.5 × Momentum_3M_Score + 0.5 × Momentum_12M_Score` |

`Composite_Score` weights are **unchanged** — `Momentum_Score` still carries 20%. The blend gives equal weight to the short-term (3M) and medium-term (12-1M) signals.

### UI: new column in Analyzer table

The S&P 500 Stock Analyzer table now shows **12M Return** alongside 3M Return when the field is present in the cached analysis. Old cached parquet files that pre-date this change will show `0.0` until the next analysis sync.

---

## 3. Files Changed

| File | Change |
|---|---|
| `shared/analysis_engine.py` | `AnalysisResult` dataclass: added `Recent_12M_Return` field; `_analyze_series`: computes 12-1M return; `_append_scores`: added `Momentum_3M_Score`, `Momentum_12M_Score`, blended `Momentum_Score`; `standardize_analysis_columns`: fills `Recent_12M_Return = 0.0` for legacy cache files |
| `services/presentation_service.py` | Analyzer display table conditionally includes `12M Return` column |

---

## 4. Refreshing the Cache

The new field only appears after running the analysis sync:

```bash
./run-analysis-sync.sh
```

Until then, `Recent_12M_Return` is `0.0` for all tickers (filled by `standardize_analysis_columns`), and `Momentum_12M_Score` ranks all tickers equally — effectively removing the 12M signal from scoring until the cache is refreshed.

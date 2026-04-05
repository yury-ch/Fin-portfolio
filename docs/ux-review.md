# UX Review — presentation_service.py

**Reviewed:** 2026-04-01
**Implemented:** 2026-04-04
**File:** `services/presentation_service.py`
**Framework:** Streamlit

---

## Summary

| Severity | Count | Status |
|---|---|---|
| Critical | 2 | ✅ Resolved |
| Inconsistency | 7 | ✅ Resolved |
| Minor | 2 | ✅ Resolved |
| Consistent / Good | 5 | ✅ No action |

---

## Critical

### UX-01 — Fan chart slider renders after the data it controls ✅

**Location:** `line 1179` — Forecast tab
**Issue:** The "Fan charts — number of tickers" `st.slider()` appeared after two summary tables. A control must precede the output it governs.
**Fix applied:** Added `st.divider()` + `st.subheader("📈 Fan Charts")` before the slider in `_render_forecast_results()`.

---

### UX-02 — "Optimizing with N tickers" banner renders unconditionally ✅

**Location:** `line 650` — Optimizer tab
**Issue:** `st.info(...)` fired every time the tab rendered, implying something was happening when nothing was.
**Fix applied:** Replaced `st.info()` with `st.caption()` to reduce visual weight.

---

## Inconsistencies

### UX-03 — Refresh button labels are not standardised ✅

**Fix applied:** Standardised all three tabs to `"🔄 Refresh"` with unique `key` params (`refresh_analyzer`, `refresh_health`, `refresh_universe`) to prevent `StreamlitDuplicateElementId`.

---

### UX-04 — Primary run button width is inconsistent ✅

**Fix applied:** Added `use_container_width=True` to the "🚀 Analyze Stocks" button (Analyzer tab).

---

### UX-05 — `st.error()` uses redundant `⚠️` emoji in some calls ✅

**Fix applied:** Removed `⚠️` from all five affected `st.error()` calls (lines ~781, ~783, ~803, ~806, ~968).

---

### UX-06 — Mixed chart types for equivalent data ✅

**Fix applied:** Replaced `st.line_chart(normalized)` in `_render_optimization_results()` with a Plotly `go.Figure` for visual parity with fan charts. Falls back to `st.line_chart` if Plotly is not importable.

---

### UX-07 — `st.success()` used for passive informational states ✅

**Fix applied:** Replaced 4 passive `st.success()` calls with `st.info()` (cached analysis freshness messages, universe sync state).

---

### UX-08 — Download button placement differs between tabs ✅

**Fix applied:** Moved Optimizer download buttons to after the Historical Performance chart (end of results section), matching Forecast tab placement.

---

### UX-09 — Ticker Universe section uses `st.markdown()` for structured metadata ✅

**Fix applied:** Replaced raw `st.markdown()` strings with 4 `st.metric()` widgets and `st.caption()` for the file path. Aligns with the rest of the Health dashboard.

---

## Minor

### UX-10 — `_micro` key suffix leaks into session state keys ✅

**Fix applied:** Renamed `key="analysis_period_micro"` → `key="analysis_period"` and `key="use_top20_micro"` → `key="use_top20"`.

---

### UX-11 — Price Preview expander missing `hide_index=True` ✅

**Fix applied:** Added `hide_index=True` to `st.dataframe(prices_df.tail(), ...)` in the Price Preview expander.

---

## What is already consistent

- All 5 tabs open with `st.header()` + `st.caption()` — uniform structure.
- All primary action buttons use the `🚀` emoji.
- Empty states uniformly use `st.info()`.
- Service connection errors uniformly use `st.warning()`.
- All `st.dataframe()` calls use `use_container_width=True`.

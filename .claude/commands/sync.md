---
name: sync
description: Run price sync and analysis sync data refresh pipeline
---

Run the full data refresh pipeline: price sync followed by analysis sync.

1. Run price sync:
   bash ./run-price-sync.sh
   Report: tickers attempted, loaded, failed, data through date.

2. If price sync succeeded (exit code 0), run analysis sync:
   bash ./run-analysis-sync.sh
   Report: periods processed, stocks analysed.

3. Summarise the outcome:
   - ✅ Both complete — show data through date and stock count
   - ⚠️  Price sync ok, analysis sync failed — show error
   - ❌ Price sync failed — show error, skip analysis sync

If $ARGUMENTS contains "prices-only", run only step 1.
If $ARGUMENTS contains "analysis-only", run only step 2.

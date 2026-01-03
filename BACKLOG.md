## Backlog

### 1. Incremental Yahoo Price Sync
- **Summary:** Implement a two-phase pipeline: an initial 5-year backfill followed by weekly deltas.
- **Details:**
  - First run: fetch the full 5-year history once, persist as the master cache.
  - Subsequent runs: read the latest timestamp per ticker, pull `[last_ts, now]` deltas from yfinance, merge, and keep the rolling 5-year window trimmed.
  - Re-run analysis only for periods touched by the delta and handle splits/dividends by validating a short overlap before appending.

### 2. Single Load → Multi-Horizon Analysis
- **Summary:** Reuse the 5-year master cache to derive 1y/2y/3y/5y analyses instead of separate downloads.
- **Details:**
  - Extend price cache format to store raw 5y data only.
  - Adjust analysis sync worker to derive shorter windows from the cached 5y frame instead of hitting Yahoo per horizon.
  - Add 5y scoring/parquet generation so the UI can surface the longer-term analysis as a new tab.

### 3. Monthly S&P Constituent Validation
- **Summary:** Schedule a monthly sanity check of the Wikipedia S&P 500 list.
- **Details:**
  - Compare the cached `sp500_constituents.csv` against a fresh download monthly; alert or log diffs.
  - Optionally integrate with CI or send Slack/email when differences are detected.
  - Record validation results for auditing (e.g., write JSON reports under `sp500_data/validation/`).

### 4. Cron-Managed Orchestration
- **Summary:** Formalize cron/launchd entries for both price and analysis syncs.
- **Details:**
  - Package sample cron scripts (or launchd plist) under `ops/`.
  - Provide health checks/log rotation guidance.
  - Ensure cron jobs handle venv activation, retries, and notifications on failure.

### 5. Evaluate Redis vs Mongo for Caching
- **Summary:** Compare in-memory (Redis) vs document-store (MongoDB) approaches for replacing parquet caches.
- **Details:**
  - Assess requirements: read/write throughput, TTL support, data model (time series vs documents), deployment complexity.
  - Prototype both options:
    - Redis: use hash/streams for price data, leverage TTL + pub/sub for invalidation.
    - MongoDB: store price/analysis snapshots per period, index on ticker + timestamp, use change streams for refresh notifications.
  - Document trade-offs (cost, ops overhead, backup strategy) and recommend one path forward.

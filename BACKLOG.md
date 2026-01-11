## Backlog

### ~~1. Incremental Yahoo Price Sync~~ ✅
- **Summary:** Implement a two-phase pipeline: an initial 5-year backfill followed by weekly deltas.
- **Status:** Delivered via `shared/price_loader.py` + `services/price_sync_service.py` (single 5y master cache, delta merges, trimming).

### ~~2. Single Load → Multi-Horizon Analysis~~ ✅
- **Summary:** Reuse the 5-year master cache to derive 1y/2y/3y/5y analyses instead of separate downloads.
- **Status:** Live via updated analysis sync + presentation constants (5y horizon added, analysis slices derived from master cache, no extra Yahoo calls).

### ~~3. Monthly S&P Constituent Validation~~ ✅
- **Summary:** Schedule a monthly sanity check of the Wikipedia S&P 500 list.
- **Status:** Implemented via `services/ticker_validation_service.py` (writes diff reports under `sp500_data/validation/` and logs warnings when sets differ). Ready to wire into cron.

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

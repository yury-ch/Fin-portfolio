---
name: status
description: Check health of all fin-portfolio microservices and data cache freshness
---

Check the health and data freshness of all fin-portfolio microservices and caches.

1. Check which services are running:
   - curl -s http://localhost:8000/health  (ticker service)
   - curl -s http://localhost:8001/health  (data service)
   - curl -s http://localhost:8002/health  (calculation service)

2. Check data cache freshness:
   - curl -s http://localhost:8001/cache-info

3. Report in a clear table:
   - Each service: running ✅ / down ❌
   - Price cache: last synced, data through, age
   - Analysis cache: per period (1y/2y/3y/5y) — last run, fresh/stale
   - Flag anything stale (> 7 days) or missing with a clear warning

---
name: logs
description: Tail or search structured logs from fin-portfolio microservices
---

Fetch and display logs from the fin-portfolio microservices.

Services and their log endpoints:
- ticker service:      http://localhost:8000/logs
- data service:        http://localhost:8001/logs
- calculation service: http://localhost:8002/logs

If $ARGUMENTS is empty, tail the last 50 lines from all three services in parallel and display them grouped by service.

If $ARGUMENTS specifies a service name (ticker / data / calculation), only fetch that service's logs.

Flags parsed from $ARGUMENTS:
- `--tail N`    — show last N lines (default 50)
- `--since T`   — show logs since timestamp or relative duration (e.g. "1h", "30m", "2026-04-01T22:00")
- `--level LEVEL` — filter to WARNING / ERROR / INFO (case-insensitive)
- `--grep TEXT` — filter lines containing TEXT

If a log endpoint returns 404 or the service is down, fall back to reading the log file directly from disk if it exists under logs/ in the project root.

Format output as:
  [SERVICE] TIMESTAMP LEVEL  message

Highlight ERROR lines clearly. Summarise: X lines shown, Y errors, Z warnings.

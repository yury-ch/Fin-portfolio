# tests/e2e/conftest.py
# -----------------------------------------------------------------------
# Session-scoped fixtures that start the three microservices as real
# subprocesses and tear them down after the test session.
#
# Design choices
# ──────────────
# • If a port is already in use we assume the service is already running
#   (e.g. the developer started them manually) and we skip launching a new
#   process.  This makes the suite runnable both against a live stack AND
#   in CI where services are started freshly.
# • Services are started in dependency order: ticker → data → calc.
# • A `network` marker is registered so tests that call Yahoo Finance can
#   be skipped with  pytest -m "not network".
# -----------------------------------------------------------------------

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (name, health-check URL, script path relative to project root)
_SERVICE_DEFS = [
    ("ticker_service", "http://localhost:8000/health", "services/ticker_service.py"),
    ("data_service",   "http://localhost:8001/health", "services/data_service.py"),
    ("calc_service",   "http://localhost:8002/health", "services/calculation_service.py"),
]

BASE_URLS = {
    "ticker": "http://localhost:8000",
    "data":   "http://localhost:8001",
    "calc":   "http://localhost:8002",
}


# ── helpers ────────────────────────────────────────────────────────────

def _port_open(port: int) -> bool:
    """Return True if something is already listening on localhost:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("localhost", port)) == 0


def _wait_healthy(url: str, timeout_s: int = 45) -> bool:
    """Poll url every second until HTTP 200 or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ── session fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def services():
    """
    Start ticker, data and calculation services.

    Yields BASE_URLS dict so tests can resolve service endpoints.
    Services that are already running on their ports are left alone.
    """
    launched: list[tuple[str, subprocess.Popen]] = []

    for name, health_url, script in _SERVICE_DEFS:
        port = int(health_url.split(":")[2].split("/")[0])

        if _port_open(port):
            # Already running — just verify it's healthy
            if not _wait_healthy(health_url, timeout_s=5):
                pytest.fail(
                    f"{name} port {port} is in use but /health did not return 200"
                )
        else:
            proc = subprocess.Popen(
                [sys.executable, script],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            launched.append((name, proc))
            if not _wait_healthy(health_url, timeout_s=45):
                for _, p in launched:
                    p.terminate()
                pytest.fail(f"{name} did not become healthy within 45 s")

    yield BASE_URLS

    for _, proc in launched:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def http(services):  # noqa: F811  (services fixture used for startup side-effect)
    """Long-lived httpx client shared across all e2e tests."""
    with httpx.Client(timeout=30) as client:
        yield client

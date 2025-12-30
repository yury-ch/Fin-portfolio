"""Ticker Service - dedicated Wikipedia S&P 500 loader."""

import logging
from datetime import datetime
from typing import List, Optional
import os
import sys

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.ticker_provider import WikipediaTickerProvider, DEFAULT_SP500_SAMPLE  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ticker Service",
    description="Dedicated microservice for managing S&P 500 constituent downloads",
    version="1.0.0",
)

ticker_provider = WikipediaTickerProvider(fallback=list(DEFAULT_SP500_SAMPLE))


def get_provider() -> WikipediaTickerProvider:
    """Return the shared ticker provider instance."""
    return ticker_provider


class CacheInfo(BaseModel):
    """Metadata about the cached constituents file."""

    path: str
    exists: bool
    last_modified: Optional[datetime] = None
    size_bytes: Optional[int] = None


class TickerResponse(BaseModel):
    """Response model for ticker requests."""

    tickers: List[str]
    count: int
    cache: CacheInfo
    source: str
    prefer_remote: bool
    fallback_used: bool


class RefreshResponse(BaseModel):
    """Response for refresh operations."""

    success: bool
    count: int
    cache: CacheInfo
    source: str = "wikipedia"


def build_cache_info(provider: WikipediaTickerProvider) -> CacheInfo:
    info = provider.get_cache_info()
    return CacheInfo(
        path=info["path"],
        exists=info["exists"],
        last_modified=info.get("last_modified"),
        size_bytes=info.get("size_bytes"),
    )


@app.get("/health")
async def health(provider: WikipediaTickerProvider = Depends(get_provider)):
    """Simple health endpoint with cache diagnostics."""
    cache = build_cache_info(provider)
    return {
        "status": "healthy",
        "service": "ticker_service",
        "cache": cache.dict(),
    }


@app.get("/sp500-tickers", response_model=TickerResponse)
async def get_sp500_tickers(
    prefer_remote: bool = Query(
        True, description="Always attempt a fresh Wikipedia download before using cached data"
    ),
    allow_fallback: bool = Query(
        True, description="Return cached/fallback tickers if Wikipedia download fails"
    ),
    provider: WikipediaTickerProvider = Depends(get_provider),
):
    """Return the cached (or freshly downloaded) list of constituents."""
    tickers: Optional[List[str]] = None
    source = "cache"
    fallback_used = False

    if prefer_remote:
        try:
            remote = provider.get_remote_constituents()
            tickers = remote
            source = "wikipedia"
        except Exception as exc:
            logger.warning("Wikipedia download failed: %s", exc)
            if not allow_fallback:
                raise HTTPException(status_code=503, detail="Wikipedia download failed") from exc

    if tickers is None:
        cached = provider.get_cached_constituents()
        if cached:
            tickers = cached
            source = "cache"
        else:
            fallback = provider.get_fallback_constituents()
            tickers = fallback
            source = "fallback"
            fallback_used = True

    if not tickers:
        raise HTTPException(status_code=503, detail="No tickers available")

    cache = build_cache_info(provider)
    return TickerResponse(
        tickers=tickers,
        count=len(tickers),
        cache=cache,
        source=source,
        prefer_remote=prefer_remote,
        fallback_used=fallback_used,
    )


@app.post("/refresh", response_model=RefreshResponse)
async def refresh_tickers(provider: WikipediaTickerProvider = Depends(get_provider)):
    """Explicit refresh endpoint that always pulls the latest file."""
    try:
        tickers = provider.get_remote_constituents()
    except Exception as exc:
        logger.error("Failed to refresh Wikipedia constituents: %s", exc)
        raise HTTPException(status_code=503, detail="Wikipedia download failed") from exc

    cache = build_cache_info(provider)
    return RefreshResponse(
        success=True,
        count=len(tickers),
        cache=cache,
        source="wikipedia",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

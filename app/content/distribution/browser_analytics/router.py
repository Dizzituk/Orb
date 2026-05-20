# FILE: app/content/distribution/browser_analytics/router.py
"""
FastAPI routes for browser-scraped analytics.

Endpoints (Phase 2):
    GET  /platforms              — which platforms this module knows about
    POST /recon/{platform}       — Phase 1: dump page text to disk, no DB
    POST /recon-all              — recon every platform sequentially
    POST /scrape/{platform}      — NEW: navigate, parse, write ChannelAnalytics row
    POST /scrape-all             — NEW: scrape every platform sequentially
    GET  /channel-summary/{platform} — NEW: latest ChannelAnalytics row
    GET  /channel-history/{platform} — NEW: recent ChannelAnalytics rows
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.content.distribution.browser_analytics.recon import run_recon
from app.content.distribution.browser_analytics.scrape import run_scrape
from app.content.distribution.browser_analytics.urls import INSIGHTS_URLS
from app.content.distribution.browser_analytics.models import ChannelAnalytics
from app.content.distribution.browser_analytics.parsers import PARSERS

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content/distribution/browser_analytics",
    tags=["browser_analytics"],
)


@router.get("/platforms")
async def list_platforms(_auth=Depends(require_auth)) -> dict:
    """Platforms with known insights URLs (editable in urls.py)."""
    return {
        "platforms": [
            {
                "platform": p,
                "insights_url": url,
                "has_parser": p in PARSERS,
            }
            for p, url in INSIGHTS_URLS.items()
        ],
    }


# ─── Recon (Phase 1 — file-only, no DB) ──────────────────────────────


@router.post("/recon/{platform}")
async def recon_platform(
    platform: str,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> dict:
    if platform not in INSIGHTS_URLS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown platform '{platform}'. Known: {list(INSIGHTS_URLS)}",
        )
    result = await run_recon(db, platform)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "recon failed"))
    return result


@router.post("/recon-all")
async def recon_all(
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> dict:
    results: dict[str, dict] = {}
    for platform in INSIGHTS_URLS:
        try:
            results[platform] = await run_recon(db, platform)
        except Exception as e:
            logger.exception(f"[recon_all] {platform} crashed")
            results[platform] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    ok_count = sum(1 for r in results.values() if r.get("ok"))
    return {"total": len(results), "successful": ok_count, "results": results}


# ─── Scrape (Phase 2 — parse + DB write) ─────────────────────────────


@router.post("/scrape/{platform}")
async def scrape_platform(
    platform: str,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> dict:
    """
    Navigate the platform's session to its insights page, parse the
    metrics, and write a new ChannelAnalytics row. Also saves the
    raw DOM snapshot to disk for debugging.
    """
    if platform not in INSIGHTS_URLS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown platform '{platform}'. Known: {list(INSIGHTS_URLS)}",
        )
    result = await run_scrape(db, platform)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "scrape failed"))
    return result


@router.post("/scrape-all")
async def scrape_all(
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> dict:
    """
    Scrape every configured platform sequentially. Platforms without a
    parser still run (they save the debug file) but produce no DB row.
    """
    results: dict[str, dict] = {}
    for platform in INSIGHTS_URLS:
        try:
            results[platform] = await run_scrape(db, platform)
        except Exception as e:
            logger.exception(f"[scrape_all] {platform} crashed")
            results[platform] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    ok_count = sum(1 for r in results.values() if r.get("ok"))
    parsed_count = sum(1 for r in results.values() if r.get("parsed"))
    return {
        "total": len(results),
        "successful": ok_count,
        "parsed": parsed_count,
        "results": results,
    }


# ─── Read endpoints ──────────────────────────────────────────────────


@router.get("/channel-summary/{platform}")
async def get_channel_summary(
    platform: str,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> dict:
    """Latest ChannelAnalytics row for a platform. 404 if never scraped."""
    row = (
        db.query(ChannelAnalytics)
        .filter(ChannelAnalytics.platform == platform)
        .order_by(ChannelAnalytics.captured_at.desc())
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No ChannelAnalytics data for '{platform}' — run /scrape/{platform} first",
        )
    return _row_to_dict(row)


@router.get("/channel-history/{platform}")
async def get_channel_history(
    platform: str,
    limit: int = Query(30, ge=1, le=200),
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> dict:
    """Recent ChannelAnalytics rows for a platform, newest first."""
    rows = (
        db.query(ChannelAnalytics)
        .filter(ChannelAnalytics.platform == platform)
        .order_by(ChannelAnalytics.captured_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "platform": platform,
        "count": len(rows),
        "rows": [_row_to_dict(r) for r in rows],
    }


def _row_to_dict(row: ChannelAnalytics) -> dict:
    return {
        "id": row.id,
        "platform": row.platform,
        "captured_at": row.captured_at.isoformat() if row.captured_at else None,
        "period": row.period,
        "views": row.views,
        "likes": row.likes,
        "comments": row.comments,
        "shares": row.shares,
        "saves": row.saves,
        "profile_views": row.profile_views,
        "reach": row.reach,
        "content_interactions": row.content_interactions,
        "followers_total": row.followers_total,
        "followers_delta": row.followers_delta,
        "estimated_earnings": row.estimated_earnings,
        "source": row.source,
        "source_url": row.source_url,
        "metrics_json": row.metrics_json,
    }

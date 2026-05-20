# FILE: app/content/distribution/browser_analytics/scrape.py
"""
Phase 2 orchestrator — multi-attempt data capture and DB write.

For each platform, we walk an ordered strategy list (see strategies.py).
Each attempt: navigate + optional scroll + snapshot. We parse each
snapshot and use the FIRST attempt that produces a meaningful result
(at least one extracted metric with a non-None, non-zero value).

If all attempts come back empty, we still write a row to the DB —
just with all metrics None and a marker in metrics_json so the
frontend can tell "we tried and nothing came back" apart from "no
scrape has ever run."

Raw snapshots from every attempt are dumped to disk for debugging.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.web_automation import session_registry
from app.content.distribution.browser_analytics.strategies import get_strategy
from app.content.distribution.browser_analytics.acquire import acquire_snapshot
from app.content.distribution.browser_analytics.parsers import get_parser
from app.content.distribution.browser_analytics.models import ChannelAnalytics

logger = logging.getLogger(__name__)

SCRAPE_DIR = Path("D:/Orb/data/browser_recon")


async def run_scrape(db: Session, platform: str) -> dict[str, Any]:
    """
    End-to-end scrape: try each strategy attempt, parse, write DB row.

    Response shape:
        {
            "ok": bool,
            "platform": str,
            "parsed": bool,                 # False if no parser registered
            "strategy_used": str | None,    # label of the winning attempt
            "attempts": [                    # every attempt we made
                {"label": str, "debug_path": str, "landed_url": str,
                 "ok": bool, "metrics_summary": dict | None},
                ...
            ],
            "analytics_id": str | None,
            "captured_at": iso8601 | None,
            "metrics": dict | None,
            "empty": bool,                  # True if all attempts produced nothing
            "error": str,                   # only on hard failure
        }
    """
    # ── Resolve session
    sess = session_registry.get_session_by_platform(db, platform)
    if sess is None:
        return {"ok": False, "error": f"no session for platform '{platform}'"}
    session_id = sess.id

    strategy = get_strategy(platform)
    if not strategy:
        return {"ok": False, "error": f"no strategy defined for '{platform}'"}

    parser_fn = get_parser(platform)
    logger.info(
        f"[scrape] starting {platform} — {len(strategy)} attempt(s), "
        f"parser={'yes' if parser_fn else 'no'}"
    )

    attempts_log: list[dict[str, Any]] = []
    winning_metrics: dict[str, Any] | None = None
    winning_url: str = ""
    winning_label: str | None = None

    # ── Walk the strategy
    for attempt in strategy:
        label = attempt.get("label", "unknown")
        url = attempt["url"]
        scroll = attempt.get("scroll", False)

        logger.info(f"[scrape] {platform} · attempt {label} -> {url}")

        snap = await acquire_snapshot(session_id, url, scroll=scroll)

        debug_path = _save_debug_dump(platform, label, url, snap)

        attempt_record: dict[str, Any] = {
            "label": label,
            "debug_path": str(debug_path),
            "landed_url": snap["landed_url"],
            "ok": snap["ok"],
            "metrics_summary": None,
        }
        attempts_log.append(attempt_record)

        if not snap["ok"]:
            logger.warning(f"[scrape] {platform} · {label} · acquire failed: {snap['error']}")
            continue

        if parser_fn is None:
            # No parser — we're just capturing dumps. Keep going only if
            # the strategy has more attempts; no point parsing we can't do.
            continue

        # Parse this snapshot
        try:
            metrics = parser_fn(snap["snapshot"])
        except Exception as e:
            logger.exception(f"[scrape] {platform} · {label} · parser crashed")
            attempt_record["error"] = f"{type(e).__name__}: {e}"
            continue

        attempt_record["metrics_summary"] = _summarise_metrics(metrics)

        if _is_meaningful(metrics):
            winning_metrics = metrics
            winning_url = snap["landed_url"]
            winning_label = label
            logger.info(
                f"[scrape] {platform} · {label} · captured real metrics "
                f"(views={metrics.get('views')}, likes={metrics.get('likes')})"
            )
            break

        logger.info(f"[scrape] {platform} · {label} · parser returned empty, trying next attempt")

    # ── Handle the outcome
    if parser_fn is None:
        return {
            "ok": True,
            "platform": platform,
            "parsed": False,
            "strategy_used": None,
            "attempts": attempts_log,
            "empty": False,
        }

    if winning_metrics is None:
        # All attempts came back empty. Write a marker row so the UI
        # can show "last tried at X, nothing surfaced" instead of
        # pretending no scrape happened.
        empty_row = _write_channel_analytics(
            db, platform,
            metrics={
                "period": None,
                "metrics_json": {
                    "empty_all_strategies": True,
                    "attempts": [
                        {"label": a["label"], "landed_url": a["landed_url"]}
                        for a in attempts_log
                    ],
                },
            },
            source_url=attempts_log[-1]["landed_url"] if attempts_log else "",
        )
        logger.info(f"[scrape] {platform} · all {len(strategy)} strategies empty")
        return {
            "ok": True,
            "platform": platform,
            "parsed": True,
            "empty": True,
            "analytics_id": empty_row.id,
            "captured_at": empty_row.captured_at.isoformat(),
            "strategy_used": None,
            "attempts": attempts_log,
            "metrics": None,
        }

    # Winner found — write the real row.
    row = _write_channel_analytics(db, platform, winning_metrics, winning_url,
                                   strategy_label=winning_label)
    logger.info(
        f"[scrape] {platform} OK via {winning_label}: "
        f"views={winning_metrics.get('views')} "
        f"likes={winning_metrics.get('likes')} "
        f"period={winning_metrics.get('period')}"
    )
    return {
        "ok": True,
        "platform": platform,
        "parsed": True,
        "empty": False,
        "analytics_id": row.id,
        "captured_at": row.captured_at.isoformat(),
        "strategy_used": winning_label,
        "attempts": attempts_log,
        "metrics": _flatten_metrics(winning_metrics),
    }


# ─── Meaningfulness & summary helpers ────────────────────────────────


# Columns we consider "real data". If any of these are a non-zero
# number after parsing, the attempt is worth keeping.
_SIGNAL_FIELDS = (
    "views", "likes", "comments", "shares", "saves",
    "profile_views", "reach", "content_interactions",
    "followers_total", "followers_delta",
)


def _is_meaningful(metrics: dict[str, Any]) -> bool:
    """
    Did this parser run produce any real numeric signal?

    We allow zero values but they don't *trigger* meaningfulness on
    their own — having {views: 0, likes: 0, shares: 0} everywhere is
    indistinguishable from "parser ran but found nothing" in Meta's
    case. For TikTok though, genuine-zero is a real signal (no posts
    yet), so we special-case: if a per-post-fallback strategy ran
    and returned any posts_summed count, or if aggregates mapped,
    treat as meaningful even if values are 0.
    """
    mj = metrics.get("metrics_json") or {}
    strategy = mj.get("source_strategy")

    # Meta fallback path: posts were summed means we extracted structured data
    if mj.get("posts_summed"):
        return True
    # Meta aggregate path: mapped labels means we saw the cards
    if strategy == "aggregates" and mj.get("raw_aggregates"):
        return True
    # TikTok: raw_cards populated means all 6 metric cards parsed
    if mj.get("raw_cards"):
        return True
    # Generic signal: any of the known columns has a non-zero value
    for k in _SIGNAL_FIELDS:
        v = metrics.get(k)
        if v is not None and v != 0:
            return True

    return False


def _summarise_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compact preview of what the parser extracted. Used in API response."""
    return {
        "period": metrics.get("period"),
        "strategy": (metrics.get("metrics_json") or {}).get("source_strategy"),
        **{k: metrics.get(k) for k in _SIGNAL_FIELDS if metrics.get(k) is not None},
    }


# ─── Debug dumps ─────────────────────────────────────────────────────


def _save_debug_dump(platform: str, label: str, requested_url: str,
                     snap: dict[str, Any]) -> Path:
    """Write the raw DOM snapshot to disk. Always runs, even on failure."""
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = SCRAPE_DIR / f"{platform}-{label}-{stamp}.txt"

    lines = [
        f"# SCRAPE DUMP — {platform} (strategy: {label})",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Requested URL: {requested_url}",
        f"Actual URL after nav: {snap.get('landed_url', '')}",
        f"Page title: {snap.get('page_title', '')}",
        f"Popups dismissed ({len(snap.get('dismissed_popups', []))}): "
        f"{snap.get('dismissed_popups', [])}",
        f"Acquisition ok: {snap.get('ok', False)}",
    ]
    if snap.get("error"):
        lines.append(f"Acquisition error: {snap['error']}")
    lines += [
        "",
        "=" * 80,
        "## DOM / ACCESSIBILITY TREE",
        "=" * 80,
        json.dumps(snap.get("snapshot", {}), indent=2, ensure_ascii=False)[:200_000],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ─── DB write ────────────────────────────────────────────────────────


def _write_channel_analytics(
    db: Session,
    platform: str,
    metrics: dict[str, Any],
    source_url: str,
    strategy_label: str | None = None,
) -> ChannelAnalytics:
    """Map parsed metrics onto ChannelAnalytics columns and commit."""
    mj = dict(metrics.get("metrics_json") or {})
    if strategy_label:
        mj["strategy_label"] = strategy_label

    row = ChannelAnalytics(
        platform=platform,
        period=metrics.get("period"),
        views=_int_or_none(metrics.get("views")),
        likes=_int_or_none(metrics.get("likes")),
        comments=_int_or_none(metrics.get("comments")),
        shares=_int_or_none(metrics.get("shares")),
        saves=_int_or_none(metrics.get("saves")),
        profile_views=_int_or_none(metrics.get("profile_views")),
        reach=_int_or_none(metrics.get("reach")),
        content_interactions=_int_or_none(metrics.get("content_interactions")),
        followers_total=_int_or_none(metrics.get("followers_total")),
        followers_delta=_int_or_none(metrics.get("followers_delta")),
        estimated_earnings=metrics.get("estimated_earnings"),
        source="browser_scrape",
        source_url=source_url,
        metrics_json=mj or None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if k != "metrics_json"}

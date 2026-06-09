# FILE: app/llm/routing/domain_context.py
"""
Domain Context Providers — pull real data from ASTRA's domain services
for injection into LLM system prompts.

Each domain registers a context provider function that returns a text
block summarising the current state of that domain. This gets injected
into the system prompt so the LLM can answer questions with real data.

v3.0 (2026-03-13): Initial — 8 domain providers.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Registry of domain context providers
# Each provider: (db: Session) -> str  (returns context text block)
_DOMAIN_PROVIDERS: Dict[str, Callable[[Session], str]] = {}


def register_domain(name: str, provider: Callable[[Session], str]) -> None:
    """Register a domain context provider."""
    _DOMAIN_PROVIDERS[name] = provider
    logger.debug("[domain_context] Registered domain: %s", name)


def get_domain_context(domain: str, db: Session) -> str:
    """Pull context for a specific domain.

    Returns a text block ready for injection into a system prompt.
    If the domain provider fails, returns an error note rather than crashing.
    """
    provider = _DOMAIN_PROVIDERS.get(domain)
    if not provider:
        return f"[Domain '{domain}' has no context provider registered]"

    try:
        return provider(db)
    except Exception as e:
        logger.error("[domain_context] Provider '%s' failed: %s", domain, e)
        return f"[Domain '{domain}' context unavailable: {e}]"


def get_all_domain_names() -> list:
    """List all registered domain names."""
    return list(_DOMAIN_PROVIDERS.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN PROVIDERS
# ═══════════════════════════════════════════════════════════════════════════════


def _finance_context(db: Session) -> str:
    """Pull a summary of the delivery work ledger for the current tax year."""
    lines = ["## Work Ledger Context"]
    try:
        from app.finance.work_router import build_dashboard, list_tax_years
        ty = list_tax_years(db)["current"]
        d = build_dashboard(db, ty)
        s = d["stats"]
        lines.append(f"Tax year: {ty}")
        lines.append(f"Gross pay YTD: £{s['gross_pay_ytd']:.2f}")
        lines.append(f"Total parcels: {s['total_parcels']}")
        lines.append(f"Days worked: {s['days_worked']}")
        lines.append(f"Hours worked: {s['total_hours']:.1f}")
        lines.append(f"Work miles: {s['total_work_miles']:.0f}")
        lines.append(f"Estimated net YTD: £{s['estimated_net']:.2f}")
        lines.append(f"Net hourly: £{s['net_hourly']:.2f}")
        lines.append(f"Fixed weekly burn: £{d['fixed_costs']['weekly_burn']:.2f}")
    except Exception as e:
        lines.append(f"Work ledger data: {e}")
    return "\n".join(lines)


def _investments_context(db: Session) -> str:
    """Pull investment data directly from DB snapshots.

    Reads the latest PortfolioSnapshot and PositionSnapshot records.
    No async API calls needed — the data is already cached in the DB.
    """
    lines = ["## Investments Context"]
    try:
        from app.investments.models import PortfolioSnapshot, PositionSnapshot

        # Latest portfolio snapshot
        snap = (
            db.query(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.captured_at.desc())
            .first()
        )
        if snap:
            lines.append(f"Portfolio value: £{snap.total_value_gbp or 0:,.2f}")
            lines.append(f"Invested: £{snap.invested_value_gbp or 0:,.2f}")
            lines.append(f"Cash: £{snap.cash_balance_gbp or 0:,.2f}")
            lines.append(f"P&L: £{snap.total_pnl_gbp or 0:,.2f} ({snap.total_pnl_pct or 0:+.2f}%)")
            lines.append(f"Crypto: £{snap.crypto_value_gbp or 0:,.2f}")
            lines.append(f"Snapshot: {snap.captured_at.strftime('%d/%m/%Y %H:%M') if snap.captured_at else 'N/A'}")

            # Get positions from this snapshot
            positions = (
                db.query(PositionSnapshot)
                .filter(PositionSnapshot.snapshot_id == snap.id)
                .order_by(PositionSnapshot.pnl_pct.desc())
                .all()
            )
            lines.append(f"Positions: {len(positions)}")

            # Top 5 performers by % gain
            if positions:
                top5 = [p for p in positions if (p.pnl_pct or 0) > 0][:5]
                if top5:
                    lines.append("\nTop 5 performers (by % gain):")
                    for p in top5:
                        lines.append(
                            f"  - {p.name or p.ticker}: "
                            f"£{p.value_gbp or 0:,.2f} "
                            f"(P&L: £{p.pnl_gbp or 0:+,.2f}, {p.pnl_pct or 0:+.1f}%)"
                        )

                # Worst 5 performers
                worst5 = [p for p in reversed(positions) if (p.pnl_pct or 0) < 0][:5]
                if worst5:
                    lines.append("\nWorst 5 performers (by % loss):")
                    for p in worst5:
                        lines.append(
                            f"  - {p.name or p.ticker}: "
                            f"£{p.value_gbp or 0:,.2f} "
                            f"(P&L: £{p.pnl_gbp or 0:+,.2f}, {p.pnl_pct or 0:+.1f}%)"
                        )
        else:
            lines.append("No portfolio snapshots found. Take a snapshot first.")
    except Exception as e:
        lines.append(f"Investments data: {e}")
    return "\n".join(lines)


def _content_context(db: Session) -> str:
    """Pull content pipeline state: pieces in review, scheduled, published."""
    lines = ["## Content Context"]
    try:
        from app.content.models import ContentPiece, ContentOutput
        pieces = db.query(ContentPiece).order_by(ContentPiece.updated_at.desc()).limit(10).all()

        by_status = {}
        for p in pieces:
            by_status.setdefault(p.status, []).append(p.title)

        for status, titles in by_status.items():
            lines.append(f"{status}: {len(titles)} piece(s)")
            for t in titles[:3]:
                lines.append(f"  - {t}")

        # Scheduled outputs
        from datetime import datetime, timezone
        scheduled = (
            db.query(ContentOutput)
            .filter(ContentOutput.scheduled_at.isnot(None), ContentOutput.published_at.is_(None))
            .order_by(ContentOutput.scheduled_at.asc())
            .limit(5)
            .all()
        )
        if scheduled:
            lines.append(f"\nScheduled for publishing: {len(scheduled)}")
            for o in scheduled:
                sched = o.scheduled_at.strftime("%d/%m %H:%M") if o.scheduled_at else "?"
                lines.append(f"  - {o.output_format} on {o.platform} at {sched}")
    except Exception as e:
        lines.append(f"Content data: {e}")
    return "\n".join(lines)


def _social_context(db: Session) -> str:
    """Pull social media state: recent engagement, pending comments."""
    lines = ["## Social Media Context"]
    try:
        from app.content.models import ContentOutput, ContentAnalytics
        # Recent published outputs with analytics
        recent = (
            db.query(ContentOutput)
            .filter(ContentOutput.published_at.isnot(None))
            .order_by(ContentOutput.published_at.desc())
            .limit(5)
            .all()
        )
        for o in recent:
            latest = (
                db.query(ContentAnalytics)
                .filter(ContentAnalytics.output_id == o.id)
                .order_by(ContentAnalytics.snapshot_at.desc())
                .first()
            )
            if latest:
                lines.append(
                    f"  - {o.platform}: {latest.views} views, "
                    f"{latest.likes} likes, {latest.comments} comments"
                )
        if not recent:
            lines.append("No recently published content with analytics.")
    except Exception as e:
        lines.append(f"Social data: {e}")
    return "\n".join(lines)


def _lifestyle_context(db: Session) -> str:
    """Pull lifestyle data: goals, fitness, diet status."""
    lines = ["## Lifestyle Context"]
    try:
        from app.lifestyle.models import LifestyleGoal, LifestyleEntry
        goals = db.query(LifestyleGoal).filter(LifestyleGoal.status == "active").all()
        for g in goals:
            lines.append(f"  - Goal: {g.title} (target: {g.target_value} {g.unit})")

        # Recent entries
        recent = (
            db.query(LifestyleEntry)
            .order_by(LifestyleEntry.created_at.desc())
            .limit(5)
            .all()
        )
        if recent:
            lines.append(f"\nRecent entries:")
            for e in recent:
                lines.append(f"  - {e.entry_type}: {e.value} {e.unit} ({e.created_at.strftime('%d/%m')})")
    except Exception as e:
        lines.append(f"Lifestyle data: {e}")
    return "\n".join(lines)


def _debug_context(db: Session) -> str:
    """Pull debug state: recent errors, build status."""
    lines = ["## Debug Context"]
    try:
        from pathlib import Path
        log_dir = Path("D:/Orb/logs")
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                text = log_files[0].read_text(encoding="utf-8", errors="replace")
                error_lines = [l for l in text.splitlines() if "ERROR" in l][-5:]
                if error_lines:
                    lines.append("Recent errors:")
                    for el in error_lines:
                        lines.append(f"  {el[:200]}")
                else:
                    lines.append("No recent errors in logs.")
    except Exception as e:
        lines.append(f"Debug data: {e}")
    return "\n".join(lines)


def _education_context(db: Session) -> str:
    """Pull education state: study plans, progress."""
    lines = ["## Education Context"]
    try:
        from app.education.models import StudyPlan
        plans = db.query(StudyPlan).order_by(StudyPlan.updated_at.desc()).limit(5).all()
        for p in plans:
            lines.append(f"  - {p.title}: {p.status} ({p.progress_pct}% complete)")
        if not plans:
            lines.append("No active study plans.")
    except Exception as e:
        lines.append(f"Education data: {e}")
    return "\n".join(lines)


def _builds_context(db: Session) -> str:
    """Pull build pipeline state: recent builds, status."""
    lines = ["## Builds Context"]
    try:
        from app.builds.models import BuildProject
        builds = db.query(BuildProject).order_by(BuildProject.updated_at.desc()).limit(5).all()
        for b in builds:
            lines.append(f"  - {b.name}: {b.status} (updated {b.updated_at.strftime('%d/%m %H:%M')})")
        if not builds:
            lines.append("No recent builds.")
    except Exception as e:
        lines.append(f"Builds data: {e}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-REGISTER ALL PROVIDERS
# ═══════════════════════════════════════════════════════════════════════════════

register_domain("finance", _finance_context)
register_domain("investments", _investments_context)
register_domain("content", _content_context)
register_domain("social", _social_context)
register_domain("lifestyle", _lifestyle_context)
register_domain("debug", _debug_context)
register_domain("education", _education_context)
register_domain("builds", _builds_context)

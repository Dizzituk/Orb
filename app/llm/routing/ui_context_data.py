# FILE: app/llm/routing/ui_context_data.py
# Purpose: Lightweight data fetchers for UI context injection.
# Called-by: app.llm.routing.chat_routing
# Depends-on: app.investments.models
# Last-renovated: 2026-06-11
"""
Lightweight data fetchers for UI context injection.

When the chat panel sends ui_context, we can optionally inject
a summary of the data visible on that tab so the LLM can answer
questions directly without asking the user to repeat what's on screen.

v1.0 (2026-02-26): Initial — investments snapshot injection.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def fetch_investments_summary(db: Session) -> Optional[str]:
    """
    Read the latest portfolio snapshot from the DB and format a concise
    summary string for injection into the system prompt.

    Returns None if no data is available.
    """
    try:
        from app.investments.models import PortfolioSnapshot, PositionSnapshot

        # Get latest snapshot
        snap = (
            db.query(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.captured_at.desc())
            .first()
        )
        if not snap:
            return None

        # Get positions from that snapshot
        positions = (
            db.query(PositionSnapshot)
            .filter(PositionSnapshot.snapshot_id == snap.id)
            .order_by(PositionSnapshot.value_gbp.desc())
            .all()
        )

        if not positions:
            return None

        lines = [
            f"## PORTFOLIO DATA (snapshot: {snap.captured_at.strftime('%Y-%m-%d %H:%M')} UTC)",
        ]
        if snap.total_value_gbp:
            lines.append(f"Total value: \u00a3{snap.total_value_gbp:,.2f}")
        if snap.total_pnl_gbp is not None:
            lines.append(f"Total P&L: \u00a3{snap.total_pnl_gbp:+,.2f}")
            if snap.total_pnl_pct is not None:
                lines[-1] += f" ({snap.total_pnl_pct:+.1f}%)"
        if snap.cash_balance_gbp is not None:
            lines.append(f"Cash balance: \u00a3{snap.cash_balance_gbp:,.2f}")

        # Position table
        lines.append("")
        lines.append("Positions (sorted by value):")
        for p in positions:
            ticker = p.ticker or ''
            name = p.name or ticker
            qty = p.quantity or 0
            value = p.value_gbp or 0
            pnl = p.pnl_gbp
            pnl_pct = p.pnl_pct

            line = f"  {ticker} ({name}): qty={qty}, value=\u00a3{value:,.2f}"
            if pnl is not None:
                line += f", P&L=\u00a3{pnl:+,.2f}"
            if pnl_pct is not None:
                line += f" ({pnl_pct:+.1f}%)"
            lines.append(line)

        return "\n".join(l for l in lines if l is not None)

    except Exception as e:
        logger.debug(f"[ui_context_data] Failed to fetch investments summary: {e}")
        return None


def fetch_tab_data(job_type: str, db: Session) -> Optional[str]:
    """
    Dispatch to the appropriate data fetcher based on tab type.

    Returns a formatted string for system prompt injection, or None.
    """
    if job_type == "investments":
        return fetch_investments_summary(db)

    # Future: accounts, content, etc.
    return None


def fetch_editor_state_block() -> Optional[str]:
    """Live editor-pane state for prompt injection (2026-06-12).

    When a document is open in the desktop Univer pane, returns a short
    block telling the model WHAT is open and WHICH tools act on it, so
    'can you see what I'm looking at' resolves to the open file instead
    of a request to upload. In-memory read — no round trip to the pane.
    Returns None when nothing is open (zero cost, zero noise).
    """
    try:
        from app.documents import editor_actions
        state = editor_actions.editor_state()
        if not state.get("open"):
            return None
        kind = "spreadsheet" if state.get("kind") == "sheet" else "document"
        name = state.get("name") or "a file"
        return (
            "## EDITOR PANE (live)\n"
            f"The user has {name} open in the desktop editor pane right now — "
            f"a {kind} at {state.get('path')}.\n"
            "When they say 'this file', 'this document', 'this spreadsheet', or "
            "'what I'm looking at', they mean THIS open file. You can see and "
            "edit it through tools: read_document (documents) or "
            "read_sheet_range / list_sheet_names (spreadsheets) to read; "
            "edit_document_text or set_sheet_range to change it; save_document "
            "to save. Never ask the user to upload or paste it — read it with "
            "the tools."
        )
    except Exception as e:
        logger.debug(f"[ui_context_data] editor state fetch failed: {e}")
        return None

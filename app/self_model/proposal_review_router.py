# FILE: app/self_model/proposal_review_router.py
"""
HTTP API for reviewing proposed identity-store writes.

Exposes:
  GET  /self-model/proposals                   list proposals (filterable)
  GET  /self-model/proposals/shadow-log        shadow-mode observations only
  GET  /self-model/proposals/stats             counts by status
  POST /self-model/proposals/{id}/accept       commit a queued proposal
  POST /self-model/proposals/{id}/reject       reject a queued proposal
  POST /self-model/proposals/expire-old        expire stale proposals (admin)

In Phase 3 shadow mode, /shadow-log is the main thing Taz looks at — it's
the live audit trail of "what would the arbiter do if it were enforcing?"
without any actual behaviour change.

Authoritative since Phase 3 (2026-05-02).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/self-model/proposals", tags=["self_model"])


# =============================================================================
# GET /self-model/proposals
# =============================================================================

@router.get("")
async def list_proposals(
    status: Optional[str] = Query(None, description="filter by status"),
    field_name: Optional[str] = Query(None, alias="field"),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    """List proposals (most recent first), optionally filtered."""
    from app.self_model.proposed_facts import get_proposed_facts_store
    store = get_proposed_facts_store()
    items = store.list(status=status, field_name=field_name, limit=limit)
    return {
        "count": len(items),
        "filter": {"status": status, "field": field_name, "limit": limit},
        "items": [_view(p) for p in items],
    }


# =============================================================================
# GET /self-model/proposals/shadow-log
# =============================================================================

@router.get("/shadow-log")
async def shadow_log(
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    """
    Show only shadow-mode observations: what the arbiter would have done
    in enforce mode. Grouped by would_have_status so it's easy to scan
    "what would have been queued / rejected as contradictions / etc."
    """
    from app.self_model.proposed_facts import get_proposed_facts_store
    store = get_proposed_facts_store()
    items = store.list(status="shadow_logged", limit=limit)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for p in items:
        key = p.would_have_status or "unknown"
        grouped.setdefault(key, []).append(_view(p))

    return {
        "total_shadow_observations": len(items),
        "by_would_have_status": {k: len(v) for k, v in grouped.items()},
        "grouped": grouped,
    }


# =============================================================================
# GET /self-model/proposals/stats
# =============================================================================

@router.get("/stats")
async def stats() -> Dict[str, Any]:
    """Return status histogram + arbiter mode."""
    from app.self_model.proposed_facts import get_proposed_facts_store
    from app.self_model.write_arbiter import get_arbiter_mode
    return {
        "mode": get_arbiter_mode(),
        "counts": get_proposed_facts_store().stats(),
    }


# =============================================================================
# GET /self-model/proposals/reconcile
# =============================================================================

@router.get("/reconcile")
async def reconcile_endpoint() -> Dict[str, Any]:
    """
    Run the cross-store reconciler over the live identity + user_facts data.
    Returns warnings for every detected disagreement (Tier 1 vs user_facts.json,
    address vs current_location, shape-validation failures, etc.).

    This is read-only and never mutates anything. Use this to see what the
    Phase 5 migration will need to clean up.
    """
    from app.self_model.reconciler import reconcile
    report = reconcile()
    return report.to_dict()


# =============================================================================
# POST /self-model/proposals/{id}/accept
# =============================================================================

@router.post("/{proposal_id}/accept")
async def accept_proposal(proposal_id: str) -> Dict[str, Any]:
    """Accept a queued proposal — commits to identity.json."""
    from app.self_model.write_arbiter import commit_proposal
    result = commit_proposal(proposal_id, by="user")
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("reason"))
    return result


# =============================================================================
# POST /self-model/proposals/{id}/reject
# =============================================================================

@router.post("/{proposal_id}/reject")
async def reject_proposal_endpoint(proposal_id: str) -> Dict[str, Any]:
    """Reject a queued proposal — never gets committed."""
    from app.self_model.write_arbiter import reject_proposal
    result = reject_proposal(proposal_id, by="user")
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("reason"))
    return result


# =============================================================================
# POST /self-model/proposals/expire-old
# =============================================================================

@router.post("/expire-old")
async def expire_old(
    max_age_days: int = Query(30, ge=1, le=365),
) -> Dict[str, Any]:
    """Move stale shadow_logged/queued proposals to expired. Admin-ish."""
    from app.self_model.proposed_facts import get_proposed_facts_store
    n = get_proposed_facts_store().expire_old(max_age_days=max_age_days)
    return {"expired": n, "max_age_days": max_age_days}


# =============================================================================
# Helpers
# =============================================================================

def _view(p) -> Dict[str, Any]:
    """Project a ProposedFact into a stable dict for JSON output."""
    return {
        "id": p.id,
        "field": p.field_name,
        "proposed_value": p.proposed_value,
        "current_value": p.current_value,
        "source": p.source,
        "proposed_at": p.proposed_at,
        "status": p.status,
        "would_have_status": p.would_have_status,
        "would_have_reason": p.would_have_reason,
        "decided_at": p.decided_at,
        "decided_by": p.decided_by,
        "evidence": p.evidence,
    }

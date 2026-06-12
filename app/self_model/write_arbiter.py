# FILE: app/self_model/write_arbiter.py
# Purpose: Write arbiter — single chokepoint for all proposed Tier 1 identity writes.
# Called-by: app.debug.executors.memory_tools, app.self_model.hooks, app.self_model.identity_capture, app.self_model.proposal_review_router
# Depends-on: app.self_model.canonical_schema, app.self_model.contradiction_detector, app.self_model.identity, app.self_model.proposed_facts
# Last-renovated: 2026-06-11
"""
Write arbiter — single chokepoint for all proposed Tier 1 identity writes.

Every auto-writer (identity_capture, knowledge_extractor, save_to_memory,
etc.) calls arbiter.propose() instead of writing to identity.json directly.
The arbiter then:

  1. Validates the proposed value against canonical_schema.
  2. Checks for contradictions with the existing value.
  3. Decides what to do based on tier rules and ASTRA_ARBITER_MODE:
       - shadow  : log only, don't change live store (Phase 3 default)
       - enforce : actually act on the decision (Phase 4)
       - off     : refuse all proposals (panic disable)

Tier rules (when in enforce mode):
  - Tier 1 + first time     -> queued (human confirmation required)
  - Tier 1 + soft contra    -> queued
  - Tier 1 + hard contra    -> queued (with prominent warning)
  - Tier 1 + identical      -> shadow_logged with reason="no-op"
  - Tier 1 + validation fail -> rejected_validation

The arbiter never bypasses confirmation for Tier 1 writes. That's the whole
point. Auto-acceptance is reserved for Tier 2 preference writes (which
already have their own confidence/decay machinery and a forgiving model).

Authoritative since Phase 3 (2026-05-02).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Modes
# =============================================================================

# ASTRA_ARBITER_MODE env var controls behaviour.
# Default is "shadow" — Phase 3 baseline. Phase 4 will set this to "enforce"
# in .env after a week of shadow observation.
ARBITER_MODE_SHADOW = "shadow"
ARBITER_MODE_ENFORCE = "enforce"
ARBITER_MODE_OFF = "off"


def get_arbiter_mode() -> str:
    """Read ASTRA_ARBITER_MODE; default shadow."""
    raw = os.getenv("ASTRA_ARBITER_MODE", ARBITER_MODE_SHADOW).strip().lower()
    if raw not in (ARBITER_MODE_SHADOW, ARBITER_MODE_ENFORCE, ARBITER_MODE_OFF):
        logger.warning(
            "[arbiter] unknown ASTRA_ARBITER_MODE=%r, defaulting to shadow",
            raw,
        )
        return ARBITER_MODE_SHADOW
    return raw


# =============================================================================
# Result type
# =============================================================================

@dataclass
class ProposalResult:
    """What the caller gets back from arbiter.propose()."""
    proposal_id: str
    field_name: str
    status: str
    mode: str  # "shadow" | "enforce" | "off"
    would_have_status: Optional[str] = None
    reason: str = ""
    committed_to_store: bool = False  # True if identity.json was actually written
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "field_name": self.field_name,
            "status": self.status,
            "mode": self.mode,
            "would_have_status": self.would_have_status,
            "reason": self.reason,
            "committed_to_store": self.committed_to_store,
            **self.extra,
        }


# =============================================================================
# Arbiter
# =============================================================================

def _read_current_value(field_name: str) -> Any:
    """Look up the current identity-store value for `field_name`. None if absent."""
    try:
        from app.self_model.identity import get_identity_store
        return get_identity_store().get_value(field_name)
    except Exception as exc:
        logger.debug("[arbiter] could not read current value for %s: %s", field_name, exc)
        return None


def propose(
    field_name: str,
    proposed_value: Any,
    source: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> ProposalResult:
    """
    Submit a proposed write to the arbiter.

    The arbiter never silently drops a proposal — every call produces a
    ProposalResult and creates a record in the proposed_facts store. The
    record's `would_have_status` always tells you what enforce-mode would
    have done, regardless of current mode.

    Args:
        field_name:     Tier 1 field name (or alias).
        proposed_value: The value the writer wants to store.
        source:         Identifier of the writer (e.g. "identity_capture",
                        "knowledge_extractor:session:53", "tool:save_to_memory").
        evidence:       Optional dict of context for the human reviewer.

    Returns:
        ProposalResult.
    """
    from app.self_model.canonical_schema import (
        get_field_schema, validate_value, is_deprecated,
    )
    from app.self_model.contradiction_detector import check_contradiction
    from app.self_model.proposed_facts import (
        ProposedFact, get_proposed_facts_store,
    )

    mode = get_arbiter_mode()
    store = get_proposed_facts_store()
    schema = get_field_schema(field_name)

    # ── Unknown field → log and reject without staging ─────────────
    if schema is None:
        logger.warning(
            "[arbiter] unknown field %r from source=%s — refusing to propose",
            field_name, source,
        )
        return ProposalResult(
            proposal_id="",
            field_name=field_name,
            status="rejected_validation",
            mode=mode,
            reason=f"unknown field {field_name!r} (not in canonical_schema)",
        )

    # ── Validate ────────────────────────────────────────────────────
    valid, validate_reason = validate_value(field_name, proposed_value)

    # ── Read current value for contradiction check ─────────────────
    current = _read_current_value(field_name)

    # ── Compute would_have_status ──────────────────────────────────
    if not valid:
        would = "rejected_validation"
        reason = f"validation failed: {validate_reason}"
        contradiction = None
    else:
        contradiction = check_contradiction(field_name, current, proposed_value)
        if contradiction.severity == "none":
            # Identical value or first-time write
            if current is None:
                would = "queued"
                reason = "first-time Tier 1 write (human confirmation required)"
            else:
                would = "shadow_logged"  # no-op duplicate
                reason = "value identical to current (no-op)"
        elif contradiction.is_hard:
            would = "rejected_contradiction" if is_deprecated(field_name) else "queued"
            reason = f"hard contradiction: {contradiction.reason}"
        else:
            would = "queued"
            reason = f"soft contradiction: {contradiction.reason}"

    # In shadow mode, the recorded status is always "shadow_logged".
    # In enforce mode, the status equals would_have_status.
    # In off mode, everything is "rejected_validation" with mode-off reason.
    if mode == ARBITER_MODE_OFF:
        record_status = "rejected_validation"
        reason = "ARBITER_MODE=off (writes refused)"
    elif mode == ARBITER_MODE_SHADOW:
        record_status = "shadow_logged"
    else:  # enforce
        record_status = would

    # ── Stage the proposal ─────────────────────────────────────────
    proposal = ProposedFact(
        field_name=field_name,
        proposed_value=proposed_value,
        current_value=current,
        source=source,
        status=record_status,
        would_have_status=would,
        would_have_reason=reason,
        evidence=evidence or {},
    )
    store.add(proposal)

    committed = False

    # ── In enforce mode, act on auto-acceptable decisions ──────────
    if mode == ARBITER_MODE_ENFORCE and would == "shadow_logged":
        # No-op duplicate — nothing to write, but return as committed (the
        # value is already the proposed one).
        committed = True

    # NOTE: enforce-mode does NOT commit Tier 1 writes here. Tier 1 commits
    # only happen via the review API (proposal_review_router.accept). This is
    # intentional — Tier 1 ALWAYS requires a human in the loop. Auto-accept
    # for Tier 1 would defeat the purpose of the arbiter.

    return ProposalResult(
        proposal_id=proposal.id,
        field_name=field_name,
        status=record_status,
        mode=mode,
        would_have_status=would,
        reason=reason,
        committed_to_store=committed,
        extra={
            "valid": valid,
            "contradiction_severity": (
                contradiction.severity if contradiction else "n/a"
            ),
        },
    )


# =============================================================================
# Confirmation API (called by the review router, not by writers)
# =============================================================================

def commit_proposal(proposal_id: str, by: str = "user") -> Dict[str, Any]:
    """
    Mark a queued proposal as user_accepted AND write the value to identity.json.

    This is the ONLY path that commits Tier 1 writes. Called by the review
    router when Taz accepts a proposal.
    """
    from app.self_model.identity import get_identity_store
    from app.self_model.proposed_facts import get_proposed_facts_store

    proposed_facts = get_proposed_facts_store()
    proposal = proposed_facts.get(proposal_id)
    if not proposal:
        return {"ok": False, "reason": f"unknown proposal_id {proposal_id!r}"}

    if proposal.status not in ("queued", "shadow_logged"):
        return {
            "ok": False,
            "reason": f"proposal in status {proposal.status!r}, cannot commit",
        }

    # Write to the identity store using a source string that's on the
    # kill-switch allowlist (identity.py recognises "user_confirmed" and
    # "audit_fix" prefixes).
    src = f"user_confirmed_via_arbiter:{proposal.source}"
    write_result = get_identity_store().set(
        proposal.field_name, proposal.proposed_value, source=src,
    )

    proposed_facts.update_status(
        proposal.id, "user_accepted", decided_by=by,
    )

    logger.info(
        "[arbiter] committed proposal %s field=%s by=%s -> %s",
        proposal_id, proposal.field_name, by, write_result.get("action"),
    )
    return {
        "ok": True,
        "proposal_id": proposal_id,
        "write_result": write_result,
    }


def reject_proposal(proposal_id: str, by: str = "user") -> Dict[str, Any]:
    """Mark a queued / shadow_logged proposal as user_rejected."""
    from app.self_model.proposed_facts import get_proposed_facts_store

    store = get_proposed_facts_store()
    target = store.update_status(proposal_id, "user_rejected", decided_by=by)
    if not target:
        return {"ok": False, "reason": f"unknown proposal_id {proposal_id!r}"}
    return {"ok": True, "proposal_id": proposal_id}

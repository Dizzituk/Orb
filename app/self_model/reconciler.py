# FILE: app/self_model/reconciler.py
# Purpose: Read-side reconciler for the self-model stores.
# Called-by: app.self_model.proposal_review_router
# Depends-on: app.self_model.canonical_schema, app.self_model.identity
# Last-renovated: 2026-06-11
"""
Read-side reconciler for the self-model stores.

Detects when the same conceptual fact is stored differently in two places
(identity.json vs user_facts.json, address vs current_location, etc.) and
attaches warnings to whatever the caller is about to display.

This is the read-side counterpart to the write arbiter (Phase 3). The
arbiter prevents NEW corruptions; the reconciler surfaces EXISTING ones
so they get cleaned up over time.

It's safe to ship live — read-only, only ever ATTACHES warnings to a
copy of the data, never mutates the underlying stores.

Authoritative since Phase 6 (2026-05-02).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Result types
# =============================================================================

@dataclass
class ReconciliationWarning:
    """One detected disagreement or anomaly."""
    rule: str           # which rule fired (R1/R2/R3/R4)
    severity: str       # "info" | "soft" | "hard"
    field: str          # the field involved (or pair, like "address~current_location")
    detail: str
    sources: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "field": self.field,
            "detail": self.detail,
            "sources": self.sources,
        }


@dataclass
class ReconciliationReport:
    """Bundle of warnings + the (read-only) reconciled view."""
    warnings: List[ReconciliationWarning] = field(default_factory=list)

    @property
    def has_hard_warning(self) -> bool:
        return any(w.severity == "hard" for w in self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warning_count": len(self.warnings),
            "hard_warning_count": sum(1 for w in self.warnings if w.severity == "hard"),
            "warnings": [w.to_dict() for w in self.warnings],
        }


# =============================================================================
# Rules
# =============================================================================

def _normalise_str(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())


def _r1_tier_dominance(
    identity: Dict[str, Any],
    user_facts: List[Dict[str, Any]],
) -> List[ReconciliationWarning]:
    """
    Rule R1: same field in Tier 1 (identity) and lower tier (user_facts) with
    different values. Tier 1 wins; flag the lower for cleanup.
    """
    out: List[ReconciliationWarning] = []
    if not user_facts:
        return out
    identity_keys = set(identity.keys())
    biographical = [f for f in user_facts if f.get("category") == "biographical"]
    for fact in biographical:
        key = fact.get("key", "")
        if key not in identity_keys:
            continue
        ident_val = identity[key]
        uf_val = fact.get("value")
        if _normalise_str(ident_val) == _normalise_str(uf_val):
            continue
        out.append(ReconciliationWarning(
            rule="R1",
            severity="soft",
            field=key,
            detail=(f"Tier 1 identity has {ident_val!r} but user_facts.json has "
                    f"{uf_val!r} (Tier 1 wins; user_facts entry is stale)"),
            sources={
                "identity_value": ident_val,
                "user_facts_value": uf_val,
                "user_facts_source": fact.get("source"),
            },
        ))
    return out


def _r2_semantic_pairs(identity: Dict[str, Any]) -> List[ReconciliationWarning]:
    """
    Rule R2: known overlapping field pairs. Flag hard disagreements.
    """
    out: List[ReconciliationWarning] = []

    # address vs current_location (pre-Phase 5 retirement)
    addr = identity.get("address")
    cl = identity.get("current_location")
    if addr and cl:
        addr_norm = _normalise_str(addr)
        cl_norm = _normalise_str(cl)
        if cl_norm and cl_norm not in addr_norm:
            # If current_location's town isn't a substring of address,
            # they disagree. (Soft, because current_location is being
            # retired anyway.)
            out.append(ReconciliationWarning(
                rule="R2",
                severity="soft",
                field="address~current_location",
                detail=(f"current_location {cl!r} doesn't appear inside "
                        f"address {addr!r} (current_location is being "
                        f"retired in Phase 5)"),
                sources={"address": addr, "current_location": cl},
            ))

    # birthplace vs residence_history[0]
    bp = identity.get("birthplace")
    rh = identity.get("residence_history")
    if bp and isinstance(rh, list) and rh:
        first = rh[0]
        if isinstance(first, dict):
            first_place = first.get("place", "")
            bp_norm = _normalise_str(bp)
            fp_norm = _normalise_str(first_place)
            if bp_norm and fp_norm and bp_norm != fp_norm:
                # Token overlap test for "different country" detection.
                # Strip punctuation before tokenising so "Barnstaple, Devon"
                # and "Barnstaple" tokenise consistently.
                import re
                bp_clean = re.sub(r"[^a-z0-9\s]", " ", bp_norm)
                fp_clean = re.sub(r"[^a-z0-9\s]", " ", fp_norm)
                bp_tokens = set(bp_clean.split())
                fp_tokens = set(fp_clean.split())
                if bp_tokens and fp_tokens:
                    # Use min() to ask "is the SHORTER name fully contained
                    # in the longer one?". 'Barnstaple' has 1 token;
                    # 'Barnstaple, North Devon' has 3. If all 1 of the
                    # short tokens appear in the long one, overlap=1.0
                    # and we treat as agreement (one is a refinement of
                    # the other, e.g. with country added).
                    overlap = len(bp_tokens & fp_tokens) / min(
                        len(bp_tokens), len(fp_tokens),
                    )
                    if overlap < 1.0:
                        # Soft is fine — they CAN differ (born in hospital,
                        # raised elsewhere). Hard only if zero overlap.
                        severity = "hard" if overlap == 0.0 else "info"
                        out.append(ReconciliationWarning(
                            rule="R2",
                            severity=severity,
                            field="birthplace~residence_history[0]",
                            detail=(f"birthplace {bp!r} differs from first residence "
                                    f"{first_place!r} (overlap {overlap:.2f})"),
                            sources={"birthplace": bp, "first_residence": first_place},
                        ))

    return out


def _r3_shape_validation(identity: Dict[str, Any]) -> List[ReconciliationWarning]:
    """
    Rule R3: every read of a Tier 1 field is shape-checked against canonical_schema.
    Values that fail the shape check are flagged so they can be excluded from the
    prompt block.
    """
    from app.self_model.canonical_schema import (
        is_tier_1_field, validate_value,
    )
    out: List[ReconciliationWarning] = []
    for key, value in identity.items():
        if value in (None, "", []):
            continue
        if not is_tier_1_field(key):
            continue
        # List fields use trivial validators; skip them here (they have their
        # own write-side dedupe).
        if isinstance(value, list):
            continue
        valid, reason = validate_value(key, value)
        if not valid:
            out.append(ReconciliationWarning(
                rule="R3",
                severity="hard",
                field=key,
                detail=f"current value fails shape check: {reason}",
                sources={"value": value},
            ))
    return out


# =============================================================================
# Public API
# =============================================================================

def reconcile(
    identity_dict: Optional[Dict[str, Any]] = None,
    user_facts: Optional[List[Dict[str, Any]]] = None,
) -> ReconciliationReport:
    """
    Run all rules over the provided stores. Returns a report.

    If identity_dict is None, reads from the live identity store.
    If user_facts is None, reads from user_facts.json (or [] if that file
    has been retired by the Phase 5 migration).
    """
    if identity_dict is None:
        identity_dict = _load_identity_dict()
    if user_facts is None:
        user_facts = _load_user_facts()

    report = ReconciliationReport()
    report.warnings.extend(_r1_tier_dominance(identity_dict, user_facts))
    report.warnings.extend(_r2_semantic_pairs(identity_dict))
    report.warnings.extend(_r3_shape_validation(identity_dict))
    return report


def _load_identity_dict() -> Dict[str, Any]:
    """Read identity.json into a flat {field: value} dict."""
    try:
        from app.self_model.identity import get_identity_store
        store = get_identity_store()
        return {k: f.value for k, f in store.all().items()}
    except Exception as exc:
        logger.debug("[reconciler] identity load failed: %s", exc)
        return {}


def _load_user_facts() -> List[Dict[str, Any]]:
    """Read user_facts.json (post-migration this returns []) ."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "data" / "self_model" / "user_facts.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("[reconciler] user_facts load failed: %s", exc)
        return []

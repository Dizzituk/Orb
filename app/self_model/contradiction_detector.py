# FILE: app/self_model/contradiction_detector.py
# Purpose: Detect when a proposed identity-store write contradicts an existing
# Called-by: app.self_model.write_arbiter
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Detect when a proposed identity-store write contradicts an existing
high-confidence value, and decide whether to require human confirmation.

This is the "would have caught basketball overwriting Barnstaple, North
Devon" check. The arbiter calls into here for every proposal; if the
detector returns a contradiction, the arbiter forces queued status
(human confirmation required) regardless of tier rules.

The detector is intentionally PERMISSIVE for first-time writes (no
existing value = no contradiction, just queue normally) and STRICT for
overwrites of values with established confidence.

Authoritative since Phase 3 (2026-05-02).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContradictionResult:
    """Outcome of a contradiction check."""
    has_contradiction: bool
    severity: str  # "none" | "soft" | "hard"
    reason: str = ""

    @property
    def is_hard(self) -> bool:
        return self.severity == "hard"


# =============================================================================
# Per-field contradiction logic
# =============================================================================

def _normalise(s: Any) -> str:
    """Lowercase + strip + collapse whitespace, for fuzzy comparison."""
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())


def _string_similarity(a: str, b: str) -> float:
    """
    Cheap token-overlap similarity. Returns 0.0 (disjoint) to 1.0 (identical).
    Good enough for "is the user restating the same fact" checks.
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _check_birthplace(current: Any, proposed: Any) -> ContradictionResult:
    """
    Hard contradiction if the proposed value mentions a different country,
    or has zero token overlap with the existing value.
    """
    if not current:
        return ContradictionResult(False, "none")
    a, b = _normalise(current), _normalise(proposed)
    if a == b:
        return ContradictionResult(False, "none", "identical")
    sim = _string_similarity(a, b)
    if sim < 0.2:
        return ContradictionResult(
            True, "hard",
            f"existing {current!r} vs proposed {proposed!r} (similarity {sim:.2f})",
        )
    return ContradictionResult(
        True, "soft",
        f"refinement of existing value (similarity {sim:.2f})",
    )


def _check_address(current: Any, proposed: Any) -> ContradictionResult:
    """
    Address contradictions: any change at all is hard, because addresses
    don't drift — you either move or you don't. This forces human
    confirmation for any address overwrite, which is the right default.
    """
    if not current:
        return ContradictionResult(False, "none")
    a, b = _normalise(current), _normalise(proposed)
    if a == b:
        return ContradictionResult(False, "none", "identical")
    return ContradictionResult(
        True, "hard",
        f"address change requires confirmation: {current!r} -> {proposed!r}",
    )


def _check_date_of_birth(current: Any, proposed: Any) -> ContradictionResult:
    """
    DOB never changes. Any difference is a hard contradiction.
    """
    if not current:
        return ContradictionResult(False, "none")
    if str(current).strip() == str(proposed).strip():
        return ContradictionResult(False, "none", "identical")
    return ContradictionResult(
        True, "hard",
        f"DOB cannot change: existing {current!r}, proposed {proposed!r}",
    )


def _check_name(current: Any, proposed: Any) -> ContradictionResult:
    """
    Names rarely change. Any difference is hard unless the proposed is a
    substring of the current (e.g. learning the surname after the first name
    was set, or vice versa).
    """
    if not current:
        return ContradictionResult(False, "none")
    a, b = _normalise(current), _normalise(proposed)
    if a == b:
        return ContradictionResult(False, "none", "identical")
    if a in b or b in a:
        return ContradictionResult(
            True, "soft",
            f"one is substring of the other: {current!r} <-> {proposed!r}",
        )
    return ContradictionResult(
        True, "hard",
        f"name change: existing {current!r}, proposed {proposed!r}",
    )


def _check_email(current: Any, proposed: Any) -> ContradictionResult:
    """Email change is hard — different email is a different account."""
    if not current:
        return ContradictionResult(False, "none")
    a, b = _normalise(current), _normalise(proposed)
    if a == b:
        return ContradictionResult(False, "none")
    return ContradictionResult(
        True, "hard", f"email change: {current!r} -> {proposed!r}",
    )


def _check_default(current: Any, proposed: Any) -> ContradictionResult:
    """
    Generic check for fields without specific logic: hard if values differ
    after normalisation. Conservative — always asks for confirmation on
    overwrite.
    """
    if not current:
        return ContradictionResult(False, "none")
    if _normalise(current) == _normalise(proposed):
        return ContradictionResult(False, "none", "identical")
    return ContradictionResult(
        True, "soft",
        f"value differs from current: {current!r} -> {proposed!r}",
    )


# =============================================================================
# Dispatcher
# =============================================================================

_CHECKERS = {
    "birthplace": _check_birthplace,
    "address": _check_address,
    "current_location": _check_birthplace,  # same logic as birthplace
    "date_of_birth": _check_date_of_birth,
    "name": _check_name,
    "preferred_name": _check_name,
    "email": _check_email,
    "phone": _check_email,         # same "different = hard" logic
    "current_occupation": _check_default,
    "occupation": _check_default,
    "nationality": _check_default,
}


def check_contradiction(
    field_name: str,
    current_value: Any,
    proposed_value: Any,
) -> ContradictionResult:
    """
    Public entry point. Routes to the field-specific checker; falls back
    to _check_default for unknown fields.
    """
    checker = _CHECKERS.get(field_name, _check_default)
    try:
        return checker(current_value, proposed_value)
    except Exception as exc:
        logger.error(
            "[contradiction_detector] checker for %s failed: %s",
            field_name, exc,
        )
        # Fail-safe: any error means "treat as contradiction so a human looks at it".
        return ContradictionResult(True, "hard", f"checker error: {exc}")

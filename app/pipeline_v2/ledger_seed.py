# FILE: app/pipeline_v2/ledger_seed.py
# Purpose: ASTRA v2.2 Decision Ledger — seeding helpers.
# Called-by: app.pipeline_v2.orchestrator
# Depends-on: app.pipeline_v2.ledger
# Last-renovated: 2026-06-11
"""
ASTRA v2.2 Decision Ledger — seeding helpers.

At pipeline start, the orchestrator calls seed_ledger_from_spec() to populate
the ledger with decisions already present in the finalised spec:

  - Each spec output becomes a decision: key="output.<n>"
  - Each spec step becomes a constraint (binding on downstream implementation)
  - Each acceptance criterion becomes a constraint

Keeping this separate from ledger.py so the core module stays small and the
seeding policy can evolve independently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.pipeline_v2.ledger import (
    DecisionLedger,
    LedgerConflictError,
    ledger_append,
    save_ledger,
)

logger = logging.getLogger(__name__)


def _normalise_output(item: Any) -> Optional[Dict[str, Any]]:
    """Accept dict or string output descriptors, return a normalised dict."""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"name": item, "description": item}
    return None


def seed_ledger_from_spec(
    ledger: DecisionLedger,
    spec: Dict[str, Any],
    job_dir: str,
    profile: Any = None,
) -> int:
    """Seed the ledger with decisions implied by the finalised spec.

    Idempotent by a marker decision (Derek p7 review fix): decisions restate
    cleanly on a same-key+value append, but STEP/ACCEPTANCE constraints carry
    no key and would DUPLICATE on every call. Since Derek p4 the orchestrator
    calls this on every run (agent evidence may pre-populate the ledger, so
    entry_count==0 no longer means 'unseeded'). The marker makes repeat calls
    a no-op regardless of caller.

    Returns number of entries added (0 when already seeded).
    """
    if not isinstance(spec, dict):
        logger.debug("[ledger_seed] spec not a dict; skipping seed")
        return 0

    _SEED_MARKER_KEY = "_ledger.seeded"
    if any(
        getattr(e, "type", "") == "decision" and getattr(e, "key", "") == _SEED_MARKER_KEY
        for e in ledger.entries
    ):
        logger.debug("[ledger_seed] already seeded (marker present) — skipping")
        return 0

    added = 0
    try:
        ledger_append(
            ledger, entry_type="decision", stage="spec_gate", relevant_to=["all"],
            summary="Ledger seeded from finalised spec", key=_SEED_MARKER_KEY, value="1",
        )
    except LedgerConflictError:
        return 0

    # ---- Profile-level decisions (binding for everything)
    if profile is not None:
        try:
            added += _append_profile_decisions(ledger, profile)
        except LedgerConflictError as e:
            logger.warning("[ledger_seed] Profile decision conflict: %s", e)

    # ---- Output decisions
    for output in spec.get("outputs", []) or []:
        norm = _normalise_output(output)
        if not norm:
            continue
        name = norm.get("name") or norm.get("path") or "(unnamed)"
        key = f"output.{name}"
        value = norm.get("path") or norm.get("name") or str(name)
        try:
            ledger_append(
                ledger,
                entry_type="decision",
                stage="spec_gate",
                relevant_to=["all"],
                summary=f"Expected output: {name}",
                key=key,
                value=str(value),
                rationale=norm.get("description") or "Declared in spec.outputs",
                constrains=["file layout", "build artefacts"],
            )
            added += 1
        except LedgerConflictError as e:
            logger.warning(
                "[ledger_seed] Output decision conflict for %s: %s", name, e,
            )

    # ---- Steps -> constraints
    for idx, step in enumerate(spec.get("steps", []) or [], start=1):
        step_text = step if isinstance(step, str) else (
            step.get("description") or step.get("text") or str(step)
        )
        ledger_append(
            ledger,
            entry_type="constraint",
            stage="spec_gate",
            relevant_to=["all"],
            summary=f"Step {idx}: {step_text}"[:200],
        )
        added += 1

    # ---- Acceptance -> constraints
    for idx, crit in enumerate(spec.get("acceptance_criteria", []) or [], start=1):
        crit_text = crit if isinstance(crit, str) else (
            crit.get("description") or crit.get("text") or str(crit)
        )
        ledger_append(
            ledger,
            entry_type="constraint",
            stage="spec_gate",
            relevant_to=["all"],
            summary=f"Acceptance {idx}: {crit_text}"[:200],
            category="acceptance",
        )
        added += 1

    # ---- Objective -> codebase fact (informational, not a decision)
    objective = spec.get("objective")
    if objective:
        ledger_append(
            ledger,
            entry_type="codebase_fact",
            stage="spec_gate",
            relevant_to=["all"],
            summary="Build objective",
            value=str(objective)[:500],
        )
        added += 1

    if added:
        save_ledger(ledger, job_dir)
        logger.info("[ledger_seed] Seeded %d entries from spec", added)

    return added


def _append_profile_decisions(ledger: DecisionLedger, profile: Any) -> int:
    """Record the BuildTargetProfile as a set of binding decisions."""
    added = 0
    pairs = [
        ("profile.project",   getattr(profile, "project_id", None),   ["all"]),
        ("profile.language",  getattr(profile, "language", None),     ["all"]),
        ("profile.framework", getattr(profile, "framework", None),    ["all"]),
        ("profile.root",      getattr(profile, "project_root", None), ["file layout", "paths"]),
    ]
    for key, value, constrains in pairs:
        if value is None:
            continue
        ledger_append(
            ledger,
            entry_type="decision",
            stage="orchestrator",
            relevant_to=["all"],
            summary=f"Build target {key.split('.')[-1]}",
            key=key,
            value=str(value),
            rationale="Resolved from build target profile",
            constrains=constrains,
        )
        added += 1
    return added

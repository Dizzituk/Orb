# FILE: app/translation/intents.py
# Purpose: Canonical intent definitions for ASTRA Translation Layer.
# Called-by: app.translation, app.translation._intents_utils_2, app.translation.feedback, app.translation.gates (+3 more)
# Depends-on: app.translation._intents_conversational, app.translation._intents_domains, app.translation._intents_infrastructure, app.translation._intents_operations (+2 more)
# Last-renovated: 2026-06-11
"""
Canonical intent definitions for ASTRA Translation Layer.
This is the CLOSED SET of allowed intents. No free-form execution.

v2.0 (2026-02): Split into domain files (_intents_infrastructure, _intents_operations,
                _intents_conversational). This file merges and re-exports.
v1.3 (2026-01): Added LATEST_ARCHITECTURE_MAP and LATEST_CODEBASE_REPORT_FULL intents
v1.2 (2026-01): Fixed Overwatcher gating
v1.1 (2026-01): Added Spec Gate flow intents
"""
from __future__ import annotations
from typing import Dict
from .schemas import CanonicalIntent, IntentDefinition
from app.translation._intents_utils_2 import (
    get_all_command_intents,
    get_high_stakes_intents,
    get_intent_by_trigger_phrase,
    get_intent_definition,
    get_overwatcher_intents,
    get_spec_gate_flow_intents,
)

# Domain sub-dicts
from app.translation._intents_infrastructure import INFRASTRUCTURE_INTENTS
from app.translation._intents_operations import OPERATIONS_INTENTS
from app.translation._intents_conversational import CONVERSATIONAL_INTENTS
from app.translation._intents_domains import DOMAIN_INTENTS


# =============================================================================
# CANONICAL INTENT DEFINITIONS — merged from domain files
# =============================================================================

INTENT_DEFINITIONS: Dict[CanonicalIntent, IntentDefinition] = {
    **INFRASTRUCTURE_INTENTS,
    **OPERATIONS_INTENTS,
    **CONVERSATIONAL_INTENTS,
    **DOMAIN_INTENTS,
}


# =============================================================================
# DUPLICATE GUARD — catch silent overwrites at import time
# =============================================================================

def _check_no_duplicate_intents() -> None:
    """Detect duplicate keys across the three domain dicts."""
    all_dicts = [
        ("infrastructure", INFRASTRUCTURE_INTENTS),
        ("operations", OPERATIONS_INTENTS),
        ("conversational", CONVERSATIONAL_INTENTS),
    ]
    seen: dict = {}
    for source_name, d in all_dicts:
        for key in d:
            key_str = str(key)
            if key_str in seen:
                import warnings
                warnings.warn(
                    f"[intents.py] DUPLICATE INTENT KEY {key_str} "
                    f"in '{source_name}' (first seen in '{seen[key_str]}'). "
                    f"The second definition silently overwrites the first.",
                    stacklevel=2,
                )
            seen[key_str] = source_name

_check_no_duplicate_intents()
del _check_no_duplicate_intents

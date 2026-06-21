# FILE: app/translation/gates.py
# Purpose: Safety gates for ASTRA Translation Layer (re-export shim; context gate retained here).
# Called-by: app.translation, app.translation.tier0_rules, app.translation.translator
# Depends-on: app, app.translation.confirmation_log, app.translation.intents, app.translation.schemas
# Last-renovated: 2026-06-21
"""
Safety gates for ASTRA Translation Layer.
- Directive vs Story Gate: blocks past tense/questions/future planning  -> gates_directive.py
- Context Gate: ensures required context is present                      -> here
- Confirmation Gate: requires explicit Yes for high-stakes operations    -> gates_confirmation.py
- Overwatcher Gate: validated spec + pipeline completion (v1.2)          -> gates_overwatcher.py

Split 2026-06-21 (BATCH 6): directive / overwatcher / confirmation gates moved to
single-responsibility modules; the context gate stays here. All public names are
re-exported so importers resolve unchanged.
"""
from __future__ import annotations
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from .schemas import (
    GateResult,
    DirectiveGateResult,
    ContextGateResult,
    ConfirmationGateResult,
    CanonicalIntent,
)
from .intents import get_intent_definition

# Re-export the moved gates so `from app.translation.gates import X` keeps resolving.
from .gates_directive import (
    NON_DIRECTIVE_PATTERNS,
    _COMPILED_NON_DIRECTIVE,
    check_directive_gate,
    is_obvious_chat,
)
from .gates_overwatcher import (
    check_overwatcher_gate,
    _resolve_validated_spec,
    _resolve_spec_from_jobs,
    _check_critical_pipeline_completed,
    _check_pipeline_from_jobs,
)
from .gates_confirmation import (
    ConfirmationState,
    check_confirmation_gate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT GATE
# =============================================================================

def check_context_gate(
    intent: CanonicalIntent,
    provided_context: Dict[str, Any],
) -> ContextGateResult:
    """
    Check if required context is present for the given intent.
    
    Args:
        intent: The resolved canonical intent
        provided_context: Context provided (from UI, previous messages, etc.)
        
    Returns:
        ContextGateResult indicating if context requirements are met
    """
    defn = get_intent_definition(intent)
    required = defn.requires_context
    
    if not required:
        return ContextGateResult(
            passed=True,
            gate_name="context",
            reason="No context required for this intent",
            provided_context=provided_context,
        )
    
    missing = []
    for key in required:
        if key not in provided_context or provided_context[key] is None:
            missing.append(key)
    
    if missing:
        return ContextGateResult(
            passed=False,
            gate_name="context",
            reason=f"Missing required context: {', '.join(missing)}",
            missing_context=missing,
            provided_context=provided_context,
        )
    
    return ContextGateResult(
        passed=True,
        gate_name="context",
        reason="All required context present",
        missing_context=[],
        provided_context=provided_context,
    )


def extract_context_from_text(
    text: str,
    intent: CanonicalIntent,
) -> Dict[str, Any]:
    """
    Attempt to extract context from the message text itself.
    E.g., "Run critical pipeline for job abc123" -> {"job_id": "abc123"}
    
    This is a best-effort extraction. Missing context will still
    require clarification.
    """
    context = {}
    text_lower = text.lower()
    
    # Extract job_id
    job_patterns = [
        r"for job\s+([a-zA-Z0-9\-_]+)",
        r"job[:\s]+([a-zA-Z0-9\-_]+)",
        r"job_id[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in job_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["job_id"] = match.group(1)
            break
    
    # Extract sandbox_id
    sandbox_patterns = [
        r"for sandbox\s+([a-zA-Z0-9\-_]+)",
        r"sandbox[:\s]+([a-zA-Z0-9\-_]+)",
        r"sandbox_id[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in sandbox_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["sandbox_id"] = match.group(1)
            break
    
    # Extract change_set_id (optional - Overwatcher can derive if missing)
    changeset_patterns = [
        r"change(?:_?set)?[:\s]+([a-zA-Z0-9\-_]+)",
        r"changes[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in changeset_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["change_set_id"] = match.group(1)
            break
    
    # Web search: extract query from Tier 0 rule if available
    # (The Tier 0 rule already parsed the query out of the natural language;
    #  this fallback catches cases where the full message IS the query.)
    if intent == CanonicalIntent.WEB_SEARCH and "extracted_query" not in context:
        context["extracted_query"] = text.strip()
    
    return context

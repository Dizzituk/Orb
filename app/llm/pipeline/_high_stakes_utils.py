from __future__ import annotations
import hashlib
import logging
import os
from datetime import datetime, timezone
logger = logging.getLogger(__name__)


AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"

def _maybe_complete_trace(audit_logger, trace, *, success: bool = True, error_message: str = "") -> None:
    """Complete an audit trace if one exists."""
    if not trace or not audit_logger:
        return
    try:
        audit_logger.complete_trace(trace, success=success, error_message=error_message)
    except Exception:
        pass

def _trace_step(trace, step: str, **kv) -> None:
    """Log a step/warning to the trace."""
    if not trace:
        return
    try:
        # RoutingTrace doesn't have add_step, use log_warning for step tracking
        trace.log_warning(f"step:{step}", **kv)
    except Exception:
        pass

def _trace_error(trace, step: str, message: str) -> None:
    """Log an error to the trace."""
    if not trace:
        return
    try:
        trace.log_error(step, message)
    except Exception:
        pass

def _format_fulfilled_evidence(context: "JobContext") -> str:
    """Format fulfilled evidence results for re-injection into stage prompt.

    After the orchestrator dispatches tool calls from EVIDENCE_REQUEST blocks,
    the results are stored in context.fulfilled_evidence. This function formats
    them into a system message so the LLM can incorporate the evidence on its
    next pass.
    """
    parts = []
    parts.append("=" * 60)
    parts.append("FULFILLED EVIDENCE — Orchestrator Results")
    parts.append("=" * 60)
    parts.append("")
    parts.append(
        "The orchestrator has fulfilled the following EVIDENCE_REQUESTs. "
        "Use these results to CITE evidence in your architecture. "
        "Replace the corresponding EVIDENCE_REQUEST blocks with CITED "
        "claims or DECISION blocks as appropriate. "
        "Do NOT re-emit EVIDENCE_REQUESTs for fulfilled items."
    )
    parts.append("")

    for req_id, info in context.fulfilled_evidence.items():
        parts.append(f"--- Evidence for {req_id} ---")
        tools_called = info.get("tools_called", [])
        results = info.get("results", [])
        if tools_called:
            parts.append(f"  Tools called: {', '.join(tools_called)}")
        for i, result in enumerate(results):
            if isinstance(result, dict):
                # Truncate large content payloads
                for key, val in result.items():
                    if isinstance(val, str) and len(val) > 3000:
                        result[key] = val[:3000] + "... [truncated]"
                parts.append(f"  Result {i + 1}: {result}")
            else:
                parts.append(f"  Result {i + 1}: {result}")
        parts.append("")

    parts.append("=" * 60)
    parts.append(
        "NOW: Re-generate the architecture incorporating this evidence. "
        "CITE the evidence using [CITED file=\"...\" lines=\"...\"] tags. "
        "Any claims that are still unresolved should use EVIDENCE_REQUEST "
        "(if more evidence is needed) or DECISION (if you can decide now)."
    )
    parts.append("=" * 60)
    return "\n".join(parts)

def _format_force_resolve(context: "JobContext") -> str:
    """Format force-resolve instructions for unresolved CRITICAL requests.

    After max_loops, any remaining CRITICAL EVIDENCE_REQUESTs with
    fallback_if_not_found=DECISION_ALLOWED must be resolved by the stage.
    The orchestrator NEVER fabricates decisions.
    """
    parts = []
    parts.append("=" * 60)
    parts.append("FORCE RESOLVE — Evidence Not Found")
    parts.append("=" * 60)
    parts.append("")
    parts.append(
        "The following evidence requests could NOT be fulfilled after "
        "exhausting all search loops. You MUST now resolve each one as "
        "either a DECISION block (with rationale and revisit_if) or a "
        "HUMAN_REQUIRED block. Do NOT emit EVIDENCE_REQUEST for these items."
    )
    parts.append("")

    for req_id, info in context.force_resolve.items():
        parts.append(f"--- {req_id} ---")
        parts.append(f"  Original need: {info.get('original_need', 'unknown')}")
        parts.append(f"  Instruction: {info.get('instruction', '')}")
        parts.append("")

    parts.append("=" * 60)
    return "\n".join(parts)

def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content (truncated to 16 chars)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def _utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

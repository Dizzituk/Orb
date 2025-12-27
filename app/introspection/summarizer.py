# FILE: app/introspection/summarizer.py
"""
Log Summarizer - LLM-based log explanation.

Uses the cheap OpenAI frontier model for internal summarization.
Does NOT override temperature (model default used to avoid 400 errors).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from app.introspection.schemas import JobLogBundle, LogEvent

logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt
# =============================================================================

LOG_SUMMARIZER_SYSTEM_PROMPT = """You are a log analysis assistant. Explain internal logs in clear plain English.

Your task:
1. Describe what happened in chronological order, job-by-job.
2. Mention which stages ran and whether they succeeded or failed.
3. Note any key error messages.
4. If spec-gate hash verification events are present, clearly state:
   - Whether the expected and observed hashes matched or mismatched
   - If a mismatch occurred, note that the pipeline was aborted before applying output
5. Do NOT invent events that are not present in the logs.
6. If something is missing or unclear, say so.

Keep your explanation concise but complete. Focus on what the operator needs to know."""


# =============================================================================
# Bundle Formatting
# =============================================================================

def _format_event_for_llm(event: LogEvent) -> str:
    """Format a single event for LLM consumption."""
    parts = [f"[{event.timestamp.isoformat()}] {event.event_type}"]
    
    if event.stage_name:
        parts.append(f"stage={event.stage_name}")
    if event.status:
        parts.append(f"status={event.status}")
    if event.spec_id:
        parts.append(f"spec_id={event.spec_id[:12]}...")
    if event.expected_spec_hash:
        parts.append(f"expected_hash={event.expected_spec_hash[:16]}...")
    if event.observed_spec_hash:
        parts.append(f"observed_hash={event.observed_spec_hash[:16]}...")
    if event.verified is not None:
        parts.append(f"verified={event.verified}")
    if event.error:
        parts.append(f"error={event.error[:100]}")
    
    # Add relevant metadata
    for key in ["provider", "model", "inputs", "outputs"]:
        if key in event.metadata:
            val = event.metadata[key]
            if isinstance(val, dict):
                val = json.dumps(val, ensure_ascii=False)[:100]
            parts.append(f"{key}={val}")
    
    return " | ".join(parts)


def _format_bundle_for_llm(bundle: JobLogBundle) -> str:
    """Format a job bundle for LLM consumption."""
    lines = [f"=== JOB: {bundle.job_id} ==="]
    
    if bundle.job_type:
        lines.append(f"Type: {bundle.job_type}")
    if bundle.state:
        lines.append(f"State: {bundle.state}")
    if bundle.created_at:
        lines.append(f"Created: {bundle.created_at.isoformat()}")
    if bundle.completed_at:
        lines.append(f"Completed: {bundle.completed_at.isoformat()}")
    
    # Spec-hash summary
    if bundle.spec_hash_computed or bundle.spec_hash_verified or bundle.spec_hash_mismatch:
        lines.append(f"Spec-Gate: computed={bundle.spec_hash_computed}, verified={bundle.spec_hash_verified}, mismatch={bundle.spec_hash_mismatch}")
    
    lines.append(f"Events ({len(bundle.events)}):")
    for event in bundle.events:
        lines.append(f"  {_format_event_for_llm(event)}")
    
    return "\n".join(lines)


def format_bundles_for_llm(bundles: list[JobLogBundle]) -> str:
    """Format multiple job bundles for LLM summarization."""
    if not bundles:
        return "No log events found."
    
    parts = []
    for bundle in bundles:
        parts.append(_format_bundle_for_llm(bundle))
    
    return "\n\n".join(parts)


# =============================================================================
# Summarizer
# =============================================================================

async def summarize_logs(bundles: list[JobLogBundle]) -> str:
    """
    Generate a plain-English summary of log bundles using LLM.
    
    Uses the cheap OpenAI frontier model (same as internal system summarization).
    Does NOT set temperature to avoid 400 errors from some models.
    
    Args:
        bundles: List of job log bundles to summarize
    
    Returns:
        Plain-English summary string
    """
    if not bundles:
        return "No logs found for that request."
    
    # Format logs for LLM
    formatted_logs = format_bundles_for_llm(bundles)
    
    # Truncate if too long (keep under ~8K tokens)
    max_chars = 24000
    if len(formatted_logs) > max_chars:
        formatted_logs = formatted_logs[:max_chars] + "\n\n<<truncated - showing first portion of logs>>"
    
    # Check if we should skip LLM (env flag for testing/debugging)
    if os.getenv("ORB_INTROSPECTION_SKIP_LLM", "").lower() in ("1", "true"):
        logger.info("[summarizer] Skipping LLM call (ORB_INTROSPECTION_SKIP_LLM=1)")
        return _generate_fallback_summary(bundles)
    
    try:
        from app.providers.registry import llm_call
        
        # Use cheap OpenAI model for summarization
        model_id = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
        
        logger.info(f"[summarizer] Calling LLM for log summary: provider=openai model={model_id} log_chars={len(formatted_logs)}")
        
        result = await llm_call(
            provider_id="openai",
            model_id=model_id,
            messages=[
                {"role": "user", "content": f"Summarize these internal logs:\n\n{formatted_logs}"}
            ],
            system_prompt=LOG_SUMMARIZER_SYSTEM_PROMPT,
            # NOTE: Do NOT set temperature - some OpenAI models reject non-default values
            max_tokens=1500,
            timeout_seconds=60,
        )
        
        # Log the result for debugging
        logger.info(f"[summarizer] LLM result: status={getattr(result, 'status', 'unknown')} content_len={len(result.content or '')} error={result.error_message}")
        
        if result.is_success():
            content = (result.content or "").strip()
            if content:
                return content
            else:
                logger.warning("[summarizer] LLM returned empty content, using fallback")
                return _generate_fallback_summary(bundles)
        else:
            logger.warning(f"[summarizer] LLM call failed: {result.error_message}")
            return _generate_fallback_summary(bundles)
            
    except ImportError as e:
        logger.warning(f"[summarizer] Provider registry not available: {e}")
        return _generate_fallback_summary(bundles)
    except Exception as e:
        logger.exception("[summarizer] Error calling LLM for summary")
        return _generate_fallback_summary(bundles)


def _generate_fallback_summary(bundles: list[JobLogBundle]) -> str:
    """Generate a basic summary without LLM if provider unavailable."""
    if not bundles:
        return "No logs found for that request."
    
    lines = []
    
    # Count stats
    total_jobs = len(bundles)
    succeeded = sum(1 for b in bundles if b.state == "succeeded")
    failed = sum(1 for b in bundles if b.state == "failed")
    hash_verified = sum(1 for b in bundles if b.spec_hash_verified)
    hash_mismatch = sum(1 for b in bundles if b.spec_hash_mismatch)
    
    # Summary line
    lines.append(f"Found {total_jobs} job(s): {succeeded} succeeded, {failed} failed.")
    
    if hash_verified:
        lines.append(f"✓ {hash_verified} job(s) had spec hash verified.")
    if hash_mismatch:
        lines.append(f"⚠️ {hash_mismatch} job(s) had spec hash MISMATCH - pipeline aborted before applying output.")
    
    # Job details
    lines.append("")
    for bundle in bundles:
        status_icon = "✓" if bundle.state == "succeeded" else "✗" if bundle.state == "failed" else "○"
        job_line = f"{status_icon} Job `{bundle.job_id[:12]}...`"
        
        if bundle.job_type:
            job_line += f" ({bundle.job_type})"
        
        job_line += f" - {len(bundle.events)} events"
        
        if bundle.spec_hash_mismatch:
            job_line += " ⚠️ HASH MISMATCH"
        elif bundle.spec_hash_verified:
            job_line += " ✓ hash verified"
        
        lines.append(job_line)
        
        # Show errors
        errors = [e for e in bundle.events if e.status == "error" or e.error]
        for err in errors[:2]:  # Show first 2 errors
            lines.append(f"  └─ {err.event_type}: {err.error or 'error'}")
    
    return "\n".join(lines)


__all__ = [
    "summarize_logs",
    "format_bundles_for_llm",
    "LOG_SUMMARIZER_SYSTEM_PROMPT",
]

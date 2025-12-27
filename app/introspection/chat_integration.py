# FILE: app/introspection/chat_integration.py
"""
Chat Layer Integration for Log Introspection

Detects log introspection intents from natural language:
- "show me the log for the last action"
- "show me the logs for today"
- "show me the logs for job <id>"

This module provides:
- Intent detection
- Parameter extraction
- Response formatting for chat context
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.introspection.schemas import LogRequestType
from app.introspection.service import (
    get_job_logs,
    get_jobs_in_range,
    get_last_job_logs,
    get_time_window_for_description,
)
from app.introspection.summarizer import summarize_logs

logger = logging.getLogger(__name__)


# =============================================================================
# Intent Detection
# =============================================================================

# Trigger patterns for log introspection
_LOG_TRIGGER_PATTERNS = [
    r"show\s+(?:me\s+)?(?:the\s+)?logs?\s+for",
    r"what\s+(?:are|were)\s+(?:the\s+)?logs?\s+for",
    r"get\s+(?:me\s+)?(?:the\s+)?logs?\s+for",
    r"display\s+(?:the\s+)?logs?\s+for",
    r"view\s+(?:the\s+)?logs?\s+for",
    r"fetch\s+(?:the\s+)?logs?\s+for",
]

_LAST_JOB_PATTERNS = [
    r"last\s+(?:action|job|task|run)",
    r"most\s+recent\s+(?:action|job|task|run)",
    r"previous\s+(?:action|job|task|run)",
    r"latest\s+(?:action|job|task|run)",
]

_TIME_PATTERNS = [
    (r"(?:the\s+)?last\s+hour", "last hour"),
    (r"(?:the\s+)?past\s+hour", "last hour"),
    (r"today", "today"),
    (r"yesterday", "yesterday"),
    (r"this\s+week", "this week"),
    (r"(?:the\s+)?past\s+week", "this week"),
    (r"(?:the\s+)?last\s+week", "this week"),
]

# UUID pattern for job IDs
_UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"


@dataclass
class LogIntentResult:
    """Result of log intent detection."""
    is_log_request: bool
    request_type: Optional[LogRequestType] = None
    job_id: Optional[str] = None
    time_description: Optional[str] = None


def detect_log_intent(message: str) -> LogIntentResult:
    """
    Detect if a message is a log introspection request.
    
    Args:
        message: User message text
    
    Returns:
        LogIntentResult with detection results
    """
    if not message:
        return LogIntentResult(is_log_request=False)
    
    msg_lower = message.lower().strip()
    
    # Check if this looks like a log request
    is_log_trigger = any(
        re.search(pattern, msg_lower)
        for pattern in _LOG_TRIGGER_PATTERNS
    )
    
    if not is_log_trigger:
        return LogIntentResult(is_log_request=False)
    
    # Check for "last job" patterns
    for pattern in _LAST_JOB_PATTERNS:
        if re.search(pattern, msg_lower):
            return LogIntentResult(
                is_log_request=True,
                request_type=LogRequestType.LAST_JOB,
            )
    
    # Check for time patterns
    for pattern, description in _TIME_PATTERNS:
        if re.search(pattern, msg_lower):
            return LogIntentResult(
                is_log_request=True,
                request_type=LogRequestType.TIME_WINDOW,
                time_description=description,
            )
    
    # Check for job ID
    uuid_match = re.search(_UUID_PATTERN, message)
    if uuid_match:
        return LogIntentResult(
            is_log_request=True,
            request_type=LogRequestType.JOB_ID,
            job_id=uuid_match.group(0),
        )
    
    # Default to last job if we detected a log trigger but couldn't parse specifics
    return LogIntentResult(
        is_log_request=True,
        request_type=LogRequestType.LAST_JOB,
    )


# =============================================================================
# Response Generation
# =============================================================================

async def handle_log_request(
    db: Session,
    intent: LogIntentResult,
) -> Tuple[str, dict]:
    """
    Handle a log introspection request and generate response.
    
    Args:
        db: Database session
        intent: Detected log intent
    
    Returns:
        (summary_text, structured_data)
        
        summary_text: Plain-English explanation for the user
        structured_data: Compact log data for programmatic use
    """
    if not intent.is_log_request:
        return "This doesn't appear to be a log request.", {}
    
    try:
        if intent.request_type == LogRequestType.LAST_JOB:
            result = get_last_job_logs(db)
            
        elif intent.request_type == LogRequestType.TIME_WINDOW:
            start, end = get_time_window_for_description(
                intent.time_description or "last hour"
            )
            result = get_jobs_in_range(db, start, end)
            
        elif intent.request_type == LogRequestType.JOB_ID:
            if not intent.job_id:
                return "No job ID specified.", {}
            result = get_job_logs(db, intent.job_id)
            
        else:
            return "Unknown log request type.", {}
        
        if result.error:
            return result.error, {}
        
        if not result.bundles:
            return "No logs found for that request.", {}
        
        # Generate summary
        summary = await summarize_logs(result.bundles)
        
        # Build structured data
        structured = {
            "request_type": intent.request_type.value if intent.request_type else None,
            "total_events": result.total_events,
            "jobs": [
                {
                    "job_id": b.job_id,
                    "job_type": b.job_type,
                    "state": b.state,
                    "event_count": len(b.events),
                    "spec_hash_verified": b.spec_hash_verified,
                    "spec_hash_mismatch": b.spec_hash_mismatch,
                }
                for b in result.bundles
            ],
        }
        
        return summary, structured
        
    except Exception as e:
        logger.exception("[chat_integration] Error handling log request")
        return f"Error retrieving logs: {e}", {}


def format_log_response_for_chat(summary: str, structured: dict) -> str:
    """
    Format log introspection response for chat display.
    
    Args:
        summary: Plain-English summary
        structured: Structured log data
    
    Returns:
        Formatted response string
    """
    parts = [summary]
    
    if structured and structured.get("jobs"):
        parts.append("\n\n---\n**Log Details:**")
        for job in structured["jobs"]:
            status_icon = "✓" if job.get("state") == "succeeded" else "✗" if job.get("state") == "failed" else "○"
            hash_status = ""
            if job.get("spec_hash_mismatch"):
                hash_status = " ⚠️ HASH MISMATCH"
            elif job.get("spec_hash_verified"):
                hash_status = " ✓ hash verified"
            
            parts.append(
                f"\n- {status_icon} `{job['job_id'][:12]}...` "
                f"({job.get('job_type', 'unknown')}) "
                f"[{job.get('event_count', 0)} events]{hash_status}"
            )
    
    return "".join(parts)


__all__ = [
    "detect_log_intent",
    "handle_log_request",
    "format_log_response_for_chat",
    "LogIntentResult",
]

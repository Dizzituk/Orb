# FILE: app/introspection/__init__.py
"""
Log Introspection Feature (Read-Only)

Provides natural-language log queries for operators:
- "show me the log for the last action"
- "show me the logs for today"
- "show me the logs for job <id>"

Returns:
- Plain-English summary (via LLM)
- Compact structured log dump

100% read-only - never triggers jobs or modifies state.
"""

from app.introspection.service import (
    get_last_job_logs,
    get_jobs_in_range,
    get_job_logs,
    LogQueryResult,
)
from app.introspection.summarizer import summarize_logs
from app.introspection.schemas import (
    LogEvent,
    JobLogBundle,
    LogQueryRequest,
    LogQueryResponse,
    LogRequestType,
)
from app.introspection.chat_integration import (
    detect_log_intent,
    handle_log_request,
    format_log_response_for_chat,
    LogIntentResult,
)

__all__ = [
    # Service
    "get_last_job_logs",
    "get_jobs_in_range",
    "get_job_logs",
    "LogQueryResult",
    # Summarizer
    "summarize_logs",
    # Schemas
    "LogEvent",
    "JobLogBundle",
    "LogQueryRequest",
    "LogQueryResponse",
    "LogRequestType",
    # Chat integration
    "detect_log_intent",
    "handle_log_request",
    "format_log_response_for_chat",
    "LogIntentResult",
]

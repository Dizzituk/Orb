# FILE: app/llm/local_tools/zobie/tool_commentary.py
"""Tool Commentary Renderer (Stage 2).

Wires deterministic tool outputs into lightweight chat layer (Gemini 2.0 Flash)
for readable, conversational commentary.

Key Design Principles:
- Tools remain the DRIVER, LLM is the COMMENTATOR/RENDERER
- No agent behavior - LLM cannot request actions or make decisions
- Strict formatter only - uses only provided fields, no inventions
- Switchable via env flag: ORB_COMMENTARY_ENABLED (default OFF)

Usage:
    from app.llm.local_tools.zobie.tool_commentary import (
        render_tool_commentary,
        ToolResult,
        is_commentary_enabled,
    )
    
    if is_commentary_enabled():
        result = ToolResult(
            ok=True,
            action="append",
            path="C:\\path\\to\\file.py",
            source="remote_agent",
            ...
        )
        async for chunk in render_tool_commentary(result, user_message):
            yield sse_token(chunk)

v1.1 (2026-01): Fixed is_commentary_enabled() to properly parse env values
  - ON if: 1, true, yes, on
  - OFF if: 0, false, no, off, "" (default)
v1.0 (2026-01): Initial Stage 2 implementation
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, AsyncGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def is_commentary_enabled() -> bool:
    """
    Check if tool commentary is enabled via env flag.
    
    Returns True if ORB_COMMENTARY_ENABLED is set to: 1, true, yes, on
    Returns False if set to: 0, false, no, off, "" or unset (default OFF)
    """
    val = os.getenv("ORB_COMMENTARY_ENABLED", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def get_commentary_config() -> Dict[str, Any]:
    """Get commentary configuration from stage_models."""
    try:
        from app.llm.stage_models import get_chat_config
        config = get_chat_config()
        return {
            "provider": config.provider,
            "model": config.model,
            "max_tokens": min(config.max_output_tokens, 1000),  # Keep commentary short
            "timeout": config.timeout_seconds,
        }
    except ImportError:
        # Fallback defaults matching .env CHAT config
        return {
            "provider": os.getenv("CHAT_PROVIDER", "google"),
            "model": os.getenv("CHAT_MODEL", "gemini-2.0-flash"),
            "max_tokens": 1000,
            "timeout": 30,
        }


# =============================================================================
# TOOL RESULT SCHEMA
# =============================================================================

@dataclass
class BeforeAfterState:
    """State snapshot before/after an operation."""
    lines: int = 0
    bytes: int = 0
    excerpt: str = ""  # First/last few lines preview


@dataclass
class WriteInfo:
    """Write operation details."""
    status_code: int = 200
    bytes_written: Optional[int] = None
    method: str = ""  # local, remote_agent


@dataclass
class ErrorInfo:
    """Error details."""
    type: str = ""
    detail: str = ""


@dataclass
class ToolResult:
    """Standard tool result schema for commentary rendering.
    
    All fields are optional except 'ok' and 'action'.
    Commentary renderer uses only provided fields.
    """
    # Required fields
    ok: bool
    action: str  # read, append, overwrite, delete_area, delete_lines, list, find
    
    # Path context
    path: str = ""
    
    # Execution source
    source: str = "local"  # local, remote_agent, db
    
    # Human-safe message (brief summary)
    message: str = ""
    
    # State tracking for write operations
    before: Optional[BeforeAfterState] = None
    after: Optional[BeforeAfterState] = None
    
    # Write operation metadata
    write: Optional[WriteInfo] = None
    
    # Errors (if any)
    errors: List[ErrorInfo] = field(default_factory=list)
    
    # Debug/diagnostic info (not shown to user but available)
    debug: Dict[str, Any] = field(default_factory=dict)
    
    # User's original message (for context)
    user_message: str = ""
    
    # Query-specific fields
    query_type: str = ""  # read, head, lines, list, find, append, etc.
    
    # Read operation fields
    total_lines: int = 0
    shown_lines: int = 0
    truncated: bool = False
    content_preview: str = ""  # First few lines for read operations
    
    # List operation fields
    folders_count: int = 0
    files_count: int = 0
    
    # Find operation fields
    search_term: str = ""
    results_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != "" and value != [] and value != {}:
                if isinstance(value, (BeforeAfterState, WriteInfo, ErrorInfo)):
                    result[key] = asdict(value)
                elif isinstance(value, list) and value and isinstance(value[0], ErrorInfo):
                    result[key] = [asdict(e) for e in value]
                else:
                    result[key] = value
        return result
    
    def summary_line(self) -> str:
        """Generate a one-line summary."""
        if self.ok:
            if self.action == "read":
                return f"✅ Read {self.shown_lines}/{self.total_lines} lines from {self.path}"
            elif self.action in ("append", "overwrite"):
                bytes_info = f" ({self.write.bytes_written} bytes)" if self.write and self.write.bytes_written else ""
                return f"✅ {self.action.title()} succeeded{bytes_info}: {self.path}"
            elif self.action in ("delete_area", "delete_lines"):
                return f"✅ {self.action.replace('_', ' ').title()} succeeded: {self.path}"
            elif self.action == "list":
                return f"✅ Listed {self.folders_count} folders, {self.files_count} files in {self.path}"
            elif self.action == "find":
                return f"✅ Found {self.results_count} matches for '{self.search_term}'"
            else:
                return f"✅ {self.action.title()} completed: {self.path}"
        else:
            error_msg = self.errors[0].detail if self.errors else "Unknown error"
            return f"❌ {self.action.title()} failed: {error_msg}"


# =============================================================================
# COMMENTARY PROMPT
# =============================================================================

COMMENTARY_SYSTEM_PROMPT = """You are a strict tool result formatter for the ASTRA system. Your ONLY job is to format tool execution results into readable commentary.

CRITICAL RULES:
1. You are a RENDERER, not an agent. You CANNOT request actions or make decisions.
2. Use ONLY the fields provided in the tool result. Never invent, assume, or hallucinate data.
3. Keep responses SHORT (2-4 sentences max for simple operations).
4. Always mention the path and action.
5. Always state whether remote_agent was used if source="remote_agent".
6. Show proof of success (line counts, byte counts, before/after excerpts).
7. For errors, explain what went wrong clearly.
8. Use bullet format for multi-part results.
9. Never suggest next steps or additional actions.

FORMAT GUIDELINES:
- Start with status emoji: ✅ for success, ❌ for error
- Mention path clearly
- For writes: show before/after line counts
- For reads: show lines shown vs total
- For remote_agent: explicitly state "via remote agent (OneDrive sync)"
- Keep technical details brief but accurate
"""


def build_commentary_prompt(result: ToolResult) -> str:
    """Build the prompt for commentary generation."""
    parts = [
        f"User request: {result.user_message}",
        "",
        "Tool execution result:",
        f"  action: {result.action}",
        f"  ok: {result.ok}",
        f"  path: {result.path}",
        f"  source: {result.source}",
    ]
    
    if result.message:
        parts.append(f"  message: {result.message}")
    
    if result.query_type:
        parts.append(f"  query_type: {result.query_type}")
    
    # Add action-specific fields
    if result.action in ("read", "head", "lines"):
        parts.extend([
            f"  total_lines: {result.total_lines}",
            f"  shown_lines: {result.shown_lines}",
            f"  truncated: {result.truncated}",
        ])
        if result.content_preview:
            parts.append(f"  content_preview: {result.content_preview[:200]}...")
    
    elif result.action in ("append", "overwrite", "delete_area", "delete_lines"):
        if result.before:
            parts.append(f"  before: {result.before.lines} lines, {result.before.bytes} bytes")
            if result.before.excerpt:
                parts.append(f"    excerpt: {result.before.excerpt[:100]}...")
        if result.after:
            parts.append(f"  after: {result.after.lines} lines, {result.after.bytes} bytes")
            if result.after.excerpt:
                parts.append(f"    excerpt: {result.after.excerpt[:100]}...")
        if result.write:
            parts.append(f"  write.status_code: {result.write.status_code}")
            if result.write.bytes_written is not None:
                parts.append(f"  write.bytes_written: {result.write.bytes_written}")
    
    elif result.action == "list":
        parts.extend([
            f"  folders_count: {result.folders_count}",
            f"  files_count: {result.files_count}",
        ])
    
    elif result.action == "find":
        parts.extend([
            f"  search_term: {result.search_term}",
            f"  results_count: {result.results_count}",
        ])
    
    # Add errors if present
    if result.errors:
        parts.append("  errors:")
        for err in result.errors:
            parts.append(f"    - {err.type}: {err.detail}")
    
    parts.extend([
        "",
        "Generate a brief, readable commentary for this result. Follow the system rules strictly.",
    ])
    
    return "\n".join(parts)


# =============================================================================
# COMMENTARY RENDERER
# =============================================================================

async def render_tool_commentary(
    result: ToolResult,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """
    Render tool result as conversational commentary via lightweight LLM.
    
    Yields commentary text chunks for SSE streaming.
    
    Args:
        result: ToolResult with execution details
        user_message: Original user message for context
    
    Yields:
        Commentary text chunks
    """
    # Set user message if not already set
    if user_message and not result.user_message:
        result.user_message = user_message
    
    # Get config
    config = get_commentary_config()
    provider = config["provider"]
    model = config["model"]
    max_tokens = config["max_tokens"]
    
    logger.info(f"[COMMENTARY] Rendering for action={result.action} ok={result.ok} via {provider}/{model}")
    
    try:
        # Build prompt
        prompt = build_commentary_prompt(result)
        
        # Call LLM via providers registry
        from app.providers.registry import llm_call
        
        llm_result = await llm_call(
            provider_id=provider,
            model_id=model,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=COMMENTARY_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
        
        if llm_result.is_success():
            # Yield the commentary
            commentary = llm_result.content.strip()
            yield commentary
        else:
            # Fallback to simple summary if LLM fails
            logger.warning(f"[COMMENTARY] LLM call failed: {llm_result.error_message}")
            yield result.summary_line()
            
    except Exception as e:
        logger.exception(f"[COMMENTARY] Error rendering commentary: {e}")
        # Fallback to simple summary
        yield result.summary_line()


async def render_tool_commentary_sync(
    result: ToolResult,
    user_message: str = "",
) -> str:
    """
    Synchronous version that collects all chunks and returns full commentary.
    
    Use render_tool_commentary() for streaming.
    """
    chunks = []
    async for chunk in render_tool_commentary(result, user_message):
        chunks.append(chunk)
    return "".join(chunks)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_read_result(
    ok: bool,
    path: str,
    source: str,
    total_lines: int,
    shown_lines: int,
    truncated: bool,
    content_preview: str = "",
    error: str = "",
    user_message: str = "",
) -> ToolResult:
    """Create a ToolResult for read operations."""
    errors = [ErrorInfo(type="read_error", detail=error)] if error else []
    return ToolResult(
        ok=ok,
        action="read",
        path=path,
        source=source,
        query_type="read",
        total_lines=total_lines,
        shown_lines=shown_lines,
        truncated=truncated,
        content_preview=content_preview,
        errors=errors,
        user_message=user_message,
        message=f"Read {shown_lines}/{total_lines} lines" if ok else error,
    )


def create_write_result(
    ok: bool,
    action: str,  # append, overwrite, delete_area, delete_lines
    path: str,
    source: str,
    before_lines: int = 0,
    before_bytes: int = 0,
    before_excerpt: str = "",
    after_lines: int = 0,
    after_bytes: int = 0,
    after_excerpt: str = "",
    status_code: int = 200,
    bytes_written: Optional[int] = None,
    error: str = "",
    user_message: str = "",
) -> ToolResult:
    """Create a ToolResult for write operations."""
    errors = [ErrorInfo(type="write_error", detail=error)] if error else []
    return ToolResult(
        ok=ok,
        action=action,
        path=path,
        source=source,
        query_type=action,
        before=BeforeAfterState(lines=before_lines, bytes=before_bytes, excerpt=before_excerpt),
        after=BeforeAfterState(lines=after_lines, bytes=after_bytes, excerpt=after_excerpt),
        write=WriteInfo(status_code=status_code, bytes_written=bytes_written, method=source),
        errors=errors,
        user_message=user_message,
        message=f"{action.title()} succeeded" if ok else error,
    )


def create_list_result(
    ok: bool,
    path: str,
    source: str,
    folders_count: int,
    files_count: int,
    error: str = "",
    user_message: str = "",
) -> ToolResult:
    """Create a ToolResult for list operations."""
    errors = [ErrorInfo(type="list_error", detail=error)] if error else []
    return ToolResult(
        ok=ok,
        action="list",
        path=path,
        source=source,
        query_type="list",
        folders_count=folders_count,
        files_count=files_count,
        errors=errors,
        user_message=user_message,
        message=f"Listed {folders_count} folders, {files_count} files" if ok else error,
    )


def create_find_result(
    ok: bool,
    search_term: str,
    results_count: int,
    path: str = "",
    source: str = "db",
    error: str = "",
    user_message: str = "",
) -> ToolResult:
    """Create a ToolResult for find operations."""
    errors = [ErrorInfo(type="find_error", detail=error)] if error else []
    return ToolResult(
        ok=ok,
        action="find",
        path=path,
        source=source,
        query_type="find",
        search_term=search_term,
        results_count=results_count,
        errors=errors,
        user_message=user_message,
        message=f"Found {results_count} matches for '{search_term}'" if ok else error,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "is_commentary_enabled",
    "get_commentary_config",
    
    # Schema
    "ToolResult",
    "BeforeAfterState",
    "WriteInfo",
    "ErrorInfo",
    
    # Renderer
    "render_tool_commentary",
    "render_tool_commentary_sync",
    
    # Factory functions
    "create_read_result",
    "create_write_result",
    "create_list_result",
    "create_find_result",
]

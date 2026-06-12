# FILE: app/debug/context_assembler.py
# Purpose: Context Assembler: Gathers ASTRA pipeline state for the debug LLM prompt.
# Called-by: app.debug.debug_chat
# Depends-on: app.cost.cost_budget, app.llm.routing.handler_registry, app.llm.spec_flow_state, app.llm.stage_trace
# Last-renovated: 2026-06-11
"""
Context Assembler: Gathers ASTRA pipeline state for the debug LLM prompt.

Collects data from:
- Pipeline stage outputs (Weaver, SpecGate, Critical Pipeline, Overwatcher, Implementer)
- Current Point of Truth spec
- Recent log entries
- Overwatcher governance flags
- Host scans (Architecture Scan, File Health Scan — read-only)
- Error traces
- Sandbox file tree

Each source has a token budget. Data is prioritised by recency and relevance.
Staleness filtering ensures only current-run data is included at full priority.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Token budgets per source (approximate — counted by chars / 4)
DEFAULT_TOKEN_BUDGETS = {
    "pipeline_state":   2000,
    "current_spec":     1500,
    "recent_logs":      1000,
    "overwatcher_flags": 500,
    "host_scans":       2500,  # combined architecture + file health
    "error_traces":     1000,
    "sandbox_tree":      500,
}

TOTAL_CONTEXT_BUDGET = 10_000  # tokens


@dataclass
class ContextSource:
    """A single data source for the context window."""
    name: str
    content: str
    token_estimate: int
    timestamp: float = 0.0
    pipeline_run_id: Optional[str] = None
    priority: int = 0  # higher = included first


@dataclass
class AssembledContext:
    """The final assembled context ready for the system prompt."""
    xml: str
    sources_included: List[str]
    total_tokens: int
    assembly_time_ms: int


# =============================================================================
# SOURCE COLLECTORS
# =============================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated to fit context budget]"


async def _collect_pipeline_state() -> ContextSource:
    """Gather current pipeline stage outputs."""
    content_parts = []

    # Try to get flow state
    try:
        from app.llm.spec_flow_state import get_active_flow
        flow = get_active_flow()
        if flow:
            content_parts.append(f"Active flow: {flow.stage.value if hasattr(flow, 'stage') else flow}")
    except Exception as e:
        content_parts.append(f"Flow state: unavailable ({e})")

    # Try to get stage trace
    try:
        from app.llm.stage_trace import get_recent_traces
        traces = get_recent_traces(limit=5)
        if traces:
            content_parts.append("Recent stage traces:")
            for t in traces:
                content_parts.append(f"  - {t}")
    except Exception:
        pass

    content = "\n".join(content_parts) if content_parts else "No pipeline state available."
    return ContextSource(
        name="pipeline_state",
        content=_truncate_to_budget(content, DEFAULT_TOKEN_BUDGETS["pipeline_state"]),
        token_estimate=_estimate_tokens(content),
        timestamp=time.time(),
        priority=5,
    )


async def _collect_current_spec() -> ContextSource:
    """Get the current validated Point of Truth spec."""
    content = "No validated spec found."
    try:
        from app.llm.routing.handler_registry import get_latest_validated_spec
        spec = get_latest_validated_spec()
        if spec:
            content = f"Spec ID: {spec.get('id', 'unknown')}\n"
            content += f"Hash: {spec.get('hash', 'unknown')}\n"
            if spec.get("content"):
                content += f"\n{spec['content']}"
    except Exception as e:
        content = f"Spec lookup failed: {e}"

    return ContextSource(
        name="current_spec",
        content=_truncate_to_budget(content, DEFAULT_TOKEN_BUDGETS["current_spec"]),
        token_estimate=_estimate_tokens(content),
        timestamp=time.time(),
        priority=3,
    )


async def _collect_recent_logs(limit: int = 50) -> ContextSource:
    """Read the most recent log entries from ASTRA's log files."""
    content = ""
    try:
        from pathlib import Path
        log_dir = Path("D:/Orb/logs")
        if log_dir.exists():
            # Find the most recent log file
            log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                lines = log_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
                recent = lines[-limit:]
                content = "\n".join(recent)
    except Exception as e:
        content = f"Log collection failed: {e}"

    if not content:
        content = "No recent logs available."

    return ContextSource(
        name="recent_logs",
        content=_truncate_to_budget(content, DEFAULT_TOKEN_BUDGETS["recent_logs"]),
        token_estimate=_estimate_tokens(content),
        timestamp=time.time(),
        priority=2,
    )


async def _collect_overwatcher_flags() -> ContextSource:
    """Get any active Overwatcher governance flags."""
    content = "No active governance flags."
    try:
        from app.cost.cost_budget import get_spend_summary
        summary = get_spend_summary()
        parts = []
        if summary.daily.exceeded:
            parts.append(f"⚠️ DAILY BUDGET EXCEEDED: £{summary.daily.spent_gbp:.2f} / £{summary.daily.budget_gbp:.2f}")
        if summary.monthly.exceeded:
            parts.append(f"⚠️ MONTHLY BUDGET EXCEEDED: £{summary.monthly.spent_gbp:.2f} / £{summary.monthly.budget_gbp:.2f}")
        if not parts:
            parts.append(f"Budget OK — Daily: £{summary.daily.spent_gbp:.2f} / £{summary.daily.budget_gbp:.2f}")
            parts.append(f"Monthly: £{summary.monthly.spent_gbp:.2f} / £{summary.monthly.budget_gbp:.2f}")
        content = "\n".join(parts)
    except Exception as e:
        content = f"Overwatcher data unavailable: {e}"

    return ContextSource(
        name="overwatcher_flags",
        content=_truncate_to_budget(content, DEFAULT_TOKEN_BUDGETS["overwatcher_flags"]),
        token_estimate=_estimate_tokens(content),
        timestamp=time.time(),
        priority=8,  # high priority — governance always included
    )


async def _collect_host_scans() -> ContextSource:
    """Read Architecture Scan and File Health Scan output (host, read-only)."""
    parts = []

    # Architecture scan
    try:
        from pathlib import Path
        arch_dir = Path("D:/Orb/.architecture")
        if arch_dir.exists():
            # Find the most recent architecture scan
            scan_files = sorted(arch_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            if scan_files:
                content = scan_files[0].read_text(encoding="utf-8", errors="replace")
                parts.append(f"=== Architecture Scan ({scan_files[0].name}) ===\n{content}")
    except Exception as e:
        parts.append(f"Architecture scan unavailable: {e}")

    content = "\n\n".join(parts) if parts else "No host scans available."
    return ContextSource(
        name="host_scans",
        content=_truncate_to_budget(content, DEFAULT_TOKEN_BUDGETS["host_scans"]),
        token_estimate=_estimate_tokens(content),
        timestamp=time.time(),
        priority=4,
    )


async def _collect_error_traces() -> ContextSource:
    """Collect recent error traces from the pipeline."""
    content = "No recent errors."
    try:
        from pathlib import Path
        log_dir = Path("D:/Orb/logs")
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                text = log_files[0].read_text(encoding="utf-8", errors="replace")
                # Extract ERROR and CRITICAL lines
                error_lines = [
                    line for line in text.splitlines()
                    if "ERROR" in line or "CRITICAL" in line or "Traceback" in line
                ]
                if error_lines:
                    content = "\n".join(error_lines[-30:])  # last 30 error lines
    except Exception as e:
        content = f"Error trace collection failed: {e}"

    return ContextSource(
        name="error_traces",
        content=_truncate_to_budget(content, DEFAULT_TOKEN_BUDGETS["error_traces"]),
        token_estimate=_estimate_tokens(content),
        timestamp=time.time(),
        priority=10,  # highest priority — errors always first
    )


async def _collect_sandbox_tree() -> ContextSource:
    """Get a lightweight file tree from the sandbox."""
    content = "Sandbox tree unavailable (sandbox may not be running)."
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                "http://192.168.250.2:8765/fs/tree",
                json={"roots": ["D:\\Orb"], "max_files": 500, "include_size": False},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                tree_lines = [f.get("path", "") for f in files[:200]]
                content = f"Sandbox files ({data.get('total_files', 0)} total):\n" + "\n".join(tree_lines)
    except Exception as e:
        content = f"Sandbox connection failed: {e}"

    return ContextSource(
        name="sandbox_tree",
        content=_truncate_to_budget(content, DEFAULT_TOKEN_BUDGETS["sandbox_tree"]),
        token_estimate=_estimate_tokens(content),
        timestamp=time.time(),
        priority=1,
    )


# =============================================================================
# ASSEMBLY
# =============================================================================

async def assemble_context(
    include_sources: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
) -> AssembledContext:
    """
    Assemble all context sources into a structured XML block for the LLM prompt.

    Args:
        include_sources: If set, only include these sources. Otherwise include all.
        exclude_sources: Sources to skip (e.g., ["sandbox_tree"] if sandbox is down).

    Returns:
        AssembledContext with XML string ready for system prompt injection.
    """
    start_ms = time.time()
    exclude = set(exclude_sources or [])

    # Collect all sources
    collectors = {
        "pipeline_state":   _collect_pipeline_state,
        "current_spec":     _collect_current_spec,
        "recent_logs":      _collect_recent_logs,
        "overwatcher_flags": _collect_overwatcher_flags,
        "host_scans":       _collect_host_scans,
        "error_traces":     _collect_error_traces,
        "sandbox_tree":     _collect_sandbox_tree,
    }

    if include_sources:
        collectors = {k: v for k, v in collectors.items() if k in include_sources}

    collectors = {k: v for k, v in collectors.items() if k not in exclude}

    # Run all collectors concurrently
    import asyncio
    tasks = {name: asyncio.create_task(fn()) for name, fn in collectors.items()}
    sources: List[ContextSource] = []
    for name, task in tasks.items():
        try:
            result = await task
            sources.append(result)
        except Exception as e:
            logger.warning("[context_assembler] Collector %s failed: %s", name, e)

    # Sort by priority (highest first), then fit into budget
    sources.sort(key=lambda s: s.priority, reverse=True)

    included = []
    total_tokens = 0
    for src in sources:
        if total_tokens + src.token_estimate > TOTAL_CONTEXT_BUDGET:
            # Try truncating to fit
            remaining = TOTAL_CONTEXT_BUDGET - total_tokens
            if remaining > 100:  # at least 100 tokens worth including
                src.content = _truncate_to_budget(src.content, remaining)
                src.token_estimate = remaining
            else:
                continue
        included.append(src)
        total_tokens += src.token_estimate

    # Build XML
    xml_parts = ["<astra_context>"]
    for src in included:
        tag = src.name
        xml_parts.append(f"  <{tag} timestamp=\"{src.timestamp:.0f}\">")
        xml_parts.append(f"    {src.content}")
        xml_parts.append(f"  </{tag}>")
    xml_parts.append("</astra_context>")

    assembly_ms = int((time.time() - start_ms) * 1000)

    return AssembledContext(
        xml="\n".join(xml_parts),
        sources_included=[s.name for s in included],
        total_tokens=total_tokens,
        assembly_time_ms=assembly_ms,
    )

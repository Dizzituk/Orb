# FILE: app/web_automation/memory/checks.py
"""
Check evaluators — the verification layer that turns "I clicked something"
into "the page is now in the expected state."

A check has a kind (dom_includes, url_excludes, text_includes, always_pass,
etc.) and a list of expected substrings. The runner polls each check until
it passes or its timeout elapses, so transient UI lag doesn't kill a step.

Why substring matching rather than CSS selectors? The dom_snapshot tool
returns a flattened JSON with element labels, roles, and text. A substring
match against that text is robust to:
  * minor DOM tree restructuring (the label still appears somewhere)
  * Meta A/B test variants (different exact selectors, same labels)
  * locale changes (so long as the canonical English label is included)

When DOM substrings prove unreliable for a specific platform/task, a
'vision_question' check kind can be added later (slow, expensive, but
catches what DOM misses). The kind is pluggable here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional, Tuple

from app.web_automation.memory.models import Check, CheckResult

logger = logging.getLogger(__name__)

# Type alias: an async tool dispatcher. Wired from the chat tool layer
# at call time to avoid circular imports with action_executor.
ToolDispatcher = Callable[[str, dict], Awaitable[str]]


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

async def evaluate_check(
    check: Optional[Check],
    session: Optional[str],
    executor: ToolDispatcher,
) -> CheckResult:
    """
    Poll a check until it passes or its timeout elapses.

    Args:
        check    : Check definition; None / kind=='always_pass' is a no-op pass.
        session  : Web session id (for browser-state checks). Ignored otherwise.
        executor : Async tool dispatcher (action_executor.execute_tool).

    Returns:
        CheckResult with ok, observed_summary, elapsed_ms, timed_out.
    """
    if check is None or check.kind == "always_pass":
        return CheckResult(
            ok=True,
            kind="always_pass",
            expected=[],
            observed_summary="(no check)",
            elapsed_ms=0,
        )

    start = time.monotonic()
    poll_interval = max(0.1, check.poll_interval_ms / 1000.0)
    timeout_seconds = check.timeout_ms / 1000.0

    last_observed = ""
    last_error: Optional[str] = None

    while True:
        try:
            observed, matched = await _evaluate_once(check, session, executor)
            last_observed = observed
            last_error = None
            if matched:
                return CheckResult(
                    ok=True,
                    kind=check.kind,
                    expected=list(check.expected),
                    observed_summary=_summarise(observed),
                    elapsed_ms=int((time.monotonic() - start) * 1000),
                )
        except Exception as exc:
            last_error = f"check evaluation raised: {exc}"
            logger.debug("[checks] %s during %s", last_error, check.kind)

        if (time.monotonic() - start) >= timeout_seconds:
            summary = _summarise(last_observed)
            if last_error:
                summary = f"{summary} [last error: {last_error}]"
            return CheckResult(
                ok=False,
                kind=check.kind,
                expected=list(check.expected),
                observed_summary=summary,
                elapsed_ms=int((time.monotonic() - start) * 1000),
                timed_out=True,
            )

        await asyncio.sleep(poll_interval)


# =============================================================================
# ONE-SHOT EVALUATORS
# =============================================================================

async def _evaluate_once(
    check: Check,
    session: Optional[str],
    executor: ToolDispatcher,
) -> Tuple[str, bool]:
    """Fetch fresh state for the check kind and report match status."""
    kind = check.kind

    # ─── DOM-based ──────────────────────────────────────────────────
    if kind in ("dom_includes", "dom_excludes"):
        dom = await executor("web_dom_snapshot", {"session": session})
        haystack = dom or ""
        if kind == "dom_includes":
            return haystack, _all_in(haystack, check.expected)
        return haystack, _none_in(haystack, check.expected)

    # ─── URL-based ──────────────────────────────────────────────────
    if kind in ("url_includes", "url_excludes"):
        state_raw = await executor("web_current_state", {"session": session})
        url = _extract_url(state_raw)
        if kind == "url_includes":
            return url, _all_in(url, check.expected)
        return url, _none_in(url, check.expected)

    # ─── Visible-text-based ─────────────────────────────────────────
    if kind in ("text_includes", "text_absent"):
        text = await executor("web_extract_text", {"session": session})
        haystack = text or ""
        if kind == "text_includes":
            return haystack, _all_in(haystack, check.expected)
        return haystack, _none_in(haystack, check.expected)

    return f"(unknown check kind: {kind})", False


# =============================================================================
# UTILITIES
# =============================================================================

def _all_in(haystack: str, needles) -> bool:
    return all(n in haystack for n in needles)


def _none_in(haystack: str, needles) -> bool:
    return not any(n in haystack for n in needles)


def _extract_url(state_raw: str) -> str:
    """current_state returns JSON-ish; pull current_url out best-effort."""
    if not state_raw:
        return ""
    try:
        obj = json.loads(state_raw)
        return obj.get("current_url") or obj.get("url") or ""
    except Exception:
        # web_current_state may return a formatted string in some impls;
        # fall back to a substring sweep for "url".
        return state_raw


def _summarise(observed: str, max_chars: int = 600) -> str:
    """Trim and collapse whitespace so observed snapshots fit a tool result."""
    if not observed:
        return "(empty)"
    flat = " ".join(observed.split())
    if len(flat) <= max_chars:
        return flat
    return f"{flat[:max_chars]}… [+{len(flat) - max_chars} chars]"

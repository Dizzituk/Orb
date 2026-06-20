# FILE: app/llm/loop_guard.py
# Purpose: Circuit-breaker for the Debug tool loops. Detect when the brain is "going
#          in circles" (the same move / the same failing edit repeated) and stop EARLY
#          with a graceful, resumable summary -- instead of silently grinding to
#          MAX_TOOL_ROUNDS and dumping a dead "[max rounds]" line with work half-done.
# Called-by: app.llm.chat_tool_loop (the OpenAI debug-brain loop)
# Depends-on: app.llm._streaming_utils_3 (stream_llm, lazy), stdlib
# Last-renovated: 2026-06-19
"""Loop guard -- stop the Debug brain when it repeats itself, not when it's working.

Two separate jobs:

1. DETECTION (LoopGuard). Every tool call is keyed by (name + target), where target is
   the file path (file tools) or the command head (run_command). We count calls and
   FAILURES per key -- a failure is the executor's own signal (a `BLOCKED` / `SYNTAX
   GUARD` / `old_text not found` / non-zero `Exit code` / `Tool error` string). The
   breaker trips when the SAME target has FAILED `DEBUG_LOOP_REPEAT_LIMIT` times
   (default 3) -- the real "going in circles" tell -- or when the identical call is made
   far too many times with no progress. It does NOT trip on varied, succeeding work, so
   genuine progress is never halted (the exact concern: it might have been fixing it).

2. GRACEFUL EXIT (stream_loop_wrapup). When the loop ends -- breaker tripped OR round cap
   reached -- one final, tool-FREE generation tells the user, in the brain's own voice,
   that it stopped (and why), what it DID change, what's blocking, and the single next
   step. That hand-off is what lets the work be picked up where it left off.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

# Mutating / command tools: a repeated FAILURE here is a real loop. (Reads can
# legitimately repeat, so for those only exact-identical repetition counts.)
_MUTATING = {
    "edit_file", "write_file", "run_command",
    "start_sandbox_clone", "stop_sandbox_clone",
    "read_sandbox_boot", "inspect_sandbox_boot",
}

# Substrings the executors put in a RESULT string when an action failed
# (see app/debug/executors/filesystem.py + phase_narration refusals).
_FAILURE_MARKERS = (
    "SYNTAX GUARD", "old_text not found", "old_text found", "would have syntax",
    "never closed", "not allowed", "Cannot ", "not found", "Tool error",
    "REFUSED", "Command timed out", "Command execution error",
    "Write failed", "Write error", "write error",
)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.getenv(name) or "").strip() or default))
    except Exception:
        return default


def repeat_limit() -> int:
    """How many times the same failing move is tolerated before stopping.
    DEBUG_LOOP_REPEAT_LIMIT (default 3 -- 'the third time I've seen this')."""
    return _int_env("DEBUG_LOOP_REPEAT_LIMIT", 3)


def is_failure_result(name: str, result: str) -> bool:
    """True if a tool result signals a failed action (so a repeat means a loop)."""
    if not result:
        return False
    head = result.lstrip()
    if head.startswith("Error:") or head.startswith("BLOCKED"):
        return True
    if name == "run_command" and head.startswith("Exit code:"):
        return head.split("\n", 1)[0].strip() != "Exit code: 0"
    if name in _MUTATING and any(m in result for m in _FAILURE_MARKERS):
        return True
    return False


def call_signature(name: str, tool_input: Dict) -> str:
    """Stable key: name + target (path / command-head). Retries that differ only in
    whitespace/line-endings of old_text still map to the SAME key, so the failure
    counter accumulates instead of being fooled by trivial variation."""
    target = ""
    if isinstance(tool_input, dict):
        if tool_input.get("path"):
            target = str(tool_input.get("path")).strip()
        elif tool_input.get("command"):
            target = " ".join(str(tool_input["command"]).split())[:80]
        else:
            try:
                blob = json.dumps(tool_input, sort_keys=True, default=str)
            except Exception:
                blob = str(tool_input)
            target = hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()[:10]
    return f"{name}|{target}"


class LoopGuard:
    """Per-turn tracker of tool-call repetition + failure, for the tool loop."""

    def __init__(self, limit: Optional[int] = None) -> None:
        self.limit = limit or repeat_limit()
        self.calls: Dict[str, int] = {}
        self.fails: Dict[str, int] = {}
        self._tripped: Optional[str] = None

    def record(self, name: str, signature: str, failed: bool) -> None:
        """Record one tool call + whether it failed; set the trip reason if a loop emerges."""
        self.calls[signature] = self.calls.get(signature, 0) + 1
        if failed:
            self.fails[signature] = self.fails.get(signature, 0) + 1
        if self._tripped:
            return
        target = signature.split("|", 1)[-1] or "the same target"
        if self.fails.get(signature, 0) >= self.limit:
            self._tripped = (
                f"the same action ({name} on {target}) has failed "
                f"{self.fails[signature]} times in a row"
            )
        elif self.calls.get(signature, 0) >= max(self.limit + 2, 5):
            self._tripped = (
                f"the same action ({name} on {target}) has been repeated "
                f"{self.calls[signature]} times with no progress"
            )

    def tripped(self) -> Optional[str]:
        """The human-readable loop reason, or None if no loop has been detected."""
        return self._tripped


async def stream_loop_wrapup(
    current_messages: List[Dict],
    provider: str,
    model: str,
    *,
    reason: Optional[str],
    rounds: int,
    enable_reasoning: bool = False,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """Final, tool-FREE generation after the loop ends.

    `reason` set  -> the circuit-breaker tripped (going in circles).
    `reason` None -> MAX_TOOL_ROUNDS was reached.

    Either way the model writes a short, honest hand-off (stopped + why / what changed /
    what's blocking / next step), streamed as `token` chunks then a `done`. Never raises:
    on any failure it emits a plain fallback line so the turn always closes cleanly.
    """
    from app.llm._streaming_utils_3 import stream_llm

    if reason:
        directive = (
            "STOP — you are going in circles: " + reason + ". Do NOT call any more tools. "
            "Tell the user plainly, in your own voice and in 2–4 sentences: that you have hit "
            "the same problem repeatedly and are stopping so you don't loop, exactly what is "
            "blocking you, what you DID manage to change so far, and the single concrete next "
            "step (or what you need from them). Be honest that it is not finished."
        )
    else:
        directive = (
            f"You have used all {rounds} tool rounds for this turn. Do NOT call any more tools. "
            "In 2–4 sentences, summarise what you changed, what is still left, and the next step, "
            "so the user can pick up exactly where you left off."
        )

    system_prompt = next(
        (m.get("content", "") for m in current_messages if m.get("role") == "system"), ""
    )
    msgs = [m for m in current_messages if m.get("role") != "system"]
    msgs.append({"role": "user", "content": directive})

    streamed = False
    try:
        async for chunk in stream_llm(
            provider=provider,
            model=model,
            messages=msgs,
            system_prompt=system_prompt,
            tools=None,
            enable_reasoning=enable_reasoning,
            max_tokens=max_tokens,
        ):
            ct = chunk.get("type")
            if ct == "token":
                streamed = True
                yield chunk
            elif ct == "reasoning":
                yield chunk
            # swallow 'done'/'metadata'/stray tool events; we emit our own done below
    except Exception as e:
        logger.warning("[loop_guard] wrap-up generation failed: %s", e)

    if not streamed:
        if reason:
            yield {"type": "token", "text": (
                "\n\nI've stopped here — " + reason + " — so I don't keep going in circles. "
                "This isn't fully done; tell me how you'd like to proceed and I'll pick up "
                "from where I left off."
            )}
        else:
            yield {"type": "token", "text": (
                "\n\nI've used all my tool rounds for this turn and stopped so we don't loop. "
                "Ask me to continue and I'll pick up where I left off."
            )}
    yield {"type": "done", "provider": provider, "model": model}

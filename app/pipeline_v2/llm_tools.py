# FILE: app/pipeline_v2/llm_tools.py
"""
LLM Tool-Calling Loop for the Agentic Builder.

Handles the OpenAI function-calling protocol:
  1. Send messages + tool definitions to the API
  2. If the model returns tool_calls, execute them
  3. Send tool results back as tool messages
  4. Repeat until the model returns a text response (no more tool calls)

Supports OpenAI (GPT-5.4) natively. Anthropic/Google would need
adapter logic (not implemented yet — GPT-5.4 is primary).

v1.0 (2026-03-07): Initial implementation for ASTRA v2.1.
v1.1 (2026-03-08): Added write integrity checking — pre-flight content
                    validation + post-write read-back verification.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the codebase. Returns the file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path (e.g. 'app/debug/models.py' or 'src/components/debug/DebugView.tsx')",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates or overwrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a PowerShell command. Use for syntax checks, booting app, npm commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "PowerShell command to execute",
                    },
                },
                "required": ["cmd"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Main tool-calling loop
# ---------------------------------------------------------------------------

async def run_tool_loop(
    system_prompt: str,
    initial_user_message: str,
    provider: str,
    model: str,
    max_iterations: int = 50,
    max_tokens: int = 128000,
    on_tool_call: Optional[Callable[[str, Dict], None]] = None,
    on_text: Optional[Callable[[str], None]] = None,
    existing_messages: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], int, int]:
    """Run an agentic tool-calling loop.

    Sends the initial prompt with tool definitions, then loops:
    model returns tool_calls -> we execute -> send results -> model continues.

    Args:
        system_prompt: System message.
        initial_user_message: First user message (ignored if existing_messages).
        provider: "openai" (primary).
        model: Model name (e.g. "gpt-5.4").
        max_iterations: Max tool-call rounds before stopping.
        max_tokens: Max output tokens per call.
        on_tool_call: Callback(tool_name, args) for progress tracking.
        on_text: Callback(text) for model's text responses.
        existing_messages: If provided, continues from this conversation history
            instead of starting fresh. The initial_user_message is appended as
            a new user message to this history. This keeps the builder in the
            same context window so it knows what it already built.

    Returns:
        (messages_history, total_input_tokens, total_output_tokens)
    """
    if existing_messages:
        messages = list(existing_messages)
        messages.append({"role": "user", "content": initial_user_message})
        logger.info(
            "[llm_tools] CONTINUATION MODE: %d existing messages + new instruction",
            len(existing_messages),
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]

    total_input_tokens = 0
    total_output_tokens = 0

    for iteration in range(max_iterations):
        logger.info("[llm_tools] Iteration %d — calling %s/%s", iteration + 1, provider, model)

        response, in_tok, out_tok = await _call_api(
            provider=provider,
            model=model,
            messages=messages,
            tools=TOOLS,
            max_tokens=max_tokens,
        )
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        if response is None:
            logger.error("[llm_tools] API call returned None at iteration %d", iteration + 1)
            break

        tool_calls = _extract_tool_calls(response, provider)

        if not tool_calls:
            text = _extract_text(response, provider)
            if on_text and text:
                on_text(text)
            messages.append(_make_assistant_message(response, provider))
            logger.info("[llm_tools] Text response at iteration %d (%d chars)", iteration + 1, len(text or ""))
            break

        messages.append(_make_assistant_message(response, provider))

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            if on_tool_call:
                on_tool_call(tool_name, tool_args)

            result = await _execute_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
            })

            logger.info("[llm_tools] Tool %s -> %d chars result", tool_name, len(result))

    else:
        logger.warning("[llm_tools] Hit max iterations (%d)", max_iterations)

    return messages, total_input_tokens, total_output_tokens


# ---------------------------------------------------------------------------
# API call (OpenAI)
# ---------------------------------------------------------------------------

async def _call_api(
    provider: str,
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
) -> Tuple[Optional[Any], int, int]:
    """Call the LLM API with tools. Returns (response, input_tokens, output_tokens)."""

    if provider == "openai":
        return await _call_openai(model, messages, tools, max_tokens)
    else:
        logger.warning("[llm_tools] Provider %s does not support tool calling — using text mode", provider)
        from app.pipeline_v2.llm_caller import call_llm
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        try:
            text = await call_llm(provider=provider, model=model,
                                  system_prompt=system, user_prompt=user,
                                  max_tokens=max_tokens)
            return {"text": text}, 0, 0
        except Exception as e:
            logger.error("[llm_tools] Fallback call failed: %s", e)
            return None, 0, 0


async def _call_openai(
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
) -> Tuple[Optional[Any], int, int]:
    """Call OpenAI API with function calling."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("[llm_tools] openai package not installed")
        return None, 0, 0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("[llm_tools] OPENAI_API_KEY not set")
        return None, 0, 0

    client = AsyncOpenAI(api_key=api_key, timeout=180.0)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_completion_tokens=min(max_tokens, 16384),
            temperature=0,
        )

        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0

        return response, in_tok, out_tok

    except Exception as e:
        logger.error("[llm_tools] OpenAI API error: %s", e)
        return None, 0, 0


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_tool_calls(response: Any, provider: str) -> List[Dict]:
    """Extract tool calls from API response."""
    if provider == "openai" and hasattr(response, "choices"):
        msg = response.choices[0].message
        if msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })
            return calls
    return []


def _extract_text(response: Any, provider: str) -> Optional[str]:
    """Extract text content from API response."""
    if provider == "openai" and hasattr(response, "choices"):
        return response.choices[0].message.content
    if isinstance(response, dict) and "text" in response:
        return response["text"]
    return None


def _make_assistant_message(response: Any, provider: str) -> Dict:
    """Convert API response to an assistant message for the conversation."""
    if provider == "openai" and hasattr(response, "choices"):
        msg = response.choices[0].message
        result = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return result
    text = _extract_text(response, provider) or ""
    return {"role": "assistant", "content": text}


# ---------------------------------------------------------------------------
# Tool execution (with write integrity checking)
# ---------------------------------------------------------------------------

# Module-level profile for tool execution routing (host vs sandbox).
_tool_profile: Optional["BuildTargetProfile"] = None

# v2.4 (2026-04-12): Phase 2 Job 8 — per-segment profile routing.
# When the orchestrator is iterating a multi-target manifest, it sets the
# current segment here. _resolve_active_profile() prefers the segment's
# target over the global profile, so each segment routes to its own repo.
_current_segment: Optional[Any] = None

# v2.4b (2026-04-12): Phase 2 Job 8b — manifest-aware auto-routing.
# When the LLM calls write_file/read_file but no segment is explicitly set,
# we can still route correctly IF we know which manifest is active: look up
# which segment owns the path, and use that segment's target. This lets the
# legacy single-loop agentic_builder flow benefit from Job 5's per-segment
# target tagging without needing the Job 12 DAG orchestrator rewrite first.
_active_manifest: Optional[Any] = None


def set_tool_manifest(manifest: Optional[Any]) -> None:
    """Set the segment manifest for path-based auto-routing (Phase 2 Job 8b).

    Called once per build, right after the manifest is finalised. Enables
    write_file/read_file to figure out which segment a path belongs to,
    even when set_tool_segment hasn't been called explicitly.
    """
    global _active_manifest
    _active_manifest = manifest


def _find_segment_for_path(path: str):
    """Find the manifest segment whose file_scope contains `path`.

    Uses os.path.normpath on both sides so "\\" vs "/" and mixed separators
    don't cause spurious misses. Case-insensitive on Windows.

    Returns None if no manifest is active or no segment owns the path.
    """
    if _active_manifest is None:
        return None
    # Resolve manifest.segments whether manifest is a dict or a dataclass
    if isinstance(_active_manifest, dict):
        segs = _active_manifest.get("segments", [])
    else:
        segs = getattr(_active_manifest, "segments", None) or []

    import os as _os
    target = _os.path.normpath(path).lower()
    for seg in segs:
        file_scope = seg.get("file_scope", []) if isinstance(seg, dict) else getattr(seg, "file_scope", [])
        for fpath in file_scope:
            candidate = _os.path.normpath(fpath).lower()
            if candidate == target:
                return seg
            # also allow the stored path to be a suffix of the resolved target
            # (covers relative paths stored in segments, absolute at call time)
            if target.endswith(_os.sep + candidate) or target.endswith("/" + candidate.replace("\\", "/")):
                return seg
    return None


def set_tool_profile(profile: Optional["BuildTargetProfile"]) -> None:
    """Set the primary profile for tool execution routing."""
    global _tool_profile
    _tool_profile = profile


def set_tool_segment(segment: Optional[Any]) -> None:
    """Set the currently-active segment for per-call routing (Phase 2 Job 8).

    Called by the orchestrator at the start of each segment's tool-loop run.
    The segment's target_id is translated to a profile via target_registry.
    Passing None clears the override so the global _tool_profile is used.
    """
    global _current_segment
    _current_segment = segment
    if segment is not None:
        logger.debug(
            "[llm_tools] Segment routing active: segment_id=%s target_id=%s",
            getattr(segment, 'segment_id', '?'),
            getattr(segment, 'target_id', None),
        )


def _resolve_active_profile(path: Optional[str] = None) -> Optional["BuildTargetProfile"]:
    """Pick the profile to use for the next tool call.

    Priority (v2.4b, Phase 2 Jobs 8 + 8b):
      1. If _current_segment has a resolvable target_id -> that target's profile.
      2. If _current_segment exists but target_id is None -> REFUSE (None).
      3. If path provided AND manifest set AND path matches a segment with
         a target_id -> that segment's profile (path-based auto-routing).
      4. Otherwise fall back to _tool_profile (single-target legacy).
    """
    # Priority 1 + 2: explicit segment override
    if _current_segment is not None:
        tid = getattr(_current_segment, 'target_id', None)
        if tid is not None:
            try:
                from app.pipeline_v2.target_registry import get_profile
                seg_profile = get_profile(tid)
                if seg_profile is not None:
                    return seg_profile
            except Exception as _err:
                logger.warning(
                    "[llm_tools] target_registry lookup failed for target_id=%s: %s",
                    tid, _err,
                )
        else:
            sid = getattr(_current_segment, 'segment_id', '?')
            logger.error(
                "[llm_tools] Segment %s has no target_id — refusing to route.",
                sid,
            )
            return None
    # Priority 3: path-based auto-routing via manifest
    if path and _active_manifest is not None:
        auto_seg = _find_segment_for_path(path)
        if auto_seg is not None:
            tid = auto_seg.get('target_id') if isinstance(auto_seg, dict) else getattr(auto_seg, 'target_id', None)
            if tid:
                try:
                    from app.pipeline_v2.target_registry import get_profile
                    seg_profile = get_profile(tid)
                    if seg_profile is not None:
                        sid = auto_seg.get('segment_id') if isinstance(auto_seg, dict) else getattr(auto_seg, 'segment_id', '?')
                        logger.debug(
                            "[llm_tools] Auto-routed path=%s -> segment=%s target=%s",
                            path, sid, tid,
                        )
                        return seg_profile
                except Exception as _err:
                    logger.warning("[llm_tools] auto-route lookup failed: %s", _err)
    # Priority 4: legacy fallback
    return _tool_profile


async def _execute_tool(name: str, args: Dict) -> str:
    """Execute a tool call and return the result as a string.

    Write operations include integrity checking:
      1. Pre-flight: scans content for patterns likely to be corrupted
      2. Write: sends content to sandbox
      3. Verify: reads file back and compares to original

    If corruption is detected, the error message tells the LLM exactly
    what went wrong so it can fix it (e.g. "backticks were stripped").
    """
    from app.pipeline_v2 import sandbox_tools

    try:
        if name == "read_file":
            path = args.get("path", "")
            _active = _resolve_active_profile(path=path)
            content = await sandbox_tools.read_file(path, profile=_active)
            if content is None:
                return "ERROR: File not found: " + path
            if len(content) > 30000:
                return (
                    "[" + str(len(content)) + " chars total — showing first 15K + last 5K]\n"
                    + content[:15000] + "\n...\n" + content[-5000:]
                )
            return content

        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")

            # --- Write integrity: pre-flight check ---
            from app.pipeline_v2.write_integrity import (
                check_content_integrity,
                verify_write,
            )

            ext = ""
            if "." in path:
                ext = "." + path.rsplit(".", 1)[-1]

            pre_issues = check_content_integrity(content, ext)
            pre_errors = [w for w in pre_issues if w["severity"] == "error"]

            if pre_errors:
                logger.warning(
                    "[llm_tools] Pre-flight found %d issues for %s: %s",
                    len(pre_errors), path,
                    "; ".join(w["message"] for w in pre_errors[:3]),
                )

            # --- Write ---
            # v2.4 Job 8: resolve per-segment profile; refuse on ambiguous target.
            _active = _resolve_active_profile(path=path)
            if _active is None and _current_segment is not None:
                return (
                    "ERROR: Cannot write " + path + " — current segment has no target_id. "
                    "Refusing ambiguous write. The segmenter left target_id=None "
                    "(likely a mixed-target segment). Fix the segment manifest "
                    "before retrying."
                )
            ok = await sandbox_tools.write_file(path, content, profile=_active)
            if not ok:
                return "ERROR: Write failed for " + path

            # --- Write integrity: post-write verify ---
            # v2.3: Skip integrity verification for host-mode writes.
            # Host writes use direct open() — no transport corruption possible.
            # The integrity checker causes false positives from \r\n vs \n diffs.
            _skip_verify = False
            _android_profile = _resolve_active_profile() or _tool_profile
            if _android_profile is not None:
                try:
                    from app.pipeline_v2.android_sandbox import is_android_build
                    _skip_verify = is_android_build(_android_profile)
                except Exception:
                    pass

            if _skip_verify:
                verify_ok = True
                verify_issues = []
            else:
                verify_ok, verify_issues = await verify_write(path, content)

            if verify_ok:
                # Clean write
                result_msg = "OK: Written " + str(len(content)) + " chars to " + path
                if pre_errors:
                    result_msg += (
                        "\nWARNING: Pre-flight detected potential issues (but write verified OK): "
                        + "; ".join(w["message"] for w in pre_errors[:3])
                    )
                return result_msg
            else:
                # Corruption detected!
                logger.error(
                    "[llm_tools] WRITE INTEGRITY FAILURE for %s: %s",
                    path,
                    "; ".join(i.get("message", i.get("type", "unknown")) for i in verify_issues[:5]),
                )
                error_parts = [
                    "WRITE INTEGRITY ERROR for " + path + ":",
                    "The file was written but read-back verification FAILED.",
                    "The content on disk does not match what you sent.",
                ]
                for issue in verify_issues[:5]:
                    issue_type = issue.get("type", "unknown")
                    if issue_type == "backtick_stripped":
                        error_parts.append(
                            "CORRUPTION: Backticks were stripped during write. "
                            "This is a transport-layer bug. The content was correct "
                            "but the write path corrupted it."
                        )
                        if issue.get("fix_hint"):
                            error_parts.append("FIX: " + issue["fix_hint"])
                    elif issue_type == "dollar_interpolated":
                        error_parts.append(
                            "CORRUPTION: Dollar signs ($) were consumed during write. "
                            "Variable interpolation in the transport layer ate them."
                        )
                    elif issue_type == "line_diff":
                        error_parts.append(
                            "Line " + str(issue.get("line", "?")) + " differs:"
                        )
                        error_parts.append("  Expected: " + str(issue.get("expected", ""))[:80])
                        error_parts.append("  Got:      " + str(issue.get("actual", ""))[:80])
                    else:
                        error_parts.append(
                            issue_type + ": " + issue.get("message", str(issue))
                        )

                error_parts.append(
                    "ACTION: Please re-write the file. The content you generated was correct — "
                    "the transport layer corrupted it. Try writing again."
                )
                return "\n".join(error_parts)

        elif name == "run_shell":
            cmd = args.get("cmd", "")
            # v2.3: Longer timeout for Gradle builds (first run downloads deps)
            _shell_profile = _resolve_active_profile() or _tool_profile
            _shell_timeout = 600 if (_shell_profile and _shell_profile.build_system == "gradle" and "gradlew" in cmd) else 30
            _active = _resolve_active_profile()
            result = await sandbox_tools.run_shell(cmd, timeout_sec=_shell_timeout, profile=_active)
            stdout = result.get("stdout", "")[:1000]
            stderr = result.get("stderr", "")[:500]
            rc = result.get("returncode", -1)
            parts = ["exit_code=" + str(rc)]
            if stdout:
                parts.append("STDOUT:\n" + stdout)
            if stderr:
                parts.append("STDERR:\n" + stderr)
            return "\n".join(parts)

        else:
            return "ERROR: Unknown tool: " + name

    except Exception as e:
        return "ERROR: " + name + " failed: " + str(e)

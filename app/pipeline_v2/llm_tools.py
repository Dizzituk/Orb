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
v1.4 (2026-06-09): Tools+reasoning now routes through the Responses API
                    (llm_responses_api.py) — gpt-5.x rejects reasoning_effort
                    with function tools on /v1/chat/completions. All API calls
                    wrapped in an asyncio.wait_for hard timeout: the client's
                    own timeout was observed not firing (a wedged call froze a
                    build for 17 minutes).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.pipeline_v2.llm_response_parsing import (
    _extract_tool_calls,
    _extract_text,
    _make_assistant_message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions — moved to llm_tool_defs.py (2026-06-10 file-size split;
# Jobs 8-9 needed room in this file). Imported here so existing
# `from app.pipeline_v2.llm_tools import TOOLS` callers keep working.
# JOB 9: the toolset now includes edit_file (surgical unique replace).
# ---------------------------------------------------------------------------

from app.pipeline_v2.llm_tool_defs import TOOLS


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
    reasoning: Optional[Dict] = None,
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
    if provider == "anthropic":
        # JOB 8 (2026-06-10): Claude models run through the Anthropic adapter
        # (tool_use / tool_result blocks, thinking, image tool results).
        # NOTE: existing_messages must be Anthropic-native format when used
        # with this provider — the formats are not interchangeable.
        from app.pipeline_v2.llm_tools_anthropic import run_anthropic_tool_loop
        return await run_anthropic_tool_loop(
            system_prompt=system_prompt,
            initial_user_message=initial_user_message,
            model=model,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            on_tool_call=on_tool_call,
            on_text=on_text,
            existing_messages=existing_messages,
            thinking=(True if reasoning else None),
        )

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
            reasoning=reasoning,
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

# Hard per-call timeout (Issue #3, 2026-06-09): the AsyncOpenAI client's own
# timeout=180 was observed NOT firing — a wedged request froze a build for
# 17 minutes with no error. asyncio.wait_for guarantees cancellation.
HARD_CALL_TIMEOUT = int(os.getenv("ASTRA_V2_BUILDER_CALL_TIMEOUT", "300"))


def _sanitize_for_chat_api(messages: List[Dict]) -> List[Dict]:
    """Strip private keys (e.g. _responses_raw_items) — chat completions
    rejects unknown message fields."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


async def _call_api(
    provider: str,
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
    reasoning: Optional[Dict] = None,
) -> Tuple[Optional[Any], int, int]:
    """Call the LLM API with tools. Returns (response, input_tokens, output_tokens)."""

    if provider == "openai":
        return await _call_openai(model, messages, tools, max_tokens, reasoning=reasoning)
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
    reasoning: Optional[Dict] = None,
) -> Tuple[Optional[Any], int, int]:
    """Call OpenAI with function calling.

    v1.4 (2026-06-09): reasoning + tools => Responses API (chat completions
    rejects that combination on gpt-5.x). On Responses failure, falls back to
    chat completions WITHOUT reasoning — same behaviour the 400-fallback gave
    before, but now logged loudly. Every call is hard-capped by
    asyncio.wait_for so a wedged request cannot freeze the build.
    """
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

    _use_reasoning = bool(
        reasoning
        and isinstance(reasoning, dict)
        and str(reasoning.get("effort", "")).lower() in ("low", "medium", "high", "xhigh", "max")
    )

    # --- Tools + reasoning: Responses API path (v1.4) ---
    if _use_reasoning and tools:
        try:
            from app.pipeline_v2.llm_responses_api import call_openai_responses
            adapted, in_tok, out_tok = await asyncio.wait_for(
                call_openai_responses(
                    client, model, messages, tools, max_tokens,
                    str(reasoning["effort"]).lower(),
                ),
                timeout=HARD_CALL_TIMEOUT,
            )
            return adapted, in_tok, out_tok
        except asyncio.TimeoutError:
            logger.error(
                "[llm_tools] Responses API call hard-timed out after %ss — cancelled",
                HARD_CALL_TIMEOUT,
            )
            return None, 0, 0
        except Exception as e:
            logger.warning(
                "[llm_tools] Responses API path failed (%s) — falling back to "
                "chat completions WITHOUT reasoning",
                str(e)[:300],
            )

    # --- Chat Completions path (no reasoning, or Responses fallback) ---
    create_kwargs: Dict[str, Any] = dict(
        model=model,
        messages=_sanitize_for_chat_api(messages),
        tools=tools,
        max_completion_tokens=min(max_tokens, 16384),
        temperature=0,
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(**create_kwargs),
            timeout=HARD_CALL_TIMEOUT,
        )
        usage = response.usage
        return response, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)
    except asyncio.TimeoutError:
        logger.error(
            "[llm_tools] Chat completions call hard-timed out after %ss — cancelled",
            HARD_CALL_TIMEOUT,
        )
        return None, 0, 0
    except Exception as e:
        # Some gpt-5.x variants only accept the default temperature — retry once without it.
        msg = str(e)
        if "temperature" in msg:
            create_kwargs.pop("temperature", None)
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**create_kwargs),
                    timeout=HARD_CALL_TIMEOUT,
                )
                usage = response.usage
                return response, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)
            except Exception as e2:
                logger.error("[llm_tools] OpenAI API error (after temperature retry): %s", e2)
                return None, 0, 0
        logger.error("[llm_tools] OpenAI API error: %s", e)
        return None, 0, 0


# ---------------------------------------------------------------------------
# Response parsing — moved to llm_response_parsing.py (2026-06-09 file-size
# split; llm_tools hit the 30KB ceiling). Imported at the top of this file.
# ---------------------------------------------------------------------------

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

        elif name == "edit_file":
            # JOB 9 (2026-06-10): surgical exact-string replacement with
            # unique-match enforcement + dated .bak. The verifier patches
            # precisely; it never rewrites whole files.
            path = args.get("path", "")
            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")
            _active = _resolve_active_profile(path=path)
            if _active is None and _current_segment is not None:
                return (
                    "ERROR: Cannot edit " + path + " — current segment has no target_id. "
                    "Refusing ambiguous edit."
                )
            ok, msg = await sandbox_tools.edit_file(path, old_str, new_str, profile=_active)
            return ("OK: " + msg) if ok else ("EDIT REFUSED: " + msg)

        elif name == "run_shell":
            cmd = args.get("cmd", "")
            # v2.4 (2026-04-18): Bumped Gradle timeout from 600s to 1500s.
            # Cold-start Gradle compiles (dependency downloads, first-time
            # Kotlin compilation of new modules) can exceed 10 minutes on
            # the Driver CoPilot build. The previous 600s cap caused the
            # agent's shell call to time out on the first Gradle invocation
            # even when Gradle was still making progress, forcing the agent
            # to bail out just before BVL's Tier 1 would have succeeded
            # (BVL retry built in ~12s because caches were already warm).
            # 1500s gives comfortable headroom; matches 1200s used by
            # sandbox_tools.build_check with extra margin for agent-driven
            # explorations (e.g. --refresh-dependencies).
            _shell_profile = _resolve_active_profile() or _tool_profile
            _shell_timeout = 1500 if (_shell_profile and _shell_profile.build_system == "gradle" and "gradlew" in cmd) else 30
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

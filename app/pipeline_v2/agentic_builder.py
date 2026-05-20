# FILE: app/pipeline_v2/agentic_builder.py
"""
ASTRA v2.2 Agentic Builder — one model, one loop, job done.

A single model (GPT-5.4) with full tool access works through
the build in a continuous context window:

  1. Read the spec and scaffold
  2. For each file: read existing content, write the logic, verify syntax
  3. Wire cross-file dependencies (imports, route registrations, props)
  4. Build/boot the application
  5. Read output for errors — fix if needed
  6. Hand off to Verification Model for visual check
  7. If FAIL: fix issues and loop back to step 4
  8. If PASS: done

The Builder has tools: read_file, write_file, run_shell, check_syntax.
It reasons about what to do, calls tools, reads results, continues.

v1.0 (2026-03-07): Initial implementation for ASTRA v2.1.
v2.0 (2026-03-10): Multi-project targeting. Dynamic system prompts per
    language. Profile-aware path resolution in tool execution.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from app.pipeline_v2.config import (
    BUILDER_PROVIDER, BUILDER_MODEL, BUILDER_MAX_OUTPUT,
    FALLBACK_BUILDER_PROVIDER, FALLBACK_BUILDER_MODEL,
    MAX_TOOL_CALLS, HANDOVER_THRESHOLD_PCT,
)
from app.pipeline_v2.models import (
    BuildResult, BuildSession, ScaffoldResult, ToolCall,
)

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

from app.core_principles import get_principles_block as _get_principles

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Legacy system prompt (kept for backward compat if no profile)
# ═══════════════════════════════════════════════════════════════════

BUILDER_SYSTEM = """You are ASTRA's Agentic Builder. You build features by reading, writing, and testing code.

You have tools to interact with the codebase in a Windows Sandbox:
- read_file(path): Read a file's contents
- write_file(path, content): Write or overwrite a file
- run_shell(cmd): Run a PowerShell command (syntax checks, boot app, etc.)

WORKFLOW:
1. Read the spec to understand what needs to be built
2. Read the scaffold files to see what structure exists
3. For each file, read it, write the logic, verify it compiles
4. Wire cross-file dependencies: imports, route registrations, component props
5. When you think you're done, boot the app and check for errors
6. Fix any errors and re-boot until clean

RULES:
- Read a file BEFORE modifying it. Never guess what's in a file.
- For MODIFY files (existing files), make TARGETED changes. Don't rewrite.
- For CREATE files (scaffold stubs), write the complete implementation.
- Check Python syntax after writing: run_shell("python -m py_compile <file>")
- After wiring everything, boot with: run_shell("cd D:\\Orb && .venv\\Scripts\\python.exe -c \\"from main import app; print('BOOT_OK')\\"")
- When you've fixed all errors and the app boots clean, say BUILDER_COMPLETE.

CRITICAL:
- Every import must reference a real file that exists
- Every function/class you import must actually be exported by that file
- Read the file you're importing from to verify the export exists
- Never add a route registration without reading main.py first
- Test compilation after every file write
"""


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

async def run_agentic_builder(
    spec: Dict[str, Any],
    manifest: Dict[str, Any],
    scaffold: ScaffoldResult,
    job_dir: str,
    handover_context: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    existing_messages: Optional[List[Dict]] = None,
    profile: Optional["BuildTargetProfile"] = None,
) -> BuildResult:
    """Run the Agentic Builder.

    One model, one loop. Reads the spec and scaffold, writes code,
    tests, fixes, until the build passes.

    Args:
        spec: Verified spec from SpecGate.
        manifest: Segment manifest.
        scaffold: ScaffoldResult from the Scaffold Engine.
        job_dir: Job directory.
        handover_context: If continuing from a previous session.
        on_progress: Progress callback.
        existing_messages: Conversation history for continuation.
        profile: Build target profile (determines prompts, paths).

    Returns:
        BuildResult with all files written and session details.
    """
    t_start = time.time()
    emit = on_progress or (lambda msg: None)
    result = BuildResult()

    # Build prompts from profile
    if profile:
        from app.pipeline_v2.builder_prompts import build_system_prompt, build_initial_prompt
        system_prompt = build_system_prompt(profile)
        initial_prompt = build_initial_prompt(spec, manifest, scaffold, profile, handover_context)
    else:
        system_prompt = BUILDER_SYSTEM + '\n\n' + _get_principles()
        initial_prompt = _build_initial_prompt_legacy(spec, manifest, scaffold, handover_context)

    # v2.2 Piece 1 + Piece 5: inject prior decisions + upstream stage
    # summaries via the context assembler. The builder is a consumer of
    # decisions, not a producer — reads to honour prior choices.
    try:
        from app.pipeline_v2.config import LEDGER_ENABLED, LEDGER_PROMPT_MAX_CHARS
        if LEDGER_ENABLED:
            from app.pipeline_v2.context_assembler import assemble_context
            _ctx = assemble_context(
                job_dir=job_dir,
                stage="agentic_builder",
                max_chars=LEDGER_PROMPT_MAX_CHARS,
            )
            if _ctx:
                initial_prompt = (
                    f"{_ctx}\n\n"
                    "The decisions above are binding. Honour them. If you "
                    "believe one is wrong, say so explicitly in your reply "
                    "rather than silently contradicting it.\n\n"
                    f"{initial_prompt}"
                )
    except Exception as _lerr:
        logger.debug("[agentic_builder] Context assembly skipped: %s", _lerr)

    lang = profile.language if profile else "python"
    emit(f"🤖 Agentic Builder: Starting {lang} build loop...")
    emit(f"   Model: {BUILDER_PROVIDER}/{BUILDER_MODEL}")
    emit(f"   Scaffold: {scaffold.total_files} files")
    emit(f"   Max tool calls: {MAX_TOOL_CALLS}")
    emit(f"   Initial prompt: {len(initial_prompt):,} chars")

    # v2.3 (2026-04-18): Resolve reasoning config from CRITICAL_PIPELINE stage.
    # The Agentic Builder is load-bearing — it writes production code. Per
    # Taz's directive, reasoning=high is on by default for this stage.
    # Falls back to None (no reasoning) if stage_models lookup fails, so the
    # builder still runs even if the alias system isn't available.
    _builder_reasoning = None
    try:
        from app.llm.stage_models import get_stage_config
        _builder_reasoning = get_stage_config("CRITICAL_PIPELINE").reasoning
        if _builder_reasoning:
            emit(f"   Reasoning: effort={_builder_reasoning.get('effort', '?')}")
    except Exception as _rerr:
        logger.debug("[agentic_builder] reasoning lookup skipped: %s", _rerr)

    # Run the agentic loop using proper tool calling
    session = BuildSession(session_number=1)

    try:
        from app.pipeline_v2.llm_tools import run_tool_loop, set_tool_profile, set_tool_manifest, set_tool_segment
        set_tool_profile(profile)  # v2.3: Route tools to host for Android builds
        # v2.4 (2026-04-12): Phase 2 Job 9 — activate Job 8b path-based
        # auto-routing by registering the manifest. Every write_file call
        # will now be routed by inspecting the path against segment file_scopes,
        # so multi-target jobs land each file in the correct repo.
        try:
            set_tool_manifest(manifest)
        except Exception as _mf_err:
            logger.debug("[agentic_builder] set_tool_manifest skipped: %s", _mf_err)

        # v2.5 (2026-04-12): Phase 4 Job 12 — reactive per-segment tracking.
        # The tracker observes write_file calls and updates set_tool_segment
        # based on which segment owns the path just written. Gives explicit
        # per-segment awareness to the tool layer without rewriting the
        # builder's run_tool_loop flow. Fires segment_start/complete events.
        _tracker = None
        try:
            from app.pipeline_v2.segment_tracker import SegmentTracker
            from app.pot_spec.grounded.segment_schemas import SegmentManifest as _SM
            _manifest_obj = manifest
            if isinstance(manifest, dict):
                try:
                    _manifest_obj = _SM.from_dict(manifest)
                except Exception:
                    _manifest_obj = None
            if _manifest_obj is not None:
                def _on_seg_start(seg):
                    emit(f"   📍 Segment active: {seg.segment_id} (target: {seg.target_id})")
                def _on_seg_complete(seg):
                    emit(f"   ✓ Segment complete: {seg.segment_id}")
                _tracker = SegmentTracker(
                    _manifest_obj,
                    on_segment_start=_on_seg_start,
                    on_segment_complete=_on_seg_complete,
                )
        except Exception as _tracker_err:
            logger.debug("[agentic_builder] SegmentTracker skipped: %s", _tracker_err)

        files_written = set()

        def on_tool(name, args):
            path = args.get("path", "")
            summary = path if name == "read_file" else (
                f"{path} ({len(args.get('content', ''))} chars)" if name == "write_file"
                else args.get("cmd", "")[:60]
            )
            emit(f"   🔧 {name}({summary})")
            session.tool_calls.append(ToolCall(tool=name, args=summary))
            # v2.5 Job 12: feed every tool call to the segment tracker so it
            # can update set_tool_segment and fire progress callbacks.
            if _tracker is not None:
                try:
                    _tracker.on_tool_call(name, args)
                except Exception as _te:
                    logger.debug("[agentic_builder] tracker.on_tool_call raised: %s", _te)
            if name == "write_file" and path:
                files_written.add(path)
                try:
                    from app.pipeline_v2.orchestrator import _push_narrative
                    _push_narrative(
                        stage="critical_pipeline",
                        title=f"Write: {path.rsplit('/', 1)[-1] if '/' in path else path}",
                        output_summary=f"{len(args.get('content', ''))} chars written",
                        files_touched=[path],
                    )
                except Exception:
                    pass

        def on_text(text):
            if "BUILDER_COMPLETE" in text:
                emit("   ✅ Builder signalled COMPLETE")
                session.completed = True
            else:
                emit(f"   💬 {text[:150]}")

        messages, in_tok, out_tok = await run_tool_loop(
            system_prompt=system_prompt,
            initial_user_message=initial_prompt,
            provider=BUILDER_PROVIDER,
            model=BUILDER_MODEL,
            max_iterations=MAX_TOOL_CALLS,
            max_tokens=BUILDER_MAX_OUTPUT,
            on_tool_call=on_tool,
            on_text=on_text,
            existing_messages=existing_messages,
            reasoning=_builder_reasoning,
        )

        # v2.1: Guard against premature BUILDER_COMPLETE.
        # If the builder declared done but hasn't written at least 50% of
        # the scaffold files, push it back with a continuation prompt.
        _expected_files = scaffold.total_files if scaffold else 0
        _written_count = len(files_written)
        _write_ratio = _written_count / max(_expected_files, 1)
        if session.completed and _write_ratio < 0.5 and _expected_files > 3:
            session.completed = False
            emit(f"   \u26a0\ufe0f Builder declared COMPLETE but only wrote {_written_count}/{_expected_files} files")
            emit(f"   \U0001f504 Pushing back — requesting continuation...")
            _pushback = (
                f"You declared BUILDER_COMPLETE but you have only written {_written_count} "
                f"out of {_expected_files} scaffold files. You need to implement the remaining "
                f"files. Read the scaffold files you haven't written yet and fill them in. "
                f"Do NOT say BUILDER_COMPLETE until you have written ALL files."
            )
            messages_2, in_tok_2, out_tok_2 = await run_tool_loop(
                system_prompt=system_prompt,
                initial_user_message=_pushback,
                provider=BUILDER_PROVIDER,
                model=BUILDER_MODEL,
                max_iterations=MAX_TOOL_CALLS,
                max_tokens=BUILDER_MAX_OUTPUT,
                on_tool_call=on_tool,
                on_text=on_text,
                existing_messages=messages,
                reasoning=_builder_reasoning,
            )
            messages = messages_2
            in_tok += in_tok_2
            out_tok += out_tok_2
            # Check again
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and "BUILDER_COMPLETE" in (msg.get("content") or ""):
                    session.completed = True
                    break
            emit(f"   After pushback: {len(files_written)} files written")

        session.files_created = list(files_written)
        session.total_input_tokens = in_tok
        session.total_output_tokens = out_tok
        if not session.completed:
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and "BUILDER_COMPLETE" in (msg.get("content") or ""):
                    session.completed = True
                    break

    except Exception as e:
        logger.exception("[agentic_builder] Build loop failed")
        result.errors.append(f"Build loop error: {e}")
        messages = []

    # Collect results
    result.sessions.append(session)
    result.all_files_written = session.files_created + session.files_modified
    result.total_tool_calls = session.total_tool_calls
    result.total_llm_calls = len([m for m in messages if m.get("role") == "assistant"])
    result.total_input_tokens = session.total_input_tokens
    result.total_output_tokens = session.total_output_tokens
    result.total_duration_seconds = time.time() - t_start
    result.success = session.completed and len(result.all_files_written) > 0
    result.messages_history = messages

    # v2.5 Job 12: log final per-segment progress + clear tracker state.
    if _tracker is not None:
        try:
            emit(f"   📊 {_tracker.summary()}")
            for _p in _tracker.progress_report():
                _mark = "✓" if _p["complete"] else "✗"
                emit(f"      {_mark} {_p['segment_id']:<20} "
                     f"{_p['target_id'] or '?':<15} "
                     f"{_p['files_written']}/{_p['files_in_scope']} files")
        except Exception:
            pass
    # Clear segment state so the next build doesn't inherit stale routing.
    try:
        set_tool_segment(None)
    except Exception:
        pass

    emit(f"\n🤖 Builder complete: {len(result.all_files_written)} files, "
         f"{result.total_tool_calls} tool calls, "
         f"{result.total_duration_seconds:.1f}s")

    return result


# ═══════════════════════════════════════════════════════════════════
# Legacy initial prompt (no profile)
# ═══════════════════════════════════════════════════════════════════

def _build_initial_prompt_legacy(
    spec: Dict[str, Any],
    manifest: Dict[str, Any],
    scaffold: ScaffoldResult,
    handover_context: Optional[str] = None,
) -> str:
    """Build the initial prompt (legacy — no profile)."""
    parts = []

    if handover_context:
        parts.append("=== HANDOVER FROM PREVIOUS SESSION ===")
        parts.append(handover_context)
        parts.append("")

    parts.append("=== SPECIFICATION ===")
    spec_text = json.dumps(spec, indent=2) if isinstance(spec, dict) else str(spec)
    parts.append(spec_text[:15000])
    parts.append("")

    parts.append("=== FILE SCOPE ===")
    segments = manifest.get("segments", [])
    for seg in segments:
        seg_id = seg.get("segment_id", "")
        files = seg.get("file_scope", [])
        reqs = seg.get("requirements", [])
        parts.append(f"\nSegment: {seg_id}")
        for f in files:
            parts.append(f"  - {f}")
        if reqs:
            parts.append(f"  Requirements:")
            for r in reqs:
                parts.append(f"    - {r}")
    parts.append("")

    parts.append("=== SCAFFOLD FILES ===")
    parts.append("The Scaffold Engine has written skeleton files.")
    for sf in scaffold.files:
        tag = "[CREATE]" if sf.is_new else "[MODIFY]"
        parts.append(f"  {tag} {sf.path}")
    parts.append("")

    parts.append("=== YOUR JOB ===")
    parts.append("1. Read each scaffold file and the files it depends on")
    parts.append("2. Write the business logic to replace TODO markers")
    parts.append("3. Wire imports, route registrations, and component props")
    parts.append("4. Test syntax after each write")
    parts.append("5. Boot the app when all files are done")
    parts.append("6. Say BUILDER_COMPLETE when the app boots clean")

    return "\n".join(parts)

# FILE: app/llm/weaver_stream.py
"""
Weaver Stream Handler for ASTRA v4.2.0

Converts human rambling into structured job outlines.
See _weaver_prompts.py for system prompts.
See _weaver_stream_utils_12 through _17 for extracted helpers/constants.

LOCKED BEHAVIOUR:
- Text organizer only — NOT a spec builder
- ALWAYS outputs structured outline (never conversational)
- Gap handling: "Questions for user" (subjective only) vs "SpecGate must resolve" (code-answerable)
- NEVER asks technical questions (frameworks, algorithms, architecture)
"""
from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator, Dict, List, Optional, Any

from sqlalchemy.orm import Session
from app.llm._weaver_stream_utils_12 import _enforce_deduplication, _get_blocking_questions, _get_weaver_config, _serialize_sse
from app.llm._weaver_stream_utils_14 import _get_streaming_function, _sanitize_weaver_output
from app.llm._weaver_stream_utils_15 import _is_micro_file_task, _user_dismissed_questions
from app.llm._weaver_stream_utils_16 import _enforce_design_pref_hygiene, _is_refactor_task
from app.llm._weaver_stream_utils_17 import _has_core_goal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Imports with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

# Memory service for reading conversation
try:
    from app.memory import service as memory_service
    from app.memory import schemas as memory_schemas
    _MEMORY_AVAILABLE = True
except ImportError:
    memory_service = None
    memory_schemas = None
    _MEMORY_AVAILABLE = False

# Flow state for storing output and design question state
try:
    from app.llm.spec_flow_state import (
        start_weaver_flow,
        clear_weaver_design_questions,
        save_confirmed_design_prefs,
        save_weave_checkpoint,
        save_woven_user_hashes,
    )
    _FLOW_STATE_AVAILABLE = True
except ImportError:
    start_weaver_flow = None
    clear_weaver_design_questions = None
    save_confirmed_design_prefs = None
    save_weave_checkpoint = None
    save_woven_user_hashes = None
    _FLOW_STATE_AVAILABLE = False

# Import streaming functions for all providers
try:
    from app.llm.streaming import stream_openai, stream_anthropic, stream_gemini
    _STREAMING_AVAILABLE = True
except ImportError:
    try:
        from .streaming import stream_openai, stream_anthropic, stream_gemini
        _STREAMING_AVAILABLE = True
    except ImportError:
        stream_openai = None
        stream_anthropic = None
        stream_gemini = None
        _STREAMING_AVAILABLE = False

async def generate_weaver_stream(
    *,
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: str,
    is_continuation: bool = False,
    captured_answers: Optional[Dict[str, str]] = None,
    pending_user_message: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Main Weaver stream generator v4.2.0 — converts ramble into structured job outline."""
    print(f"[WEAVER] Starting weaver v4.2.0 for project_id={project_id}")
    logger.info("[WEAVER] Starting weaver v4.2.0 for project_id=%s", project_id)
    
    provider, model = _get_weaver_config()
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    if not _STREAMING_AVAILABLE:
        error_msg = "Streaming providers not available - check imports"
        yield _serialize_sse({"type": "token", "content": f"X {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    if not _MEMORY_AVAILABLE:
        error_msg = "Memory service not available - cannot read conversation"
        yield _serialize_sse({"type": "token", "content": f"X {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    stream_fn = _get_streaming_function(provider)
    if stream_fn is None:
        error_msg = f"Streaming function not available for provider: {provider}"
        yield _serialize_sse({"type": "token", "content": f"X {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    try:
        # =====================================================================
        # STEPS 1-4: Gather, filter, dedup, load state (delegated)
        # =====================================================================
        from app.llm._weaver_stream_prepare import prepare_weaver_messages

        prep = prepare_weaver_messages(db, project_id, pending_user_message, captured_answers)

        if prep.early_exit_message:
            yield _serialize_sse({"type": "token", "content": prep.early_exit_message})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return

        total_message_count = prep.total_message_count
        relevant_messages = prep.relevant_messages
        new_user_messages = prep.new_user_messages
        ramble_text = prep.ramble_text
        vision_context = prep.vision_context
        confirmed_prefs = prep.confirmed_prefs
        current_user_hashes = prep.current_user_hashes
        checkpoint = prep.checkpoint
        is_update_mode = prep.is_update_mode
        execution_mode = prep.execution_mode

        # v5.5 PHASE 4C: Progressive Memory — compact long conversations
        if len(relevant_messages) >= 15:
            try:
                from app.llm.weaver_memory import compact_conversation
                _compaction_result = await compact_conversation(relevant_messages)
                if _compaction_result.was_compacted:
                    ramble_text = _compaction_result.format_for_weaver()
            except (ImportError, Exception):
                pass
        
        # =====================================================================
        # STEP 5: Core goal check (FOR LOGGING ONLY - v3.5.0)
        # In v3.5.0, Weaver ALWAYS proceeds to weave, even if core goal unclear
        # =====================================================================
        
        has_clear_goal = _has_core_goal(ramble_text)
        if not has_clear_goal:
            print("[WEAVER] Core goal unclear - will list as ambiguity in output")
        
        # =====================================================================
        # STEP 5b: Micro-task classification (v3.6.0)
        # Detect simple file operations that should skip unnecessary questions
        # =====================================================================
        
        is_micro_task = _is_micro_file_task(ramble_text)
        is_refactor_task = _is_refactor_task(ramble_text)
        questions_dismissed = _user_dismissed_questions(ramble_text)
        
        if is_micro_task:
            print("[WEAVER] MICRO_FILE_TASK mode - minimal output, no unnecessary questions")
        if is_refactor_task:
            print("[WEAVER] v3.8 REFACTOR_TASK mode - no design questions")
        if questions_dismissed:
            print("[WEAVER] v3.8 User dismissed questions - skipping shallow questions")
        
        # =====================================================================
        # STEP 6: Get blocking questions for micro tasks (v4.0.0 simplified)
        # v4.0.0: Removed _is_design_job() and hardcoded shallow questions.
        # The LLM now generates its own contextual questions in the system prompt.
        # Only micro tasks still get deterministic blocker questions.
        # =====================================================================
        
        blocking_questions = []
        
        if is_micro_task:
            # Micro tasks: blocker-only questions (delete confirmation, move destination)
            blocking_questions = _get_blocking_questions(ramble_text, is_micro_task=True)
            if blocking_questions:
                print(f"[WEAVER] Micro-task has {len(blocking_questions)} blocking question(s)")
        elif is_refactor_task or questions_dismissed:
            # v3.8.0: Refactor tasks and dismissed questions skip question generation
            print("[WEAVER] v3.8 Skipping questions (refactor task or user dismissed)")
        else:
            # v4.0.0: Normal feature requests - LLM generates its own questions
            # No hardcoded question injection. The system prompt instructs the LLM
            # to identify genuine gaps and ask contextually relevant questions.
            print("[WEAVER] v4.0 LLM will generate contextual questions (no hardcoded injection)")
        
        # Clear any lingering question state
        if _FLOW_STATE_AVAILABLE and clear_weaver_design_questions:
            clear_weaver_design_questions(project_id)
        
        # =====================================================================
        # STEP 7: Weave - CREATE or UPDATE mode
        # v3.5.0: ALWAYS produces structured output, never conversational
        # =====================================================================
        from app.llm._weaver_stream_modes import build_weave_prompt

        system_prompt, user_prompt, start_message = build_weave_prompt(
            is_micro_task=is_micro_task,
            is_refactor_task=is_refactor_task,
            is_update_mode=is_update_mode,
            ramble_text=ramble_text,
            blocking_questions=blocking_questions,
            new_user_messages=new_user_messages,
            checkpoint=checkpoint,
            confirmed_prefs=confirmed_prefs,
            execution_mode=execution_mode,
            total_message_count=total_message_count,
        )
        yield _serialize_sse({"type": "token", "content": start_message})
        
        # v2.2: Inject project session context into Weaver
        try:
            from app.shared_context.project_session import get_project_session
            _proj_session = get_project_session(conversation_id)
            if _proj_session.is_set:
                _proj_ctx = (
                    f"\n\n## ACTIVE PROJECT CONTEXT\n"
                    f"Project: {_proj_session.project_name} ({_proj_session.project_id})\n"
                    f"Type: {_proj_session.project_type}\n"
                    f"Root: {_proj_session.project_root}\n"
                    f"All files in the manifest MUST target this project root.\n"
                )
                # Include the scoping conversation as the project brief
                # This is the substantive content the user described during scoping
                if _proj_session.scoping_messages:
                    _proj_ctx += "\n## PROJECT BRIEF (from scoping conversation)\n"
                    for msg in _proj_session.scoping_messages:
                        if msg["role"] == "user" and len(msg["content"]) > 20:
                            _proj_ctx += f"- {msg['content']}\n"
                    _proj_ctx += "\nUse the above brief as the primary source for what the user wants built.\n"
                system_prompt += _proj_ctx
                logger.info("[WEAVER] Project session injected: %s (with %d scoping messages)",
                    _proj_session.project_id, len(_proj_session.scoping_messages))
            elif _proj_session.scoping_active and _proj_session.scoping_messages:
                _scoping_ctx = (
                    f"\n\n## PROJECT SCOPING CONTEXT\n"
                    f"The user is scoping a new project. Here is the conversation so far:\n"
                )
                for msg in _proj_session.scoping_messages:
                    _scoping_ctx += f"  {msg['role']}: {msg['content']}\n"
                system_prompt += _scoping_ctx
                logger.info("[WEAVER] Scoping context injected: %d messages", len(_proj_session.scoping_messages))
        except Exception as e:
            logger.debug("[WEAVER] Project session injection failed: %s", e)

        # v3.0: Inject user memory into Weaver system prompt
        try:
            from app.experience.user_memory import get_user_context_for_conversation
            _user_conv_ctx = get_user_context_for_conversation(
                db, query=user_prompt[:300], project_id=project_id,
            )
            if _user_conv_ctx:
                system_prompt += f"\n\n{_user_conv_ctx}"
        except Exception:
            pass

        # Stream from LLM
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_chunks: List[str] = []
        
        async for chunk in stream_fn(messages=llm_messages, model=model):
            content = None
            if isinstance(chunk, dict):
                # v3.2: Surface LLM errors instead of silently swallowing them
                if chunk.get("type") == "error":
                    err_msg = chunk.get("message", "Unknown LLM error")
                    print(f"[WEAVER] LLM ERROR from {provider}/{model}: {err_msg}")
                    yield _serialize_sse({"type": "token", "content": f"\n\n⚠️ Weaver LLM error: {err_msg}"})
                    continue
                content = chunk.get("text") or chunk.get("content")
                if chunk.get("type") == "metadata":
                    continue
            elif hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
            if content:
                response_chunks.append(content)
                yield _serialize_sse({"type": "token", "content": content})
        
        # =====================================================================
        # STEP 8: Post-process and save
        # =====================================================================
        
        raw_output = "".join(response_chunks).strip()
        
        # Sanitize output to remove any prompt leakage
        sanitized_output = _sanitize_weaver_output(raw_output)
        
        # Enforce design preference hygiene (v3.4.2)
        hygiene_output = _enforce_design_pref_hygiene(sanitized_output)
        
        # Enforce deduplication (v3.5.0 - Bug 3 fix)
        dedup_output = _enforce_deduplication(hygiene_output)
        
        # v4.0.0: Slot detection/reconciliation REMOVED.
        # The LLM generates its own contextual questions and reads the user's
        # requirements directly — no need for hardcoded slot post-processing.
        job_description = dedup_output
        
        weaver_output_id = f"weaver-{uuid.uuid4().hex[:12]}"
        
        # Save woven user hashes (accumulate - don't replace)
        if _FLOW_STATE_AVAILABLE and save_woven_user_hashes:
            save_woven_user_hashes(project_id, current_user_hashes)
        
        # Save weave checkpoint
        if _FLOW_STATE_AVAILABLE and save_weave_checkpoint:
            save_weave_checkpoint(project_id, total_message_count, job_description)
        
        # Save confirmed prefs (they persist)
        if _FLOW_STATE_AVAILABLE and save_confirmed_design_prefs and confirmed_prefs:
            save_confirmed_design_prefs(project_id, confirmed_prefs)
        
        # Store in flow state for Spec Gate
        # v3.9.1: Now passing vision_context for intelligent UI classification
        if _FLOW_STATE_AVAILABLE and start_weaver_flow:
            try:
                start_weaver_flow(
                    project_id=project_id,
                    weaver_spec_id=weaver_output_id,
                    weaver_job_description=job_description,
                    vision_context=vision_context,  # v3.9.1: Pass vision context to flow state
                )
                if vision_context:
                    print(f"[WEAVER] v3.9.1 Vision context stored in flow state ({len(vision_context)} chars)")
            except Exception as e:
                logger.warning("[WEAVER] Failed to store in flow state: %s", e)
        
        # =====================================================================
        # Persist to message history for cross-model context continuity
        # =====================================================================
        if _MEMORY_AVAILABLE and memory_service and memory_schemas:
            try:
                memory_service.create_message(
                    db,
                    memory_schemas.MessageCreate(
                        project_id=project_id,
                        role="assistant",
                        content=job_description,
                        provider=provider,
                        model=model,
                    ),
                )
            except Exception as e:
                logger.warning("[WEAVER] Failed to save message to history: %s", e)

        # =====================================================================
        # Completion message
        # =====================================================================
        
        mode_indicator = "updated" if is_update_mode else "ready"
        completion_message = f"""

---

**Job description {mode_indicator}** (`{weaver_output_id}`)

This is a structured outline of what you described. Review it above.

**Next step:** Say **'Send to Spec Gate'** to validate and build a full specification."""

        yield _serialize_sse({"type": "token", "content": completion_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        
    except Exception as e:
        logger.exception("[WEAVER] Error during streaming")
        error_message = f"\n\nWeaver error: {str(e)}"
        yield _serialize_sse({"type": "token", "content": error_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})


# ---------------------------------------------------------------------------
# LEGACY COMPATIBILITY
# ---------------------------------------------------------------------------

__all__ = ["generate_weaver_stream"]

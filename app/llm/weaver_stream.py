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

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Any, Set, Tuple

from sqlalchemy.orm import Session
from app.llm._weaver_stream_utils_12 import _SLOT_RECONCILIATION_REMOVED, _enforce_deduplication, _format_execution_mode, _format_ramble, _get_blocking_questions, _get_weaver_config, _is_control_message, _serialize_sse
from app.llm._weaver_stream_utils_13 import BUILD_VERBS, INTENT_GOAL_PATTERNS, LEAKAGE_PATTERNS, MICRO_FILE_INDICATORS, NEGATION_PATTERNS, NON_MICRO_INDICATORS, REFACTOR_INDICATORS, _hash_messages
from app.llm._weaver_stream_utils_14 import CORE_GOAL_VERBS, FEATURE_COMPONENT_INDICATORS, META_MODE_PATTERNS, QUESTIONS_DISMISSED_PATTERNS, _get_streaming_function, _hash_message, _normalize_typos, _sanitize_weaver_output
from app.llm._weaver_stream_utils_15 import DESIGN_PREF_WHITELIST_PATTERNS, REFACTOR_ACTION_PATTERNS, TYPO_NORMALIZATIONS, _extract_meta_mode, _extract_vision_context, _gather_ramble_messages, _is_micro_file_task, _user_dismissed_questions
from app.llm._weaver_stream_utils_16 import CONCRETE_TARGETS, DESIGN_PREF_BLACKLIST_PATTERNS, MICRO_TASK_SYSTEM_PROMPT, REFACTOR_TASK_SYSTEM_PROMPT, VISION_CONTEXT_PATTERNS, _enforce_design_pref_hygiene, _is_refactor_task, _is_vision_context
from app.llm._weaver_stream_utils_17 import CORE_GOAL_TARGETS, _has_core_goal
from app.llm._weaver_prompts import WEAVER_UPDATE_SYSTEM_PROMPT, WEAVER_CREATE_SYSTEM_PROMPT

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
        SpecFlowStage,
        set_weaver_design_questions,
        get_weaver_design_state,
        clear_weaver_design_questions,
        get_active_flow,
        # v1.2: Persistent prefs and checkpoints
        save_confirmed_design_prefs,
        get_confirmed_design_prefs,
        save_weave_checkpoint,
        get_weave_checkpoint,
        # v1.3: Hash-based delta tracking
        save_woven_user_hashes,
        get_woven_user_hashes,
    )
    _FLOW_STATE_AVAILABLE = True
except ImportError:
    start_weaver_flow = None
    SpecFlowStage = None
    set_weaver_design_questions = None
    get_weaver_design_state = None
    clear_weaver_design_questions = None
    get_active_flow = None
    save_confirmed_design_prefs = None
    get_confirmed_design_prefs = None
    save_weave_checkpoint = None
    get_weave_checkpoint = None
    save_woven_user_hashes = None
    get_woven_user_hashes = None
    _FLOW_STATE_AVAILABLE = False

# Simple weaver function
try:
    from app.llm.weaver_simple import weave, WEAVER_SYSTEM_PROMPT, _format_messages_as_ramble
    _SIMPLE_WEAVER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[weaver_stream] weaver_simple not available: {e}")
    _SIMPLE_WEAVER_AVAILABLE = False
    weave = None

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


# ---------------------------------------------------------------------------
# All constants, helpers, hashing, classification, and prompts have been
# extracted to _weaver_stream_utils_12..17 and _weaver_prompts.py.
# ---------------------------------------------------------------------------

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
        # STEP 1: Gather ALL messages
        # =====================================================================
        
        all_messages = _gather_ramble_messages(db, project_id)
        
        # =============================================================
        # v4.1.0: INJECT PENDING USER MESSAGE (auto-reweave race fix)
        # When stream_router auto-routes to Weaver UPDATE, the user's
        # latest message may not yet be persisted to the DB (the SSE
        # handler fires before message persistence completes). We
        # inject it directly so hash-based dedup sees it as new.
        # Dedup by hash ensures no double-counting if it IS in DB.
        # =============================================================
        if pending_user_message and pending_user_message.strip():
            pending_msg = {"role": "user", "content": pending_user_message.strip()}
            # Only inject if not already present (hash check prevents duplicates)
            pending_hash = _hash_message(pending_msg)
            existing_hashes = _hash_messages(all_messages)
            if pending_hash not in existing_hashes:
                all_messages.append(pending_msg)
                print(f"[WEAVER] v4.1.0 Injected pending_user_message ({len(pending_user_message)} chars, hash={pending_hash})")
            else:
                print(f"[WEAVER] v4.1.0 pending_user_message already in DB (hash={pending_hash}), skipping injection")
        
        if not all_messages:
            no_messages_msg = (
                "**No conversation to weave**\n\n"
                "I don't see any recent messages to organize into a job description.\n\n"
                "Share what you want to build or change, then say "
                "`how does that look all together` again."
            )
            yield _serialize_sse({"type": "token", "content": no_messages_msg})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        total_message_count = len(all_messages)
        print(f"[WEAVER] Gathered {total_message_count} total messages")
        
        # =====================================================================
        # STEP 2: Extract meta-mode phrases (Bug 2 fix)
        # =====================================================================
        
        filtered_messages, extracted_modes = _extract_meta_mode(all_messages)
        execution_mode = _format_execution_mode(extracted_modes)
        
        if execution_mode:
            print(f"[WEAVER] Extracted execution_mode: {execution_mode}")
        
        # =====================================================================
        # STEP 2b: Apply typo normalization (v3.6.0)
        # Must happen BEFORE Step 4 builds ramble_text so classification sees
        # normalized text (e.g., "deck top" → "desktop")
        # =====================================================================
        
        for i, msg in enumerate(filtered_messages):
            if msg.get("role") == "user" and msg.get("content"):
                normalized_content = _normalize_typos(msg["content"])
                if normalized_content != msg["content"]:
                    filtered_messages[i] = {**msg, "content": normalized_content}
        
        # =====================================================================
        # STEP 3: Load confirmed prefs + woven hashes + checkpoint
        # =====================================================================
        
        confirmed_prefs = {}
        if _FLOW_STATE_AVAILABLE and get_confirmed_design_prefs:
            confirmed_prefs = get_confirmed_design_prefs(project_id)
            if confirmed_prefs:
                print(f"[WEAVER] Loaded confirmed prefs: {confirmed_prefs}")
        
        # Merge with any newly captured answers
        if captured_answers:
            confirmed_prefs.update(captured_answers)
            if _FLOW_STATE_AVAILABLE and save_confirmed_design_prefs:
                save_confirmed_design_prefs(project_id, captured_answers)
        
        # Load woven hashes for delta detection
        woven_hashes: Set[str] = set()
        if _FLOW_STATE_AVAILABLE and get_woven_user_hashes:
            woven_hashes = get_woven_user_hashes(project_id)
            if woven_hashes:
                print(f"[WEAVER] Loaded {len(woven_hashes)} woven user hashes")
        
        # Load checkpoint for previous output
        checkpoint = None
        if _FLOW_STATE_AVAILABLE and get_weave_checkpoint:
            checkpoint = get_weave_checkpoint(project_id)
            if checkpoint:
                print(f"[WEAVER] Loaded checkpoint: {checkpoint['message_count']} messages")
        
        # =====================================================================
        # STEP 4: Compute new messages using HASH-BASED dedup
        # v3.9.0: Now includes assistant messages with vision context
        # =====================================================================
        
        # v3.9.0: Extract vision context BEFORE filtering
        # This preserves Gemini vision analysis for SpecGate
        vision_context = _extract_vision_context(filtered_messages)
        if vision_context:
            print(f"[WEAVER] v3.9 Extracted {len(vision_context)} chars of vision context")
        
        # Filter to USER messages + assistant messages with vision context
        # v3.9.0: Changed from USER-only to include valuable vision analysis
        relevant_messages = [
            m for m in filtered_messages 
            if m.get("role") == "user" or 
               (m.get("role") == "assistant" and _is_vision_context(m.get("content", "")))
        ]
        
        # For hashing, we still only track USER messages (to determine what's new)
        user_messages_only = [m for m in filtered_messages if m.get("role") == "user"]
        print(f"[WEAVER] Filtered to {len(relevant_messages)} relevant messages ({len(user_messages_only)} USER + vision context) (from {total_message_count} total)")
        
        # Compute hashes for current user messages
        current_user_hashes = _hash_messages(user_messages_only)
        
        # Determine which messages are NEW (not in woven_hashes)
        new_user_hashes = current_user_hashes - woven_hashes
        
        # Get the actual new messages (those whose hash is in new_user_hashes)
        new_user_messages = [
            m for m in user_messages_only
            if _hash_message(m) in new_user_hashes
        ]
        
        # Determine mode: UPDATE if we have previous output AND woven hashes
        is_update_mode = bool(woven_hashes) and checkpoint is not None and checkpoint.get("last_output")
        
        if is_update_mode:
            if not new_user_messages:
                no_new_msg = (
                    "**Nothing new to weave**\n\n"
                    "I don't see any new requirements from you since the last weave.\n\n"
                    "Add more details to your conversation, then say "
                    "`how does that look all together` again."
                )
                yield _serialize_sse({"type": "token", "content": no_new_msg})
                yield _serialize_sse({"type": "done", "provider": provider, "model": model})
                return
            
            print(f"[WEAVER] UPDATE mode: {len(new_user_messages)} new USER messages (hash-based detection)")
        else:
            print("[WEAVER] CREATE mode: first weave for this project")
        
        # =====================================================================
        # v5.5 PHASE 4C: Progressive Memory — compact long conversations
        # =====================================================================
        _compaction_applied = False
        if len(relevant_messages) >= 15:
            try:
                from app.llm.weaver_memory import compact_conversation
                _compaction_result = await compact_conversation(relevant_messages)
                if _compaction_result.was_compacted:
                    ramble_text = _compaction_result.format_for_weaver()
                    _compaction_applied = True
                    logger.info(
                        "[WEAVER] v5.5 Progressive memory: %d messages compacted "
                        "(%d distilled → %d chars, %d verbatim)",
                        _compaction_result.total_messages,
                        _compaction_result.compacted_count,
                        len(_compaction_result.distilled_summary),
                        _compaction_result.preserved_count,
                    )
                    print(
                        f"[WEAVER] v5.5 COMPACTION: {_compaction_result.compacted_count} old → "
                        f"{len(_compaction_result.distilled_summary)} char summary, "
                        f"{_compaction_result.preserved_count} recent kept verbatim"
                    )
                else:
                    logger.debug("[WEAVER] v5.5 Compaction skipped: %s", _compaction_result.skip_reason)
            except (ImportError, Exception) as _compact_err:
                logger.debug("[WEAVER] v5.5 Progressive memory unavailable: %s", _compact_err)

        # Format ramble text from RELEVANT messages (USER + vision context)
        # v3.9.0: Now includes vision analysis for context
        if not _compaction_applied:
            ramble_text = _format_ramble(relevant_messages)
        
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
        
        # Build prefs context
        prefs_context = ""
        if confirmed_prefs:
            prefs_lines = [f"- {k.title()}: {v}" for k, v in confirmed_prefs.items()]
            prefs_context = "\n\nUser's confirmed design preferences:\n" + "\n".join(prefs_lines)
        
        # Build execution mode context
        exec_mode_context = ""
        if execution_mode:
            exec_mode_context = f"\n\nExecution mode (extracted from meta-phrases): {execution_mode}"
        
        # v4.0.0: No questions_context injection. The LLM generates its own
        # contextual questions based on actual gaps in the user's requirements.
        
        if is_micro_task:
            # =================================================================
            # MICRO-TASK MODE (v3.6.0) - Simple file operations, minimal output
            # =================================================================
            print(f"[WEAVER] MICRO-TASK mode: using minimal prompt for file operation")
            
            start_message = f"**Quick task detected...**\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            # Build blocking questions context if any
            blocker_context = ""
            if blocking_questions:
                blocker_context = "\n\nBLOCKING QUESTIONS (must include in output):\n" + "\n".join(f"- {q}" for q in blocking_questions)
            
            system_prompt = MICRO_TASK_SYSTEM_PROMPT
            user_prompt = f"""User request:

{ramble_text}{blocker_context}

Produce the minimal job outline:"""
        
        elif is_refactor_task:
            # =================================================================
            # REFACTOR-TASK MODE (v3.8.0) - Text replacement, no design questions
            # =================================================================
            print(f"[WEAVER] REFACTOR-TASK mode: using refactor prompt")
            
            start_message = f"**Refactor/rename task detected...**\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            system_prompt = REFACTOR_TASK_SYSTEM_PROMPT
            user_prompt = f"""User request:

{ramble_text}

Produce the refactor job outline:"""
        
        elif is_update_mode:
            # UPDATE MODE - Merge new info into existing job description
            print(f"[WEAVER] UPDATE mode: weaving {len(new_user_messages)} new messages into existing spec")
            
            # v5.5 PHASE 4C: Compact new messages if there are many
            _update_compacted = False
            if len(new_user_messages) >= 15:
                try:
                    from app.llm.weaver_memory import compact_conversation
                    _update_compact = await compact_conversation(new_user_messages)
                    if _update_compact.was_compacted:
                        new_ramble = _update_compact.format_for_weaver()
                        _update_compacted = True
                        print(f"[WEAVER] v5.5 UPDATE compaction: {_update_compact.compacted_count} distilled + {_update_compact.preserved_count} verbatim")
                except (ImportError, Exception):
                    pass
            if not _update_compacted:
                new_ramble = _format_ramble(new_user_messages)
            previous_output = checkpoint["last_output"]
            
            # DEBUG: Show what we're sending to the LLM
            print(f"[WEAVER] NEW RAMBLE CONTENT ({len(new_ramble)} chars):")
            print(f"[WEAVER] ---\n{new_ramble[:500]}{'...' if len(new_ramble) > 500 else ''}\n[WEAVER] ---")
            
            start_message = f"**Updating your job description...**\n\nIncorporating {len(new_user_messages)} new requirement(s) from you.\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            system_prompt = WEAVER_UPDATE_SYSTEM_PROMPT

            user_prompt = f"""Previous job description:

{previous_output}

New requirements from user (extract and add EVERY feature):

{new_ramble}
{prefs_context}{exec_mode_context}

Output the complete updated job description with all new features added:"""

        else:
            # CREATE MODE - First weave
            print(f"[WEAVER] CREATE mode: weaving {total_message_count} messages")
            
            start_message = f"**Organizing your thoughts...**\n\nAnalyzing {total_message_count} messages to create a job description.\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            system_prompt = WEAVER_CREATE_SYSTEM_PROMPT

            user_prompt = f"""Organize this conversation into a job description:

{ramble_text}{prefs_context}{exec_mode_context}

Remember:
- Include ALL requirements the user stated (don't drop anything)
- Preserve any ambiguities (list them, don't resolve them)
- Keep What and Outcome DIFFERENT (no duplication)
- Code-answerable gaps go in "SpecGate must resolve" (NOT questions for user)
- Only put genuinely subjective/preference questions in "Questions for user"
- When in doubt, it's a SpecGate directive, not a user question
- Preserve the user's domain terminology"""
        
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

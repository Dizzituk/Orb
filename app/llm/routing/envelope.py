# FILE: app/llm/routing/envelope.py
"""JobEnvelope construction helpers.

Extracted from app.llm.routing.core to keep the core router smaller and easier to sanity-check.
Behavior is intended to be identical to the legacy implementation.

v2.0: Added memory injection from ASTRA memory system.
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from uuid import uuid4

from app.llm.schemas import LLMTask
from app.jobs.schemas import (
    JobEnvelope,
    DataSensitivity,
    JobBudget,
    OutputContract,
    validate_job_envelope,
    ValidationError,
)
from app.llm.routing.job_routing import (
    _default_importance_for_job_type,
    _default_modalities_for_job_type,
    inject_file_map_into_messages,
)
from app.llm.pipeline.high_stakes import _map_to_phase4_job_type

# v2.0: Memory injection
from app.llm.routing.memory_injection import (
    build_memory_context,
    inject_memory_into_system_prompt,
    get_memory_injection_stats,
    MEMORY_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Environment flag to disable memory injection
MEMORY_INJECTION_ENABLED = os.getenv("ORB_MEMORY_INJECTION", "1") == "1"


def _get_db_session():
    """Get a database session for memory retrieval."""
    try:
        from app.db import SessionLocal
        return SessionLocal()
    except Exception as e:
        logger.warning(f"[envelope] Failed to get DB session: {e}")
        return None


def synthesize_envelope_from_task(
    task: LLMTask,
    session_id: Optional[str] = None,
    project_id: int = 1,
    file_map: Optional[str] = None,
    cleaned_message: Optional[str] = None,
) -> JobEnvelope:
    """
    Synthesize a JobEnvelope from LLMTask.

    v0.15.1: Added cleaned_message parameter for OVERRIDE line removal.
    v2.0: Added memory injection from ASTRA memory system.
    """
    phase4_job_type = _map_to_phase4_job_type(task.job_type)
    importance = _default_importance_for_job_type(task.job_type)
    modalities = _default_modalities_for_job_type(task.job_type)

    routing = task.routing
    max_tokens = routing.max_tokens if routing else 8000
    max_cost = routing.max_cost_usd if routing else 1.0
    timeout = routing.timeout_seconds if routing else 60

    budget = JobBudget(
        max_tokens=max_tokens,
        max_cost_estimate=float(max_cost),
        max_wall_time_seconds=timeout,
    )

    # ==========================================================================
    # v2.0: Memory injection
    # ==========================================================================
    memory_context = None
    memory_stats = None
    
    if MEMORY_AVAILABLE and MEMORY_INJECTION_ENABLED:
        db = _get_db_session()
        if db:
            try:
                # Determine job type string for preference filtering
                job_type_str = None
                if task.job_type:
                    job_type_str = task.job_type.value if hasattr(task.job_type, 'value') else str(task.job_type)
                
                memory_context = build_memory_context(
                    db=db,
                    messages=task.messages,
                    job_type=job_type_str,
                    component="llm_router",
                )
                memory_stats = get_memory_injection_stats(memory_context)
                
                if not memory_context.is_empty():
                    logger.info(
                        f"[envelope] Memory injected: depth={memory_stats['depth']} "
                        f"tokens≈{memory_stats['token_estimate']} "
                        f"prefs={len(memory_stats['preferences_applied'])} "
                        f"records={memory_stats['records_retrieved']}"
                    )
            except Exception as e:
                logger.warning(f"[envelope] Memory injection failed: {e}")
            finally:
                db.close()

    # ==========================================================================
    # Build messages with system/project context
    # ==========================================================================
    final_messages = []

    system_parts = []
    if task.system_prompt:
        system_parts.append(task.system_prompt)
    if task.project_context:
        system_parts.append(task.project_context)

    system_content = "\n\n".join(system_parts) if system_parts else None
    
    # v2.0: Inject memory into system prompt
    if memory_context and not memory_context.is_empty():
        system_content = inject_memory_into_system_prompt(system_content, memory_context)

    if system_content:
        final_messages.append({"role": "system", "content": system_content})

    # v0.15.1: Replace user message content if cleaned_message provided
    if cleaned_message is not None:
        for msg in task.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Simple string message - replace entirely
                    final_messages.append({"role": "user", "content": cleaned_message})
                elif isinstance(content, list):
                    # Multimodal message - replace only the text part
                    new_content = []
                    text_replaced = False
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text" and not text_replaced:
                            # Replace first text part with cleaned message
                            new_content.append({"type": "text", "text": cleaned_message})
                            text_replaced = True
                        else:
                            new_content.append(part)
                    # If no text part was found, append the cleaned message
                    if not text_replaced and cleaned_message:
                        new_content.append({"type": "text", "text": cleaned_message})
                    final_messages.append({"role": "user", "content": new_content})
                else:
                    final_messages.append(msg)
            else:
                final_messages.append(msg)
    else:
        final_messages.extend(task.messages)

    # v0.15.0: Inject file map if available
    if file_map:
        final_messages = inject_file_map_into_messages(final_messages, file_map)

    envelope = JobEnvelope(
        job_id=str(uuid4()),
        session_id=session_id or f"legacy-{uuid4()}",
        project_id=project_id,
        job_type=phase4_job_type,
        importance=importance,
        data_sensitivity=DataSensitivity.INTERNAL,
        modalities_in=modalities,
        budget=budget,
        output_contract=OutputContract.TEXT_RESPONSE,
        messages=final_messages,
        metadata={
            "legacy_provider_hint": task.provider.value if task.provider else None,
            "legacy_routing": routing.model_dump() if routing else None,
            "legacy_context": task.project_context,
            "file_map": file_map,
            # v2.0: Memory injection metadata
            "memory_injection": memory_stats,
        },
        allow_multi_model_review=False,
        needs_tools=[],
    )

    try:
        validate_job_envelope(envelope)
    except ValidationError as ve:
        logger.warning("[router] Envelope validation failed: %s", ve)
        raise

    return envelope

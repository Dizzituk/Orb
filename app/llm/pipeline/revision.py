# FILE: app/llm/pipeline/revision.py
"""Revision pipeline for high-stakes jobs.

Block 6 of the PoT (Proof of Thought) system:
- Spec-anchored revision (verifies suggestions against spec before implementing)
- Revision loop until critique passes or max iterations
- Legacy single-revision for backward compatibility

v1.1 (2026-01):
- Uses stage_models for provider/model configuration (env-driven)
- No more hardcoded provider - REVISION_PROVIDER/REVISION_MODEL from env

v1.0 (2025-12):
- Extracted from high_stakes.py for better maintainability
- Spec-anchored revision prompt prevents drift from reviewer suggestions
- Debug logging for pipeline visibility

SPEC ANCHORING:
When revising based on critique, the PoT Spec serves as authoritative anchor:
- Suggestions that ADD requirements not in spec → REJECTED
- Suggestions that CONTRADICT spec → REJECTED
- Suggestions that ALIGN with spec → IMPLEMENTED
This prevents "spec drift" where reviewers inadvertently change scope.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from app.llm.schemas import JobType, LLMResult, LLMTask
from app.jobs.schemas import (
    JobEnvelope,
    JobType as Phase4JobType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
)
from app.providers.registry import llm_call as registry_llm_call

# Critique schemas
from app.llm.pipeline.critique_schemas import CritiqueResult

# Import from sibling module
from app.llm.pipeline.critique import (
    call_json_critic,
    store_critique_artifact,
)

# Ledger events
try:
    from app.pot_spec.ledger import (
        emit_revision_loop_started,
        emit_arch_revised,
        emit_revision_loop_terminated,
    )
    from app.pot_spec.service import get_job_artifact_root
    LEDGER_AVAILABLE = True
except ImportError:
    LEDGER_AVAILABLE = False

# Stage models (env-driven model resolution)
try:
    from app.llm.stage_models import get_revision_config, get_architecture_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

logger = logging.getLogger(__name__)

MAX_REVISION_ITERATIONS = int(os.getenv("ORB_MAX_REVISION_ITERATIONS", "3"))


def _get_revision_model_config() -> tuple[str, str, int, int]:
    """Get revision provider/model from stage_models or env vars AT RUNTIME.
    
    Returns: (provider, model, max_tokens, timeout)
    """
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_revision_config()
            return cfg.provider, cfg.model, cfg.max_output_tokens, cfg.timeout_seconds
        except Exception:
            pass
    
    # Fallback to legacy env vars
    provider = os.getenv("REVISION_PROVIDER", "anthropic")
    model = os.getenv("REVISION_MODEL") or os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101")
    max_tokens = int(os.getenv("REVISION_MAX_OUTPUT_TOKENS") or os.getenv("OPUS_REVISION_MAX_TOKENS", "60000"))
    timeout = int(os.getenv("REVISION_TIMEOUT_SECONDS") or os.getenv("OPUS_REVISION_TIMEOUT", "300"))
    return provider, model, max_tokens, timeout


def _get_architecture_model_config() -> tuple[str, str, int, int]:
    """Get architecture provider/model from stage_models or env vars AT RUNTIME.
    
    Returns: (provider, model, max_tokens, timeout)
    """
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_architecture_config()
            return cfg.provider, cfg.model, cfg.max_output_tokens, cfg.timeout_seconds
        except Exception:
            pass
    
    # Fallback to legacy env vars
    provider = os.getenv("ARCHITECTURE_PROVIDER", "anthropic")
    model = os.getenv("ARCHITECTURE_MODEL") or os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101")
    max_tokens = int(os.getenv("ARCHITECTURE_MAX_OUTPUT_TOKENS") or os.getenv("OPUS_DRAFT_MAX_TOKENS", "60000"))
    timeout = int(os.getenv("ARCHITECTURE_TIMEOUT_SECONDS") or os.getenv("OPUS_TIMEOUT_SECONDS", "300"))
    return provider, model, max_tokens, timeout


# Legacy exports (for backward compatibility - these get value at import time)
# For truly dynamic lookup, use _get_revision_model_config() directly
OPUS_REVISION_MAX_TOKENS = int(os.getenv("REVISION_MAX_OUTPUT_TOKENS") or os.getenv("OPUS_REVISION_MAX_TOKENS", "60000"))
OPUS_REVISION_TIMEOUT = int(os.getenv("REVISION_TIMEOUT_SECONDS") or os.getenv("OPUS_REVISION_TIMEOUT", "300"))


# =============================================================================
# Block 6: Spec-Anchored Revision Prompt
# =============================================================================

def build_spec_anchored_revision_prompt(
    draft_text: str,
    original_request: str,
    critique: CritiqueResult,
    spec_json: Optional[str] = None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
) -> str:
    """Build revision prompt with spec-anchoring to prevent drift.
    
    This wrapper adds explicit spec verification instructions to ensure
    Claude Opus validates Gemini's suggestions against the authoritative spec
    before implementing them.
    """
    # Format blocking issues
    blocking_text = ""
    if critique.blocking_issues:
        blocking_items = []
        for i, issue in enumerate(critique.blocking_issues, 1):
            issue_id = getattr(issue, 'id', f'ISSUE-{i}')
            title = getattr(issue, 'title', 'Untitled')
            desc = getattr(issue, 'description', '')
            fix = getattr(issue, 'suggested_fix', '')
            blocking_items.append(f"  {issue_id}: {title}\n    Description: {desc}\n    Suggested Fix: {fix}")
        blocking_text = "\n".join(blocking_items)
    
    # Build the spec-anchored prompt
    prompt = f"""You are revising an architecture document based on a critique.

CRITICAL: SPEC-ANCHORED REVISION PROTOCOL
==========================================
The PoT Spec below is the AUTHORITATIVE source of truth. Before implementing ANY 
suggestion from the critique, you MUST verify it aligns with the spec.

RULES:
1. If a suggestion ADDS requirements not in the spec → REJECT and note why
2. If a suggestion CONTRADICTS the spec → REJECT and note why  
3. If a suggestion ALIGNS with the spec → IMPLEMENT it
4. If unsure → Default to what the spec says

This prevents "spec drift" where reviewers inadvertently add scope or change requirements.

"""

    if spec_json:
        prompt += f"""PoT SPEC (AUTHORITATIVE - DO NOT DEVIATE):
============================================
{spec_json}

"""

    prompt += f"""ORIGINAL USER REQUEST:
======================
{original_request}

CURRENT ARCHITECTURE (to be revised):
=====================================
{draft_text}

CRITIQUE FROM REVIEWER (verify each suggestion against spec before implementing):
==================================================================================
Overall Assessment: {"PASS" if critique.overall_pass else "FAIL - BLOCKING ISSUES FOUND"}
Summary: {critique.summary}

"""

    if blocking_text:
        prompt += f"""BLOCKING ISSUES (must address - but verify against spec first):
{blocking_text}

"""

    prompt += """YOUR TASK:
==========
1. Review each blocking issue
2. For each suggested fix, check: "Is this in the spec? Does this align with the spec?"
3. If YES → Implement the fix
4. If NO → Note that you're rejecting the suggestion because it's out-of-spec
5. Output the complete revised architecture document

"""

    # Add MANDATORY header requirement for Stage 3 verification
    if spec_id and spec_hash:
        prompt += f"""MANDATORY OUTPUT FORMAT:
========================
Your revised architecture MUST begin with these EXACT lines (Stage 3 verification requires this):

SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

Then continue with your architecture document. DO NOT omit these lines or Stage 3 will fail.

"""

    prompt += """Begin your response with the revised architecture (no preamble):
"""

    return prompt


# =============================================================================
# Block 6: Revision Functions
# =============================================================================

async def call_revision(
    *,
    arch_content: str,
    original_request: str,
    critique: CritiqueResult,
    spec_json: Optional[str] = None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    opus_model_id: str,
    envelope: JobEnvelope,
) -> Optional[str]:
    """Call revision model based on blocking issues.
    
    Uses REVISION_PROVIDER/REVISION_MODEL from env via stage_models.
    Uses spec-anchored prompt to prevent drift from reviewer suggestions.
    Returns revised architecture content or None on failure.
    """
    # Get config from stage_models (runtime lookup)
    revision_provider, revision_model, revision_max_tokens, revision_timeout = _get_revision_model_config()
    
    # Allow override from caller (for backward compat)
    if opus_model_id and opus_model_id != revision_model:
        revision_model = opus_model_id
    
    # DEBUG: Log revision start
    print(f"[DEBUG] [revision] Starting revision: provider={revision_provider}, model={revision_model}")
    print(f"[DEBUG] [revision] Blocking issues to address: {len(critique.blocking_issues)}")
    logger.info(f"[revision] Calling revision with {len(critique.blocking_issues)} blocking issues")
    
    # Use spec-anchored prompt instead of generic one
    revision_prompt = build_spec_anchored_revision_prompt(
        draft_text=arch_content,
        original_request=original_request,
        critique=critique,
        spec_json=spec_json,
        spec_id=spec_id,
        spec_hash=spec_hash,
    )
    
    revision_messages = [
        {"role": "system", "content": "You are revising an architecture document. The PoT Spec is authoritative - reject any suggestions that contradict or add to it."},
        {"role": "user", "content": revision_prompt},
    ]
    
    try:
        revision_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=getattr(envelope, 'session_id', 'session-unknown'),
            project_id=int(getattr(envelope, 'project_id', 0)),
            job_type=getattr(Phase4JobType, "APP_ARCHITECTURE", list(Phase4JobType)[0]),
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=revision_max_tokens,
                max_cost_estimate=0.15,
                max_wall_time_seconds=revision_timeout,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=revision_messages,
            metadata={"revision": True, "spec_anchored": True, "provider": revision_provider},
            allow_multi_model_review=False,
            needs_tools=[],
        )
        
        print(f"[DEBUG] [revision] Sending revision request to {revision_provider}/{revision_model}...")
        result = await registry_llm_call(
            provider_id=revision_provider,
            model_id=revision_model,
            messages=revision_messages,
            job_envelope=revision_envelope,
            max_tokens=revision_max_tokens,
            timeout_seconds=revision_timeout,
        )
        
        if not result or not result.content:
            print(f"[DEBUG] [revision] ERROR: Empty response from {revision_provider}")
            return None
        
        print(f"[DEBUG] [revision] Received revised architecture: {len(result.content)} chars")
        logger.info(f"[revision] Revision complete: {len(result.content)} chars")
        return result.content
        
    except Exception as exc:
        logger.warning(f"[revision] Revision call failed: {exc}")
        print(f"[DEBUG] [revision] EXCEPTION: {exc}")
        return None


async def run_revision_loop(
    *,
    db,
    job_id: str,
    project_id: int,
    arch_content: str,
    arch_id: str,
    spec_id: str,
    spec_hash: str,
    spec_json: Optional[str],
    original_request: str,
    opus_model_id: str,
    envelope: JobEnvelope,
    env_context: Optional[Dict[str, Any]] = None,
    # Callback to store revised architecture
    store_architecture_fn=None,
) -> Tuple[str, int, bool, CritiqueResult]:
    """Run the revision loop until critique passes or max iterations.
    
    Returns (final_content, final_version, passed, final_critique)
    """
    current_content = arch_content
    current_version = 1
    iterations_used = 0
    
    # Emit loop start
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            emit_revision_loop_started(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                max_iterations=MAX_REVISION_ITERATIONS,
            )
        except Exception:
            pass
    
    final_critique = CritiqueResult()
    
    for iteration in range(MAX_REVISION_ITERATIONS):
        iterations_used = iteration + 1
        logger.info(f"[revision_loop] Iteration {iterations_used}/{MAX_REVISION_ITERATIONS}")
        print(f"[DEBUG] [revision_loop] === Iteration {iterations_used}/{MAX_REVISION_ITERATIONS} ===")
        
        # 1. Critique current architecture
        print(f"[DEBUG] [revision_loop] Calling JSON critic for arch_v{current_version}...")
        critique = await call_json_critic(
            arch_content=current_content,
            original_request=original_request,
            spec_json=spec_json,
            env_context=env_context,
            envelope=envelope,
        )
        
        # 2. Store critique artifact
        print(f"[DEBUG] [revision_loop] Storing critique artifact...")
        store_critique_artifact(
            job_id=job_id,
            arch_id=arch_id,
            arch_version=current_version,
            critique=critique,
        )
        
        final_critique = critique
        
        # 3. Check if passed
        if critique.overall_pass:
            logger.info(f"[revision_loop] Critique passed at iteration {iterations_used}")
            print(f"[DEBUG] [revision_loop] ✅ CRITIQUE PASSED at iteration {iterations_used}")
            break
        
        print(f"[DEBUG] [revision_loop] ❌ Critique FAILED: {len(critique.blocking_issues)} blocking issues")
        
        # 4. If not last iteration, revise
        if iteration < MAX_REVISION_ITERATIONS - 1:
            logger.info(f"[revision_loop] Revising to address {len(critique.blocking_issues)} blocking issues")
            print(f"[DEBUG] [revision_loop] Calling Opus revision to address blocking issues...")
            
            revised_content = await call_revision(
                arch_content=current_content,
                original_request=original_request,
                critique=critique,
                spec_json=spec_json,
                spec_id=spec_id,
                spec_hash=spec_hash,
                opus_model_id=opus_model_id,
                envelope=envelope,
            )
            
            if revised_content:
                old_version = current_version
                current_version += 1
                current_content = revised_content
                
                # Store revised architecture (using callback if provided)
                new_arch_id = arch_id
                new_hash = ""
                if store_architecture_fn:
                    new_arch_id, new_hash, _ = store_architecture_fn(
                        db=db,
                        job_id=job_id,
                        project_id=project_id,
                        arch_content=current_content,
                        spec_id=spec_id,
                        spec_hash=spec_hash,
                        arch_version=current_version,
                        model=opus_model_id,
                        previous_arch_id=arch_id,
                    )
                
                # Emit revision event
                if LEDGER_AVAILABLE:
                    try:
                        job_root = get_job_artifact_root()
                        emit_arch_revised(
                            job_artifact_root=job_root,
                            job_id=job_id,
                            arch_id=arch_id,
                            old_version=old_version,
                            new_version=current_version,
                            new_hash=new_hash,
                            addressed_issues=[i.id for i in critique.blocking_issues],
                            model=opus_model_id,
                        )
                    except Exception:
                        pass
            else:
                logger.warning("[revision_loop] Revision failed, stopping loop")
                break
    
    # Emit loop termination
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            reason = "pass" if final_critique.overall_pass else "max_iterations"
            emit_revision_loop_terminated(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                final_version=current_version,
                reason=reason,
                iterations_used=iterations_used,
                final_pass=final_critique.overall_pass,
            )
        except Exception:
            pass
    
    return current_content, current_version, final_critique.overall_pass, final_critique


# =============================================================================
# Legacy Revision (Prose-based)
# =============================================================================

def _map_to_phase4_job_type(job_type: JobType) -> Phase4JobType:
    """Map router JobType to Phase 4 JobType."""
    def _p4(*names: str) -> Phase4JobType:
        for n in names:
            v = getattr(Phase4JobType, n, None)
            if v is not None:
                return v
        return list(Phase4JobType)[0]

    name = getattr(job_type, "name", "") or ""
    value = getattr(job_type, "value", "") or ""
    key = f"{name} {value} {job_type}".strip().lower()

    if "critique" in key or "review" in key:
        return _p4("CRITIQUE_REVIEW", "CHAT_RESEARCH")
    if "video" in key:
        return _p4("VIDEO_ADVANCED", "VIDEO_SIMPLE")
    if any(tok in key for tok in ("image", "vision", "ocr")):
        return _p4("VISION_COMPLEX", "VISION_SIMPLE")
    if any(tok in key for tok in ("arch", "architecture")):
        return _p4("APP_ARCHITECTURE", "ORCHESTRATION_PLAN")
    if any(tok in key for tok in ("repo", "refactor", "code")):
        return _p4("CODE_REPO", "CODE_SMALL")

    return _p4("CHAT_RESEARCH", "CHAT_SIMPLE")


async def call_opus_revision(
    original_task: LLMTask,
    draft_result: LLMResult,
    critique_result: LLMResult,
    opus_model_id: str,
    envelope: JobEnvelope,
) -> Optional[LLMResult]:
    """Call revision model based on critique (legacy).
    
    Uses REVISION_PROVIDER/REVISION_MODEL from env via stage_models.
    """
    # Get config from stage_models (runtime lookup)
    revision_provider, revision_model, revision_max_tokens, revision_timeout = _get_revision_model_config()
    
    # Allow override from caller (for backward compat)
    if opus_model_id and opus_model_id != revision_model:
        revision_model = opus_model_id
    
    print(f"[DEBUG] [revision-legacy] Using: provider={revision_provider}, model={revision_model}")
    
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""

    revision_prompt = f"""You are revising your own previous answer using a critique.

ORIGINAL REQUEST:
{original_request}

YOUR DRAFT ANSWER:
{draft_result.content}

CRITIQUE:
{critique_result.content}
"""

    revision_messages = [
        {"role": "system", "content": "You are revising your own answer. Output only the improved final answer."},
        {"role": "user", "content": revision_prompt},
    ]

    try:
        phase4_job_type = _map_to_phase4_job_type(original_task.job_type)

        revision_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=getattr(envelope, 'session_id', 'session-unknown'),
            project_id=int(getattr(envelope, 'project_id', 0)),
            job_type=phase4_job_type,
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=revision_max_tokens,
                max_cost_estimate=0.10,
                max_wall_time_seconds=revision_timeout,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=revision_messages,
            metadata={"revision_of_draft": True, "provider": revision_provider},
            allow_multi_model_review=False,
            needs_tools=[],
        )

        result = await registry_llm_call(
            provider_id=revision_provider,
            model_id=revision_model,
            messages=revision_messages,
            job_envelope=revision_envelope,
            max_tokens=revision_max_tokens,
            timeout_seconds=revision_timeout,
        )

        if not result:
            return None

        return LLMResult(
            content=result.content,
            provider=revision_provider,
            model=revision_model,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )

    except Exception as exc:
        logger.warning(f"[revision-legacy] Revision call failed: {exc}")
        return None


__all__ = [
    # Configuration
    "OPUS_REVISION_MAX_TOKENS",
    "MAX_REVISION_ITERATIONS",
    # Block 6: Revision
    "build_spec_anchored_revision_prompt",
    "call_revision",
    "run_revision_loop",
    # Legacy
    "call_opus_revision",
    "_map_to_phase4_job_type",
]
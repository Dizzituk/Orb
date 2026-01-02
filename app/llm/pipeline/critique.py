# FILE: app/llm/pipeline/critique.py
"""Critique pipeline for high-stakes jobs.

Block 5 of the PoT (Proof of Thought) system:
- JSON critique with structured blocking/non-blocking issues
- Spec-anchored critique (verifies architecture against spec)
- Legacy prose-based critique for backward compatibility

v1.1 (2026-01):
- Uses stage_models for provider/model configuration (env-driven)
- No more hardcoded provider - CRITIQUE_PROVIDER/CRITIQUE_MODEL from env

v1.0 (2025-12):
- Extracted from high_stakes.py for better maintainability
- Spec verification protocol in system messages
- Debug logging for pipeline visibility
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from app.llm.schemas import LLMResult, LLMTask
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
from app.llm.pipeline.critique_schemas import (
    CritiqueResult,
    CritiqueIssue,
    parse_critique_output,
    build_json_critique_prompt,
)

# Ledger events
try:
    from app.pot_spec.ledger import (
        emit_critique_created,
        emit_critique_pass,
        emit_critique_fail,
    )
    from app.pot_spec.service import get_job_artifact_root
    LEDGER_AVAILABLE = True
except ImportError:
    LEDGER_AVAILABLE = False

# Stage models (env-driven model resolution)
try:
    from app.llm.stage_models import get_critique_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

logger = logging.getLogger(__name__)


def _get_critique_model_config() -> tuple[str, str, int]:
    """Get critique provider/model from stage_models or env vars AT RUNTIME.
    
    Returns: (provider, model, max_tokens)
    """
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_critique_config()
            return cfg.provider, cfg.model, cfg.max_output_tokens
        except Exception:
            pass
    
    # Fallback to legacy env vars
    provider = os.getenv("CRITIQUE_PROVIDER", "google")
    model = os.getenv("CRITIQUE_MODEL") or os.getenv("GEMINI_CRITIC_MODEL", "gemini-2.0-flash")
    max_tokens = int(os.getenv("CRITIQUE_MAX_OUTPUT_TOKENS") or os.getenv("GEMINI_CRITIC_MAX_TOKENS", "60000"))
    return provider, model, max_tokens


# Legacy exports (for backward compatibility - these call runtime lookup)
# Note: These are module-level variables that get the current value at import time
# For truly dynamic lookup, use _get_critique_model_config() directly
GEMINI_CRITIC_MODEL = os.getenv("CRITIQUE_MODEL") or os.getenv("GEMINI_CRITIC_MODEL", "gemini-2.0-flash")
GEMINI_CRITIC_MAX_TOKENS = int(os.getenv("CRITIQUE_MAX_OUTPUT_TOKENS") or os.getenv("GEMINI_CRITIC_MAX_TOKENS", "60000"))


# =============================================================================
# Block 5: Structured JSON Critique
# =============================================================================

def store_critique_artifact(
    *,
    job_id: str,
    arch_id: str,
    arch_version: int,
    critique: CritiqueResult,
) -> Tuple[str, str, str]:
    """Store critique as JSON + MD artifacts.
    
    Returns (critique_id, json_path, md_path)
    """
    critique_id = str(uuid4())
    json_path = ""
    md_path = ""
    
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            critique_dir = Path(job_root) / "jobs" / job_id / "critique"
            critique_dir.mkdir(parents=True, exist_ok=True)
            
            # Write JSON artifact
            json_path = str(critique_dir / f"critique_v{arch_version}.json")
            Path(json_path).write_text(critique.to_json(), encoding="utf-8")
            
            # Write MD artifact (human-readable)
            md_path = str(critique_dir / f"critique_v{arch_version}.md")
            Path(md_path).write_text(critique.to_markdown(), encoding="utf-8")
            
            # Emit events
            emit_critique_created(
                job_artifact_root=job_root,
                job_id=job_id,
                critique_id=critique_id,
                arch_id=arch_id,
                arch_version=arch_version,
                blocking_count=len(critique.blocking_issues),
                non_blocking_count=len(critique.non_blocking_issues),
                overall_pass=critique.overall_pass,
                model=critique.critique_model,
                json_path=json_path,
                md_path=md_path,
            )
            
            # Emit pass/fail event
            if critique.overall_pass:
                emit_critique_pass(
                    job_artifact_root=job_root,
                    job_id=job_id,
                    critique_id=critique_id,
                    arch_id=arch_id,
                    arch_version=arch_version,
                )
            else:
                emit_critique_fail(
                    job_artifact_root=job_root,
                    job_id=job_id,
                    critique_id=critique_id,
                    arch_id=arch_id,
                    arch_version=arch_version,
                    blocking_issues=[i.id for i in critique.blocking_issues],
                )
            
            logger.info(f"[critique] Stored: {json_path}")
        except Exception as e:
            logger.warning(f"[critique] Failed to store artifacts: {e}")
    
    return critique_id, json_path, md_path


async def call_json_critic(
    *,
    arch_content: str,
    original_request: str,
    spec_json: Optional[str] = None,
    env_context: Optional[Dict[str, Any]] = None,
    envelope: JobEnvelope,
) -> CritiqueResult:
    """Call critic with JSON output schema.
    
    Returns structured CritiqueResult.
    Uses CRITIQUE_PROVIDER/CRITIQUE_MODEL from env via stage_models.
    """
    # Get config from stage_models (runtime lookup)
    critique_provider, critique_model, critique_max_tokens = _get_critique_model_config()
    
    # DEBUG: Log critique start
    print(f"[DEBUG] [critique] Starting JSON critic: provider={critique_provider}, model={critique_model}")
    logger.info(f"[critique] Calling JSON critic: {critique_provider}/{critique_model}")
    
    critique_prompt = build_json_critique_prompt(
        draft_text=arch_content,
        original_request=original_request,
        spec_json=spec_json,
        env_context=env_context,
    )
    
    # System message emphasizes spec as authoritative anchor
    system_message = """You are a critical architecture reviewer. Output ONLY valid JSON.

SPEC VERIFICATION PROTOCOL:
- The PoT Spec (if provided) is the AUTHORITATIVE source of requirements
- When reviewing the architecture, check if it SATISFIES the spec requirements
- Do NOT suggest adding features/requirements that aren't in the spec
- Only flag as "blocking" if the architecture FAILS to meet spec requirements
- Non-blocking issues are style/optimization suggestions that don't violate the spec

Your suggestions must align with the spec. Do not expand scope."""

    critique_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": critique_prompt},
    ]
    
    try:
        critic_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=getattr(envelope, 'session_id', 'session-unknown'),
            project_id=int(getattr(envelope, 'project_id', 0)),
            job_type=getattr(Phase4JobType, "CRITIQUE_REVIEW", list(Phase4JobType)[0]),
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=critique_max_tokens,
                max_cost_estimate=0.05,
                max_wall_time_seconds=90,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=critique_messages,
            metadata={"critic": "json", "provider": critique_provider},
            allow_multi_model_review=False,
            needs_tools=[],
        )
        
        print(f"[DEBUG] [critique] Sending request to {critique_provider}/{critique_model}...")
        result = await registry_llm_call(
            provider_id=critique_provider,
            model_id=critique_model,
            messages=critique_messages,
            job_envelope=critic_envelope,
            max_tokens=critique_max_tokens,
        )
        
        if not result or not result.content:
            logger.warning("[critic] Empty response from critic")
            print(f"[DEBUG] [critique] ERROR: Empty response from {critique_provider}")
            return CritiqueResult(
                summary="Critique failed: empty response",
                critique_model=critique_model,
            )
        
        print(f"[DEBUG] [critique] Received response: {len(result.content)} chars")
        critique = parse_critique_output(result.content, model=critique_model)
        
        # DEBUG: Log critique result
        print(f"[DEBUG] [critique] Result: overall_pass={critique.overall_pass}, blocking={len(critique.blocking_issues)}, non_blocking={len(critique.non_blocking_issues)}")
        logger.info(f"[critique] Result: pass={critique.overall_pass}, blocking={len(critique.blocking_issues)}, non_blocking={len(critique.non_blocking_issues)}")
        
        # Log full details of blocking issues for visibility
        if critique.blocking_issues:
            print(f"[DEBUG] [critique] === BLOCKING ISSUES ===")
            for issue in critique.blocking_issues:
                issue_id = getattr(issue, 'id', 'N/A')
                title = getattr(issue, 'title', 'Untitled')
                desc = getattr(issue, 'description', '')[:200]  # Truncate for logs
                category = getattr(issue, 'category', 'unknown')
                spec_ref = getattr(issue, 'spec_ref', 'N/A')
                print(f"[DEBUG] [critique]   {issue_id}: [{category}] {title}")
                print(f"[DEBUG] [critique]     → {desc}")
                print(f"[DEBUG] [critique]     spec_ref: {spec_ref}")
            print(f"[DEBUG] [critique] === END BLOCKING ===")
        
        # Log summary of non-blocking issues
        if critique.non_blocking_issues:
            print(f"[DEBUG] [critique] Non-blocking ({len(critique.non_blocking_issues)}): {[getattr(i, 'id', 'N/A') for i in critique.non_blocking_issues]}")
        
        return critique
        
    except Exception as exc:
        logger.warning(f"[critic] JSON critic call failed: {exc}")
        print(f"[DEBUG] [critique] EXCEPTION: {exc}")
        # Get model for error response
        _, model_for_error, _ = _get_critique_model_config()
        return CritiqueResult(
            summary=f"Critique failed: {exc}",
            critique_model=model_for_error,
        )


# =============================================================================
# Legacy Prompt Builders (kept for backward compatibility)
# =============================================================================

def build_critique_prompt_for_architecture(
    draft_text: str,
    original_request: str,
    env_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build architecture-specific critique prompt (legacy prose format)."""
    env_text = ""
    if env_context:
        env_text = f"\n\nENVIRONMENT CONTEXT:\n{env_context}\n"

    return textwrap.dedent(
        f"""
        You are a senior architecture reviewer. Critique the following draft response for:
        - Technical correctness
        - Completeness against the original request
        - Security implications (if relevant)
        - Practicality in the given environment (single-host, Windows 11, solo dev, local-only)
        - Clarity and actionability

        Provide a structured critique with:
        1) Critical issues (must-fix)
        2) Important improvements (should-fix)
        3) Nice-to-haves (could-fix)
        4) Missing assumptions / unknowns
        5) Suggested revised outline

        ORIGINAL REQUEST:
        {original_request}

        DRAFT RESPONSE:
        {draft_text}
        {env_text}
        """
    ).strip()


def build_critique_prompt_for_security(draft_text: str, original_request: str) -> str:
    """Build security-specific critique prompt."""
    return textwrap.dedent(
        f"""
        You are a senior security reviewer. Critique the following draft response for:
        - Security correctness (threats, mitigations, assumptions)
        - Missing controls or hardening steps
        - Risk prioritization
        - Practicality for a solo developer on Windows 11
        - Potential policy/safety issues

        ORIGINAL REQUEST:
        {original_request}

        DRAFT RESPONSE:
        {draft_text}
        """
    ).strip()


def build_critique_prompt_for_general(draft_text: str, original_request: str, job_type_str: str) -> str:
    """Build general critique prompt for non-architecture/security high-stakes."""
    return textwrap.dedent(
        f"""
        You are a critical reviewer. Critique the following draft response for:
        - Correctness
        - Completeness
        - Clarity
        - Logical consistency
        - Actionability

        Job type context: {job_type_str}

        ORIGINAL REQUEST:
        {original_request}

        DRAFT RESPONSE:
        {draft_text}
        """
    ).strip()


def build_critique_prompt(
    draft_text: str,
    original_request: str,
    job_type_str: str,
    env_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Dispatch critique prompt builder based on job type (legacy)."""
    jt = (job_type_str or "").strip().lower()
    if jt in ("architecture_design", "big_architecture", "high_stakes_infra", "architecture", "orchestrator"):
        return build_critique_prompt_for_architecture(draft_text, original_request, env_context=env_context)
    if jt in ("security_review", "compliance_review"):
        return build_critique_prompt_for_security(draft_text, original_request)
    return build_critique_prompt_for_general(draft_text, original_request, job_type_str=jt)


# =============================================================================
# Legacy Critique Pipeline (Prose-based)
# =============================================================================

async def call_gemini_critic(
    original_task: LLMTask,
    draft_result: LLMResult,
    job_type_str: str,
    envelope: JobEnvelope,
    env_context: Optional[Dict[str, Any]] = None,
) -> Optional[LLMResult]:
    """Call critic for prose-based critique (legacy format).
    
    Uses CRITIQUE_PROVIDER/CRITIQUE_MODEL from env via stage_models.
    """
    # Get config from stage_models (runtime lookup)
    critique_provider, critique_model, critique_max_tokens = _get_critique_model_config()
    
    # DEBUG: Log critique start
    print(f"[DEBUG] [critique-legacy] Starting critic: provider={critique_provider}, model={critique_model}")
    logger.info(f"[critique-legacy] Calling critic: {critique_provider}/{critique_model}")
    
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""

    critique_prompt = build_critique_prompt(
        draft_text=draft_result.content,
        original_request=original_request,
        job_type_str=job_type_str,
        env_context=env_context,
    )

    critique_messages = [
        {"role": "system", "content": "You are a critical reviewer. Provide direct critique."},
        {"role": "user", "content": critique_prompt},
    ]

    try:
        critic_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=getattr(envelope, 'session_id', 'session-unknown'),
            project_id=int(getattr(envelope, 'project_id', 0)),
            job_type=getattr(Phase4JobType, "CRITIQUE_REVIEW", list(Phase4JobType)[0]),
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=critique_max_tokens,
                max_cost_estimate=0.05,
                max_wall_time_seconds=60,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=critique_messages,
            metadata={"critic": "prose", "provider": critique_provider},
            allow_multi_model_review=False,
            needs_tools=[],
        )

        print(f"[DEBUG] [critique-legacy] Sending request to {critique_provider}/{critique_model}...")
        result = await registry_llm_call(
            provider_id=critique_provider,
            model_id=critique_model,
            messages=critique_messages,
            job_envelope=critic_envelope,
            max_tokens=critique_max_tokens,
        )

        if not result:
            print(f"[DEBUG] [critique-legacy] ERROR: No result from {critique_provider}")
            return None

        print(f"[DEBUG] [critique-legacy] Received response: {len(result.content)} chars")
        logger.info(f"[critique-legacy] Response received: {len(result.content)} chars")
        
        return LLMResult(
            content=result.content,
            provider=critique_provider,
            model=critique_model,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )

    except Exception as exc:
        logger.warning(f"[critic] Critic call failed: {exc}")
        print(f"[DEBUG] [critique-legacy] EXCEPTION: {exc}")
        return None


__all__ = [
    # Configuration
    "GEMINI_CRITIC_MODEL",
    "GEMINI_CRITIC_MAX_TOKENS",
    # Block 5: JSON critique
    "store_critique_artifact",
    "call_json_critic",
    # Legacy
    "call_gemini_critic",
    "build_critique_prompt",
    "build_critique_prompt_for_architecture",
    "build_critique_prompt_for_security",
    "build_critique_prompt_for_general",
]
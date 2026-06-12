# FILE: app/llm/pipeline/_critique_legacy.py
# Purpose: Legacy critique pipeline helpers.
# Called-by: app.llm.pipeline.critique
# Depends-on: app.jobs.schemas, app.llm.pipeline.critique_parts.model_config, app.llm.schemas, app.providers.registry
# Last-renovated: 2026-06-11
"""
Legacy critique pipeline helpers.

Extracted from critique.py: prompt builders, prose-based Gemini critic,
and segment interface contract validation.
"""
from __future__ import annotations

import logging
import os
import textwrap
from typing import Any, Dict, List, Optional
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
from app.llm.pipeline.critique_parts.model_config import _get_critique_model_config

logger = logging.getLogger(__name__)

def build_critique_prompt_for_architecture(
    draft_text: str, original_request: str,
    env_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build architecture-specific critique prompt (legacy prose format)."""
    env_text = f"\n\nENVIRONMENT CONTEXT:\n{env_context}\n" if env_context else ""
    return textwrap.dedent(f"""
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
    """).strip()


def build_critique_prompt_for_security(draft_text: str, original_request: str) -> str:
    """Build security-specific critique prompt."""
    return textwrap.dedent(f"""
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
    """).strip()


def build_critique_prompt_for_general(draft_text: str, original_request: str, job_type_str: str) -> str:
    """Build general critique prompt for non-architecture/security high-stakes."""
    return textwrap.dedent(f"""
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
    """).strip()


def build_critique_prompt(
    draft_text: str, original_request: str, job_type_str: str,
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
    original_task: LLMTask, draft_result: LLMResult,
    job_type_str: str, envelope: JobEnvelope,
    env_context: Optional[Dict[str, Any]] = None,
) -> Optional[LLMResult]:
    """Call critic for prose-based critique (legacy format)."""
    critique_provider, critique_model, critique_max_tokens = _get_critique_model_config()
    
    print(f"[DEBUG] [critique-legacy] Starting critic: provider={critique_provider}, model={critique_model}")
    
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

        result = await registry_llm_call(
            provider_id=critique_provider,
            model_id=critique_model,
            messages=critique_messages,
            job_envelope=critic_envelope,
            max_tokens=critique_max_tokens,
            timeout_seconds=180,  # v1.10: Large arch + spec inputs need room
        )

        if not result:
            return None

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
        return None


# =============================================================================
# SEGMENT INTERFACE CONTRACT VALIDATION (Phase 2)
# =============================================================================

def validate_interface_contracts(
    arch_content: str,
    segment_context: dict,
) -> list:
    """Phase 2: Validate that a segment's architecture respects its interface contracts."""
    import re as _local_re
    issues = []
    if not segment_context or not arch_content:
        return issues

    file_scope = segment_context.get("file_scope", [])
    exposes = segment_context.get("exposes") or {}
    consumes = segment_context.get("consumes") or {}
    segment_id = segment_context.get("segment_id", "unknown")

    arch_lower = arch_content.lower()

    for class_name in exposes.get("class_names", []):
        if class_name.lower() not in arch_lower:
            issues.append({
                "type": "contract_violation", "severity": "warning",
                "segment_id": segment_id,
                "message": f"Segment {segment_id} promises to expose class '{class_name}' but it is not mentioned in the architecture.",
            })

    for endpoint in exposes.get("endpoint_paths", []):
        path_part = endpoint.split()[-1] if " " in endpoint else endpoint
        if path_part.lower() not in arch_lower:
            issues.append({
                "type": "contract_violation", "severity": "warning",
                "segment_id": segment_id,
                "message": f"Segment {segment_id} promises to expose endpoint '{endpoint}' but it is not mentioned in the architecture.",
            })

    for export_name in exposes.get("export_names", []):
        if export_name.lower() not in arch_lower:
            issues.append({
                "type": "contract_violation", "severity": "warning",
                "segment_id": segment_id,
                "message": f"Segment {segment_id} promises to expose '{export_name}' but it is not mentioned in the architecture.",
            })

    for class_name in consumes.get("class_names", []):
        if class_name.lower() not in arch_lower:
            issues.append({
                "type": "contract_violation", "severity": "info",
                "segment_id": segment_id,
                "message": f"Segment {segment_id} declares it consumes '{class_name}' from upstream but doesn't reference it.",
            })

    if file_scope:
        scope_basenames = {os.path.basename(f).lower() for f in file_scope}
        file_refs = _local_re.findall(r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx|json|yaml|css)', arch_content)
        for ref in file_refs:
            ref_basename = os.path.basename(ref).lower()
            if ref_basename not in scope_basenames:
                ref_context_idx = arch_content.lower().find(ref.lower())
                if ref_context_idx >= 0:
                    context_before = arch_content[max(0, ref_context_idx - 50):ref_context_idx].lower()
                    if any(kw in context_before for kw in ["create", "modify", "write", "add to", "update"]):
                        issues.append({
                            "type": "scope_violation", "severity": "warning",
                            "segment_id": segment_id,
                            "message": f"Architecture for {segment_id} references file '{ref}' outside the segment's file_scope.",
                        })

    if issues:
        logger.info("[critique] Phase 2 contract validation for %s: %d issue(s)", segment_id, len(issues))
        for issue in issues:
            print(f"[critique] CONTRACT: [{issue['severity']}] {issue['message']}")

    return issues

# FILE: app/llm/pipeline/_revision_helpers.py
"""
Revision pipeline helpers.

Extracted from revision.py: spec-anchored prompt builder,
Opus revision caller, and job type mapping.
"""
from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.llm.schemas import LLMResult
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

logger = logging.getLogger(__name__)

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
    spec_markdown: Optional[str] = None,
    segment_contract_markdown: Optional[str] = None,  # v2.1: skeleton contracts
    env_context: Optional[Dict[str, Any]] = None,  # v2.1: tech stack context
    enrichment_markdown: Optional[str] = None,  # v5.18: AST-extracted symbols
) -> str:
    """Build revision prompt with spec-anchoring to prevent drift.
    
    v5.18: Now includes enrichment_markdown so revision has the same
    symbol knowledge as the draft stage.
    
    v2.1: Now includes segment_contract_markdown and env_context so the
    revision model has the same context as the draft and critique stages.
    
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

    if spec_markdown:
        prompt += f"""PoT SPEC MARKDOWN (AUTHORITATIVE - THIS IS THE FULL SPEC):
================================================================
The following markdown IS the complete, authoritative spec. ALL sections
including Acceptance Criteria, Constraints, and Implementation Steps are
binding requirements. Do NOT claim any section is empty or missing if it
appears below.

{spec_markdown}

================================================================
END OF AUTHORITATIVE SPEC
================================================================

"""
        # v2.2: Extract and highlight acceptance criteria for regression check
        ac_lines = []
        in_ac = False
        for line in spec_markdown.split('\n'):
            line_lower = line.strip().lower()
            if 'acceptance criteria' in line_lower or 'acceptance criterion' in line_lower:
                in_ac = True
                ac_lines.append(line)
            elif in_ac:
                if line.strip().startswith('#') and 'acceptance' not in line_lower:
                    in_ac = False
                else:
                    ac_lines.append(line)
        if ac_lines:
            ac_block = '\n'.join(ac_lines)
            prompt += f"""ACCEPTANCE CRITERIA CHECKLIST (extracted for regression verification):
====================================================================
You MUST verify your revision satisfies ALL of these. Fixing one while
breaking another is the most common revision failure mode.

{ac_block}

====================================================================

"""
    elif spec_json:
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

    # v2.1: Inject segment contract and env context (same data as draft/critique)
    if segment_contract_markdown:
        prompt += f"""SEGMENT INTERFACE CONTRACT (from skeleton — binding cross-segment bindings):
====================================================================================
{segment_contract_markdown}

IMPORTANT: Your revision MUST NOT change any function names, signatures, or exports
that are listed in this contract. Other segments depend on these exact interfaces.

"""

    if env_context:
        import json as _json
        prompt += f"""ENVIRONMENT CONTEXT (tech stack constraints):
=============================================
{_json.dumps(env_context, indent=2)}

"""

    if enrichment_markdown:
        prompt += f"""SEGMENT ENRICHMENT (AST-extracted symbols from source file):
============================================================
The following symbols were extracted from the original source file.
This list is NOT exhaustive — the source may contain additional
functions not captured by AST extraction. If a symbol logically
belongs in this segment, include it even if not listed below.

{enrichment_markdown}
============================================================

"""

    prompt += """YOUR TASK:
==========
1. Review each blocking issue
2. For each suggested fix, check: "Is this in the spec? Does this align with the spec?"
3. If YES → Implement the fix
4. If NO → Note that you're rejecting the suggestion because it's out-of-spec
5. MANDATORY REGRESSION CHECK: After making ALL fixes, re-read EVERY Acceptance Criterion
   in the spec above. For each AC, verify your revised architecture still satisfies it.
   If fixing one issue broke a different AC, fix that too before outputting.
   This is critical — do NOT fix one criterion while silently breaking another.
6. Output the complete revised architecture document

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
    
    # v2.1: opus_model_id override REMOVED — REVISION_MODEL env var is authoritative.
    
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
            stage="revision",  # v2.2: Cost tracking
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

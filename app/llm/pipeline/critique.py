# FILE: app/llm/pipeline/critique.py
"""Critique pipeline for high-stakes jobs — Orchestrator Façade.

Block 5 of the PoT (Proof of Thought) system.
Decomposed into critique_parts/ submodules for maintainability.

See critique_parts/ for individual check implementations:
- model_config.py: Provider/model configuration
- blocker_filtering.py: Approved blocker type enforcement (v1.2)
- grounding_validation.py: Spec-ref fabrication detection (v1.7)
- section_authority.py: LLM-suggestion section handling (v1.9)
- evidence_resolution.py: CRITICAL_CLAIMS validation (v2.0)
- scope_creep.py: Endpoint drift + excluded feature detection (v2.1)
- spec_compliance.py: Platform/stack/scope mismatch detection (v1.3-v2.2)
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    APPROVED_ARCHITECTURE_BLOCKER_TYPES,
)

# Evidence loop utilities (v2.0)
from app.llm.pipeline.evidence_loop import parse_evidence_requests

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

# =============================================================================
# Import from decomposed submodules
# =============================================================================
from app.llm.pipeline.critique_parts.model_config import (
    _get_critique_model_config,
    GEMINI_CRITIC_MODEL,
    GEMINI_CRITIC_MAX_TOKENS,
)
from app.llm.pipeline.critique_parts.blocker_filtering import filter_blocking_issues
from app.llm.pipeline.critique_parts.grounding_validation import validate_spec_ref_grounding
from app.llm.pipeline.critique_parts.section_authority import validate_section_authority
from app.llm.pipeline.critique_parts.evidence_resolution import (
    extract_critical_claims,
    run_evidence_resolution_check,
    VALID_RESOLUTIONS,
)
from app.llm.pipeline.critique_parts.scope_creep import run_scope_creep_check
from app.llm.pipeline.critique_parts.spec_compliance import run_deterministic_spec_compliance_check

logger = logging.getLogger(__name__)


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
    """Store critique as JSON + MD artifacts. Returns (critique_id, json_path, md_path)."""
    critique_id = str(uuid4())
    json_path = ""
    md_path = ""
    
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            critique_dir = Path(job_root) / "jobs" / job_id / "critique"
            critique_dir.mkdir(parents=True, exist_ok=True)
            
            json_path = str(critique_dir / f"critique_v{arch_version}.json")
            Path(json_path).write_text(critique.to_json(), encoding="utf-8")
            
            md_path = str(critique_dir / f"critique_v{arch_version}.md")
            Path(md_path).write_text(critique.to_markdown(), encoding="utf-8")
            
            emit_critique_created(
                job_artifact_root=job_root, job_id=job_id,
                critique_id=critique_id, arch_id=arch_id,
                arch_version=arch_version,
                blocking_count=len(critique.blocking_issues),
                non_blocking_count=len(critique.non_blocking_issues),
                overall_pass=critique.overall_pass,
                model=critique.critique_model,
                json_path=json_path, md_path=md_path,
            )
            
            if critique.overall_pass:
                emit_critique_pass(
                    job_artifact_root=job_root, job_id=job_id,
                    critique_id=critique_id, arch_id=arch_id,
                    arch_version=arch_version,
                )
            else:
                emit_critique_fail(
                    job_artifact_root=job_root, job_id=job_id,
                    critique_id=critique_id, arch_id=arch_id,
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
    spec_markdown: Optional[str] = None,
    env_context: Optional[Dict[str, Any]] = None,
    envelope: JobEnvelope,
    segment_contract_markdown: Optional[str] = None,
) -> CritiqueResult:
    """Call critic with JSON output schema. Returns structured CritiqueResult."""
    critique_provider, critique_model, critique_max_tokens = _get_critique_model_config()
    
    print(f"[DEBUG] [critique] Starting JSON critic: provider={critique_provider}, model={critique_model}")
    logger.info(f"[critique] Calling JSON critic: {critique_provider}/{critique_model}")
    
    # v1.3: Run DETERMINISTIC spec-compliance check FIRST
    deterministic_issues = run_deterministic_spec_compliance_check(
        arch_content=arch_content,
        spec_json=spec_json,
        original_request=original_request,
    )
    
    # v2.1: Run SCOPE CREEP check
    scope_creep_issues = run_scope_creep_check(
        arch_content=arch_content,
        spec_markdown=spec_markdown,
        spec_json=spec_json,
    )
    deterministic_issues.extend(scope_creep_issues)
    if scope_creep_issues:
        print(f"[DEBUG] [critique] v2.1 Scope creep check: {len(scope_creep_issues)} issue(s)")
    
    if deterministic_issues:
        print(f"[DEBUG] [critique] v1.3 EARLY FAIL: {len(deterministic_issues)} deterministic blocker(s) found")
        logger.warning("[critique] v1.3 Deterministic check BLOCKED architecture: %d issue(s)", len(deterministic_issues))
        return CritiqueResult(
            summary=f"Architecture BLOCKED by deterministic spec-compliance check: {len(deterministic_issues)} issue(s) found",
            critique_model="deterministic_check_v1.3",
            critique_failed=False,
            critique_mode="deep+deterministic",
            blocking_issues=deterministic_issues,
            non_blocking_issues=[],
        )
    
    # v2.0: Evidence resolution check
    pending_requests = parse_evidence_requests(arch_content)
    
    if not pending_requests:
        evidence_issues = run_evidence_resolution_check(arch_content=arch_content)
        blocking_evidence = [i for i in evidence_issues if i.severity == "blocking"]
        non_blocking_evidence = [i for i in evidence_issues if i.severity != "blocking"]
        
        if blocking_evidence:
            print(f"[DEBUG] [critique] v2.0 EVIDENCE CHECK FAIL: {len(blocking_evidence)} blocking issue(s)")
            return CritiqueResult(
                summary=f"Architecture BLOCKED by evidence resolution check: {len(blocking_evidence)} unresolved critical claim(s)",
                critique_model="evidence_check_v2.0",
                critique_failed=False,
                critique_mode="deep+evidence",
                blocking_issues=blocking_evidence,
                non_blocking_issues=non_blocking_evidence,
            )
        
        deterministic_issues.extend(non_blocking_evidence)
        if non_blocking_evidence:
            print(f"[DEBUG] [critique] v2.0 Evidence check: {len(non_blocking_evidence)} non-blocking issue(s) noted")
    else:
        print(f"[DEBUG] [critique] v2.0 Skipping evidence resolution check: {len(pending_requests)} pending EVIDENCE_REQUEST(s)")
    
    print(f"[DEBUG] [critique] v1.3 Deterministic check PASSED - proceeding to LLM critique")
    
    if spec_markdown:
        print(f"[DEBUG] [critique] v1.6 POT spec markdown provided ({len(spec_markdown)} chars)")
        logger.info("[critique] v1.6 POT spec markdown injected (%d chars)", len(spec_markdown))
    
    critique_prompt = build_json_critique_prompt(
        draft_text=arch_content,
        original_request=original_request,
        spec_json=spec_json,
        spec_markdown=spec_markdown,
        env_context=env_context,
    )
    
    system_message = """You are a critical architecture reviewer. Output ONLY valid JSON.

GROUNDED CRITIQUE PROTOCOL (v1.9):
==================================
The POT Spec (if provided) is the AUTHORITATIVE CONTRACT for this task.
Your critique is BOUND to that spec - you cannot add terms to the contract.

CRITICAL RULES:
1. Judge the architecture ONLY against what's in the POT spec
2. Do NOT invent constraints that aren't in the spec
3. If the spec says "use OpenAI API" or any external service, that is ALLOWED
4. Do NOT flag user-requested features as violations
5. The spec IS the contract - if user wanted local-only, spec would say so

SECTION AUTHORITY (v1.9 - CRITICAL):
====================================
The spec contains TWO types of content:
- USER REQUIREMENTS: Goal, Constraints, Scope, Implementation Stack (if LOCKED)
  These are binding. Missing these = BLOCKING.
- LLM SUGGESTIONS: 'Files to Modify', 'Reference Files', 'Implementation Steps',
  'New Files to Create', 'Patterns to Follow', 'LLM Architecture Analysis'
  These are guidance only. The architecture MAY choose completely different
  files, integration points, or approaches. Do NOT raise BLOCKING issues
  if the architecture chooses different files or approaches than these
  sections suggest.

BLOCKING ISSUES (flag these):
- Architecture MISSES something the spec's USER REQUIREMENTS sections require
- Architecture CONTRADICTS something the spec's USER REQUIREMENTS state
- Architecture references files/paths NOT in the spec evidence
- Architecture has internal contradictions or calculation errors
- Architecture calls existing functions with UNVERIFIED parameter names
- Architecture depends on data not shown to be available at the code point
- Architecture proposes regex-parsing free text instead of adding fields explicitly
- Architecture proposes naive text processing for varying format content

NOT BLOCKING (do not flag these as blocking):
- Architecture choosing different integration files than 'Files to Modify' lists
- Architecture using different approaches than 'Implementation Steps' suggests
- External API usage that the spec requested
- Technology choices that align with the spec
- Features the spec explicitly requested
- Generic "best practices" not mentioned in the spec

EVIDENCE REQUIREMENT:
- Every blocking issue MUST include both spec_ref AND arch_ref
- spec_ref: Which USER REQUIREMENT is violated (MUST exist in the spec)
- arch_ref: Which architecture section shows the violation
- If you cannot cite both, make the issue non_blocking
- If spec_ref points to an LLM SUGGESTION section, make the issue non_blocking

## EVIDENCE-OR-REQUEST CONTRACT

For every implementation-affecting claim, you MUST output exactly one of:
1. **CITED** — You have seen evidence in this context.
2. **EVIDENCE_REQUEST** — You need the orchestrator to fetch something.
3. **DECISION** — This is a genuine design choice.
4. **HUMAN_REQUIRED** — Evidence doesn't exist and guessing is high-risk.

### CRITICAL_CLAIMS Register (Required)
At the END of your output (must be the LAST block), include:

CRITICAL_CLAIMS:
  - id: "CC-001"
    claim: "Short description"
    resolution: "CITED"
    evidence:
      - file: "path/to/file.py"
        lines: "42-58"

Every critical claim must be accounted for. This register is validated deterministically.
CRITICAL_CLAIMS must be the LAST block in your output."""

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
            timeout_seconds=180,  # v1.10: Large arch + spec inputs need room
        )
        
        if not result or not result.content:
            logger.warning("[critic] Empty response from critic")
            return CritiqueResult(
                summary="Critique failed: empty response / timeout",
                critique_model=critique_model,
                critique_failed=True,
                critique_mode="deep",
                blocking_issues=[CritiqueIssue(
                    id="CRITIQUE-FAIL-001", spec_ref=None, arch_ref=None,
                    category="system", severity="blocking",
                    description="Critique could not be completed due to timeout or empty response",
                    fix_suggestion="Retry critique with different provider or increase timeout",
                )],
            )
        
        print(f"[DEBUG] [critique] Received response: {len(result.content)} chars")
        critique = parse_critique_output(result.content, model=critique_model)
        critique.critique_mode = "deep"
        
        # v1.2: Apply blocker filtering
        original_blocking_count = len(critique.blocking_issues)
        if critique.blocking_issues:
            real_blocking, downgraded = filter_blocking_issues(
                critique.blocking_issues, require_evidence=True,
            )
            critique.blocking_issues = real_blocking
            critique.non_blocking_issues.extend(downgraded)
            critique.overall_pass = len(critique.blocking_issues) == 0 and not critique.critique_failed
            if downgraded:
                print(f"[DEBUG] [critique] Blocker filtering: {original_blocking_count} → {len(real_blocking)} (downgraded {len(downgraded)})")
        
        # v1.7: Grounding validation
        if critique.blocking_issues and (spec_markdown or spec_json):
            grounded_blocking, grounding_downgraded = validate_spec_ref_grounding(
                critique.blocking_issues, spec_markdown=spec_markdown, spec_json=spec_json,
            )
            critique.blocking_issues = grounded_blocking
            critique.non_blocking_issues.extend(grounding_downgraded)
            critique.overall_pass = len(critique.blocking_issues) == 0 and not critique.critique_failed
            if grounding_downgraded:
                print(f"[DEBUG] [critique] v1.7 Grounding filter: {len(grounded_blocking)} kept, {len(grounding_downgraded)} downgraded")
        
        # v1.9: Section authority validation
        if critique.blocking_issues:
            authority_kept, authority_downgraded = validate_section_authority(critique.blocking_issues)
            critique.blocking_issues = authority_kept
            critique.non_blocking_issues.extend(authority_downgraded)
            critique.overall_pass = len(critique.blocking_issues) == 0 and not critique.critique_failed
            if authority_downgraded:
                print(f"[DEBUG] [critique] v1.9 Section authority filter: {len(authority_kept)} kept, {len(authority_downgraded)} downgraded")
        
        print(f"[DEBUG] [critique] Result: overall_pass={critique.overall_pass}, blocking={len(critique.blocking_issues)}, non_blocking={len(critique.non_blocking_issues)}")
        logger.info(f"[critique] Result: pass={critique.overall_pass}, blocking={len(critique.blocking_issues)}, non_blocking={len(critique.non_blocking_issues)}")
        
        if critique.blocking_issues:
            print(f"[DEBUG] [critique] === BLOCKING ISSUES (after filtering) ===")
            for issue in critique.blocking_issues:
                print(f"[DEBUG] [critique]   {getattr(issue, 'id', 'N/A')}: [{getattr(issue, 'category', 'unknown')}] {getattr(issue, 'title', 'Untitled')}")
                print(f"[DEBUG] [critique]     → {(getattr(issue, 'description', '') or '')[:200]}")
                print(f"[DEBUG] [critique]     spec_ref: {getattr(issue, 'spec_ref', 'N/A')}, arch_ref: {getattr(issue, 'arch_ref', 'N/A')}")
            print(f"[DEBUG] [critique] === END BLOCKING ===")
        
        if critique.non_blocking_issues:
            print(f"[DEBUG] [critique] Non-blocking ({len(critique.non_blocking_issues)}): {[getattr(i, 'id', 'N/A') for i in critique.non_blocking_issues[:5]]}...")
        
        return critique
        
    except Exception as exc:
        logger.warning(f"[critic] JSON critic call failed: {exc}")
        print(f"[DEBUG] [critique] EXCEPTION: {exc}")
        _, model_for_error, _ = _get_critique_model_config()
        return CritiqueResult(
            summary=f"Critique failed: {exc}",
            critique_model=model_for_error,
            critique_failed=True,
            critique_mode="deep",
            blocking_issues=[CritiqueIssue(
                id="CRITIQUE-FAIL-002", spec_ref=None, arch_ref=None,
                category="system", severity="blocking",
                description=f"Critique could not be completed due to exception: {exc}",
                fix_suggestion="Check critic provider configuration and retry",
            )],
        )


# =============================================================================
# Legacy Prompt Builders (kept for backward compatibility)
# =============================================================================

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


__all__ = [
    # Configuration
    "GEMINI_CRITIC_MODEL",
    "GEMINI_CRITIC_MAX_TOKENS",
    # Block 5: Blocker filtering (v1.2)
    "filter_blocking_issues",
    # Block 5b: Grounding validation (v1.7)
    "validate_spec_ref_grounding",
    # Block 5c: Section authority validation (v1.9)
    "validate_section_authority",
    # Block 5d: Evidence resolution check (v2.0)
    "extract_critical_claims",
    "run_evidence_resolution_check",
    # Block 5e: Scope creep detection (v2.1)
    "run_scope_creep_check",
    # Block 5: Deterministic spec-compliance check (v1.3)
    "run_deterministic_spec_compliance_check",
    # Block 5: JSON critique
    "store_critique_artifact",
    "call_json_critic",
    # Legacy
    "call_gemini_critic",
    "build_critique_prompt",
    "build_critique_prompt_for_architecture",
    "build_critique_prompt_for_security",
    "build_critique_prompt_for_general",
    # Phase 2: Segment interface contract validation
    "validate_interface_contracts",
]

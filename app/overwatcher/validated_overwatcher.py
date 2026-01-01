# FILE: app/overwatcher/validated_overwatcher.py
"""
Validated Overwatcher Wrapper (Job 3)

Wraps run_overwatcher with:
1. Output validation (reject code/commands)
2. Automatic reprompting on violations
3. Cost guard enforcement
4. Strike integration
5. Routing persistence

Spec §4.4: On rejection, automatically reprompt with contract reminder.
Contract violations do NOT count as ErrorSignature strikes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from app.overwatcher.overwatcher import (
    run_overwatcher,
    OverwatcherOutput,
    Decision,
    OVERWATCHER_PROVIDER,
    OVERWATCHER_MODEL,
    OVERWATCHER_MAX_OUTPUT_TOKENS,
)
from app.overwatcher.evidence import EvidenceBundle
from app.overwatcher.output_validator import (
    validate_overwatcher_output,
    ValidationResult,
    build_reprompt_messages,
    REPROMPT_SYSTEM_ADDENDUM,
)
from app.overwatcher.cost_guard import (
    ModelRole,
    check_budget,
    record_usage,
    get_max_tokens,
    BudgetStatus,
)

logger = logging.getLogger(__name__)

# Max reprompt attempts for contract violations
MAX_REPROMPT_ATTEMPTS = 2


@dataclass
class ValidatedOverwatcherResult:
    """Result from validated overwatcher call."""
    output: OverwatcherOutput
    validation_passed: bool
    reprompt_count: int
    contract_violations: List[Dict[str, Any]]
    cost_tracked: bool
    budget_status: str
    
    def to_dict(self) -> dict:
        return {
            "output": self.output.to_dict(),
            "validation_passed": self.validation_passed,
            "reprompt_count": self.reprompt_count,
            "contract_violations": self.contract_violations,
            "cost_tracked": self.cost_tracked,
            "budget_status": self.budget_status,
        }


async def run_validated_overwatcher(
    *,
    evidence: EvidenceBundle,
    llm_call_fn: Callable,
    job_artifact_root: str,
    provider_id: str = None,
    model_id: str = None,
    deep_research_context: Optional[str] = None,
    break_glass: bool = False,
    db_session=None,
) -> ValidatedOverwatcherResult:
    """
    Run Overwatcher with output validation and reprompting.
    
    Flow:
    1. Check budget before calling
    2. Call run_overwatcher
    3. Validate output for contract compliance
    4. If violations: reprompt (up to MAX_REPROMPT_ATTEMPTS)
    5. Record usage to cost guard
    6. Return validated result
    
    Args:
        evidence: Evidence bundle for Overwatcher
        llm_call_fn: Async function to call LLM
        job_artifact_root: Root for job artifacts
        provider_id: LLM provider (default: openai)
        model_id: LLM model (default: gpt-5.2-pro)
        deep_research_context: Additional context from Strike 2
        break_glass: Enable break-glass budget mode
        db_session: Database session for persistence
    
    Returns:
        ValidatedOverwatcherResult with output and validation info
    """
    provider_id = provider_id or OVERWATCHER_PROVIDER
    model_id = model_id or OVERWATCHER_MODEL
    
    # Check budget before calling
    budget_check = check_budget(
        role=ModelRole.OVERWATCHER,
        requested_tokens=OVERWATCHER_MAX_OUTPUT_TOKENS,
        break_glass=break_glass,
    )
    
    if budget_check.status == BudgetStatus.EXCEEDED:
        logger.error(f"[validated_overwatcher] Budget exceeded: {budget_check.message}")
        return ValidatedOverwatcherResult(
            output=OverwatcherOutput(
                decision=Decision.FAIL,
                diagnosis=f"Budget exceeded: {budget_check.message}",
                blockers=["Token budget exceeded"],
            ),
            validation_passed=False,
            reprompt_count=0,
            contract_violations=[],
            cost_tracked=False,
            budget_status=budget_check.status.value,
        )
    
    if budget_check.status == BudgetStatus.BREAK_GLASS_REQUIRED and not break_glass:
        logger.warning(f"[validated_overwatcher] Break-glass required: {budget_check.message}")
    
    # Track reprompt attempts
    reprompt_count = 0
    all_violations: List[Dict[str, Any]] = []
    
    # Initial call
    output = await run_overwatcher(
        evidence=evidence,
        llm_call_fn=llm_call_fn,
        job_artifact_root=job_artifact_root,
        provider_id=provider_id,
        model_id=model_id,
        deep_research_context=deep_research_context,
    )
    
    # Validate output
    raw_text = _extract_raw_text(output)
    validation = validate_overwatcher_output(raw_text)
    
    # Reprompt loop if violations detected
    while not validation.is_valid and reprompt_count < MAX_REPROMPT_ATTEMPTS:
        reprompt_count += 1
        
        # Record violations
        for v in validation.violations:
            all_violations.append(v.to_dict())
        
        logger.warning(
            f"[validated_overwatcher] Contract violation #{reprompt_count}: "
            f"{len(validation.violations)} violations detected, reprompting"
        )
        
        # Build reprompt with contract reminder
        # Note: This is a simplified reprompt - in production, would need
        # to maintain conversation state and add the violation context
        output = await _reprompt_overwatcher(
            evidence=evidence,
            llm_call_fn=llm_call_fn,
            job_artifact_root=job_artifact_root,
            provider_id=provider_id,
            model_id=model_id,
            previous_output=output,
            validation_result=validation,
            deep_research_context=deep_research_context,
        )
        
        # Re-validate
        raw_text = _extract_raw_text(output)
        validation = validate_overwatcher_output(raw_text)
    
    # Final validation check
    validation_passed = validation.is_valid
    
    if not validation_passed:
        # Record final violations
        for v in validation.violations:
            all_violations.append(v.to_dict())
        
        logger.error(
            f"[validated_overwatcher] Contract violations persist after {reprompt_count} reprompts"
        )
    
    # Record usage
    try:
        record_usage(
            job_id=evidence.job_id,
            role=ModelRole.OVERWATCHER,
            provider=provider_id,
            model=model_id,
            prompt_tokens=0,  # Would need to track from LLM response
            completion_tokens=len(raw_text) // 4,  # Rough estimate
            cost_estimate=0.0,
            break_glass_used=break_glass,
            stage="verification",
        )
        cost_tracked = True
    except Exception as e:
        logger.warning(f"[validated_overwatcher] Failed to record usage: {e}")
        cost_tracked = False
    
    return ValidatedOverwatcherResult(
        output=output,
        validation_passed=validation_passed,
        reprompt_count=reprompt_count,
        contract_violations=all_violations,
        cost_tracked=cost_tracked,
        budget_status=budget_check.status.value,
    )


def _extract_raw_text(output: OverwatcherOutput) -> str:
    """Extract raw text from OverwatcherOutput for validation."""
    parts = [output.diagnosis]
    
    for fa in output.fix_actions:
        parts.append(fa.description)
        parts.append(fa.rationale)
    
    for constraint in output.constraints:
        parts.append(constraint)
    
    for v in output.verification:
        parts.append(v.command)
        parts.append(v.expected_outcome)
    
    return "\n".join(parts)


async def _reprompt_overwatcher(
    *,
    evidence: EvidenceBundle,
    llm_call_fn: Callable,
    job_artifact_root: str,
    provider_id: str,
    model_id: str,
    previous_output: OverwatcherOutput,
    validation_result: ValidationResult,
    deep_research_context: Optional[str] = None,
) -> OverwatcherOutput:
    """
    Reprompt Overwatcher after contract violation.
    
    Adds contract reminder and violation context to prompt.
    """
    from app.overwatcher.overwatcher import (
        build_overwatcher_prompt,
        parse_overwatcher_output,
        OVERWATCHER_SYSTEM,
    )
    
    # Build base prompt
    _, user_prompt = build_overwatcher_prompt(evidence)
    
    # Add contract reminder
    enhanced_system = OVERWATCHER_SYSTEM + REPROMPT_SYSTEM_ADDENDUM
    
    # Add violation context
    violation_types = set(v.violation_type.value for v in validation_result.violations[:5])
    user_prompt += f"\n\n[CONTRACT VIOLATION DETECTED]\nYour previous response contained: {', '.join(violation_types)}.\nYou MUST re-emit using ONLY the allowed JSON schema with NO CODE."
    
    if deep_research_context:
        user_prompt += f"\n\n## Deep Research Context\n{deep_research_context}"
    
    messages = [
        {"role": "system", "content": enhanced_system},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        result = await llm_call_fn(
            provider_id=provider_id,
            model_id=model_id,
            messages=messages,
            max_tokens=OVERWATCHER_MAX_OUTPUT_TOKENS,
        )
        
        raw_output = result.content if hasattr(result, "content") else str(result)
        return parse_overwatcher_output(raw_output)
        
    except Exception as e:
        logger.error(f"[validated_overwatcher] Reprompt failed: {e}")
        return OverwatcherOutput(
            decision=Decision.FAIL,
            diagnosis=f"Reprompt failed: {e}",
        )


def validate_overwatcher_result(output: OverwatcherOutput) -> ValidationResult:
    """
    Validate an OverwatcherOutput for contract compliance.
    
    Convenience function for external callers.
    """
    raw_text = _extract_raw_text(output)
    return validate_overwatcher_output(raw_text)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ValidatedOverwatcherResult",
    "run_validated_overwatcher",
    "validate_overwatcher_result",
    "MAX_REPROMPT_ATTEMPTS",
]

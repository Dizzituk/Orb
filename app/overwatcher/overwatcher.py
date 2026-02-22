# FILE: app/overwatcher/overwatcher.py
"""Overwatcher - Supervisor that diagnoses failures without writing code.

v3.1 (2026-01): Uses centralized stage_models for all config (no hardcoded models)

Role:
- Diagnoses failures, defines fix actions, enforces constraints
- Must NOT write code, patches, diffs, or full files
- Model configured via OVERWATCHER_PROVIDER and OVERWATCHER_MODEL env vars

Output Contract:
- DECISION: PASS | FAIL | NEEDS_INFO
- DIAGNOSIS: root cause hypothesis
- FIX_ACTIONS: ordered, file-targeted actions (no code)
- CONSTRAINTS: invariants to respect
- VERIFICATION: commands + expected outcomes
- BLOCKERS / NONBLOCKERS lists
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4
from datetime import datetime, timezone

from app.overwatcher.evidence import EvidenceBundle
from app.overwatcher._overwatcher_utils_4 import CODE_PATTERNS, FixAction, OVERWATCHER_MAX_INPUT_TOKENS, OVERWATCHER_SYSTEM, OVERWATCHER_USER, VerificationStep, _get_fallback_config, _get_overwatcher_config
from app.overwatcher._overwatcher_utils_5 import build_overwatcher_prompt, contains_code, run_pot_spec_execution

# v2.0: Evidence-or-Request Contract prompt
try:
    from app.llm.pipeline.evidence_contract_prompt import EVIDENCE_CONTRACT_PROMPT
    _EVIDENCE_CONTRACT_AVAILABLE = True
except ImportError:
    _EVIDENCE_CONTRACT_AVAILABLE = False
    EVIDENCE_CONTRACT_PROMPT = ""

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration - v3.1: Uses centralized stage_models (NO HARDCODED DEFAULTS)
# =============================================================================

# Import centralized config
try:
    from app.llm.stage_models import get_overwatcher_config, get_stage_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    get_overwatcher_config = None
    get_stage_config = None
    _STAGE_MODELS_AVAILABLE = False
    logger.warning("[overwatcher] stage_models not available")


# Max input tokens (reasonable constant, not model-specific)


# =============================================================================
# Output Schema
# =============================================================================

class Decision(str, Enum):
    """Overwatcher decision outcome."""
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_INFO = "NEEDS_INFO"


@dataclass
class OverwatcherOutput:
    """Complete Overwatcher output (decision-only, no code).
    
    Spec §9.2: All outputs MUST be decision-only with no code blocks.
    """
    
    decision: Decision
    diagnosis: str  # Root cause hypothesis
    fix_actions: List[FixAction] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    verification: List[VerificationStep] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    nonblockers: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0  # 0-1 confidence in diagnosis
    needs_deep_research: bool = False  # Hint for Strike 2
    
    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "diagnosis": self.diagnosis,
            "fix_actions": [fa.to_dict() for fa in self.fix_actions],
            "constraints": self.constraints,
            "verification": [v.to_dict() for v in self.verification],
            "blockers": self.blockers,
            "nonblockers": self.nonblockers,
            "confidence": self.confidence,
            "needs_deep_research": self.needs_deep_research,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "OverwatcherOutput":
        return cls(
            decision=Decision(data.get("decision", "FAIL")),
            diagnosis=data.get("diagnosis", ""),
            fix_actions=[FixAction.from_dict(fa) for fa in data.get("fix_actions", [])],
            constraints=data.get("constraints", []),
            verification=[VerificationStep.from_dict(v) for v in data.get("verification", [])],
            blockers=data.get("blockers", []),
            nonblockers=data.get("nonblockers", []),
            confidence=data.get("confidence", 0.0),
            needs_deep_research=data.get("needs_deep_research", False),
        )
    
    def validate_no_code(self) -> List[str]:
        """Validate that output contains no code.
        
        Returns list of violations if any code detected.
        """
        violations = []
        
        # Check diagnosis
        if contains_code(self.diagnosis):
            violations.append("Diagnosis contains code")
        
        # Check fix actions
        for fa in self.fix_actions:
            if contains_code(fa.description):
                violations.append(f"FixAction '{fa.action_type}' contains code")
        
        return violations


# =============================================================================
# Code Detection
# =============================================================================


# =============================================================================
# Prompts
# =============================================================================


# =============================================================================
# Output Parsing
# =============================================================================

def parse_overwatcher_output(raw_output: str) -> OverwatcherOutput:
    """Parse Overwatcher LLM output to structured format.
    
    Handles:
    - Raw JSON
    - JSON in code fences
    - Partial/malformed JSON
    """
    logger.info(f"[overwatcher_parse] Input length: {len(raw_output) if raw_output else 0} chars")
    preview = repr(raw_output[:300]) if raw_output else 'None'
    logger.info(f"[overwatcher_parse] Input preview: {preview}")
    
    if not raw_output:
        logger.warning("[overwatcher_parse] Empty output received")
        return OverwatcherOutput(
            decision=Decision.FAIL,
            diagnosis="Empty output from Overwatcher",
        )
    
    text = raw_output.strip()
    
    # Try to extract JSON from code fence
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        text = fence_match.group(1).strip()
        logger.info(f"[overwatcher_parse] Extracted from fence: {len(text)} chars")
    
    # Try direct parse
    try:
        data = json.loads(text)
        logger.info(f"[overwatcher_parse] Direct JSON parse succeeded, keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
        try:
            return OverwatcherOutput.from_dict(data)
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"[overwatcher_parse] from_dict failed: {e}")
            logger.error(f"[overwatcher_parse] Data was: {data}")
            return OverwatcherOutput(
                decision=Decision.FAIL,
                diagnosis=f"Invalid Overwatcher output format: {e}",
            )
    except json.JSONDecodeError as e:
        logger.warning(f"[overwatcher_parse] Direct parse failed: {e}")
    
    # Try to find JSON object
    start = text.find("{")
    if start == -1:
        logger.warning(f"[overwatcher_parse] No JSON object found in output")
        return OverwatcherOutput(
            decision=Decision.FAIL,
            diagnosis=f"Could not parse Overwatcher output: {text[:200]}",
        )
    
    # Find matching closing brace
    depth = 0
    end = -1
    in_string = False
    escape = False
    
    for i, char in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    
    if end > start:
        try:
            extracted = text[start:end]
            logger.info(f"[overwatcher_parse] Brace extraction: {len(extracted)} chars")
            data = json.loads(extracted)
            logger.info(f"[overwatcher_parse] Brace JSON parse succeeded, keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            try:
                return OverwatcherOutput.from_dict(data)
            except (ValueError, KeyError, TypeError) as e:
                logger.error(f"[overwatcher_parse] from_dict failed on brace-extracted: {e}")
                return OverwatcherOutput(
                    decision=Decision.FAIL,
                    diagnosis=f"Invalid Overwatcher output format: {e}",
                )
        except json.JSONDecodeError as e:
            logger.warning(f"[overwatcher_parse] Brace-extracted JSON parse failed: {e}")
    
    return OverwatcherOutput(
        decision=Decision.FAIL,
        diagnosis=f"Malformed Overwatcher output: {text[:200]}",
    )


# =============================================================================
# Main API
# =============================================================================

async def run_overwatcher(
    *,
    evidence: EvidenceBundle,
    llm_call_fn: Callable,
    job_artifact_root: str,
    provider_id: str = None,
    model_id: str = None,
    deep_research_context: Optional[str] = None,
    file_scope: Optional[list] = None,
) -> OverwatcherOutput:
    """Run Overwatcher analysis on evidence bundle.
    
    v3.1: Uses centralized stage_models for configuration.
    
    Args:
        evidence: Evidence bundle to analyze
        llm_call_fn: Async function to call LLM
        job_artifact_root: Root for artifacts
        provider_id: LLM provider (defaults to OVERWATCHER_PROVIDER from env)
        model_id: LLM model (defaults to OVERWATCHER_MODEL from env)
        deep_research_context: Additional context from Strike 2 research
    
    Returns:
        OverwatcherOutput with decision and fix actions
    """
    from app.pot_spec.ledger import (
        emit_verify_pass,
        emit_verify_fail,
        emit_stage_started,
        emit_provider_fallback,
    )
    
    stage_run_id = str(uuid4())
    
    # v3.1: Get config from stage_models (reads from env vars)
    config = _get_overwatcher_config()
    provider_id = provider_id or config.provider
    model_id = model_id or config.model
    max_output_tokens = config.max_output_tokens
    
    logger.info(f"[overwatcher] Using {provider_id}/{model_id} (max_tokens={max_output_tokens})")
    logger.info(f"[overwatcher] Running analysis for chunk {evidence.chunk_id} (strike {evidence.strike_number})")
    
    # Emit stage started
    try:
        emit_stage_started(
            job_artifact_root=job_artifact_root,
            job_id=evidence.job_id,
            stage_id="verification",
            stage_run_id=stage_run_id,
        )
    except Exception as e:
        logger.warning(f"[overwatcher] Failed to emit stage started: {e}")
    
    # Build prompt
    system_prompt, user_prompt = build_overwatcher_prompt(evidence)
    
    # Add deep research context if available
    if deep_research_context:
        user_prompt += f"\n\n## Deep Research Context\n{deep_research_context}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Check input token limit
    input_tokens = (len(system_prompt) + len(user_prompt)) // 4
    if input_tokens > OVERWATCHER_MAX_INPUT_TOKENS:
        logger.warning(f"[overwatcher] Input exceeds limit: {input_tokens} > {OVERWATCHER_MAX_INPUT_TOKENS}")
    
    # Call LLM
    result = None
    used_fallback = False
    
    try:
        result = await llm_call_fn(
            provider_id=provider_id,
            model_id=model_id,
            messages=messages,
            max_tokens=max_output_tokens,
        )
    except Exception as e:
        logger.warning(f"[overwatcher] Primary model failed ({provider_id}/{model_id}): {e}")
        
        # Try fallback if configured
        fallback_config = _get_fallback_config()
        if fallback_config:
            try:
                emit_provider_fallback(
                    job_artifact_root=job_artifact_root,
                    job_id=evidence.job_id,
                    from_provider=provider_id,
                    from_model=model_id,
                    to_provider=fallback_config.provider,
                    to_model=fallback_config.model,
                    reason=str(e),
                )
            except Exception:
                pass
            
            try:
                result = await llm_call_fn(
                    provider_id=fallback_config.provider,
                    model_id=fallback_config.model,
                    messages=messages,
                    max_tokens=fallback_config.max_output_tokens,
                )
                used_fallback = True
            except Exception as e2:
                logger.error(f"[overwatcher] Fallback also failed: {e2}")
    
    if result is None:
        return OverwatcherOutput(
            decision=Decision.FAIL,
            diagnosis="Both primary and fallback Overwatcher models failed",
            blockers=["LLM call failed"],
        )
    
    # Parse output
    raw_output = result.content if hasattr(result, "content") else str(result)
    logger.info(f"[overwatcher] Raw output length: {len(raw_output) if raw_output else 0}")
    
    output = parse_overwatcher_output(raw_output)
    
    # Validate no code in output
    violations = output.validate_no_code()
    if violations:
        logger.warning(f"[overwatcher] Output contains code: {violations}")
        # Still use the output but log the violation
    
    # Emit verification result
    try:
        if output.decision == Decision.PASS:
            emit_verify_pass(
                job_artifact_root=job_artifact_root,
                job_id=evidence.job_id,
                chunk_id=evidence.chunk_id,
                tests_passed=evidence.test_result.passed if evidence.test_result else 0,
                lint_errors=sum(lr.errors for lr in evidence.lint_results),
                type_errors=0,
            )
        else:
            emit_verify_fail(
                job_artifact_root=job_artifact_root,
                job_id=evidence.job_id,
                chunk_id=evidence.chunk_id,
                tests_failed=evidence.test_result.failed if evidence.test_result else 0,
                lint_errors=sum(lr.errors for lr in evidence.lint_results),
                type_errors=0,
                failure_summary=output.diagnosis,
            )
    except Exception as e:
        logger.warning(f"[overwatcher] Failed to emit verification result: {e}")
    
    logger.info(f"[overwatcher] Decision: {output.decision.value}, Confidence: {output.confidence}")
    return output


# =============================================================================
# v2.0 POT Spec Sequential Execution
# =============================================================================


__all__ = [
    # Enums
    "Decision",
    # Data classes
    "FixAction",
    "VerificationStep",
    "OverwatcherOutput",
    # Functions
    "contains_code",
    "build_overwatcher_prompt",
    "parse_overwatcher_output",
    "run_overwatcher",
    "run_pot_spec_execution",  # v2.0: POT spec execution
    # Config (v3.1: now functions, not constants)
    "_get_overwatcher_config",
    "_get_fallback_config",
    "OVERWATCHER_MAX_INPUT_TOKENS",
]
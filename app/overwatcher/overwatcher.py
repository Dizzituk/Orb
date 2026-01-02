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

from app.overwatcher.evidence import EvidenceBundle

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


def _get_overwatcher_config():
    """
    Get Overwatcher configuration from centralized stage_models.
    
    v3.1: NO HARDCODED DEFAULTS. All config comes from env vars via stage_models.
    
    Returns:
        StageConfig with provider, model, max_output_tokens, timeout_seconds
    
    Raises:
        RuntimeError if stage_models unavailable
    """
    if not _STAGE_MODELS_AVAILABLE or get_overwatcher_config is None:
        raise RuntimeError(
            "FATAL: stage_models not available. Ensure app/llm/stage_models.py exists "
            "and OVERWATCHER_PROVIDER + OVERWATCHER_MODEL env vars are set."
        )
    return get_overwatcher_config()


def _get_fallback_config():
    """
    Get fallback configuration for Overwatcher.
    
    Uses OVERWATCHER_FALLBACK stage config if available.
    Returns None if no fallback configured.
    """
    if not _STAGE_MODELS_AVAILABLE or get_stage_config is None:
        return None
    try:
        # Only return fallback if explicitly configured in env
        fallback_provider = os.getenv("OVERWATCHER_FALLBACK_PROVIDER", "").strip()
        fallback_model = os.getenv("OVERWATCHER_FALLBACK_MODEL", "").strip()
        if fallback_provider and fallback_model:
            return get_stage_config("OVERWATCHER_FALLBACK")
        return None
    except Exception:
        return None


# Max input tokens (reasonable constant, not model-specific)
OVERWATCHER_MAX_INPUT_TOKENS = 120_000


# =============================================================================
# Output Schema
# =============================================================================

class Decision(str, Enum):
    """Overwatcher decision outcome."""
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_INFO = "NEEDS_INFO"


@dataclass
class FixAction:
    """A single fix action (no code allowed)."""
    
    order: int
    target_file: str
    action_type: str  # "add_function" | "modify_function" | "fix_import" | etc.
    description: str  # What to do (high-level, no code)
    rationale: str  # Why this fixes the issue
    
    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "target_file": self.target_file,
            "action_type": self.action_type,
            "description": self.description,
            "rationale": self.rationale,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FixAction":
        return cls(
            order=data.get("order", 0),
            target_file=data.get("target_file", ""),
            action_type=data.get("action_type", ""),
            description=data.get("description", ""),
            rationale=data.get("rationale", ""),
        )


@dataclass
class VerificationStep:
    """A verification step with expected outcome."""
    
    command: str
    expected_outcome: str
    timeout_seconds: int = 60
    
    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "expected_outcome": self.expected_outcome,
            "timeout_seconds": self.timeout_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VerificationStep":
        return cls(
            command=data.get("command", ""),
            expected_outcome=data.get("expected_outcome", ""),
            timeout_seconds=data.get("timeout_seconds", 60),
        )


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

CODE_PATTERNS = [
    r"```\w*\n",  # Code fences
    r"^\s{4,}def\s+\w+",  # Python function definition
    r"^\s{4,}class\s+\w+",  # Python class definition
    r"^\s{4,}import\s+",  # Import statement (indented = in code block)
    r"^\s{4,}from\s+\w+\s+import",  # From import (indented)
    r"^\+\s*def\s+",  # Diff with function
    r"^\+\s*class\s+",  # Diff with class
]


def contains_code(text: str) -> bool:
    """Check if text contains code patterns.
    
    Used to enforce Overwatcher output contract.
    """
    if not text:
        return False
    
    for pattern in CODE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    return False


# =============================================================================
# Prompts
# =============================================================================

OVERWATCHER_SYSTEM = """You are an expert software engineering supervisor (Overwatcher).

YOUR ROLE:
- Diagnose failures and define fix actions
- Enforce constraints and spec compliance
- Decide PASS, FAIL, or NEEDS_INFO

CRITICAL RULES - YOU MUST FOLLOW:
1. NEVER write code, patches, diffs, or file contents
2. NEVER include code blocks in your response
3. Only output JSON with decision, diagnosis, and fix actions
4. Fix actions describe WHAT to do, not HOW (no code)

You must respond with ONLY a valid JSON object matching this schema:
{{
  "decision": "PASS" | "FAIL" | "NEEDS_INFO",
  "diagnosis": "Root cause hypothesis (1-2 sentences)",
  "fix_actions": [
    {{
      "order": 1,
      "target_file": "path/to/file.py",
      "action_type": "add_function|modify_function|fix_import|add_error_handling|etc",
      "description": "High-level description of what to change (NO CODE)",
      "rationale": "Why this fixes the issue"
    }}
  ],
  "constraints": ["List of invariants to respect"],
  "verification": [
    {{
      "command": "pytest tests/test_foo.py -v",
      "expected_outcome": "all tests pass",
      "timeout_seconds": 60
    }}
  ],
  "blockers": ["Issues that must be fixed"],
  "nonblockers": ["Issues that can be deferred"],
  "confidence": 0.0-1.0,
  "needs_deep_research": true|false
}}

SPEC HASH LOCK: {spec_hash}
You must preserve this spec hash. Do not suggest changes that would alter the spec."""

OVERWATCHER_USER = """Analyze this evidence bundle and provide your decision.

{evidence_text}

Remember:
- Output ONLY JSON
- NO code in fix_actions descriptions
- decision must be PASS, FAIL, or NEEDS_INFO
- Strike {strike_number}/3 - {strike_hint}"""


def build_overwatcher_prompt(
    evidence: EvidenceBundle,
) -> tuple[str, str]:
    """Build Overwatcher prompts.
    
    Returns (system_prompt, user_prompt)
    """
    system = OVERWATCHER_SYSTEM.format(
        spec_hash=evidence.spec_hash,
    )
    
    strike_hint = ""
    if evidence.strike_number == 1:
        strike_hint = "Consider needs_deep_research if stuck"
    elif evidence.strike_number == 2:
        strike_hint = "Deep research available if needed"
    else:
        strike_hint = "Final strike - be thorough"
    
    user = OVERWATCHER_USER.format(
        evidence_text=evidence.to_prompt_text(),
        strike_number=evidence.strike_number,
        strike_hint=strike_hint,
    )
    
    return system, user


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
    # Config (v3.1: now functions, not constants)
    "_get_overwatcher_config",
    "_get_fallback_config",
    "OVERWATCHER_MAX_INPUT_TOKENS",
]
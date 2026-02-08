# FILE: app/llm/pipeline/critique.py
"""Critique pipeline for high-stakes jobs.

Block 5 of the PoT (Proof of Thought) system:
- JSON critique with structured blocking/non-blocking issues
- Spec-anchored critique (verifies architecture against spec)
- Legacy prose-based critique for backward compatibility
- v1.2: Blocker filtering - only approved blocker types can block
- v1.3: DETERMINISTIC spec-compliance check (catches stack/scope/platform mismatch)
- v1.4: Uses explicit implementation_stack field from spec (stack_locked anchoring)
- v1.6: GROUNDED CRITIQUE - POT spec markdown as source of truth
- v1.7: GROUNDING VALIDATION - catch hallucinated constraints in critique output
- v1.9: SECTION AUTHORITY - downgrade blockers citing LLM-suggestion sections
- v2.1: SCOPE CREEP DETECTION - endpoint drift + excluded feature enforcement

v2.2 (2026-02-07): PLATFORM DETECTION FALSE-POSITIVE FIX
- _check_platform_targeted_not_excluded() now requires STRUCTURAL platform signals
- Incidental keyword mentions ("mobile browser", "mobile audio", "mobile-friendly") no longer trigger
- New Layer 0: Tech stack override — if arch_stack contains Electron, platform is Desktop regardless
- New Layer 4: Structural signal validation — affirmative mentions must appear in declarative contexts
  (e.g., "targets: mobile", "platform: mobile", "mobile app", "for Android/iOS")
- Incidental contexts added: compound words (mobile-friendly, mobile-first, mobile-responsive),
  API/browser contexts (mobile browser, mobile audio, MediaRecorder), compatibility notes
- Fixes: Desktop/Electron architecture blocked by incidental "mobile" mentions in component descriptions

v2.1 (2026-02-06): SCOPE CREEP DETECTION - endpoint/feature drift
- Added run_scope_creep_check() - deterministic endpoint and feature comparison
- Extracts HTTP endpoint patterns (GET/POST/etc + /path) from spec and architecture
- Check 1: Extra endpoints in architecture not listed in spec (scope creep)
- Check 2: Spec endpoints missing from architecture (possible rename/drift)
- Check 3: Excluded features (wake_word, tts, websocket) appearing in architecture
- Wired into call_json_critic() after deterministic spec-compliance check
- New blocker categories: scope_creep, endpoint_rename, excluded_feature
- See voice STT integration analysis for root cause context
v1.11 (2026-02-07): CRITIQUE META-COMMENTARY EXCLUSION - feedback loop fix
- Broader future-reference pattern: future \w+ (catches 'future mobile/web clients')
- Added critique rebuttal inline patterns: **Reviewer Claim**, **DECISION REJECT**, false positive
- Added exclusion section headers: Reviewer Suggestion, Critique Rebuttal, Spec-Compliance
- Fixes feedback loop: critique blocks -> Sonnet rebuts quoting error -> critique re-scans rebuttals
- Fix for cp-964426dc


v1.9 (2026-02-05): SECTION AUTHORITY VALIDATION - LLM suggestion defense
- validate_section_authority() runs AFTER grounding validation
- Downgrade blocking issues whose spec_ref cites LLM-generated sections
  (Files to Modify, Implementation Steps, Reference Files, etc.)
- These sections are implementation SUGGESTIONS, not user requirements
- The architecture may choose alternative approaches without being blocked
- Fixes critique deadlock where Gemini raised blockers on GPT-suggested files
- Defense-in-depth layer on top of v1.3 prompt-level fix in critique_schemas.py
- See critique-pipeline-fix-jobspec.md for full root cause analysis

v1.11 (2026-02-07): CRITIQUE META-COMMENTARY EXCLUSION - feedback loop fix
- Expanded _EXCLUSION_CONTEXT_PATTERNS with broader "future \w+" pattern and critique rebuttal patterns
- Added inline exclusion patterns for **Reviewer Claim/Suggestion**, **DECISION**: **REJECT**,
  "false positive", and "allowing future <word>" (catches "allowing future mobile/web clients")
- Added exclusion section headers for Reviewer Suggestion / Critique Rebuttal sections
- Fixes feedback loop where critique blocks architecture, Sonnet rebuts with meta-commentary
  quoting the error message, critique re-scans and finds "mobile/Android/iOS" in the rebuttal,
  loop repeats 3 times exhausting iterations
- Fix for cp-964426dc: 3 wasted critique iterations on false positive feedback loop

v1.8 (2026-02-05): CONTEXT-AWARE PLATFORM CHECK - false positive fix
- _check_platform_targeted_not_excluded() replaces naive substring check in Check 3
- Platform keywords (mobile/android/ios) now checked for exclusion context
- Lines containing "out of scope", "phase 2+", "not in this phase", etc. are ignored
- Prevents false blocker when architecture EXCLUDES mobile (e.g., "Out of Scope: Mobile")
- Also handles revision meta-commentary ("critique claims", "rejecting this")
- Fix for cp-1846c85e: 3 wasted critique iterations + 2 Opus revisions on false positive

v1.7 (2026-02-04): GROUNDING VALIDATION - hallucinated constraint defense
- validate_spec_ref_grounding() runs AFTER LLM critique and blocker filtering
- Checks each blocking issue's spec_ref against actual spec content
- Fabrication pattern detection (cloud_services: false, local_only, etc.)
- Field reference validation (constraints.FIELD_NAME must exist in spec)
- Issues citing non-existent constraints are downgraded to non-blocking
- Defense-in-depth layer on top of v1.6's grounded system prompt

v1.6 (2026-02-02): GROUNDED CRITIQUE - POT spec as source of truth
- call_json_critic() now accepts spec_markdown parameter
- Full POT spec with grounded evidence injected into critique prompt
- Critique judges ONLY against what's in the spec
- If user requested "OpenAI API", critique does NOT flag it as violation
- Philosophy: "Ground and trust" - spec IS the contract
- Updated system message to emphasize spec as authoritative

v1.5 (2026-01-22): CRITICAL FIX - Phantom Constraint Bug
- Fixed _detect_stack_from_text() to use word-boundary matching
- Prevents false positives like "go" matching "going", "goal"
- Changed "go" keyword to "golang" to avoid ambiguity
- Added word boundary matching for "rust", "java", "vue"
- See PHANTOM_CONSTRAINT_BUG_FIX.md for full details

v1.4 (2026-01-22): Implementation Stack Anchoring
- _extract_spec_constraints() now checks implementation_stack field FIRST
- If implementation_stack.stack_locked == True, stack mismatch is a HARD blocker
- Falls back to heuristic detection from goal/summary if no explicit stack
- See pot_spec/schemas.py ImplementationStack for field structure

v1.3 (2026-01-22): CRITICAL FIX - Spec-Anchored Architecture Validation
- Added run_deterministic_spec_compliance_check() - runs BEFORE LLM critique
- Catches stack mismatch (e.g., Python discussed but TS/JS proposed)
- Catches scope inflation (minimal spec but full-product architecture)
- Catches platform mismatch (Desktop spec but Mobile architecture)
- Wired into call_json_critic() - blocks BEFORE LLM call if violations found
- See CRITICAL_PIPELINE_FAILURE_REPORT.md for context on why this is needed

v1.2 (2026-01):
- Added filter_blocking_issues() - enforces approved blocker types only
- Evidence-backed blocking: requires BOTH spec_ref AND arch_ref
- Blocker filtering wired into call_json_critic()

v1.1 (2026-01):
- Uses stage_models for provider/model configuration (env-driven)
- No more hardcoded provider - CRITIQUE_PROVIDER/CRITIQUE_MODEL from env

v1.0 (2025-12):
- Extracted from high_stakes.py for better maintainability
- Spec verification protocol in system messages
- Debug logging for pipeline visibility
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import yaml

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
# Block 5: Blocker Filtering (v1.2)
# =============================================================================

def filter_blocking_issues(
    issues: List[CritiqueIssue],
    require_evidence: bool = True,
) -> Tuple[List[CritiqueIssue], List[CritiqueIssue]]:
    """
    Filter blocking issues to only approved blocker types.
    
    v1.2: Implements the "Critique Contract" rules:
    1. Only approved blocker types can block (security, correctness, spec_mismatch, etc.)
    2. Blocking issues MUST have BOTH spec_ref AND arch_ref (evidence requirement)
    3. If category unknown BUT has perfect evidence AND description mentions drift/hallucination → allow
    
    Args:
        issues: List of CritiqueIssue objects marked as blocking
        require_evidence: If True, blocking issues need BOTH spec_ref AND arch_ref
    
    Returns:
        (real_blocking, downgraded_to_non_blocking)
    """
    real_blocking: List[CritiqueIssue] = []
    downgraded: List[CritiqueIssue] = []
    
    # Keywords that indicate a real blocker even if category is weird
    drift_keywords = [
        "contradict", "contradiction", "drift", "hallucination", "invented",
        "does not exist", "doesn't exist", "mismatch", "violate", "violation",
        "missing required", "spec says", "spec requires",
    ]
    
    for issue in issues:
        # Normalize category for matching
        category = (issue.category or "").lower().replace(" ", "_").replace("-", "_")
        
        # Check evidence: BOTH spec_ref AND arch_ref required (AND rule)
        has_spec_ref = bool(issue.spec_ref and issue.spec_ref.strip())
        has_arch_ref = bool(issue.arch_ref and issue.arch_ref.strip())
        has_full_evidence = has_spec_ref and has_arch_ref
        
        # Check if category is in approved blocker types
        is_approved_category = category in APPROVED_ARCHITECTURE_BLOCKER_TYPES
        
        # Check for drift keywords in description (fallback for unknown categories)
        description_lower = (issue.description or "").lower()
        has_drift_keywords = any(kw in description_lower for kw in drift_keywords)
        
        # Decision logic:
        # 1. If approved category AND has evidence → real blocker
        # 2. If unknown category BUT has full evidence AND drift keywords → real blocker
        # 3. Otherwise → downgrade
        
        should_block = False
        reason = ""
        
        if is_approved_category:
            if require_evidence and not has_full_evidence:
                # Approved category but missing evidence → downgrade
                reason = f"category={category} approved, but missing evidence (spec_ref={has_spec_ref}, arch_ref={has_arch_ref})"
            else:
                # Approved category with evidence → real blocker
                should_block = True
                reason = f"category={category} approved, evidence present"
        else:
            # Unknown category - check for drift keywords AND full evidence
            if has_full_evidence and has_drift_keywords:
                should_block = True
                reason = f"category={category} unknown but has drift keywords and full evidence"
            else:
                reason = f"category={category} not approved, no drift keywords or missing evidence"
        
        if should_block:
            real_blocking.append(issue)
            logger.debug(
                "[critique] KEPT blocker %s: %s",
                issue.id, reason
            )
        else:
            # Downgrade to non-blocking
            issue.severity = "non_blocking"
            downgraded.append(issue)
            logger.info(
                "[critique] DOWNGRADED issue %s to non_blocking: %s",
                issue.id, reason
            )
    
    # Summary log
    if downgraded:
        print(f"[DEBUG] [critique] Filtered blockers: kept={len(real_blocking)}, downgraded={len(downgraded)}")
        logger.info(
            "[critique] Blocker filtering: %d kept, %d downgraded",
            len(real_blocking), len(downgraded)
        )
    
    return real_blocking, downgraded


# =============================================================================
# Block 5b: SPEC-REF GROUNDING VALIDATION (v1.7 - HALLUCINATION DEFENSE)
# =============================================================================
#
# v1.7 (2026-02-04): Deterministic post-processing to catch hallucinated constraints.
# After the LLM critique runs, we verify that each blocking issue's spec_ref
# actually refers to something that exists in the spec. If a critic cites
# a constraint like "cloud_services: false" that isn't in the spec,
# the issue is downgraded to non-blocking.
#
# This is a DEFENSE-IN-DEPTH layer on top of v1.6's grounded system prompt.
#

def validate_spec_ref_grounding(
    issues: List[CritiqueIssue],
    spec_markdown: Optional[str] = None,
    spec_json: Optional[str] = None,
) -> Tuple[List[CritiqueIssue], List[CritiqueIssue]]:
    """
    v1.7: Validate that blocking issues cite constraints that ACTUALLY EXIST in the spec.
    
    Catches LLM hallucinations like:
    - Citing "cloud_services: false" when spec has no such field
    - Citing "local_only" constraint when spec doesn't mention it
    - Inventing restrictions not present in the spec
    
    Strategy:
    - Extract key terms from each issue's spec_ref
    - Check if those terms appear in the spec markdown or JSON
    - If spec_ref cites something not in the spec, downgrade to non-blocking
    
    Args:
        issues: List of blocking CritiqueIssue objects to validate
        spec_markdown: The POT spec markdown (primary source of truth)
        spec_json: The spec JSON string (secondary source)
    
    Returns:
        (validated_blocking, downgraded_to_non_blocking)
    """
    if not issues:
        return [], []
    
    # Build the spec text corpus to search against
    spec_corpus = ""
    if spec_markdown:
        spec_corpus += spec_markdown.lower()
    if spec_json:
        try:
            if isinstance(spec_json, str):
                spec_corpus += " " + spec_json.lower()
            else:
                spec_corpus += " " + json.dumps(spec_json).lower()
        except Exception:
            pass
    
    if not spec_corpus.strip():
        # No spec to validate against - can't check, so keep all issues
        print("[DEBUG] [critique] v1.7 No spec corpus available for grounding validation")
        return issues, []
    
    validated: List[CritiqueIssue] = []
    downgraded: List[CritiqueIssue] = []
    
    # Patterns that indicate a hallucinated/invented constraint
    # These are field:value patterns the critic might fabricate
    FABRICATION_PATTERNS = [
        # "cloud_services: false/true" - invented field
        r'cloud_services\s*[:=]\s*(false|true|no|yes)',
        # "local_only" - invented constraint
        r'local[_\s-]only\s*[:=]\s*(true|yes|required)',
        # "requires_local" - invented constraint
        r'requires?[_\s-]local',
        # "no_external_apis" - invented constraint
        r'no[_\s-]external[_\s-]apis?',
        # "offline_only" - invented constraint
        r'offline[_\s-]only',
    ]
    
    for issue in issues:
        spec_ref = (getattr(issue, 'spec_ref', '') or '').lower()
        description = (getattr(issue, 'description', '') or '').lower()
        combined_text = f"{spec_ref} {description}"
        
        is_fabricated = False
        fabrication_reason = ""
        
        # Check 1: Does the spec_ref cite a specific field:value that doesn't exist?
        for pattern in FABRICATION_PATTERNS:
            match = _re.search(pattern, combined_text)
            if match:
                cited_text = match.group(0)
                # Check if this exact term appears anywhere in the spec
                if cited_text not in spec_corpus:
                    is_fabricated = True
                    fabrication_reason = f"Cited '{cited_text}' not found in spec"
                    break
        
        # Check 2: If spec_ref mentions a constraint field, verify it exists in spec
        if not is_fabricated and spec_ref:
            # Extract quoted or field-like references from spec_ref
            # e.g., "constraints.cloud_services" or "'local_only' requirement"
            field_refs = _re.findall(r'constraints?\.([a-z_]+)', spec_ref)
            field_refs += _re.findall(r"'([a-z_]+)'", spec_ref)
            field_refs += _re.findall(r'"([a-z_]+)"', spec_ref)
            
            for field in field_refs:
                # Skip common generic fields that are always valid
                if field in ('integrations', 'platform', 'scope', 'goal', 'summary',
                             'stack', 'language', 'framework', 'requirements'):
                    continue
                # Check if this field exists in the spec corpus
                if field not in spec_corpus:
                    is_fabricated = True
                    fabrication_reason = f"Referenced field '{field}' not found in spec"
                    break
        
        if is_fabricated:
            issue.severity = "non_blocking"
            downgraded.append(issue)
            print(f"[DEBUG] [critique] v1.7 GROUNDING FAIL: Issue {getattr(issue, 'id', 'N/A')} "
                  f"downgraded - {fabrication_reason}")
            logger.info(
                "[critique] v1.7 Grounding validation failed for %s: %s",
                getattr(issue, 'id', 'N/A'), fabrication_reason
            )
        else:
            validated.append(issue)
    
    if downgraded:
        print(f"[DEBUG] [critique] v1.7 Grounding validation: {len(validated)} kept, "
              f"{len(downgraded)} downgraded (hallucinated constraints)")
        logger.info(
            "[critique] v1.7 Grounding validation: %d kept, %d downgraded",
            len(validated), len(downgraded)
        )
    
    return validated, downgraded


# =============================================================================
# Block 5c: SECTION AUTHORITY VALIDATION (v1.9 - CRITIQUE DEADLOCK FIX)
# =============================================================================
#
# v1.9 (2026-02-05): Deterministic post-processing to catch blocking issues
# that cite LLM-generated suggestion sections rather than user requirements.
#
# The POT spec markdown contains BOTH user requirements (Goal, Constraints,
# Scope) AND LLM-generated suggestions (Files to Modify, Implementation Steps,
# Reference Files). The critique should only BLOCK on user requirements.
#
# Issues citing suggestion sections are downgraded to non-blocking, because
# the architecture is free to choose alternative approaches.
#
# This is a DEFENSE-IN-DEPTH layer on top of the prompt-level fix in
# critique_schemas.py v1.3 which tells the critique about section authority.
#

# Section names that are LLM-generated suggestions, NOT user requirements.
# If a blocking issue's spec_ref contains any of these (case-insensitive),
# the issue is downgraded to non-blocking.
_LLM_SUGGESTION_SECTIONS = {
    "files to modify",
    "reference files",
    "implementation steps",
    "new files to create",
    "patterns to follow",
    "patterns extracted",
    "existing patterns",
    "integration points",
    "llm architecture analysis",
    "llm analysis",
    "evidence summary",
}


def validate_section_authority(
    issues: List[CritiqueIssue],
) -> Tuple[List[CritiqueIssue], List[CritiqueIssue]]:
    """
    v1.9: Downgrade blocking issues that cite LLM-suggestion sections.
    
    These sections (Files to Modify, Implementation Steps, Reference Files, etc.)
    are implementation guidance generated by LLM analysis during SpecGate, NOT
    user requirements. The architecture is free to choose alternative approaches.
    
    Root cause: simple_create.py generates these sections from GPT analysis,
    but the critique prompt treats the ENTIRE spec markdown as a binding contract.
    This causes deadlocks where critique raises blockers the revision correctly
    rejects, burning all iteration budget.
    
    Args:
        issues: List of blocking CritiqueIssue objects to validate
    
    Returns:
        (validated_blocking, downgraded_to_non_blocking)
    """
    if not issues:
        return [], []
    
    validated: List[CritiqueIssue] = []
    downgraded: List[CritiqueIssue] = []
    
    for issue in issues:
        spec_ref_lower = (getattr(issue, 'spec_ref', '') or '').lower()
        
        # Check if the spec_ref cites any LLM-suggestion section
        cites_suggestion_section = any(
            section in spec_ref_lower
            for section in _LLM_SUGGESTION_SECTIONS
        )
        
        if cites_suggestion_section:
            issue.severity = "non_blocking"
            downgraded.append(issue)
            print(
                f"[DEBUG] [critique] v1.9 SECTION_AUTHORITY: Downgraded {getattr(issue, 'id', 'N/A')} "
                f"— cites LLM suggestion section: {issue.spec_ref}"
            )
            logger.info(
                "[critique] v1.9 Section authority: downgraded %s — spec_ref '%s' cites LLM suggestion",
                getattr(issue, 'id', 'N/A'), issue.spec_ref
            )
        else:
            validated.append(issue)
    
    if downgraded:
        print(
            f"[DEBUG] [critique] v1.9 Section authority filter: "
            f"{len(validated)} kept, {len(downgraded)} downgraded (LLM suggestion sections)"
        )
        logger.info(
            "[critique] v1.9 Section authority: %d kept, %d downgraded",
            len(validated), len(downgraded)
        )
    
    return validated, downgraded


# =============================================================================
# Block 5d: EVIDENCE RESOLUTION CHECK (v2.0 - Evidence-or-Request Contract)
# =============================================================================

VALID_RESOLUTIONS = {"CITED", "DECISION", "HUMAN_REQUIRED"}


def extract_critical_claims(arch_content: str) -> Optional[list]:
    """Extract CRITICAL_CLAIMS register from architecture output.

    CRITICAL_CLAIMS must be the last structured block.
    Parses from rfind() for robustness against nested YAML,
    blank lines, and field reordering.
    """
    idx = arch_content.rfind("\nCRITICAL_CLAIMS:")
    if idx == -1:
        if arch_content.startswith("CRITICAL_CLAIMS:"):
            idx = 0
        else:
            return None

    yaml_text = arch_content[idx:].strip()

    try:
        parsed = yaml.safe_load(yaml_text)
        if not isinstance(parsed, dict):
            return None
        claims = parsed.get("CRITICAL_CLAIMS")
        if claims is None:
            return None
        if not isinstance(claims, list):
            return None
        return claims
    except yaml.YAMLError:
        return None


def run_evidence_resolution_check(
    arch_content: str,
) -> List[CritiqueIssue]:
    """
    v2.0: Deterministic check — validate CRITICAL_CLAIMS register.

    Replaces regex heuristics with structured register validation.

    Rules:
    - Every claim must have resolution in {CITED, DECISION, HUMAN_REQUIRED}
    - CITED claims must have at least one evidence entry
    - DECISION claims must reference a decision_id
    - Missing register entirely = non-blocking warning (transition period,
      upgrade to blocking once all stages emit it)

    IMPORTANT: Only call this on FINAL stage output — i.e. when there are
    no pending EVIDENCE_REQUESTs. The orchestrator should check
    parse_evidence_requests() returns empty before calling this.

    Wire alongside run_deterministic_spec_compliance_check() in call_json_critic().
    """
    issues: List[CritiqueIssue] = []

    claims = extract_critical_claims(arch_content)

    if claims is None:
        issues.append(CritiqueIssue(
            id="CLAIMS-MISSING",
            spec_ref="Evidence-or-Request Contract",
            arch_ref="Architecture output",
            category="missing_claims_register",
            severity="non_blocking",  # Upgrade to blocking once all stages emit it
            description="No CRITICAL_CLAIMS register found in architecture output",
            fix_suggestion="Add CRITICAL_CLAIMS block as the last section listing all implementation-affecting claims with resolution status",
        ))
        return issues

    for claim in claims:
        claim_id = claim.get("id", "UNKNOWN")
        resolution = claim.get("resolution", "MISSING")
        claim_text = claim.get("claim", "")

        if resolution not in VALID_RESOLUTIONS:
            issues.append(CritiqueIssue(
                id=f"CLAIMS-{claim_id}",
                spec_ref="Evidence-or-Request Contract",
                arch_ref=f"CRITICAL_CLAIMS.{claim_id}: {claim_text[:80]}",
                category="unresolved_critical",
                severity="blocking",
                description=f"Critical claim '{claim_id}' has invalid resolution: '{resolution}'. Must be CITED, DECISION, or HUMAN_REQUIRED.",
                fix_suggestion="Resolve this claim with evidence, an explicit decision, or flag for human input",
            ))
            continue

        if resolution == "CITED":
            evidence = claim.get("evidence", [])
            if not evidence:
                issues.append(CritiqueIssue(
                    id=f"CLAIMS-{claim_id}",
                    spec_ref="Evidence-or-Request Contract",
                    arch_ref=f"CRITICAL_CLAIMS.{claim_id}: {claim_text[:80]}",
                    category="unresolved_critical",
                    severity="blocking",
                    description=f"Critical claim '{claim_id}' marked CITED but has no evidence entries",
                    fix_suggestion="Add evidence entries with file paths and line ranges",
                ))

        elif resolution == "DECISION":
            decision_id = claim.get("decision_id")
            if not decision_id:
                issues.append(CritiqueIssue(
                    id=f"CLAIMS-{claim_id}",
                    spec_ref="Evidence-or-Request Contract",
                    arch_ref=f"CRITICAL_CLAIMS.{claim_id}: {claim_text[:80]}",
                    category="unresolved_critical",
                    severity="blocking",
                    description=f"Critical claim '{claim_id}' marked DECISION but missing decision_id reference",
                    fix_suggestion="Add decision_id referencing a DECISION block with rationale and revisit_if",
                ))

    return issues


import re as _re  # Used by Block 5e and Block 5 deterministic checks

# =============================================================================
# Block 5e: SCOPE CREEP DETECTION (v2.1 - Endpoint/Feature Drift)
# =============================================================================
#
# v2.1 (2026-02-06): Deterministic check for scope creep.
# Extracts HTTP endpoint patterns from spec markdown and architecture,
# then flags:
#   - Endpoints in architecture NOT listed in spec (scope creep)
#   - Endpoint name changes (spec says /voice/status, arch says /voice/health)
#   - Features explicitly excluded by spec constraints appearing in arch
#
# Wired into call_json_critic() alongside run_deterministic_spec_compliance_check().
#

_ENDPOINT_PATTERN = _re.compile(
    r'(?:^|\s)'
    r'(GET|POST|PUT|PATCH|DELETE|WS|WSS|WebSocket)'
    r'\s+'
    r'(/[a-zA-Z0-9/_{}\-]+)',
    _re.IGNORECASE | _re.MULTILINE
)

# Features that spec constraints might explicitly exclude
_EXCLUDED_FEATURE_KEYWORDS = {
    'wake_word': ['wake_word', 'wake word', 'hotword', 'wakeword'],
    'tts': ['text-to-speech', 'text_to_speech', 'tts', 'piper'],
    'websocket': ['websocket', 'ws /voice', 'ws /audio', '/stream'],
}


def _extract_endpoints(text: str) -> List[Tuple[str, str]]:
    """Extract (METHOD, /path) pairs from text."""
    if not text:
        return []
    results = []
    for match in _ENDPOINT_PATTERN.finditer(text):
        method = match.group(1).upper()
        path = match.group(2).rstrip('.')
        # Normalize WebSocket variants
        if method in ('WS', 'WSS', 'WEBSOCKET'):
            method = 'WS'
        results.append((method, path.lower()))
    # Dedupe preserving order
    seen = set()
    unique = []
    for ep in results:
        if ep not in seen:
            seen.add(ep)
            unique.append(ep)
    return unique


def _build_exclusion_zone_set(lines: List[str]) -> set:
    """Build a set of line indices that are inside exclusion zones.
    
    Exclusion zones are sections like "Out of Scope", "Not in this phase",
    "Future Work", etc. where mentioning a feature does NOT mean the
    architecture is implementing it.
    
    Returns set of 0-based line indices.
    """
    exclusion_headers = [
        'out of scope', 'not in scope', 'excluded', 'future work',
        'phase 2', 'phase 3', 'not in this phase', 'deferred',
        'not implemented', 'out-of-scope', 'non-goals',
    ]
    zones: set = set()
    in_exclusion = False
    exclusion_depth = 0  # heading level that started the zone
    
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        
        # Check if this line is a heading
        heading_level = 0
        if stripped.startswith('#'):
            heading_level = len(stripped) - len(stripped.lstrip('#'))
        
        if heading_level > 0:
            heading_text = stripped.lstrip('#').strip()
            # Check if this heading starts an exclusion zone
            if any(eh in heading_text for eh in exclusion_headers):
                in_exclusion = True
                exclusion_depth = heading_level
                zones.add(i)
                continue
            # Check if a new heading at same or higher level ends the zone
            if in_exclusion and heading_level <= exclusion_depth:
                in_exclusion = False
                exclusion_depth = 0
        
        if in_exclusion:
            zones.add(i)
    
    return zones


def run_scope_creep_check(
    arch_content: str,
    spec_markdown: Optional[str] = None,
    spec_json: Optional[str] = None,
) -> List[CritiqueIssue]:
    """
    v2.1: Deterministic scope creep detection.
    
    Compares endpoints and features in architecture against spec.
    Flags:
    - Extra endpoints not in spec
    - Renamed endpoints (path differs from spec)
    - Excluded features appearing in architecture
    
    Only runs if spec_markdown contains endpoint patterns (otherwise no baseline).
    """
    issues: List[CritiqueIssue] = []
    
    if not spec_markdown or not arch_content:
        return issues
    
    spec_endpoints = _extract_endpoints(spec_markdown)
    arch_endpoints = _extract_endpoints(arch_content)
    
    if not spec_endpoints:
        # Spec doesn't list specific endpoints — can't check drift
        return issues
    
    spec_paths = {path for _, path in spec_endpoints}
    arch_paths = {path for _, path in arch_endpoints}
    
    # Check 1: Endpoints in architecture not in spec
    extra_paths = arch_paths - spec_paths
    if extra_paths:
        # Filter out paths that are just minor variants (e.g., /voice/transcribe vs /voice/transcribe/b64)
        truly_extra = []
        for ep in extra_paths:
            # Check it's not a sub-path of a spec endpoint
            is_subpath = any(ep.startswith(sp + '/') for sp in spec_paths)
            if not is_subpath:
                truly_extra.append(ep)
        
        if truly_extra:
            issue_id = len(issues) + 1
            issues.append(CritiqueIssue(
                id=f"SCOPE-CREEP-{issue_id:03d}",
                spec_ref=f"Spec endpoints: {[f'{m} {p}' for m, p in spec_endpoints]}",
                arch_ref=f"Extra architecture endpoints: {truly_extra}",
                category="scope_creep",
                severity="blocking",
                description=(
                    f"SCOPE CREEP: Architecture adds {len(truly_extra)} endpoint(s) not in spec: "
                    f"{truly_extra}. Spec only lists: {[f'{m} {p}' for m, p in spec_endpoints]}. "
                    f"Remove extra endpoints or get spec approval first."
                ),
                fix_suggestion=(
                    f"Remove these endpoints from the architecture: {truly_extra}. "
                    f"Only implement what the spec lists."
                ),
            ))
            print(f"[DEBUG] [critique] v2.1 SCOPE CREEP: {len(truly_extra)} extra endpoint(s): {truly_extra}")
    
    # Check 2: Spec endpoints missing from architecture (possible rename)
    missing_paths = spec_paths - arch_paths
    if missing_paths:
        issue_id = len(issues) + 1
        issues.append(CritiqueIssue(
            id=f"SCOPE-CREEP-{issue_id:03d}",
            spec_ref=f"Missing spec endpoints: {list(missing_paths)}",
            arch_ref=f"Architecture endpoints: {[f'{m} {p}' for m, p in arch_endpoints]}",
            category="endpoint_rename",
            severity="blocking",
            description=(
                f"ENDPOINT MISMATCH: Spec requires {list(missing_paths)} but architecture "
                f"does not include them. The architecture may have renamed these endpoints. "
                f"Use the exact endpoint paths from the spec."
            ),
            fix_suggestion=(
                f"Ensure these spec endpoints are present with the exact paths: {list(missing_paths)}"
            ),
        ))
        print(f"[DEBUG] [critique] v2.1 ENDPOINT MISMATCH: Missing {list(missing_paths)}")
    
    # Check 3: Excluded features appearing in architecture
    spec_lower = spec_markdown.lower()
    arch_lower = arch_content.lower()
    
    for feature, keywords in _EXCLUDED_FEATURE_KEYWORDS.items():
        # Check if spec explicitly excludes this feature
        spec_excludes = any(
            _re.search(rf'(?:do\s+not|don.t|no)\s+(?:implement)?.*{_re.escape(kw)}', spec_lower)
            for kw in keywords
        )
        if not spec_excludes:
            continue
        
        # Check if architecture includes this feature (outside of "out of scope" sections)
        for kw in keywords:
            if kw in arch_lower:
                # Quick check: is it in an exclusion context?
                lines = arch_content.splitlines()
                exclusion_zones = _build_exclusion_zone_set(lines)
                for i, line in enumerate(lines):
                    if kw in line.lower() and i not in exclusion_zones:
                        # Found feature in non-excluded context
                        issue_id = len(issues) + 1
                        issues.append(CritiqueIssue(
                            id=f"SCOPE-CREEP-{issue_id:03d}",
                            spec_ref=f"Spec excludes: {feature}",
                            arch_ref=f"Architecture includes '{kw}' at line {i+1}",
                            category="excluded_feature",
                            severity="blocking",
                            description=(
                                f"EXCLUDED FEATURE: Spec explicitly says do not implement '{feature}', "
                                f"but architecture includes '{kw}' in a non-exclusion context (line {i+1})."
                            ),
                            fix_suggestion=f"Remove all references to '{feature}' from the architecture.",
                        ))
                        print(f"[DEBUG] [critique] v2.1 EXCLUDED FEATURE: '{feature}' found at line {i+1}")
                        break  # One issue per feature is enough
                break  # One keyword match per feature is enough
    
    return issues


# =============================================================================
# Block 5: DETERMINISTIC Spec-Compliance Check (v1.3 - CRITICAL FIX)
# =============================================================================
#
# This function runs BEFORE the LLM critique and catches obvious spec violations
# deterministically. It is the primary line of defense against architecture drift.
#
# See: CRITICAL_PIPELINE_FAILURE_REPORT.md (2026-01-22) for why this is needed.
#

# Stack detection keywords - maps keywords to canonical stack names
# v1.1 FIX: Use word-boundary aware patterns to prevent false matches
# e.g., "go" should not match "going", "goal", etc.
_STACK_KEYWORDS = {
    # Python ecosystem
    "python": "Python",
    "pygame": "Python+Pygame",
    "tkinter": "Python+Tkinter",
    "pyqt": "Python+PyQt",
    "flask": "Python+Flask",
    "fastapi": "Python+FastAPI",
    "django": "Python+Django",
    
    # JavaScript/TypeScript ecosystem
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "react": "TypeScript/React",
    "electron": "TypeScript/Electron",
    "node.js": "Node.js",  # v1.1: More specific to avoid false matches
    "nodejs": "Node.js",   # v1.1: Alternative spelling
    "next.js": "TypeScript/Next.js",
    "vue": "TypeScript/Vue",
    
    # Other stacks - NOTE: Some keywords need word-boundary checking
    "rust": "Rust",
    "golang": "Go",        # v1.1: Prefer "golang" to avoid "going/goal" false matches
    "c++": "C++",
    "c#": "C#",
    "java": "Java",         # v1.1: Use word boundary matching to avoid "javascript" false match
}

# Scope inflation keywords - things that indicate feature creep
_SCOPE_INFLATION_KEYWORDS = [
    "electron-builder", "packaging", "installer", ".exe", ".msi",
    "telemetry", "analytics", "crash-report", "remote",
    "vite", "webpack", "bundler",
    "playwright", "e2e", "end-to-end",
    "authentication", "auth", "oauth",
    "database", "sqlite", "persistence",
    "%appdata%", "local storage",
    "settings ui", "menus", "overlays",
]


def _detect_stack_from_text(text: str) -> List[str]:
    """Detect mentioned technology stacks from text (case-insensitive).
    
    v1.1 FIX: Use word-boundary matching to prevent false positives.
    e.g., "go" should not match "going", "goal", "algorithm".
    """
    if not text:
        return []
    
    text_lower = text.lower()
    detected = []
    
    # Keywords that need exact word boundary matching
    # (to avoid false positives from partial matches)
    _WORD_BOUNDARY_KEYWORDS = {"rust", "java", "vue"}
    
    for keyword, stack_name in _STACK_KEYWORDS.items():
        # v1.1: Use word boundary for ambiguous keywords
        if keyword.strip() in _WORD_BOUNDARY_KEYWORDS:
            # Use regex word boundary to avoid partial matches
            pattern = r'\b' + _re.escape(keyword.strip()) + r'\b'
            if _re.search(pattern, text_lower):
                detected.append(stack_name)
        else:
            # Direct substring match for unambiguous keywords
            if keyword in text_lower:
                detected.append(stack_name)
    
    return list(set(detected))


def _extract_spec_constraints(spec_json: Optional[str]) -> Dict[str, Any]:
    """
    Extract key constraints from SpecGate JSON.
    
    v1.4 (2026-01-22): Now checks implementation_stack field FIRST
    - If implementation_stack is present and stack_locked=True, that's authoritative
    - Falls back to heuristic detection from goal/summary text
    
    Returns dict with:
    - platform: e.g., "Desktop"
    - scope: e.g., "bare minimum playable"
    - known_requirements: list of user-specified requirements
    - discussed_stack: list of tech stack hints from Weaver
    - implementation_stack: dict with explicit stack choice (if present)
    - stack_locked: bool indicating if stack was explicitly confirmed
    """
    if not spec_json:
        return {}
    
    try:
        spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
    except Exception:
        return {}
    
    constraints = {
        "platform": None,
        "scope": None,
        "known_requirements": [],
        "discussed_stack": [],
        "implementation_stack": None,  # v1.4: Explicit stack choice
        "stack_locked": False,          # v1.4: Whether stack was explicitly confirmed
        "raw_goal": spec_data.get("goal", ""),
        "raw_summary": spec_data.get("summary", ""),
    }
    
    # =========================================================================
    # v1.4: Check for EXPLICIT implementation_stack field FIRST
    # =========================================================================
    impl_stack = spec_data.get("implementation_stack")
    if impl_stack and isinstance(impl_stack, dict):
        constraints["implementation_stack"] = impl_stack
        constraints["stack_locked"] = impl_stack.get("stack_locked", False)
        
        # Build discussed_stack from explicit field
        explicit_stack = []
        if impl_stack.get("language"):
            explicit_stack.append(impl_stack["language"])
        if impl_stack.get("framework"):
            lang = impl_stack.get("language", "")
            framework = impl_stack["framework"]
            explicit_stack.append(f"{lang}+{framework}" if lang else framework)
        if impl_stack.get("runtime"):
            explicit_stack.append(impl_stack["runtime"])
        
        if explicit_stack:
            constraints["discussed_stack"] = explicit_stack
            logger.info(
                "[critique] v1.4 Using EXPLICIT implementation_stack: %s (locked=%s)",
                explicit_stack, constraints["stack_locked"]
            )
            print(f"[DEBUG] [critique] v1.4 EXPLICIT stack from spec: {explicit_stack} (locked={constraints['stack_locked']})")
    
    # Extract from known requirements (Weaver's output)
    goal = spec_data.get("goal", "").lower()
    summary = spec_data.get("summary", "").lower()
    all_text = f"{goal} {summary}"
    
    # Detect platform
    if "desktop" in all_text:
        constraints["platform"] = "Desktop"
    elif "web" in all_text or "browser" in all_text:
        constraints["platform"] = "Web"
    elif "mobile" in all_text or "android" in all_text or "ios" in all_text:
        constraints["platform"] = "Mobile"
    
    # Detect scope keywords
    scope_keywords = [
        "minimal", "minimum", "bare", "simple", "basic",
        "playable", "prototype", "mvp", "first version",
    ]
    for kw in scope_keywords:
        if kw in all_text:
            constraints["scope"] = "minimal"
            break
    
    # =========================================================================
    # v1.4: Only use heuristic detection if no explicit stack was provided
    # =========================================================================
    if not constraints["discussed_stack"]:
        heuristic_stack = _detect_stack_from_text(all_text)
        if heuristic_stack:
            constraints["discussed_stack"] = heuristic_stack
            logger.info(
                "[critique] v1.4 Using HEURISTIC stack detection: %s",
                heuristic_stack
            )
            print(f"[DEBUG] [critique] v1.4 HEURISTIC stack from text: {heuristic_stack}")
    
    # Extract known requirements list
    known_reqs = spec_data.get("known_requirements", []) or spec_data.get("constraints_from_weaver", [])
    if isinstance(known_reqs, list):
        constraints["known_requirements"] = known_reqs
    
    return constraints


# Patterns that indicate a mention is an EXCLUSION, not a target.
# If a line containing a platform keyword ALSO contains one of these,
# it's almost certainly saying "NOT this platform" rather than targeting it.
_EXCLUSION_CONTEXT_PATTERNS = [
    # Explicit exclusion
    r'out\s+of\s+scope',
    r'not\s+(?:included|supported|targeted|in\s+(?:this|scope|phase))',
    r'excluded',
    r'will\s+not',
    r"won't",
    r"don't",
    r'no\s+mobile',
    # Future / deferred
    r'phase\s+[2-9]',
    r'future\s+(?:work|phase|release|version|enhancement|consideration)',
    r'planned\s+for\s+(?:future|later)',
    r'later\s+(?:phase|version|release)',
    r'beyond\s+(?:scope|phase\s*1|v1|mvp)',
    # v1.11: Broader future-reference pattern (catches "future mobile/web clients" etc.)
    r'future\s+\w+',
    # Revision notes discussing the mismatch itself (meta-commentary)
    r'critique\s+(?:claims?|says?|states?|flagged|reported)',
    r'erroneous\s+assessment',
    r'factually\s+incorrect',
    r'rejecting\s+this',
    r'revision\s+notes?',
    r'review\s*:.*(?:rejected|incorrect|wrong)',
    # v1.11: Critique meta-commentary (architecture rebuttals to critique feedback)
    r'reviewer\s+(?:claim|suggestion|feedback|assertion)',
    r'false\s+positive',
    r'\bREJECT\b',
    r'this\s+is\s+(?:a\s+)?(?:false|erroneous|incorrect)',
    r'appears\s+to\s+be\s+an\s+error',
    r'spec[_-]?compliance',
    r'platform\s+mismatch\s*:',
    # v1.10: Additional inline exclusion indicators
    r'explicitly\s+not',
    r'not\s+(?:yet|now)',
    r'must\s+not\s+block',
    r'should\s+not\s+block',
    r'without\s+(?:blocking|preventing)',
    # v1.11: "allowing" pattern (forward-looking compatibility statements)
    r'allowing\s+(?:future|other|additional|external)',
]

# v1.10: Patterns for INLINE exclusion — checked on the keyword line itself only.
# These are strong enough signals that we don't need surrounding context.
_INLINE_EXCLUSION_PATTERNS = [
    _re.compile(r'^\s*[\u274c\u2717\u2718\u2573\u00d7]', _re.UNICODE),  # Line starts with ❌ ✗ ✘ ╳ ×
    _re.compile(r'no\s+mobile', _re.IGNORECASE),
    _re.compile(r'not\s+(?:in\s+)?(?:this|phase|scope|v1|mvp)', _re.IGNORECASE),
    _re.compile(r'phase\s+[2-9]', _re.IGNORECASE),
    _re.compile(r'must\s+not\s+block', _re.IGNORECASE),
    _re.compile(r'future\s+phase', _re.IGNORECASE),
    _re.compile(r'explicitly\s+not', _re.IGNORECASE),
    # v1.11: Critique meta-commentary patterns (inline)
    _re.compile(r'\*\*Reviewer\s+(?:Claim|Suggestion)', _re.IGNORECASE),  # **Reviewer Claim**: ...
    _re.compile(r'\*\*DECISION\*\*\s*:\s*\*\*REJECT', _re.IGNORECASE),  # **DECISION**: **REJECT**
    _re.compile(r'false\s+positive', _re.IGNORECASE),
    _re.compile(r'allowing\s+future\s+\w+', _re.IGNORECASE),  # "allowing future mobile/web clients"
]

# v1.10: Section headers that define "exclusion zones".
# Everything under these headers until the next header is excluded.
_EXCLUSION_SECTION_HEADERS = [
    _re.compile(r'^#+\s*.*(?:out\s+of\s+scope|future\s+consideration|not\s+in\s+(?:scope|phase)|excluded|deferred|limitation)', _re.IGNORECASE),
    _re.compile(r'^#+\s*.*(?:revision\s+(?:notes?|log|history))', _re.IGNORECASE),
    _re.compile(r'^\d+\.\s*(?:future\s+consideration|out\s+of\s+scope|revision\s+(?:notes?|log))', _re.IGNORECASE),
    _re.compile(r'^\*\*(?:out\s+of\s+scope|future|excluded|not\s+in\s+phase)', _re.IGNORECASE),
    # v1.11: Critique rebuttal sections (architecture responding to critique feedback)
    _re.compile(r'^#+\s*.*(?:reviewer\s+suggestion|revision\s+response|critique\s+rebuttal|spec[_-]?compliance)', _re.IGNORECASE),
    _re.compile(r'^#+\s*.*(?:platform\s+mismatch\s+(?:claim|analysis))', _re.IGNORECASE),
]

# Any markdown header pattern (to detect section boundaries)
_SECTION_HEADER_RE = _re.compile(r'^(?:#{1,6}\s|\d+\.\s)')

# Pre-compile context patterns for performance
_EXCLUSION_CONTEXT_RE = [_re.compile(p, _re.IGNORECASE) for p in _EXCLUSION_CONTEXT_PATTERNS]


def _build_exclusion_zone_set(lines):
    """
    v1.10: Pre-scan document to identify "exclusion zones".
    An exclusion zone starts at a section header matching _EXCLUSION_SECTION_HEADERS
    and extends until the next section header.
    Returns set of line indices inside exclusion zones.
    """
    excluded_lines = set()
    in_exclusion_zone = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_any_header = bool(_SECTION_HEADER_RE.match(stripped))
        if is_any_header:
            is_exclusion_header = any(pat.match(stripped) for pat in _EXCLUSION_SECTION_HEADERS)
            if is_exclusion_header:
                in_exclusion_zone = True
                excluded_lines.add(i)
                continue
            else:
                in_exclusion_zone = False
        if in_exclusion_zone:
            excluded_lines.add(i)
    return excluded_lines


# v2.2: Patterns indicating INCIDENTAL use of platform keywords.
# These are compound phrases or technical contexts where "mobile" etc. appear
# but do NOT indicate the architecture targets a mobile platform.
_INCIDENTAL_PLATFORM_PATTERNS = [
    # Compound adjectives (mobile as modifier, not platform target)
    _re.compile(r'mobile[\s-](?:friendly|first|responsive|optimized|compatible|aware|ready)', _re.IGNORECASE),
    # Browser/API contexts (describing web APIs, not targeting mobile)
    _re.compile(r'mobile\s+(?:browser|browsers|safari|chrome|device|devices|screen|viewport)', _re.IGNORECASE),
    # Audio/media API contexts (common in voice/audio features)
    _re.compile(r'mobile\s+(?:audio|microphone|media|recording|input)', _re.IGNORECASE),
    _re.compile(r'(?:audio|microphone|media|recording)\s+(?:on|for|from)\s+mobile', _re.IGNORECASE),
    # Compatibility/support mentions (not targeting, just noting support)
    _re.compile(r'(?:support|handle|detect|check)\s+(?:for\s+)?mobile', _re.IGNORECASE),
    _re.compile(r'mobile\s+(?:support|compatibility|fallback)', _re.IGNORECASE),
    # CSS/responsive design contexts
    _re.compile(r'(?:responsive|breakpoint|media\s+query|@media).*mobile', _re.IGNORECASE),
    _re.compile(r'mobile.*(?:responsive|breakpoint|media\s+query|@media)', _re.IGNORECASE),
    # User-agent / detection contexts
    _re.compile(r'(?:user[\s-]?agent|navigator|window).*mobile', _re.IGNORECASE),
    _re.compile(r'mobile.*(?:user[\s-]?agent|detection)', _re.IGNORECASE),
    # WebRTC / MediaRecorder contexts (very common in voice features)
    _re.compile(r'(?:MediaRecorder|getUserMedia|WebRTC|navigator\.mediaDevices).*mobile', _re.IGNORECASE),
    _re.compile(r'mobile.*(?:MediaRecorder|getUserMedia|WebRTC)', _re.IGNORECASE),
]

# v2.2: Patterns indicating STRUCTURAL platform targeting.
# These are declarative statements that the architecture IS for mobile.
_STRUCTURAL_PLATFORM_PATTERNS = [
    # Explicit platform declarations
    _re.compile(r'(?:target|platform|deploy|build)\s*(?::|for|to)\s*(?:.*\b)?(?:mobile|android|ios)', _re.IGNORECASE),
    _re.compile(r'(?:mobile|android|ios)\s+(?:app|application|client|platform|target|deployment)', _re.IGNORECASE),
    # App store / native mobile indicators
    _re.compile(r'(?:app\s+store|play\s+store|google\s+play|apple\s+store|apk|ipa\b)', _re.IGNORECASE),
    _re.compile(r'(?:react\s+native|flutter|kotlin|swift|xcode|android\s+studio)', _re.IGNORECASE),
    _re.compile(r'(?:cordova|capacitor|ionic|expo)\b', _re.IGNORECASE),
    # Mobile-specific architecture patterns
    _re.compile(r'\b(?:ios|android)\s+(?:sdk|api|permission|manifest)', _re.IGNORECASE),
    _re.compile(r'(?:push\s+notification|geofencing|nfc)\s+.*(?:mobile|android|ios)', _re.IGNORECASE),
]

# v2.2: Tech stacks that DEFINITIVELY indicate a platform.
# If the arch_stack contains any of these, platform is resolved immediately.
_DESKTOP_DEFINITIVE_STACKS = {'TypeScript/Electron', 'Python+PyQt', 'Python+Tkinter', 'C#'}
_MOBILE_DEFINITIVE_STACKS = {'React Native', 'Flutter', 'Kotlin', 'Swift'}


def _check_platform_targeted_not_excluded(
    arch_content: str,
    platform_keywords: List[str],
    context_window_lines: int = 3,
    arch_stack: Optional[List[str]] = None,
) -> bool:
    """
    v2.2: Structural platform detection with tech stack awareness.
    
    Check whether platform keywords (e.g., 'mobile', 'android', 'ios') appear
    in the architecture in a STRUCTURAL/TARGETING context, as opposed to
    incidental mentions (API descriptions, compatibility notes, etc.).
    
    v2.2 fixes the v1.10 false positive where incidental mentions of "mobile"
    in technical contexts (e.g., "mobile browser", "mobile audio handling",
    "MediaRecorder mobile support") were counted as affirmative platform signals
    for a Desktop/Electron architecture.
    
    Five-layer detection:
    0. TECH STACK OVERRIDE: If arch_stack contains definitive desktop tech
       (e.g., Electron), the architecture is Desktop regardless of keywords.
    1. EXCLUSION ZONES: Pre-scanned section headers ("Out of Scope", etc.)
    2. INLINE PATTERNS: Line-level exclusion signals (❌ prefix, etc.)
    3. CONTEXT WINDOW: Surrounding lines checked for exclusion language.
    4. STRUCTURAL SIGNAL: After exclusion filtering, remaining mentions must
       match STRUCTURAL platform patterns ("targets: mobile", "mobile app",
       "for Android/iOS") to count. Incidental mentions ("mobile browser",
       "mobile audio", "mobile-friendly") are filtered out.
    
    Args:
        arch_content: The architecture document text
        platform_keywords: Keywords to search for (e.g., ['mobile', 'android', 'ios'])
        context_window_lines: Number of lines above/below to check for exclusion context
        arch_stack: Detected tech stack from architecture (e.g., ['TypeScript/Electron'])
    
    Returns:
        True if the platform appears to be TARGETED (not just excluded/incidental)
    """
    if not arch_content:
        return False
    
    # =========================================================================
    # Layer 0: Tech stack override (v2.2)
    # If the architecture uses definitively desktop tech, it's not mobile.
    # =========================================================================
    if arch_stack:
        arch_stack_set = set(arch_stack)
        has_desktop_stack = bool(arch_stack_set & _DESKTOP_DEFINITIVE_STACKS)
        has_mobile_stack = bool(arch_stack_set & _MOBILE_DEFINITIVE_STACKS)
        
        if has_desktop_stack and not has_mobile_stack:
            print(
                f"[DEBUG] [critique] v2.2 Tech stack override: {arch_stack_set & _DESKTOP_DEFINITIVE_STACKS} "
                f"is definitively Desktop — skipping mobile keyword scan"
            )
            logger.info(
                "[critique] v2.2 Tech stack override: %s is Desktop, not scanning for mobile keywords",
                arch_stack_set & _DESKTOP_DEFINITIVE_STACKS,
            )
            return False
    
    lines = arch_content.splitlines()
    
    # Layer 1: Pre-scan for exclusion zones (section-level)
    exclusion_zones = _build_exclusion_zone_set(lines)
    if exclusion_zones:
        logger.debug(
            "[critique] v1.10 Exclusion zones: %d lines in %d total",
            len(exclusion_zones), len(lines)
        )
    
    structural_count = 0
    incidental_count = 0
    excluded_count = 0
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check if this line contains any platform keyword
        has_keyword = False
        for kw in platform_keywords:
            if kw in line_lower:
                has_keyword = True
                break
        
        if not has_keyword:
            continue
        
        # --- Layer 1: Exclusion zone check ---
        if i in exclusion_zones:
            excluded_count += 1
            logger.debug(
                "[critique] v1.10 Line %d EXCLUDED (in exclusion zone): %s",
                i + 1, line.strip()[:120]
            )
            continue
        
        # --- Layer 2: Inline exclusion patterns (line-level, no context needed) ---
        is_inline_excluded = any(
            pat.search(line) for pat in _INLINE_EXCLUSION_PATTERNS
        )
        if is_inline_excluded:
            excluded_count += 1
            logger.debug(
                "[critique] v1.10 Line %d EXCLUDED (inline pattern): %s",
                i + 1, line.strip()[:120]
            )
            continue
        
        # --- Layer 3: Context window check (original v1.8 approach, widened) ---
        start = max(0, i - context_window_lines)
        end = min(len(lines), i + context_window_lines + 1)
        context_block = " ".join(lines[start:end]).lower()
        
        is_context_excluded = False
        for pattern in _EXCLUSION_CONTEXT_RE:
            if pattern.search(context_block):
                is_context_excluded = True
                break
        
        if is_context_excluded:
            excluded_count += 1
            logger.debug(
                "[critique] v1.10 Line %d EXCLUDED (context window): %s",
                i + 1, line.strip()[:120]
            )
            continue
        
        # --- Layer 4: Structural vs incidental classification (v2.2) ---
        # The mention passed all exclusion checks. But is it actually a
        # platform declaration, or just an incidental technical reference?
        
        is_incidental = any(
            pat.search(line) for pat in _INCIDENTAL_PLATFORM_PATTERNS
        )
        if is_incidental:
            incidental_count += 1
            logger.debug(
                "[critique] v2.2 Line %d INCIDENTAL (technical context): %s",
                i + 1, line.strip()[:120]
            )
            continue
        
        is_structural = any(
            pat.search(line) for pat in _STRUCTURAL_PLATFORM_PATTERNS
        )
        if is_structural:
            structural_count += 1
            logger.info(
                "[critique] v2.2 Line %d STRUCTURAL platform signal: %s",
                i + 1, line.strip()[:120]
            )
            continue
        
        # Mention is not excluded, not incidental, not structural.
        # v2.2: Treat ambiguous mentions as incidental (conservative).
        # This is the key change — previously these counted as affirmative.
        incidental_count += 1
        logger.debug(
            "[critique] v2.2 Line %d AMBIGUOUS (treated as incidental): %s",
            i + 1, line.strip()[:120]
        )
    
    total = structural_count + incidental_count + excluded_count
    
    if total == 0:
        return False
    
    if structural_count > 0:
        print(
            f"[DEBUG] [critique] v2.2 Platform detection: {structural_count} structural, "
            f"{incidental_count} incidental, {excluded_count} excluded mentions"
        )
        return True
    
    # No structural signals — only incidental/excluded mentions
    print(
        f"[DEBUG] [critique] v2.2 Platform keywords found but NO structural signals "
        f"({incidental_count} incidental, {excluded_count} excluded) — NOT a platform mismatch"
    )
    return False


def run_deterministic_spec_compliance_check(
    arch_content: str,
    spec_json: Optional[str] = None,
    original_request: str = "",
) -> List[CritiqueIssue]:
    """
    v1.3 CRITICAL FIX: Deterministic spec-compliance check.
    v1.4: Now uses implementation_stack.stack_locked for stricter enforcement.
    
    Runs BEFORE the LLM critique to catch obvious spec violations:
    1. Platform mismatch (Desktop spec but Web architecture)
    2. Stack mismatch (Python discussed but TypeScript proposed)
    3. Scope inflation (minimal spec but full product architecture)
    
    This is a deterministic check - NO LLM calls. It catches issues the LLM
    might miss due to being distracted by "best practices" suggestions.
    
    v1.4 Enhancement:
    - If implementation_stack.stack_locked == True, stack mismatch is a HARD blocker
    - The spec_ref will indicate "LOCKED" to show this was explicitly confirmed
    
    Args:
        arch_content: The architecture document to check
        spec_json: The SpecGate JSON spec
        original_request: The original user request (for context)
    
    Returns:
        List of CritiqueIssue objects (all blocking if found)
    """
    issues: List[CritiqueIssue] = []
    issue_counter = 0
    
    if not arch_content:
        return issues
    
    arch_lower = arch_content.lower()
    
    # Extract spec constraints
    constraints = _extract_spec_constraints(spec_json)
    spec_platform = constraints.get("platform")
    spec_scope = constraints.get("scope")
    spec_stack = constraints.get("discussed_stack", [])
    stack_locked = constraints.get("stack_locked", False)  # v1.4: Explicit lock flag
    impl_stack = constraints.get("implementation_stack")   # v1.4: Full stack object
    
    # Also check original request for context (only if no explicit stack)
    if not stack_locked:
        request_stack = _detect_stack_from_text(original_request)
        all_discussed_stack = list(set(spec_stack + request_stack))
    else:
        # If stack is locked, ONLY use the explicit stack (no heuristics)
        all_discussed_stack = spec_stack
    
    # Detect architecture stack
    arch_stack = _detect_stack_from_text(arch_content)
    
    logger.info(
        "[critique] v1.4 Deterministic check: spec_platform=%s, spec_scope=%s, "
        "discussed_stack=%s, arch_stack=%s, stack_locked=%s",
        spec_platform, spec_scope, all_discussed_stack, arch_stack, stack_locked
    )
    print(f"[DEBUG] [critique] v1.4 Deterministic spec check:")
    print(f"[DEBUG] [critique]   spec_platform={spec_platform}")
    print(f"[DEBUG] [critique]   spec_scope={spec_scope}")
    print(f"[DEBUG] [critique]   discussed_stack={all_discussed_stack}")
    print(f"[DEBUG] [critique]   arch_stack={arch_stack}")
    print(f"[DEBUG] [critique]   stack_locked={stack_locked}")
    
    # =========================================================================
    # Check 1: STACK MISMATCH
    # =========================================================================
    # If user discussed Python+Pygame but architecture proposes Electron+React,
    # that's a BLOCKING issue.
    
    # Check for Python discussion but JS/TS architecture
    python_discussed = any("Python" in s for s in all_discussed_stack)
    ts_js_in_arch = any(
        s for s in arch_stack 
        if "TypeScript" in s or "JavaScript" in s or "Node" in s or "Electron" in s or "React" in s
    )
    
    if python_discussed and ts_js_in_arch and not any("Python" in s for s in arch_stack):
        issue_counter += 1
        # v1.4: Indicate if stack was explicitly locked
        lock_indicator = " [LOCKED]" if stack_locked else ""
        spec_ref = f"Discussed stack: Python-based implementation{lock_indicator}"
        
        issues.append(CritiqueIssue(
            id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
            spec_ref=spec_ref,
            arch_ref="Architecture proposes: JavaScript/TypeScript stack",
            category="stack_mismatch",
            severity="blocking",
            description=(
                f"STACK MISMATCH: The user {'EXPLICITLY CONFIRMED' if stack_locked else 'discussed'} a Python-based implementation "
                f"(detected: {[s for s in all_discussed_stack if 'Python' in s]}), "
                f"but the architecture proposes a JavaScript/TypeScript stack "
                f"(detected: {[s for s in arch_stack if 'TypeScript' in s or 'JavaScript' in s or 'Electron' in s or 'React' in s]}). "
                f"{'This stack choice was LOCKED by user confirmation and CANNOT be changed.' if stack_locked else 'Architecture must use the stack discussed with the user unless explicitly changed.'}"
            ),
            fix_suggestion=(
                "Rewrite architecture to use Python + the libraries discussed with the user. "
                f"{'The user explicitly confirmed this stack choice - it is non-negotiable.' if stack_locked else 'Do not substitute tech stacks without explicit user approval.'}"
            ),
        ))
        print(f"[DEBUG] [critique] v1.4 BLOCKER: Stack mismatch (Python discussed, TS/JS proposed, locked={stack_locked})")
    
    # Check for Pygame discussed but Electron in architecture
    pygame_discussed = any("Pygame" in s for s in all_discussed_stack)
    electron_in_arch = "electron" in arch_lower
    
    if pygame_discussed and electron_in_arch:
        issue_counter += 1
        # v1.4: Indicate if stack was explicitly locked
        lock_indicator = " [LOCKED]" if stack_locked else ""
        spec_ref = f"Discussed stack: Python + Pygame{lock_indicator}"
        
        issues.append(CritiqueIssue(
            id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
            spec_ref=spec_ref,
            arch_ref="Architecture proposes: Electron framework",
            category="stack_mismatch",
            severity="blocking",
            description=(
                f"STACK MISMATCH: User {'EXPLICITLY CONFIRMED' if stack_locked else 'discussed'} Python + Pygame for the implementation, "
                "but architecture proposes Electron (JavaScript/TypeScript). "
                f"{'This stack choice was LOCKED by user confirmation - the architecture MUST use Python + Pygame.' if stack_locked else 'This is a completely different technology stack that ignores the user\'s intent.'}"
            ),
            fix_suggestion=(
                "Rewrite architecture to use Python + Pygame as discussed. "
                f"{'This is a LOCKED requirement from user confirmation.' if stack_locked else 'Pygame is a Python library for making games and is what the user chose.'}"
            ),
        ))
        print(f"[DEBUG] [critique] v1.4 BLOCKER: Pygame discussed but Electron proposed (locked={stack_locked})")
    
    # =========================================================================
    # v1.4: NEW CHECK - Explicit stack_locked violation (any stack mismatch)
    # =========================================================================
    # If stack is explicitly locked and architecture uses ANY different stack,
    # that's a blocker even if our specific checks above didn't catch it.
    
    if stack_locked and impl_stack:
        locked_language = (impl_stack.get("language") or "").lower()
        locked_framework = (impl_stack.get("framework") or "").lower()
        
        # Check if architecture uses the locked language
        if locked_language:
            arch_uses_locked_language = any(
                locked_language in s.lower() for s in arch_stack
            )
            if not arch_uses_locked_language and arch_stack:
                issue_counter += 1
                issues.append(CritiqueIssue(
                    id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
                    spec_ref=f"LOCKED implementation_stack.language: {impl_stack.get('language')}",
                    arch_ref=f"Architecture uses: {arch_stack}",
                    category="stack_mismatch",
                    severity="blocking",
                    description=(
                        f"LOCKED STACK VIOLATION: The spec explicitly requires '{impl_stack.get('language')}' "
                        f"(stack_locked=True), but the architecture proposes different technology: {arch_stack}. "
                        f"This stack choice was confirmed by the user and CANNOT be overridden."
                    ),
                    fix_suggestion=(
                        f"Rewrite architecture to use {impl_stack.get('language')} as explicitly required. "
                        f"Source: {impl_stack.get('source', 'user confirmation')}"
                    ),
                ))
                print(f"[DEBUG] [critique] v1.4 BLOCKER: LOCKED stack violation (required={impl_stack.get('language')}, found={arch_stack})")
        
        # Check if architecture uses the locked framework
        if locked_framework:
            arch_uses_locked_framework = any(
                locked_framework in s.lower() for s in arch_stack
            ) or locked_framework in arch_lower
            if not arch_uses_locked_framework and arch_stack:
                issue_counter += 1
                issues.append(CritiqueIssue(
                    id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
                    spec_ref=f"LOCKED implementation_stack.framework: {impl_stack.get('framework')}",
                    arch_ref=f"Architecture content does not include: {impl_stack.get('framework')}",
                    category="stack_mismatch",
                    severity="blocking",
                    description=(
                        f"LOCKED FRAMEWORK VIOLATION: The spec explicitly requires '{impl_stack.get('framework')}' "
                        f"(stack_locked=True), but this framework is not mentioned in the architecture. "
                        f"This choice was confirmed by the user and CANNOT be overridden."
                    ),
                    fix_suggestion=(
                        f"Rewrite architecture to use {impl_stack.get('framework')} as explicitly required. "
                        f"Source: {impl_stack.get('source', 'user confirmation')}"
                    ),
                ))
                print(f"[DEBUG] [critique] v1.4 BLOCKER: LOCKED framework violation (required={impl_stack.get('framework')})")
    
    # =========================================================================
    # Check 2: SCOPE INFLATION
    # =========================================================================
    # If spec says "minimal" or "bare minimum" but architecture has packaging,
    # installers, telemetry, etc., that's scope creep.
    
    if spec_scope == "minimal":
        inflation_found = []
        
        for keyword in _SCOPE_INFLATION_KEYWORDS:
            if keyword in arch_lower:
                inflation_found.append(keyword)
        
        # Only flag if multiple inflation indicators found (avoid false positives)
        if len(inflation_found) >= 3:
            issue_counter += 1
            issues.append(CritiqueIssue(
                id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
                spec_ref="Scope: minimal / bare minimum playable",
                arch_ref=f"Architecture includes: {inflation_found[:5]}",
                category="scope_inflation",
                severity="blocking",
                description=(
                    f"SCOPE INFLATION: Spec requires 'minimal' or 'bare minimum' implementation, "
                    f"but architecture includes scope-inflating features: {inflation_found[:5]}. "
                    f"A minimal implementation should not include packaging, installers, "
                    f"telemetry, or other production features."
                ),
                fix_suggestion=(
                    "Remove scope-inflating features. Focus on core functionality only. "
                    "Packaging, installers, persistence, and telemetry can be added later "
                    "if/when the user requests them."
                ),
            ))
            print(f"[DEBUG] [critique] v1.3 BLOCKER: Scope inflation ({len(inflation_found)} indicators: {inflation_found[:5]})")
    
    # =========================================================================
    # Check 3: PLATFORM CONTEXT (less strict, just warn if significant mismatch)
    # =========================================================================
    # This is informational - platform choices are often flexible.
    # But if spec says "Desktop" and arch says "Mobile", that's wrong.
    #
    # v1.8 FIX: Context-aware platform detection.
    # Previous naive substring check ("mobile" in arch_lower) caused false positives
    # when the architecture EXCLUDED mobile (e.g., "Out of Scope: Mobile app").
    # Now we check whether mobile/android/ios mentions are affirmative (targeting)
    # vs exclusionary (out of scope, phase 2, future, etc.).
    #
    # See: Voice-to-text architecture false positive (cp-1846c85e) where 3 critique
    # iterations + 2 Opus revision calls were wasted on a non-existent problem.
    
    if spec_platform == "Desktop":
        mobile_is_targeted = _check_platform_targeted_not_excluded(
            arch_content=arch_content,
            platform_keywords=["mobile", "android", "ios"],
            arch_stack=arch_stack,  # v2.2: Pass stack for tech-stack-aware detection
        )
        if mobile_is_targeted:
            issue_counter += 1
            issues.append(CritiqueIssue(
                id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
                spec_ref=f"Platform: {spec_platform}",
                arch_ref="Architecture targets: Mobile platform",
                category="platform_mismatch",
                severity="blocking",
                description=(
                    f"PLATFORM MISMATCH: Spec requires Desktop platform, "
                    f"but architecture targets Mobile (Android/iOS)."
                ),
                fix_suggestion="Rewrite architecture to target Desktop platform as specified.",
            ))
            print(f"[DEBUG] [critique] v2.2 BLOCKER: Platform mismatch (Desktop spec, Mobile arch)")
    
    # Summary
    if issues:
        print(f"[DEBUG] [critique] v1.3 Deterministic check found {len(issues)} BLOCKING issue(s)")
        logger.warning(
            "[critique] v1.3 Deterministic spec-compliance check: %d blocking issues",
            len(issues)
        )
    else:
        print(f"[DEBUG] [critique] v1.3 Deterministic check: No obvious spec violations")
        logger.info("[critique] v1.3 Deterministic spec-compliance check: PASSED (no obvious violations)")
    
    return issues


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
    spec_markdown: Optional[str] = None,
    env_context: Optional[Dict[str, Any]] = None,
    envelope: JobEnvelope,
) -> CritiqueResult:
    """Call critic with JSON output schema.
    
    Returns structured CritiqueResult.
    Uses CRITIQUE_PROVIDER/CRITIQUE_MODEL from env via stage_models.
    
    v1.6: Now accepts spec_markdown - the full POT spec with grounded evidence.
          Critique judges ONLY against what's in the spec, not invented constraints.
    v1.2: Now applies blocker filtering to ensure only real blockers block.
    v1.3: Added run_deterministic_spec_compliance_check() - runs BEFORE LLM critique.
    """
    # Get config from stage_models (runtime lookup)
    critique_provider, critique_model, critique_max_tokens = _get_critique_model_config()
    
    # DEBUG: Log critique start
    print(f"[DEBUG] [critique] Starting JSON critic: provider={critique_provider}, model={critique_model}")
    logger.info(f"[critique] Calling JSON critic: {critique_provider}/{critique_model}")
    
    # =========================================================================
    # v1.3 CRITICAL FIX: Run DETERMINISTIC spec-compliance check FIRST
    # =========================================================================
    # This catches obvious spec violations (stack mismatch, scope inflation, etc.)
    # BEFORE the LLM critique. The LLM may be distracted by "best practices"
    # and miss these critical issues.
    
    deterministic_issues = run_deterministic_spec_compliance_check(
        arch_content=arch_content,
        spec_json=spec_json,
        original_request=original_request,
    )
    
    # =========================================================================
    # v2.1: Run SCOPE CREEP check (endpoint drift + excluded features)
    # =========================================================================
    scope_creep_issues = run_scope_creep_check(
        arch_content=arch_content,
        spec_markdown=spec_markdown,
        spec_json=spec_json,
    )
    deterministic_issues.extend(scope_creep_issues)
    if scope_creep_issues:
        print(f"[DEBUG] [critique] v2.1 Scope creep check: {len(scope_creep_issues)} issue(s)")
    
    # If deterministic check found blocking issues, FAIL IMMEDIATELY
    # Don't even bother calling the LLM - the architecture is fundamentally wrong.
    if deterministic_issues:
        print(f"[DEBUG] [critique] v1.3 EARLY FAIL: {len(deterministic_issues)} deterministic blocker(s) found")
        logger.warning(
            "[critique] v1.3 Deterministic check BLOCKED architecture: %d issue(s)",
            len(deterministic_issues)
        )
        
        return CritiqueResult(
            summary=f"Architecture BLOCKED by deterministic spec-compliance check: {len(deterministic_issues)} issue(s) found",
            critique_model="deterministic_check_v1.3",
            critique_failed=False,  # Not a failure - we successfully detected issues
            critique_mode="deep+deterministic",
            blocking_issues=deterministic_issues,
            non_blocking_issues=[],
        )
    
    # =========================================================================
    # v2.0: Evidence resolution check (CRITICAL_CLAIMS register validation)
    # Tweak #1: Only run when output is "final" — no pending EVIDENCE_REQUESTs
    # =========================================================================
    pending_requests = parse_evidence_requests(arch_content)
    has_claims_register = "CRITICAL_CLAIMS:" in arch_content
    
    if not pending_requests:
        # Output is final (no EVIDENCE_REQUESTs pending) — safe to validate
        evidence_issues = run_evidence_resolution_check(arch_content=arch_content)
        
        # Only blocking evidence issues feed into the deterministic check.
        # During transition, missing_claims_register is non_blocking so it
        # won't cause an early fail — it flows through to the LLM critique.
        blocking_evidence = [i for i in evidence_issues if i.severity == "blocking"]
        non_blocking_evidence = [i for i in evidence_issues if i.severity != "blocking"]
        
        if blocking_evidence:
            print(f"[DEBUG] [critique] v2.0 EVIDENCE CHECK FAIL: {len(blocking_evidence)} blocking issue(s)")
            logger.warning(
                "[critique] v2.0 Evidence resolution check BLOCKED: %d issue(s)",
                len(blocking_evidence)
            )
            return CritiqueResult(
                summary=f"Architecture BLOCKED by evidence resolution check: {len(blocking_evidence)} unresolved critical claim(s)",
                critique_model="evidence_check_v2.0",
                critique_failed=False,
                critique_mode="deep+evidence",
                blocking_issues=blocking_evidence,
                non_blocking_issues=non_blocking_evidence,
            )
        
        # Non-blocking evidence issues get collected to pass through
        deterministic_issues.extend(non_blocking_evidence)
        if non_blocking_evidence:
            print(f"[DEBUG] [critique] v2.0 Evidence check: {len(non_blocking_evidence)} non-blocking issue(s) noted")
    else:
        print(f"[DEBUG] [critique] v2.0 Skipping evidence resolution check: {len(pending_requests)} pending EVIDENCE_REQUEST(s)")
    
    # Deterministic check passed - proceed with LLM critique
    print(f"[DEBUG] [critique] v1.3 Deterministic check PASSED - proceeding to LLM critique")
    
    # v1.6: Log spec_markdown injection
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
    
    # v1.9: System message emphasizes spec as AUTHORITATIVE + section authority levels
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
- Architecture calls existing functions with UNVERIFIED parameter names (no CITED evidence
  showing the function signature accepts those exact parameters)
- Architecture depends on data (e.g., project_id) being available at a code point but has
  no CITED evidence showing callers actually pass that data in
- Architecture proposes regex-parsing free text to extract structured data instead of adding
  the field explicitly
- Architecture proposes naive text processing (e.g., split on ". ") for content that may
  arrive in varying formats (JSON, markdown, structured LLM output)

NOT BLOCKING (do not flag these as blocking):
- Architecture choosing different integration files than 'Files to Modify' lists
- Architecture using different approaches than 'Implementation Steps' suggests
- External API usage that the spec requested (e.g., "use OpenAI API")
- Technology choices that align with the spec
- Features the spec explicitly requested
- Generic "best practices" not mentioned in the spec

EVIDENCE REQUIREMENT:
- Every blocking issue MUST include both spec_ref AND arch_ref
- spec_ref: Which USER REQUIREMENT is violated (MUST exist in the spec)
- arch_ref: Which architecture section shows the violation
- If you cannot cite both, make the issue non_blocking
- If spec_ref points to an LLM SUGGESTION section, make the issue non_blocking

Your critique must align with the spec. Do not expand scope or invent constraints.

## EVIDENCE-OR-REQUEST CONTRACT

For every implementation-affecting claim, you MUST output exactly one of:

1. **CITED** — You have seen evidence in this context.
   Format: [CITED file="path/to/file.py" lines="42-58"]
           [CITED doc="runtime_facts.yaml" key="sandbox.primary_paths.desktop"]
           [CITED bundle="architecture_map" lines="80-110"]
           [CITED rag_then_read="searched 'query', confirmed in path/file.py" lines="15-38"]

2. **EVIDENCE_REQUEST** — You need the orchestrator to fetch something.
   (See format below. You will be re-prompted with results.)

3. **DECISION** — This is a genuine design choice, not a discoverable fact.
   Format:
   DECISION:
     id: "D-NNN"
     topic: "What you're deciding"
     choice: "What you chose"
     why: "Rationale"
     consequences: ["Impact 1", "Impact 2"]
     revisit_if: "Condition that would change this decision"

4. **HUMAN_REQUIRED** — Evidence doesn't exist and guessing is high-risk.
   Format:
   HUMAN_REQUIRED:
     id: "HR-NNN"
     question: "One precise question"
     why: "What breaks if we guess"
     searched: ["What you already tried"]
     default_if_no_answer: "Safe fallback if human doesn't respond"

**NEVER** silently assume a path, port, format, encoding, library API, threading model, or integration point.

### Evidence Hierarchy
- File read / sandbox read = proof (use for any implementation-affecting claim about code)
- RAG search = pointer (must follow up with file read to become a citation)
- Policy documents (runtime_facts.yaml, third_party_policy.yaml) = citation for non-code stable facts
- Architecture map / codebase report = citation for structure claims (they ARE file reads)

### Severity
- **CRITICAL**: paths, ports, formats, encodings, threading, security boundaries, API schemas, data flow contracts.
  Must be resolved (CITED / DECISION / HUMAN_REQUIRED) before implementation.
- **NONCRITICAL**: UI copy, CSS, optional optimizations, naming conventions, comment content.
  Warn if unverified, does not block.

### CRITICAL_CLAIMS Register (Required)
At the END of your output (must be the LAST block — nothing follows it), include:

CRITICAL_CLAIMS:
  - id: "CC-001"
    claim: "Short description of what you claimed"
    resolution: "CITED"
    evidence:
      - file: "path/to/file.py"
        lines: "42-58"

Every critical claim must be accounted for. This register is validated deterministically.
Do NOT output CRITICAL_CLAIMS until you are done requesting evidence.
CRITICAL_CLAIMS must be the LAST block in your output. Nothing should follow it."""

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
            # FAIL-CLOSED: Empty/timeout response = critique failed, NOT passed
            return CritiqueResult(
                summary="Critique failed: empty response / timeout",
                critique_model=critique_model,
                critique_failed=True,  # CRITICAL: Mark as failed so overall_pass=False
                critique_mode="deep",
                blocking_issues=[CritiqueIssue(
                    id="CRITIQUE-FAIL-001",
                    spec_ref=None,
                    arch_ref=None,
                    category="system",
                    severity="blocking",
                    description="Critique could not be completed due to timeout or empty response from critic model",
                    fix_suggestion="Retry critique with different provider or increase timeout",
                )],
            )
        
        print(f"[DEBUG] [critique] Received response: {len(result.content)} chars")
        critique = parse_critique_output(result.content, model=critique_model)
        critique.critique_mode = "deep"
        
        # =====================================================================
        # v1.2: Apply blocker filtering
        # =====================================================================
        
        original_blocking_count = len(critique.blocking_issues)
        
        if critique.blocking_issues:
            real_blocking, downgraded = filter_blocking_issues(
                critique.blocking_issues,
                require_evidence=True,  # Enforce AND rule: spec_ref AND arch_ref
            )
            
            # Update critique with filtered results
            critique.blocking_issues = real_blocking
            critique.non_blocking_issues.extend(downgraded)
            
            # Recalculate overall_pass (only passes if no blocking AND critique succeeded)
            critique.overall_pass = len(critique.blocking_issues) == 0 and not critique.critique_failed
            
            if downgraded:
                print(f"[DEBUG] [critique] Blocker filtering: {original_blocking_count} → {len(real_blocking)} (downgraded {len(downgraded)})")
        
        # =====================================================================
        # v1.7: Grounding validation - catch hallucinated constraints
        # =====================================================================
        if critique.blocking_issues and (spec_markdown or spec_json):
            grounded_blocking, grounding_downgraded = validate_spec_ref_grounding(
                critique.blocking_issues,
                spec_markdown=spec_markdown,
                spec_json=spec_json,
            )
            critique.blocking_issues = grounded_blocking
            critique.non_blocking_issues.extend(grounding_downgraded)
            critique.overall_pass = len(critique.blocking_issues) == 0 and not critique.critique_failed
            if grounding_downgraded:
                print(f"[DEBUG] [critique] v1.7 Grounding filter: {len(grounded_blocking)} kept, {len(grounding_downgraded)} downgraded")
        
        # =====================================================================
        # v1.9: Section authority validation - downgrade LLM-suggestion blockers
        # =====================================================================
        # Issues citing 'Files to Modify', 'Implementation Steps', etc. are
        # LLM-generated suggestions, not user requirements. Downgrade to
        # non-blocking so the architecture can choose alternative approaches.
        if critique.blocking_issues:
            authority_kept, authority_downgraded = validate_section_authority(
                critique.blocking_issues
            )
            critique.blocking_issues = authority_kept
            critique.non_blocking_issues.extend(authority_downgraded)
            critique.overall_pass = len(critique.blocking_issues) == 0 and not critique.critique_failed
            if authority_downgraded:
                print(f"[DEBUG] [critique] v1.9 Section authority filter: {len(authority_kept)} kept, {len(authority_downgraded)} downgraded")
        
        # DEBUG: Log critique result
        print(f"[DEBUG] [critique] Result: overall_pass={critique.overall_pass}, blocking={len(critique.blocking_issues)}, non_blocking={len(critique.non_blocking_issues)}")
        logger.info(f"[critique] Result: pass={critique.overall_pass}, blocking={len(critique.blocking_issues)}, non_blocking={len(critique.non_blocking_issues)}")
        
        # Log full details of blocking issues for visibility
        if critique.blocking_issues:
            print(f"[DEBUG] [critique] === BLOCKING ISSUES (after filtering) ===")
            for issue in critique.blocking_issues:
                issue_id = getattr(issue, 'id', 'N/A')
                title = getattr(issue, 'title', 'Untitled')
                desc = getattr(issue, 'description', '')[:200]  # Truncate for logs
                category = getattr(issue, 'category', 'unknown')
                spec_ref = getattr(issue, 'spec_ref', 'N/A')
                arch_ref = getattr(issue, 'arch_ref', 'N/A')
                print(f"[DEBUG] [critique]   {issue_id}: [{category}] {title}")
                print(f"[DEBUG] [critique]     → {desc}")
                print(f"[DEBUG] [critique]     spec_ref: {spec_ref}, arch_ref: {arch_ref}")
            print(f"[DEBUG] [critique] === END BLOCKING ===")
        
        # Log summary of non-blocking issues
        if critique.non_blocking_issues:
            print(f"[DEBUG] [critique] Non-blocking ({len(critique.non_blocking_issues)}): {[getattr(i, 'id', 'N/A') for i in critique.non_blocking_issues[:5]]}...")
        
        return critique
        
    except Exception as exc:
        logger.warning(f"[critic] JSON critic call failed: {exc}")
        print(f"[DEBUG] [critique] EXCEPTION: {exc}")
        # Get model for error response
        _, model_for_error, _ = _get_critique_model_config()
        # FAIL-CLOSED: Exception = critique failed, NOT passed
        return CritiqueResult(
            summary=f"Critique failed: {exc}",
            critique_model=model_for_error,
            critique_failed=True,  # CRITICAL: Mark as failed so overall_pass=False
            critique_mode="deep",
            blocking_issues=[CritiqueIssue(
                id="CRITIQUE-FAIL-002",
                spec_ref=None,
                arch_ref=None,
                category="system",
                severity="blocking",
                description=f"Critique could not be completed due to exception: {exc}",
                fix_suggestion="Check critic provider configuration and retry",
            )],
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


# =============================================================================
# SEGMENT INTERFACE CONTRACT VALIDATION (Phase 2 — Pipeline Segmentation)
# =============================================================================

def validate_interface_contracts(
    arch_content: str,
    segment_context: dict,
) -> list:
    """
    Phase 2: Validate that a segment's architecture respects its interface contracts.

    Checks:
    1. Does the architecture create/modify files consistent with 'exposes' contracts?
    2. Does the architecture reference interfaces from 'consumes' contracts?

    Returns a list of CritiqueIssue-style dicts (same shape as existing critique issues).
    This is a NEW function — zero changes to existing critique functions.

    Called by the segment loop after the standard critique pass.
    Backward compatible: only called when segment_context is present.

    v1.0 (2026-02-08): Initial implementation — Phase 2 Pipeline Segmentation.
    """
    issues = []
    if not segment_context or not arch_content:
        return issues

    file_scope = segment_context.get("file_scope", [])
    exposes = segment_context.get("exposes") or {}
    consumes = segment_context.get("consumes") or {}
    segment_id = segment_context.get("segment_id", "unknown")

    arch_lower = arch_content.lower()

    # --- Check 'exposes' contracts ---
    # Verify that class names, endpoints, and exports promised by this segment
    # are mentioned in the architecture document.
    for class_name in exposes.get("class_names", []):
        if class_name.lower() not in arch_lower:
            issues.append({
                "type": "contract_violation",
                "severity": "warning",
                "segment_id": segment_id,
                "message": (
                    f"Segment {segment_id} promises to expose class '{class_name}' "
                    f"but it is not mentioned in the architecture document."
                ),
            })

    for endpoint in exposes.get("endpoint_paths", []):
        # Check for the path portion (e.g. "/voice/transcribe")
        path_part = endpoint.split()[-1] if " " in endpoint else endpoint
        if path_part.lower() not in arch_lower:
            issues.append({
                "type": "contract_violation",
                "severity": "warning",
                "segment_id": segment_id,
                "message": (
                    f"Segment {segment_id} promises to expose endpoint '{endpoint}' "
                    f"but it is not mentioned in the architecture document."
                ),
            })

    for export_name in exposes.get("export_names", []):
        if export_name.lower() not in arch_lower:
            issues.append({
                "type": "contract_violation",
                "severity": "warning",
                "segment_id": segment_id,
                "message": (
                    f"Segment {segment_id} promises to expose '{export_name}' "
                    f"but it is not mentioned in the architecture document."
                ),
            })

    # --- Check 'consumes' contracts ---
    # Verify that consumed interfaces from upstream segments are referenced.
    for class_name in consumes.get("class_names", []):
        if class_name.lower() not in arch_lower:
            issues.append({
                "type": "contract_violation",
                "severity": "info",
                "segment_id": segment_id,
                "message": (
                    f"Segment {segment_id} declares it consumes '{class_name}' from upstream "
                    f"but doesn't reference it in the architecture. This may be intentional."
                ),
            })

    # --- Check file_scope alignment ---
    # Verify that the architecture doesn't mention files outside the segment's scope.
    # This is advisory, not blocking.
    if file_scope:
        scope_basenames = {os.path.basename(f).lower() for f in file_scope}
        # Look for file paths in the architecture that aren't in scope
        import re as _re
        file_refs = _re.findall(
            r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx|json|yaml|css)',
            arch_content,
        )
        for ref in file_refs:
            ref_basename = os.path.basename(ref).lower()
            if ref_basename not in scope_basenames:
                # Only flag if it looks like a creation/modification, not a reference
                # Skip common reference patterns like imports
                ref_context_idx = arch_content.lower().find(ref.lower())
                if ref_context_idx >= 0:
                    context_before = arch_content[max(0, ref_context_idx - 50):ref_context_idx].lower()
                    if any(kw in context_before for kw in ["create", "modify", "write", "add to", "update"]):
                        issues.append({
                            "type": "scope_violation",
                            "severity": "warning",
                            "segment_id": segment_id,
                            "message": (
                                f"Architecture for {segment_id} references file '{ref}' "
                                f"which is outside the segment's file_scope. "
                                f"This may cause cross-segment contamination."
                            ),
                        })

    if issues:
        logger.info(
            "[critique] Phase 2 contract validation for %s: %d issue(s)",
            segment_id, len(issues),
        )
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
    # Block 5: Deterministic spec-compliance check (v1.3 - CRITICAL FIX)
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

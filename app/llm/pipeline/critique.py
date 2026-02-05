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

v1.9 (2026-02-05): SECTION AUTHORITY VALIDATION - LLM suggestion defense
- validate_section_authority() runs AFTER grounding validation
- Downgrade blocking issues whose spec_ref cites LLM-generated sections
  (Files to Modify, Implementation Steps, Reference Files, etc.)
- These sections are implementation SUGGESTIONS, not user requirements
- The architecture may choose alternative approaches without being blocked
- Fixes critique deadlock where Gemini raised blockers on GPT-suggested files
- Defense-in-depth layer on top of v1.3 prompt-level fix in critique_schemas.py
- See critique-pipeline-fix-jobspec.md for full root cause analysis

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
# Block 5: DETERMINISTIC Spec-Compliance Check (v1.3 - CRITICAL FIX)
# =============================================================================
#
# This function runs BEFORE the LLM critique and catches obvious spec violations
# deterministically. It is the primary line of defense against architecture drift.
#
# See: CRITICAL_PIPELINE_FAILURE_REPORT.md (2026-01-22) for why this is needed.
#

import re as _re

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
    r'future\s+(?:work|phase|release|version|enhancement)',
    r'planned\s+for\s+(?:future|later)',
    r'later\s+(?:phase|version|release)',
    r'beyond\s+(?:scope|phase\s*1|v1|mvp)',
    # Revision notes discussing the mismatch itself (meta-commentary)
    r'critique\s+(?:claims?|says?|states?|flagged|reported)',
    r'erroneous\s+assessment',
    r'factually\s+incorrect',
    r'rejecting\s+this',
]

# Pre-compile for performance
_EXCLUSION_CONTEXT_RE = [_re.compile(p, _re.IGNORECASE) for p in _EXCLUSION_CONTEXT_PATTERNS]


def _check_platform_targeted_not_excluded(
    arch_content: str,
    platform_keywords: List[str],
    context_window_lines: int = 2,
) -> bool:
    """
    v1.8: Context-aware platform detection.
    
    Check whether platform keywords (e.g., 'mobile', 'android', 'ios') appear
    in the architecture in an AFFIRMATIVE/TARGETING context, as opposed to an
    exclusion context ("Out of Scope", "Phase 2+", "not in this phase", etc.).
    
    Strategy:
    - Find every line containing a platform keyword
    - Check the line AND surrounding lines for exclusion indicators
    - If ALL mentions are in exclusion context → not targeted (return False)
    - If ANY mention is in affirmative context → targeted (return True)
    
    Args:
        arch_content: The architecture document text
        platform_keywords: Keywords to search for (e.g., ['mobile', 'android', 'ios'])
        context_window_lines: Number of lines above/below to check for exclusion context
    
    Returns:
        True if the platform appears to be TARGETED (not just excluded)
    """
    if not arch_content:
        return False
    
    lines = arch_content.splitlines()
    affirmative_count = 0
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
        
        # Build context window: the line itself + surrounding lines
        start = max(0, i - context_window_lines)
        end = min(len(lines), i + context_window_lines + 1)
        context_block = " ".join(lines[start:end]).lower()
        
        # Check if this mention is in an exclusion context
        is_excluded = False
        for pattern in _EXCLUSION_CONTEXT_RE:
            if pattern.search(context_block):
                is_excluded = True
                break
        
        if is_excluded:
            excluded_count += 1
            logger.debug(
                "[critique] v1.8 Platform keyword on line %d is EXCLUDED (context: %s)",
                i + 1, context_block[:120]
            )
        else:
            affirmative_count += 1
            logger.debug(
                "[critique] v1.8 Platform keyword on line %d appears AFFIRMATIVE: %s",
                i + 1, line.strip()[:120]
            )
    
    total = affirmative_count + excluded_count
    
    if total == 0:
        # No mentions at all
        return False
    
    if affirmative_count > 0:
        print(
            f"[DEBUG] [critique] v1.8 Platform detection: {affirmative_count} affirmative, "
            f"{excluded_count} excluded mentions"
        )
        return True
    
    # All mentions are in exclusion context
    print(
        f"[DEBUG] [critique] v1.8 Platform keywords found but ALL in exclusion context "
        f"({excluded_count} mentions) - NOT a platform mismatch"
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
            print(f"[DEBUG] [critique] v1.8 BLOCKER: Platform mismatch (Desktop spec, Mobile arch)")
    
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

Your critique must align with the spec. Do not expand scope or invent constraints."""

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
]

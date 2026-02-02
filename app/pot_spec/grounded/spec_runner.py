# FILE: app/pot_spec/grounded/spec_runner.py
"""
SpecGate v4.0 - Direct Spec Builder

NO GATES. NO CLASSIFICATION. NO RISK ASSESSMENT.

Flow:
1. Get Weaver spec (what to do)
2. Run scan (evidence of where)
3. Build POT spec (output for Implementer)

Only ask questions if something CRITICAL is missing.

v4.0 (2026-02-01): Stripped all gates - simple but powerful
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SPEC_RUNNER_BUILD_ID = "2026-02-02-v4.4-goal-extraction-fix"
print(f"[SPEC_RUNNER_LOADED] BUILD_ID={SPEC_RUNNER_BUILD_ID}")


# =============================================================================
# IMPORTS
# =============================================================================

from .spec_models import GroundedFact, FileTarget, GroundedPOTSpec
from .domain_detection import detect_domains
from .sandbox_discovery import extract_sandbox_hints
from .evidence_gathering import gather_filesystem_evidence, sandbox_read_file
from .multi_file_detection import _detect_multi_file_intent, _build_multi_file_operation
from .weaver_parser import parse_weaver_intent, _is_placeholder_goal

# Direct spec builder (no LLM, no classification)
try:
    from .simple_refactor import build_direct_spec, SIMPLE_REFACTOR_BUILD_ID
    _DIRECT_BUILDER_AVAILABLE = True
except ImportError:
    _DIRECT_BUILDER_AVAILABLE = False
    build_direct_spec = None

# CREATE spec builder (grounded feature specs)
try:
    from .simple_create import build_grounded_create_spec, SIMPLE_CREATE_BUILD_ID
    _CREATE_BUILDER_AVAILABLE = True
except ImportError:
    _CREATE_BUILDER_AVAILABLE = False
    build_grounded_create_spec = None

# Evidence collector
try:
    from ..evidence_collector import EvidenceBundle, load_evidence
    _EVIDENCE_AVAILABLE = True
except ImportError:
    _EVIDENCE_AVAILABLE = False
    EvidenceBundle = None
    load_evidence = None

# SpecGateResult type
try:
    from ..spec_gate_types import SpecGateResult
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class SpecGateResult:
        ready_for_pipeline: bool = False
        open_questions: List[str] = field(default_factory=list)
        spot_markdown: Optional[str] = None
        db_persisted: bool = False
        spec_id: Optional[str] = None
        spec_hash: Optional[str] = None
        spec_version: Optional[int] = None
        hard_stopped: bool = False
        hard_stop_reason: Optional[str] = None
        notes: Optional[str] = None
        blocking_issues: List[str] = field(default_factory=list)
        validation_status: str = "pending"
        grounding_data: Optional[Dict] = None


__all__ = ["run_spec_gate_grounded"]


# =============================================================================
# PATH EXTRACTION - v4.3 SCOPE-AWARE
# =============================================================================

# Scope indicators: UI/frontend vs backend
# Key insight: If user explicitly says "UI" or "frontend", DON'T include backend
SCOPE_FRONTEND = {
    'the ui': True, 'on the ui': True, 'in the ui': True, 'to the ui': True,
    'a ui': True, 'ui button': True, 'ui feature': True, 'ui text': True,
    'the frontend': True, 'front-end': True, 'frontend ui': True,
    'context window': True, 'input window': True, 'text input': True,
    'the app': True, "app's": True, 'desktop app': True, 'electron': True,
}

SCOPE_BACKEND = {
    'the backend': True, 'back-end': True, 'backend api': True,
    'fastapi': True, 'api endpoint': True, 'server': True,
}

# Explicit project name patterns (only match these, not bare 'orb'/'astra')
EXPLICIT_PROJECT_PATTERNS = {
    'orb desktop': ['D:\\orb-desktop', 'D:\\Orb Desktop'],
    'orb-desktop': ['D:\\orb-desktop'],
    'astra desktop': ['D:\\astra-desktop', 'D:\\Astra Desktop'],
    'astra-desktop': ['D:\\astra-desktop'],
}

# Paths for each scope
FRONTEND_PATHS = ['D:\\orb-desktop']
BACKEND_PATHS = ['D:\\Orb']
ALL_PATHS = ['D:\\orb-desktop', 'D:\\Orb']


def _detect_search_replace_terms(text: str) -> tuple:
    """
    v4.3.2: Detect search and replacement terms to exclude from path matching.
    
    Returns (search_term, replace_term) or (None, None)
    
    v4.3.2: Require at least 3 characters to avoid false positives like 's' -> 'p'
    """
    text_lower = text.lower()
    
    # Pattern: "X to Y" refactor/rename
    # v4.3.2: Use {3,} instead of + to require at least 3 chars
    patterns = [
        r'from\s+["\']?([a-z]{3,})["\']?\s+to\s+["\']?([a-z]{3,})["\']?',
        r'rename\s+["\']?([a-z]{3,})["\']?\s+to\s+["\']?([a-z]{3,})["\']?',
        r'replace\s+["\']?([a-z]{3,})["\']?\s+with\s+["\']?([a-z]{3,})["\']?',
        r'["\']?([a-z]{3,})["\']?\s+to\s+["\']?([a-z]{3,})["\']?\s*\(refactor',
        r'rebrand.*?from\s+["\']?([a-z]{3,})["\']?\s+to\s+["\']?([a-z]{3,})["\']?',
        r'change.*?["\']?([a-z]{3,})["\']?.*?to.*?["\']?([a-z]{3,})["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return (match.group(1), match.group(2))
    
    return (None, None)


def _extract_project_paths(text: str, search_term: str = None, replace_term: str = None) -> List[str]:
    """
    v4.3: Scope-aware path extraction.
    
    Key improvements:
    1. Detect search/replace terms and EXCLUDE them from path matching
    2. Detect explicit scope (UI/frontend vs backend) and RESPECT it
    3. Only match bare 'orb'/'astra' when they're explicit project names
    
    Examples:
    - "change the front-end UI so it's called Astra" -> D:\\orb-desktop ONLY
    - "rename Orb to Astra in Orb Desktop" -> D:\\orb-desktop ONLY  
    - "rename Orb to Astra across the codebase" -> D:\\orb-desktop + D:\\Orb
    """
    if not text:
        return []
    
    text_lower = text.lower()
    paths = []
    
    print(f"[spec_runner] v4.3.4 SCOPE-AWARE PATH EXTRACTION: input={len(text)} chars")
    
    # Step 1: Detect search/replace terms (don't treat these as project names)
    detected_search, detected_replace = _detect_search_replace_terms(text)
    if detected_search:
        search_term = detected_search
        replace_term = detected_replace
        print(f"[spec_runner] v4.3.4 DETECTED SEARCH/REPLACE: '{search_term}' -> '{replace_term}'")
    
    excluded_terms = set()
    if search_term:
        excluded_terms.add(search_term.lower())
    if replace_term:
        excluded_terms.add(replace_term.lower())
    print(f"[spec_runner] v4.3.4 EXCLUDED TERMS: {excluded_terms}")
    
    # Step 2: Check for EXPLICIT scope indicators
    has_frontend_scope = any(pattern in text_lower for pattern in SCOPE_FRONTEND)
    has_backend_scope = any(pattern in text_lower for pattern in SCOPE_BACKEND)
    
    print(f"[spec_runner] v4.3.4 SCOPE: frontend={has_frontend_scope}, backend={has_backend_scope}")
    
    # Step 3: Check for explicit project name patterns
    for pattern, project_paths in EXPLICIT_PROJECT_PATTERNS.items():
        if pattern in text_lower:
            print(f"[spec_runner] v4.3.4 EXPLICIT PROJECT: '{pattern}' -> {project_paths}")
            paths.extend(project_paths)
    
    # Step 4: Check for explicit paths like "D:\orb-desktop" or "D:\Orb"
    # v4.3.4: Only match short, valid folder names (max 20 chars)
    # This prevents garbage like "D:\Orb Desktop front-end UI text"
    for match in re.findall(r'([A-Za-z]:[\\/][A-Za-z][A-Za-z0-9_\-]{0,17})', text):
        cleaned = match.rstrip(' \t')
        # Skip if too short or too long
        if len(cleaned) < 4 or len(cleaned) > 20:
            continue
        # Skip if it contains newlines  
        if '\n' in cleaned or '\r' in cleaned:
            continue
        # Check if this looks like a path to a known project
        name_part = cleaned[3:].lower().replace(' ', '-').replace('\\', '')
        if name_part not in excluded_terms:
            print(f"[spec_runner] v4.3.4 EXPLICIT PATH: '{cleaned}'")
            paths.append(cleaned)
            # Also add hyphenated version
            if ' ' in cleaned:
                drive = cleaned[:3]
                folder = cleaned[3:].replace(' ', '-').lower()
                paths.append(drive + folder)
    
    # Step 5: If no explicit paths found, use scope to determine paths
    if not paths:
        if has_frontend_scope and not has_backend_scope:
            # User explicitly mentioned UI/frontend -> frontend only
            print(f"[spec_runner] v4.3.4 SCOPE-BASED: frontend only")
            paths = FRONTEND_PATHS.copy()
        elif has_backend_scope and not has_frontend_scope:
            # User explicitly mentioned backend -> backend only
            print(f"[spec_runner] v4.3.4 SCOPE-BASED: backend only")
            paths = BACKEND_PATHS.copy()
        elif has_frontend_scope and has_backend_scope:
            # User mentioned both -> all paths
            print(f"[spec_runner] v4.3.4 SCOPE-BASED: both frontend + backend")
            paths = ALL_PATHS.copy()
        # else: no scope indicators and no explicit paths -> return empty
    
    # Step 6: "X drive" + project name detection (fallback)
    if not paths:
        drive_match = re.search(r'\b([A-Za-z])\s+drive\b', text, re.IGNORECASE)
        if drive_match:
            drive = drive_match.group(1).upper()
            # Only match explicit project names, not search terms
            if re.search(r'\borb[\s-]*desktop\b', text_lower) and 'orb' not in excluded_terms:
                paths.extend([f"{drive}:\\orb-desktop"])
            if re.search(r'\bastra[\s-]*desktop\b', text_lower) and 'astra' not in excluded_terms:
                paths.extend([f"{drive}:\\astra-desktop"])
    
    # Dedupe while preserving order
    seen = set()
    unique = []
    for p in paths:
        key = p.lower().replace('/', '\\').rstrip('\\')
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    print(f"[spec_runner] v4.3.4 FINAL PATHS: {unique}")
    return unique


# =============================================================================
# SIMPLE SPEC BUILDER (for non-scan jobs)
# =============================================================================

def _build_simple_spec(
    goal: str,
    what_to_do: str,
    evidence: Optional[Any] = None,
) -> str:
    """
    Build a simple POT spec for CREATE/MODIFY jobs.
    
    No scan results - just what Weaver said to do.
    """
    lines = []
    
    lines.append("# SPoT Spec")
    lines.append("")
    
    lines.append("## Goal")
    lines.append("")
    if goal:
        lines.append(goal.split('\n')[0].strip())
    else:
        lines.append(what_to_do.split('\n')[0].strip() if what_to_do else "Complete the requested task")
    lines.append("")
    
    lines.append("## What to do")
    lines.append("")
    if what_to_do:
        for line in what_to_do.split('\n')[:10]:
            if line.strip():
                lines.append(line.strip())
    lines.append("")
    
    lines.append("## Acceptance")
    lines.append("")
    lines.append("- [ ] Task completed as specified")
    lines.append("- [ ] No errors")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_spec_gate_grounded(
    db: Session,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: int,
    constraints_hint: Optional[Dict] = None,
    spec_version: int = 1,
    user_answers: Optional[Dict[str, str]] = None,
) -> SpecGateResult:
    """
    v4.0: Direct spec builder - NO GATES.
    
    Flow:
    1. Parse Weaver spec
    2. Run scan if needed
    3. Build POT spec
    4. Return result
    
    Only asks questions if something CRITICAL is missing (e.g., no target path).
    """
    try:
        round_n = max(1, min(3, int(spec_version or 1)))
        
        logger.info("[spec_runner] v4.0 Starting: job=%s, round=%d", job_id, round_n)
        print(f"[spec_runner] v4.0 DIRECT PATH: No gates, no classification")
        
        # =================================================================
        # STEP 1: Get Weaver spec
        # =================================================================
        
        weaver_job_text = (constraints_hint or {}).get('weaver_job_description_text', '')
        combined_text = f"{user_intent or ''} {weaver_job_text}"
        
        intent = parse_weaver_intent(constraints_hint or {})
        
        # v4.4: Extract goal with placeholder filtering
        # Priority: 1) Intent goal, 2) weaver text, 3) user intent
        # But NEVER use placeholder text like "Job Description from Weaver"
        goal = ""
        
        # Try intent goal first
        intent_goal = intent.get("goal", "")
        if intent_goal and not _is_placeholder_goal(intent_goal):
            goal = intent_goal
            logger.info("[spec_runner] v4.4 Using goal from intent: %s", goal[:80])
        
        # Fallback to weaver text (but filter out placeholder headers)
        if not goal and weaver_job_text:
            # Try to extract real content from weaver text
            # Skip lines that are just headers/placeholders
            for line in weaver_job_text.split('\n'):
                line = line.strip()
                if line and not _is_placeholder_goal(line):
                    # Found a real line of content
                    goal = line[:200]
                    logger.info("[spec_runner] v4.4 Using goal from weaver text: %s", goal[:80])
                    break
        
        # Final fallback to user intent
        if not goal and user_intent:
            goal = user_intent[:200]
            logger.info("[spec_runner] v4.4 Using goal from user intent: %s", goal[:80])
        
        logger.info("[spec_runner] v4.0 Weaver goal: %s", goal[:100])
        
        # =================================================================
        # STEP 2: Detect if this needs a scan
        # =================================================================
        
        project_paths = _extract_project_paths(combined_text)
        
        multi_file_meta = _detect_multi_file_intent(
            combined_text=combined_text,
            constraints_hint=constraints_hint,
            project_paths=project_paths,
            vision_results=constraints_hint.get('vision_results') if constraints_hint else None,
        )
        
        # =================================================================
        # STEP 3: Run scan if multi-file operation
        # =================================================================
        
        multi_file_op = None
        spot_markdown = None
        
        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            logger.info(
                "[spec_runner] v4.0 Multi-file detected: %s '%s' → '%s'",
                multi_file_meta.get("operation_type"),
                multi_file_meta.get("search_pattern"),
                multi_file_meta.get("replacement_pattern"),
            )
            print(f"[spec_runner] v4.0 SCANNING: {multi_file_meta.get('search_pattern')}")
            
            # Run scan
            multi_file_op = await _build_multi_file_operation(
                operation_type=multi_file_meta.get("operation_type", "search"),
                search_pattern=multi_file_meta.get("search_pattern", ""),
                replacement_pattern=multi_file_meta.get("replacement_pattern", ""),
                file_filter=multi_file_meta.get("file_filter"),
                sandbox_client=None,
                job_description=weaver_job_text or combined_text,
                provider_id=provider_id,
                model_id=model_id,
                explicit_roots=project_paths if project_paths else None,
                vision_context=constraints_hint.get("vision_context", "") if constraints_hint else "",
            )
            
            logger.info(
                "[spec_runner] v4.0 Scan complete: %d files, %d matches",
                multi_file_op.total_files,
                multi_file_op.total_occurrences,
            )
            print(f"[spec_runner] v4.0 FOUND: {multi_file_op.total_occurrences} matches in {multi_file_op.total_files} files")
            
            # =================================================================
            # CRITICAL CHECK: Do we have what we need?
            # =================================================================
            
            if multi_file_op.total_occurrences == 0 and not multi_file_op.error_message:
                # No matches found - this might be a problem
                logger.warning("[spec_runner] v4.0 NO MATCHES found for '%s'", multi_file_meta.get("search_pattern"))
                
                # Ask the user if no matches were found
                return SpecGateResult(
                    ready_for_pipeline=False,
                    open_questions=[
                        f"No matches found for '{multi_file_meta.get('search_pattern')}' in {project_paths}. "
                        f"Is the search term correct? Is the path correct?"
                    ],
                    spec_version=round_n,
                    validation_status="needs_clarification",
                    notes="v4.0: No scan matches found",
                )
            
            if multi_file_op.error_message:
                # Scan error - this is a real problem
                return SpecGateResult(
                    ready_for_pipeline=False,
                    blocking_issues=[f"Scan error: {multi_file_op.error_message}"],
                    spec_version=round_n,
                    validation_status="blocked",
                    notes="v4.0: Scan failed",
                )
            
            # =================================================================
            # STEP 4: Build POT spec from evidence
            # =================================================================
            
            if _DIRECT_BUILDER_AVAILABLE and multi_file_op.raw_matches:
                # Use direct builder - no LLM, no classification
                spot_markdown = build_direct_spec(
                    search_term=multi_file_op.search_pattern,
                    replace_term=multi_file_op.replacement_pattern,
                    raw_matches=multi_file_op.raw_matches,
                    goal=goal,
                    total_files=multi_file_op.total_files,
                )
                logger.info("[spec_runner] v4.0 Direct spec built: %d chars", len(spot_markdown))
                print(f"[spec_runner] v4.0 POT SPEC READY: {len(spot_markdown)} chars")
            else:
                # Fallback: use classification markdown if available
                spot_markdown = multi_file_op.classification_markdown
                if not spot_markdown:
                    # Build minimal spec
                    spot_markdown = f"""# SPoT Spec — {multi_file_op.search_pattern} → {multi_file_op.replacement_pattern}

## Goal
{goal}

## Evidence
Found **{multi_file_op.total_occurrences} occurrences** in **{multi_file_op.total_files} files**

## Replace
- `{multi_file_op.search_pattern}` → `{multi_file_op.replacement_pattern}`

## Acceptance
- [ ] App boots
- [ ] Changes applied
- [ ] No errors
"""
        else:
            # =================================================================
            # Non-scan job: CREATE, MODIFY, etc.
            # v4.1: Use grounded CREATE spec if we have project paths
            # =================================================================
            
            logger.info("[spec_runner] v4.3 Non-scan job, checking for CREATE grounding")
            print(f"[spec_runner] v4.3 NON-SCAN JOB: project_paths={project_paths}")
            
            # Check if we have enough info
            if not goal and not weaver_job_text and not user_intent:
                return SpecGateResult(
                    ready_for_pipeline=False,
                    open_questions=["What would you like me to do?"],
                    spec_version=round_n,
                    validation_status="needs_clarification",
                    notes="v4.1: No goal specified",
                )
            
            # v4.3: Try grounded CREATE spec if we have project paths
            valid_paths = [p for p in project_paths if os.path.isdir(p)]
            print(f"[spec_runner] v4.3 VALID PATHS: {valid_paths}")
            print(f"[spec_runner] v4.3 CREATE_BUILDER_AVAILABLE: {_CREATE_BUILDER_AVAILABLE}")
            
            if _CREATE_BUILDER_AVAILABLE and valid_paths:
                logger.info("[spec_runner] v4.3 Using grounded CREATE builder for paths: %s", valid_paths)
                print(f"[spec_runner] v4.3 GROUNDED CREATE: scanning {len(valid_paths)} project(s)")
                
                try:
                    spot_markdown, create_evidence = await build_grounded_create_spec(
                        goal=goal,
                        what_to_do=weaver_job_text or user_intent,
                        project_paths=valid_paths,
                        sandbox_client=None,
                    )
                    print(f"[spec_runner] v4.3 CREATE SPEC READY: {len(spot_markdown)} chars")
                except Exception as create_err:
                    logger.warning("[spec_runner] v4.3 Grounded CREATE failed, falling back: %s", create_err)
                    print(f"[spec_runner] v4.3 CREATE FAILED: {create_err}")
                    spot_markdown = _build_simple_spec(
                        goal=goal,
                        what_to_do=weaver_job_text or user_intent,
                    )
            else:
                # No project paths or CREATE builder not available - use simple spec
                logger.info("[spec_runner] v4.3 No project paths, using simple spec")
                print(f"[spec_runner] v4.3 FALLBACK: No valid paths or builder unavailable")
                spot_markdown = _build_simple_spec(
                    goal=goal,
                    what_to_do=weaver_job_text or user_intent,
                )
        
        # =================================================================
        # STEP 5: Return result
        # =================================================================
        
        spec_id = f"sg-{uuid.uuid4().hex[:12]}"
        spec_hash = hashlib.sha256(spot_markdown.encode()).hexdigest()
        
        grounding_data = {
            "job_kind": "refactor" if multi_file_op else "other",
            "multi_file": {
                "is_multi_file": multi_file_op.is_multi_file if multi_file_op else False,
                "operation_type": multi_file_op.operation_type if multi_file_op else None,
                "search_pattern": multi_file_op.search_pattern if multi_file_op else None,
                "replacement_pattern": multi_file_op.replacement_pattern if multi_file_op else None,
                "total_files": multi_file_op.total_files if multi_file_op else 0,
                "total_occurrences": multi_file_op.total_occurrences if multi_file_op else 0,
            } if multi_file_op else None,
            "goal": goal,
        }
        
        logger.info("[spec_runner] v4.0 DONE: ready_for_pipeline=True")
        print("[spec_runner] v4.0 SUCCESS: POT spec ready for Implementer")
        
        return SpecGateResult(
            ready_for_pipeline=True,
            open_questions=[],
            spot_markdown=spot_markdown,
            db_persisted=False,
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_version=round_n,
            notes="v4.0: Direct path, no gates",
            blocking_issues=[],
            validation_status="validated",
            grounding_data=grounding_data,
        )
        
    except Exception as e:
        logger.exception("[spec_runner] v4.0 HARD STOP: %s", e)
        return SpecGateResult(
            ready_for_pipeline=False,
            hard_stopped=True,
            hard_stop_reason=str(e),
            spec_version=int(spec_version) if isinstance(spec_version, int) else None,
            validation_status="error",
        )

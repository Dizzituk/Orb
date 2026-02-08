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

import glob
import hashlib
import json
import logging
import os
import re
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SPEC_RUNNER_BUILD_ID = "2026-02-08-v5.3-revert-scope-hardcoding-add-zone-todo"
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
# PATH EXTRACTION - v4.5 DYNAMIC PROJECT DISCOVERY
# =============================================================================
#
# v4.5 (2026-02-04): DYNAMIC PROJECT DISCOVERY
# - Replaced hardcoded EXPLICIT_PROJECT_PATTERNS with architecture-driven discovery
# - Reads INDEX.json from .architecture/ to discover project roots
# - Classifies roots as frontend/backend from file zone metadata
# - Generates product name aliases from folder names + configurable synonyms
# - Falls back to codebase report JSON if INDEX.json unavailable
# - Hardcoded paths kept ONLY as last-resort fallback
#
# Key insight: "Astra" and "Orb" are the same product. Future jobs may be
# for completely different projects. System must discover, not assume.
#

# --- Architecture document locations (configurable via env) ---
_ARCH_INDEX_DIR = os.getenv("ASTRA_ARCH_INDEX_DIR", os.path.join("D:\\", "Orb", ".architecture"))
_ARCH_REPORT_DIR = os.getenv("ASTRA_ARCH_REPORT_DIR", os.path.join("D:\\", "Orb.architecture"))

# --- Product synonyms: names that refer to the same product ---
# Format: comma-separated pairs like "orb=astra,foo=bar"
# These are BIDIRECTIONAL: orb=astra means both 'orb' and 'astra' map to the same roots
_PRODUCT_SYNONYMS_RAW = os.getenv("ASTRA_PRODUCT_SYNONYMS", "orb=astra")


def _parse_product_synonyms() -> Dict[str, List[str]]:
    """Parse product synonyms from env config into a lookup table.

    Returns dict mapping each name to ALL its synonyms (including itself).
    E.g., {"orb": ["orb", "astra"], "astra": ["orb", "astra"]}
    """
    synonyms: Dict[str, List[str]] = {}
    if not _PRODUCT_SYNONYMS_RAW:
        return synonyms

    for pair in _PRODUCT_SYNONYMS_RAW.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        names = [n.strip().lower() for n in pair.split("=") if n.strip()]
        # Each name maps to the full group
        for name in names:
            synonyms[name] = names
    return synonyms


_PRODUCT_SYNONYMS = _parse_product_synonyms()


def _generate_aliases_for_root(folder_name: str, root_path: str) -> List[str]:
    """Generate product name aliases for a discovered root path.

    For folder 'orb-desktop' with synonyms orb=astra:
      → ['orb desktop', 'orb-desktop', 'astra desktop', 'astra-desktop']
    For folder 'Orb' with synonyms orb=astra:
      → ['orb', 'astra']
    For folder 'my-project' with no synonyms:
      → ['my project', 'my-project']
    """
    aliases: List[str] = []
    folder_lower = folder_name.lower()

    # Split folder name into base + suffix: "orb-desktop" → ("orb", "desktop")
    parts = re.split(r'[-_\s]', folder_lower)
    base_name = parts[0]
    suffix = '-'.join(parts[1:]) if len(parts) > 1 else ''

    # Get all name variants (base name + its synonyms)
    name_variants = _PRODUCT_SYNONYMS.get(base_name, [base_name])

    for variant in name_variants:
        if suffix:
            aliases.append(f"{variant} {suffix}")   # "orb desktop" / "astra desktop"
            aliases.append(f"{variant}-{suffix}")    # "orb-desktop" / "astra-desktop"
        else:
            aliases.append(variant)                   # "orb" / "astra"

    return aliases


@lru_cache(maxsize=1)
def _discover_project_roots() -> Dict[str, Any]:
    """Dynamically discover project roots from architecture index.

    Reads the ground-truth architecture documents to find:
    - Which directories are project roots
    - Which are frontend vs backend
    - What product names map to which paths

    Sources (in priority order):
    1. INDEX.json from .architecture/ (structured, has zone metadata)
    2. CODEBASE_REPORT_FULL_*.json from .architecture/ (scan metadata)
    3. Hardcoded fallback (last resort)

    Returns dict with:
        roots: list of absolute paths to project roots
        frontend_paths: list of frontend root paths
        backend_paths: list of backend root paths
        all_paths: combined list
        aliases: dict mapping product name patterns → list of root paths
    """
    result: Dict[str, Any] = {
        "roots": [],
        "frontend_paths": [],
        "backend_paths": [],
        "all_paths": [],
        "aliases": {},
        "source": "none",
    }

    # --- Source 1: INDEX.json (best - has per-file zone metadata) ---
    index_path = os.path.join(_ARCH_INDEX_DIR, "INDEX.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            roots = index_data.get("roots", [])
            # Filter to roots that actually exist on disk
            roots = [r for r in roots if os.path.isdir(r)]

            if roots:
                result["roots"] = roots
                result["all_paths"] = roots
                result["source"] = f"INDEX.json ({index_path})"

                # Classify roots by zone from file entries
                frontend_roots: set = set()
                backend_roots: set = set()
                for file_entry in index_data.get("files", [])[:200]:  # Sample first 200 for speed
                    zone = file_entry.get("zone", "")
                    root = file_entry.get("root", "")
                    if zone == "frontend" and root:
                        frontend_roots.add(root)
                    elif zone == "backend" and root:
                        backend_roots.add(root)

                # Use zone metadata if available, else heuristic
                if frontend_roots:
                    result["frontend_paths"] = [r for r in roots if r in frontend_roots]
                else:
                    result["frontend_paths"] = [r for r in roots if 'desktop' in r.lower()]

                if backend_roots:
                    result["backend_paths"] = [r for r in roots if r in backend_roots]
                else:
                    result["backend_paths"] = [r for r in roots if 'desktop' not in r.lower()]

                print(f"[spec_runner] v4.5 DISCOVERY from INDEX.json: roots={roots}")
        except Exception as e:
            print(f"[spec_runner] v4.5 Failed to read INDEX.json: {e}")

    # --- Source 2: Codebase report JSON (fallback) ---
    if not result["roots"]:
        report_pattern = os.path.join(_ARCH_REPORT_DIR, "CODEBASE_REPORT_FULL_*.json")
        report_files = sorted(glob.glob(report_pattern), reverse=True)
        if report_files:
            try:
                with open(report_files[0], 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                roots = report_data.get("metadata", {}).get("roots_scanned", [])
                roots = [r for r in roots if os.path.isdir(r)]

                if roots:
                    result["roots"] = roots
                    result["all_paths"] = roots
                    result["source"] = f"CODEBASE_REPORT ({report_files[0]})"
                    result["frontend_paths"] = [r for r in roots if 'desktop' in r.lower()]
                    result["backend_paths"] = [r for r in roots if 'desktop' not in r.lower()]
                    print(f"[spec_runner] v4.5 DISCOVERY from codebase report: roots={roots}")
            except Exception as e:
                print(f"[spec_runner] v4.5 Failed to read codebase report: {e}")

    # --- Source 3: Hardcoded fallback (last resort) ---
    if not result["roots"]:
        print("[spec_runner] v4.5 WARNING: No architecture index found, using hardcoded fallback")
        result["roots"] = ['D:\\orb-desktop', 'D:\\Orb']
        result["frontend_paths"] = ['D:\\orb-desktop']
        result["backend_paths"] = ['D:\\Orb']
        result["all_paths"] = ['D:\\orb-desktop', 'D:\\Orb']
        result["source"] = "hardcoded_fallback"

    # --- Build product name aliases from discovered roots ---
    for root in result["roots"]:
        folder_name = os.path.basename(root)
        if not folder_name:  # Handle "D:\\" edge case
            continue
        aliases = _generate_aliases_for_root(folder_name, root)
        for alias in aliases:
            if alias not in result["aliases"]:
                result["aliases"][alias] = []
            if root not in result["aliases"][alias]:
                result["aliases"][alias].append(root)

    print(f"[spec_runner] v4.5 DISCOVERY COMPLETE: "
          f"roots={result['roots']}, "
          f"frontend={result['frontend_paths']}, "
          f"backend={result['backend_paths']}, "
          f"aliases={list(result['aliases'].keys())}, "
          f"source={result['source']}")

    return result


# --- Scope indicators: UI/frontend vs backend ---
# Key insight: If user explicitly says "UI" or "frontend", DON'T include backend
#
# v4.6: TIGHTENED FRONTEND DETECTION
# Only set frontend=True when the user requests CHANGES to the frontend.
# Merely MENTIONING the frontend (e.g., "the desktop app will call it",
# "the frontend will handle sending") does NOT mean frontend scope.
# Removed: 'the app', 'desktop app', "app's" — too broad, triggers on
# consumer/client mentions without requesting frontend code changes.
#
SCOPE_FRONTEND = {
    # UI modification patterns (user wants to CHANGE the UI)
    'the ui': True, 'on the ui': True, 'in the ui': True, 'to the ui': True,
    'a ui': True, 'ui button': True, 'ui feature': True, 'ui text': True,
    'change the ui': True, 'modify the ui': True, 'update the ui': True,
    'the frontend': True, 'front-end': True, 'frontend ui': True,
    # v5.2: REMOVED 'text input' — too ambiguous.
    # 'text input' in AI/ML context means text-type data ingestion, not a UI field.
    # Also REMOVED 'context window' and 'input window' in v4.7 for same reason.
    # Users requesting frontend changes will say 'the UI', 'the frontend', etc.
    # Electron is specific enough — only mentioned when discussing frontend code
    'electron': True,
    # NOTE: 'the app', 'desktop app', "app's" deliberately REMOVED.
    # These match consumer mentions like "works from the desktop app".
}

SCOPE_BACKEND = {
    'the backend': True, 'back-end': True, 'backend api': True,
    'fastapi': True, 'api endpoint': True, 'server': True,
    # NOTE: Static keyword dicts are a dead end for scope detection.
    # The real fix is to use INDEX.json zone data (every file is already
    # tagged as 'backend' or 'frontend'). When the job mentions files or
    # modules, look up their zone. That's intelligent scope detection.
    # TODO: Replace keyword-based scope with zone-aware detection using
    # the integration points already extracted by the spec runner.
}

# LEGACY FALLBACK: Only used if dynamic discovery fails completely
_FALLBACK_FRONTEND_PATHS = ['D:\\orb-desktop']
_FALLBACK_BACKEND_PATHS = ['D:\\Orb']
_FALLBACK_ALL_PATHS = ['D:\\orb-desktop', 'D:\\Orb']


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
    
    # Step 3: Check for project name patterns via DYNAMIC DISCOVERY
    # v4.5: Uses architecture index instead of hardcoded patterns
    discovery = _discover_project_roots()
    for alias, alias_paths in discovery["aliases"].items():
        if alias in text_lower:
            # Don't match aliases that are search/replace terms
            if alias not in excluded_terms:
                print(f"[spec_runner] v4.5 DISCOVERED PROJECT: '{alias}' -> {alias_paths}")
                paths.extend(alias_paths)
    
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
    
    # Step 5: If no explicit paths found, use scope with DISCOVERED paths
    # v4.5: Uses dynamically discovered frontend/backend paths
    if not paths:
        fe_paths = discovery["frontend_paths"] or _FALLBACK_FRONTEND_PATHS
        be_paths = discovery["backend_paths"] or _FALLBACK_BACKEND_PATHS
        all_paths = discovery["all_paths"] or _FALLBACK_ALL_PATHS

        if has_frontend_scope and not has_backend_scope:
            # User explicitly mentioned UI/frontend -> frontend only
            print(f"[spec_runner] v4.5 SCOPE-BASED: frontend only -> {fe_paths}")
            paths = list(fe_paths)
        elif has_backend_scope and not has_frontend_scope:
            # User explicitly mentioned backend -> backend only
            print(f"[spec_runner] v4.5 SCOPE-BASED: backend only -> {be_paths}")
            paths = list(be_paths)
        elif has_frontend_scope and has_backend_scope:
            # User mentioned both -> all paths
            print(f"[spec_runner] v4.5 SCOPE-BASED: both frontend + backend -> {all_paths}")
            paths = list(all_paths)
        # else: no scope indicators and no explicit paths -> return empty
    
    # Step 6: "X drive" + project name detection (fallback)
    # v4.5: Uses discovered aliases to resolve "D drive" + project name
    if not paths:
        drive_match = re.search(r'\b([A-Za-z])\s+drive\b', text, re.IGNORECASE)
        if drive_match:
            drive = drive_match.group(1).upper()
            # Check discovered aliases for project name patterns in the text
            for alias, alias_paths in discovery["aliases"].items():
                # Only check multi-word aliases ("orb desktop", not bare "orb")
                if ' ' in alias or '-' in alias:
                    # Build regex from alias: "orb desktop" -> r'\borb[\s-]*desktop\b'
                    alias_parts = re.split(r'[-\s]', alias)
                    alias_pattern = r'\b' + r'[\s-]*'.join(re.escape(p) for p in alias_parts) + r'\b'
                    if re.search(alias_pattern, text_lower) and alias_parts[0] not in excluded_terms:
                        # Use the discovered root but on the specified drive
                        for ap in alias_paths:
                            # Replace drive letter with user-specified drive
                            folder_part = ap[2:]  # Strip "D:" prefix
                            paths.append(f"{drive}:{folder_part}")
                            print(f"[spec_runner] v4.5 DRIVE+ALIAS: '{alias}' on {drive}: -> {drive}:{folder_part}")
    
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
# v4.7: ER DEDUPLICATION — collapse duplicate EVIDENCE_REQUEST blocks by id
# =============================================================================
#
# LLM outputs sometimes emit the same ER block twice (e.g., ER-001 appears
# in both scaffold and LLM analysis sections). Duplicate ERs confuse the
# Critical Pipeline and inflate the CRITICAL ER count.
#
# Strategy: Parse all EVIDENCE_REQUEST blocks from the spec markdown,
# keep the first occurrence of each id, drop duplicates, and reconstruct.
#

def _dedup_evidence_requests(spec_markdown: str) -> str:
    """
    v4.7: Remove duplicate EVIDENCE_REQUEST blocks from spec markdown.

    Scans for EVIDENCE_REQUEST blocks (delimited by 'EVIDENCE_REQUEST:' headers),
    extracts the 'id' field from each, and removes duplicates (keeping first occurrence).

    Returns the cleaned markdown with duplicates removed.
    """
    if not spec_markdown or 'EVIDENCE_REQUEST' not in spec_markdown:
        return spec_markdown

    lines = spec_markdown.split('\n')
    output_lines = []
    seen_er_ids = set()
    in_er_block = False
    er_block_lines = []
    er_block_id = None
    skip_current_block = False
    duplicates_removed = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect start of an EVIDENCE_REQUEST block
        if stripped.startswith('EVIDENCE_REQUEST:') or stripped == 'EVIDENCE_REQUEST:':
            # If we were already in an ER block, flush it
            if in_er_block and er_block_lines:
                if not skip_current_block:
                    output_lines.extend(er_block_lines)
                else:
                    duplicates_removed += 1

            # Start new ER block
            in_er_block = True
            er_block_lines = [line]
            er_block_id = None
            skip_current_block = False
            i += 1
            continue

        if in_er_block:
            # Check if this line has the id field
            id_match = re.match(r'\s+id:\s*["\']?(ER-[\w-]+)["\']?', line)
            if id_match:
                er_block_id = id_match.group(1)
                if er_block_id in seen_er_ids:
                    skip_current_block = True
                    logger.info("[spec_runner] v4.7 DEDUP: dropping duplicate %s", er_block_id)
                    print(f"[spec_runner] v4.7 ER DEDUP: dropping duplicate {er_block_id}")
                else:
                    seen_er_ids.add(er_block_id)

            # Check if we've hit the end of this ER block
            # An ER block ends when we hit: another EVIDENCE_REQUEST, a markdown header,
            # an empty line followed by non-indented content, or end of file
            next_is_new_section = False
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                next_is_new_section = (
                    next_stripped.startswith('EVIDENCE_REQUEST:') or
                    next_stripped.startswith('# ') or
                    next_stripped.startswith('## ') or
                    next_stripped.startswith('### ') or
                    # A new top-level YAML key after the ER block (not indented)
                    (next_stripped and not next_stripped.startswith(' ') and
                     not next_stripped.startswith('-') and
                     not next_stripped.startswith('EVIDENCE_REQUEST') and
                     ':' not in next_stripped and
                     stripped == '')
                )

            er_block_lines.append(line)

            # If next line starts a new section, or we're at EOF, flush this block
            if next_is_new_section or i == len(lines) - 1:
                if not skip_current_block:
                    output_lines.extend(er_block_lines)
                else:
                    duplicates_removed += 1
                in_er_block = False
                er_block_lines = []
                er_block_id = None
                skip_current_block = False
        else:
            output_lines.append(line)

        i += 1

    # Flush any remaining ER block
    if in_er_block and er_block_lines:
        if not skip_current_block:
            output_lines.extend(er_block_lines)
        else:
            duplicates_removed += 1

    if duplicates_removed > 0:
        print(f"[spec_runner] v4.7 ER DEDUP COMPLETE: removed {duplicates_removed} duplicate block(s), "
              f"{len(seen_er_ids)} unique ER(s) remain")
        logger.info("[spec_runner] v4.7 ER dedup: removed %d duplicate(s), %d unique remain",
                    duplicates_removed, len(seen_er_ids))

    return '\n'.join(output_lines)


# =============================================================================
# SEGMENTATION HELPERS (v4.8 — Pipeline Segmentation Phase 1)
# =============================================================================

def _get_job_dir_for_segmentation(job_id: str) -> str:
    """
    v4.9 PHASE 2: Get job directory path for segmentation manifest references.
    Uses the same path construction as _write_segmentation_output().
    """
    try:
        from ..spec_gate_persistence import artifact_root as _ar, job_dir as _jd
        return _jd(_ar(), job_id)
    except ImportError:
        _root = os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))
        return os.path.join(_root, "jobs", job_id)


def _extract_file_scope_from_spec(
    spec_markdown: Optional[str],
    grounding_data: Optional[Dict] = None,
    multi_file_op: Optional[Any] = None,
) -> List[str]:
    """
    v4.9: Extract file paths mentioned as targets in the spec.
    
    Looks for patterns like:
    - Relative paths: `app/foo/bar.py`, `src/components/Foo.tsx`
    - Absolute Windows paths: `D:\Orb\app\foo.py`, `D:/Orb/app/foo.py`
    - Backtick-wrapped paths in markdown
    - Paths from multi_file_op target files
    - Paths from grounding_data if available
    
    v4.9 (2026-02-08): Added absolute path extraction and multi_file_op support.
    Previous version only matched relative paths starting with app/, src/, etc.
    This meant CREATE jobs (which use absolute paths from simple_create.py)
    returned empty file scope and segmentation never triggered.
    """
    paths: List[str] = []
    seen_normalised: set = set()  # Normalised keys for dedup
    
    def _add_path(p: str) -> None:
        """Add path with dedup (case-insensitive, separator-normalised)."""
        key = p.lower().replace('/', '\\').rstrip('\\')
        if key not in seen_normalised:
            seen_normalised.add(key)
            paths.append(p)
    
    if not spec_markdown:
        # Still check multi_file_op even without spec markdown
        if multi_file_op and hasattr(multi_file_op, 'raw_matches'):
            for match_entry in (multi_file_op.raw_matches or []):
                if isinstance(match_entry, dict) and 'file' in match_entry:
                    _add_path(match_entry['file'])
        return paths
    
    # Pattern 1: Relative paths (existing — app/, src/, etc.)
    rel_pattern = re.compile(
        r'(?:^|[\s`|])'
        r'((?:app|src|orb-desktop|tests|scripts|config)[/\\]'
        r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|md|css))'
        r'(?:[\s`|,]|$)',
        re.MULTILINE,
    )
    for match in rel_pattern.finditer(spec_markdown):
        _add_path(match.group(1).replace('/', os.sep))
    
    # Pattern 2: Absolute Windows paths (D:\Orb\app\foo.py or D:/Orb/app/foo.py)
    # Grounded CREATE specs from simple_create.py use full absolute paths.
    abs_pattern = re.compile(
        r'(?:^|[\s`|])'
        r'([A-Za-z]:[/\\]'
        r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|md|css))'
        r'(?:[\s`|,]|$)',
        re.MULTILINE,
    )
    for match in abs_pattern.finditer(spec_markdown):
        abs_path = match.group(1)
        _add_path(abs_path)
        # Also extract the relative portion for layer classification.
        # e.g. D:\Orb\app\services\foo.py → app\services\foo.py
        # This ensures classify_file_layer() works correctly since it
        # matches on relative path patterns like "app/services/".
        normalised = abs_path.replace('/', '\\')
        for prefix in ('app\\', 'src\\'):
            idx = normalised.lower().find(prefix)
            if idx >= 0:
                rel_part = normalised[idx:]
                _add_path(rel_part)
                break
    
    # Pattern 3: Extract from multi_file_op target files if available
    if multi_file_op:
        if hasattr(multi_file_op, 'raw_matches'):
            for match_entry in (multi_file_op.raw_matches or []):
                if isinstance(match_entry, dict) and 'file' in match_entry:
                    _add_path(match_entry['file'])
        # Also check target_files attribute directly
        if hasattr(multi_file_op, 'target_files'):
            for f in (multi_file_op.target_files or []):
                if isinstance(f, str):
                    _add_path(f)
    
    # Pattern 4: Extract from grounding_data if provided
    if grounding_data and isinstance(grounding_data, dict):
        multi = grounding_data.get('multi_file', {})
        if multi and isinstance(multi, dict):
            for f in multi.get('target_files', []):
                if isinstance(f, str):
                    _add_path(f)
    
    return paths


def _extract_requirements_from_spec(spec_markdown: Optional[str]) -> List[str]:
    """
    v4.8: Extract requirement lines from spec markdown.
    
    Looks for bullet points under Goal, Requirements, What to do sections.
    """
    requirements: List[str] = []
    if not spec_markdown:
        return requirements
    
    in_section = False
    for line in spec_markdown.split('\n'):
        stripped = line.strip()
        # Detect relevant sections
        if stripped.startswith('## ') and any(
            kw in stripped.lower() for kw in ['goal', 'requirement', 'what to do', 'scope']
        ):
            in_section = True
            continue
        elif stripped.startswith('## '):
            in_section = False
            continue
        
        if in_section and stripped and (stripped.startswith('- ') or stripped.startswith('* ')):
            req_text = stripped.lstrip('- *').strip()
            if req_text and len(req_text) > 5:  # Skip trivially short items
                requirements.append(req_text)
    
    return requirements


def _extract_acceptance_from_spec(spec_markdown: Optional[str]) -> List[str]:
    """
    v4.8: Extract acceptance criteria from spec markdown.
    
    Looks for checkbox items under Acceptance sections.
    """
    criteria: List[str] = []
    if not spec_markdown:
        return criteria
    
    in_section = False
    for line in spec_markdown.split('\n'):
        stripped = line.strip()
        if stripped.startswith('## ') and any(
            kw in stripped.lower() for kw in ['acceptance', 'verification', 'criteria']
        ):
            in_section = True
            continue
        elif stripped.startswith('## '):
            in_section = False
            continue
        
        if in_section and stripped:
            # Match checkbox items: - [ ] or - [x] or just bullet points
            check_match = re.match(r'^[-*]\s*\[.?\]\s*(.*)', stripped)
            if check_match:
                criteria.append(check_match.group(1).strip())
            elif stripped.startswith('- ') or stripped.startswith('* '):
                criteria.append(stripped.lstrip('- *').strip())
    
    return criteria


def _write_segmentation_output(job_id: str, manifest) -> None:
    """
    v4.9: Write manifest and segment specs to the job directory.
    
    Creates:
        <artifact_root>/jobs/<job-id>/segments/manifest.json
        <artifact_root>/jobs/<job-id>/segments/seg-XX/spec.json (per segment)
    
    v4.9 (2026-02-08): Uses spec_gate_persistence.artifact_root() + job_dir()
    for path construction instead of relative paths. The previous version used
    os.path.join('jobs', job_id) which resolves relative to cwd — wrong when
    the FastAPI server's working directory differs from the project root.
    """
    from .segment_schemas import SegmentManifest
    
    # Use the same path construction as spec_gate_persistence.py
    try:
        from ..spec_gate_persistence import artifact_root as _artifact_root, job_dir as _job_dir
        job_dir_path = _job_dir(_artifact_root(), job_id)
    except ImportError:
        # Fallback: replicate the logic directly
        _root = os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))
        job_dir_path = os.path.join(_root, "jobs", job_id)
    
    os.makedirs(job_dir_path, exist_ok=True)
    
    segments_dir = os.path.join(job_dir_path, 'segments')
    os.makedirs(segments_dir, exist_ok=True)
    
    # Write manifest
    manifest_path = os.path.join(segments_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(manifest.to_json(indent=2))
    
    logger.info("[spec_runner] v4.8 Wrote manifest: %s", manifest_path)
    
    # Write per-segment specs
    for seg in manifest.segments:
        seg_dir = os.path.join(segments_dir, seg.segment_id)
        os.makedirs(seg_dir, exist_ok=True)
        seg_spec_path = os.path.join(seg_dir, 'spec.json')
        with open(seg_spec_path, 'w', encoding='utf-8') as f:
            f.write(seg.to_json(indent=2))
        logger.info("[spec_runner] v4.8 Wrote segment spec: %s", seg_spec_path)


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
                        provider_id=provider_id,
                        model_id=model_id,
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
        # STEP 4b: Segmentation check (Phase 1 — Pipeline Segmentation)
        # =================================================================
        # If the spec is large enough, SpecGate decomposes it into segments.
        # Each segment goes through the pipeline independently.
        # If segmentation fails validation, we fall back to single-pass.
        
        segmentation_manifest = None
        try:
            from .segmentation import needs_segmentation, generate_segments
            
            # Extract file scope from the spec (all files mentioned as targets)
            _file_scope = _extract_file_scope_from_spec(
                spot_markdown, grounding_data=None, multi_file_op=multi_file_op,
            )
            
            if _file_scope:
                _should_segment, _seg_reason = needs_segmentation(_file_scope)
                if _should_segment:
                    logger.info("[spec_runner] v4.8 Segmentation triggered: %s", _seg_reason)
                    print(f"[spec_runner] v4.8 SEGMENTATION: {_seg_reason}")
                    
                    # Extract requirements and acceptance criteria from spec
                    _requirements = _extract_requirements_from_spec(spot_markdown)
                    _acceptance = _extract_acceptance_from_spec(spot_markdown)
                    
                    segmentation_manifest = generate_segments(
                        file_scope=_file_scope,
                        requirements=_requirements,
                        acceptance_criteria=_acceptance,
                        parent_spec_id=f"sg-{uuid.uuid4().hex[:12]}",
                        parent_spec_hash=hashlib.sha256(spot_markdown.encode()).hexdigest() if spot_markdown else None,
                    )
                    
                    if segmentation_manifest:
                        # Write manifest and segment specs to job directory
                        _write_segmentation_output(job_id, segmentation_manifest)
                        logger.info(
                            "[spec_runner] v4.8 Segmentation complete: %s",
                            segmentation_manifest.summary(),
                        )
                        print(f"[spec_runner] v4.8 SEGMENTED: {segmentation_manifest.summary()}")

                        # v4.9 PHASE 2: Return early with "segmented" status.
                        # This prevents the spec from falling through to single-pass.
                        # The caller (spec_gate_stream.py) routes to the segment loop
                        # instead of the critical pipeline.
                        _seg_spec_id = f"sg-{uuid.uuid4().hex[:12]}"
                        _seg_spec_hash = hashlib.sha256(spot_markdown.encode()).hexdigest() if spot_markdown else ""
                        _seg_grounding = {
                            "job_kind": "architecture",
                            "job_kind_confidence": 0.9,
                            "job_kind_reason": "Segmented job — Phase 2 segment loop",
                            "goal": goal,
                            "segmentation": {
                                "segmented": True,
                                "total_segments": segmentation_manifest.total_segments,
                                "segment_ids": [s.segment_id for s in segmentation_manifest.segments],
                                "manifest_path": os.path.join(
                                    _get_job_dir_for_segmentation(job_id),
                                    'segments', 'manifest.json',
                                ),
                            },
                        }
                        logger.info(
                            "[spec_runner] v4.9 PHASE 2: Returning segmented result for segment loop"
                        )
                        print("[spec_runner] v4.9 PHASE 2: Segmented — routing to segment loop")
                        return SpecGateResult(
                            ready_for_pipeline=True,
                            open_questions=[],
                            spot_markdown=spot_markdown,
                            db_persisted=False,
                            spec_id=_seg_spec_id,
                            spec_hash=_seg_spec_hash,
                            spec_version=round_n,
                            notes="v4.9: Job segmented — use segment loop for execution",
                            blocking_issues=[],
                            validation_status="segmented",
                            grounding_data=_seg_grounding,
                        )
                    else:
                        logger.info("[spec_runner] v4.8 Segmentation returned None — single pass")
                        print("[spec_runner] v4.8 Segmentation validation failed or not needed — single pass")
                else:
                    logger.info("[spec_runner] v4.8 No segmentation needed: %s", _seg_reason)
        except ImportError:
            logger.debug("[spec_runner] v4.8 Segmentation module not available")
        except Exception as seg_err:
            # Segmentation failure is NEVER fatal — fall back to single pass
            logger.warning("[spec_runner] v4.8 Segmentation failed (non-fatal): %s", seg_err)
            print(f"[spec_runner] v4.8 SEGMENTATION FAILED (non-fatal): {seg_err}")
            segmentation_manifest = None
        
        # =================================================================
        # STEP 5: Return result
        # =================================================================
        
        spec_id = f"sg-{uuid.uuid4().hex[:12]}"
        spec_hash = hashlib.sha256(spot_markdown.encode()).hexdigest()
        
        # v4.8: Proper job_kind classification for grounding_data
        # CREATE jobs that went through simple_create should be classified as
        # "architecture" so Critical Pipeline routes them correctly.
        # Previously hardcoded to "other" which caused 0.0 confidence and
        # downstream parsing failures.
        if multi_file_op:
            _job_kind = "refactor"
            _job_kind_confidence = 0.85
            _job_kind_reason = "Multi-file operation detected"
        elif spot_markdown and _CREATE_BUILDER_AVAILABLE and valid_paths:
            _job_kind = "architecture"
            _job_kind_confidence = 0.9
            _job_kind_reason = "Grounded CREATE spec with project paths"
        else:
            _job_kind = "other"
            _job_kind_confidence = 0.5
            _job_kind_reason = "Simple spec without grounded evidence"
        
        grounding_data = {
            "job_kind": _job_kind,
            "job_kind_confidence": _job_kind_confidence,
            "job_kind_reason": _job_kind_reason,
            "multi_file": {
                "is_multi_file": multi_file_op.is_multi_file if multi_file_op else False,
                "operation_type": multi_file_op.operation_type if multi_file_op else None,
                "search_pattern": multi_file_op.search_pattern if multi_file_op else None,
                "replacement_pattern": multi_file_op.replacement_pattern if multi_file_op else None,
                "total_files": multi_file_op.total_files if multi_file_op else 0,
                "total_occurrences": multi_file_op.total_occurrences if multi_file_op else 0,
            } if multi_file_op else None,
            "goal": goal,
            "segmentation": {
                "segmented": segmentation_manifest is not None,
                "total_segments": segmentation_manifest.total_segments if segmentation_manifest else 0,
                "segment_ids": [s.segment_id for s in segmentation_manifest.segments] if segmentation_manifest else [],
            } if segmentation_manifest is not None else None,
        }
        
        # =================================================================
        # v4.7: DEDUP EVIDENCE_REQUESTs before counting
        # =================================================================
        if spot_markdown:
            spot_markdown = _dedup_evidence_requests(spot_markdown)

        # =================================================================
        # v5.0: STATUS SEMANTICS — check for unfulfilled EVIDENCE_REQUESTs
        # =================================================================
        # v4.0 of simple_create.py now fulfils ERs during spec generation,
        # so CRITICAL ERs should no longer appear in the final spec. Any
        # surviving ERs are either:
        #   a) Force-resolved (FORCED_RESOLUTION markers) — already handled
        #   b) Edge cases where fulfilment wasn't available (import failure)
        #
        # v5.0 CHANGE: NEVER return "pending_evidence" — it caused a deadlock
        # where SpecGate and Critical Pipeline each told the user to go to the
        # other. Instead:
        #   - No EVIDENCE_REQUESTs → validated
        #   - Surviving CRITICAL ERs → force-resolve them HERE as a safety net,
        #     then set validated_with_gaps (proceeds with honest acknowledgment)
        #   - Only non-CRITICAL ERs → validated (nice-to-have, not blocking)
        
        has_critical_er = False
        critical_er_count = 0
        if spot_markdown:
            # v4.6.1: Robust CRITICAL detection — multiple strategies to avoid
            # false negatives from YAML formatting variations.
            # Catches: severity: "CRITICAL", severity: CRITICAL,
            #          severity:CRITICAL, severity : 'critical', etc.
            
            _spot_lower = spot_markdown.lower()
            
            # Strategy 1: Line-level scan (handles any indentation/quoting)
            # Look for lines containing both "severity" and "critical"
            for line in _spot_lower.split('\n'):
                stripped = line.strip()
                if stripped.startswith('severity') and 'critical' in stripped:
                    critical_er_count += 1
            
            # Strategy 2: Regex fallback (catches inline/compact YAML)
            # Only used if Strategy 1 found nothing — avoids double-counting
            if critical_er_count == 0:
                er_blocks = re.findall(
                    r'severity\s*:\s*["\']?critical["\']?',
                    _spot_lower,
                )
                critical_er_count = len(er_blocks)
            
            if critical_er_count > 0:
                has_critical_er = True
                print(f"[spec_runner] v5.0 CRITICAL EVIDENCE_REQUEST survived fulfilment: {critical_er_count} block(s)")
                logger.warning(
                    "[spec_runner] v5.0 Spec has %d CRITICAL EVIDENCE_REQUEST(s) after fulfilment — force-resolving",
                    critical_er_count
                )
        
        if has_critical_er:
            # v5.0: Force-resolve surviving CRITICAL ERs instead of deadlocking
            # Import the stripping utility to convert ERs to FORCED_RESOLUTION markers
            try:
                from app.llm.pipeline.evidence_loop import (
                    parse_evidence_requests,
                    strip_forced_stop_requests,
                )
                remaining_ers = parse_evidence_requests(spot_markdown)
                if remaining_ers:
                    remaining_ids = {r.get("id", "UNKNOWN") for r in remaining_ers}
                    spot_markdown = strip_forced_stop_requests(spot_markdown, remaining_ids)
                    # Add a visible note to the spec about unfulfilled evidence
                    gap_note = (
                        "\n\n## ⚠️ Evidence Gaps\n\n"
                        "The following evidence requests could not be fulfilled during spec generation "
                        "and have been force-resolved. The Critical Pipeline's architecture stage "
                        "should gather this evidence directly.\n\n"
                    )
                    for r in remaining_ers:
                        gap_note += f"- **{r.get('id', '?')}**: {r.get('need', 'No description')}\n"
                    spot_markdown += gap_note
                    logger.info("[spec_runner] v5.0 Force-resolved %d surviving ER(s): %s",
                                len(remaining_ids), remaining_ids)
                    print(f"[spec_runner] v5.0 Force-resolved {len(remaining_ids)} surviving ER(s)")
            except ImportError as _imp_err:
                logger.warning("[spec_runner] v5.0 Cannot import evidence_loop for force-resolve: %s", _imp_err)
                # Can't strip, but still don't deadlock — proceed with gaps acknowledged
            
            final_status = "validated_with_gaps"
            final_notes = "v5.0: Spec has force-resolved CRITICAL EVIDENCE_REQUESTs. Proceeding with acknowledged gaps."
            print("[spec_runner] v5.0 STATUS: validated_with_gaps (CRITICAL ERs force-resolved)")
        else:
            final_status = "validated"
            final_notes = "v5.0: Direct path, evidence fulfilled"
            print("[spec_runner] v5.0 SUCCESS: POT spec ready for pipeline")
        
        logger.info("[spec_runner] v4.6 DONE: ready_for_pipeline=True, status=%s", final_status)
        
        return SpecGateResult(
            ready_for_pipeline=True,
            open_questions=[],
            spot_markdown=spot_markdown,
            db_persisted=False,
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_version=round_n,
            notes=final_notes,
            blocking_issues=[],
            validation_status=final_status,
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

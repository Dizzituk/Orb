# FILE: app/orchestrator/arch_template/engine.py
# Purpose: Architecture Template Engine — main entry point.
# Called-by: app.agentic_pipeline.pipeline, app.llm.critical_pipeline.stream_handler, app.pipeline_v2.stages.architect
# Depends-on: app.orchestrator.arch_template.sections, app.orchestrator.cross_segment_interfaces
# Last-renovated: 2026-06-11
"""
Architecture Template Engine — main entry point.

v1.0 (2026-03-01): Assembles a partially-filled architecture document
from deterministic sources. The LLM then completes only the [LLM_FILL]
sections instead of generating the entire document from scratch.

Deterministic sections (~65-70%):
  - File Inventory (from segment spec + grounding data)
  - Import Map (from skeleton contracts)
  - Dependency Context (from skeleton contracts)
  - Design Tokens (from CSS registry)
  - Pattern Reference (from evidence files)
  - Frozen Import blocks (from deterministic_imports)

LLM sections (~30-35%):
  - Architecture Vision (creative description)
  - Detailed File-by-File Architecture (component logic, props, code)
  - Critical Claims (evidence citations)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .sections import (
    generate_dependency_context,
    generate_design_token_section,
    generate_file_inventory,
    generate_header,
    generate_import_map,
    generate_llm_fill_sections,
    generate_pattern_reference,
)

logger = logging.getLogger(__name__)

ENGINE_BUILD_ID = "2026-03-01-v1.0-template-engine"

# Frontend path prefix required by the architecture output format
_FRONTEND_PREFIX = "orb-desktop/"


def _apply_frontend_prefix(file_scope: List[str]) -> List[str]:
    """Ensure frontend files have the orb-desktop/ prefix.

    The segment specs store bare paths like 'src/components/...'
    but the architecture output format requires 'orb-desktop/src/...'.
    """
    result = []
    for f in file_scope:
        norm = f.replace("\\", "/")
        if norm.startswith("src/") and not norm.startswith(_FRONTEND_PREFIX):
            result.append(f"{_FRONTEND_PREFIX}{norm}")
        elif norm.startswith("orb-desktop/"):
            result.append(norm)
        else:
            result.append(norm)
    return result


def _detect_is_frontend_segment(segment_spec: Dict[str, Any]) -> bool:
    """Detect whether this segment is frontend-only."""
    file_scope = segment_spec.get("file_scope", [])
    if isinstance(file_scope, str):
        file_scope = [f.strip() for f in file_scope.split(",") if f.strip()]

    frontend_exts = {".tsx", ".ts", ".jsx", ".js", ".css", ".scss", ".html", ".svg"}
    return bool(file_scope) and all(
        any(f.lower().endswith(ext) for ext in frontend_exts)
        for f in file_scope
    )


def _find_pattern_reference(
    segment_spec: Dict[str, Any],
    evidence_context: Optional[Dict[str, str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Find a suitable pattern reference file from evidence.

    Returns (file_path, file_content) or (None, None).
    """
    if not evidence_context:
        return None, None

    # Prefer files in similar directories or with similar naming
    file_scope = segment_spec.get("file_scope", [])
    if not file_scope:
        return None, None

    # Get the directory of the first file in scope
    first_file = file_scope[0] if isinstance(file_scope, list) else file_scope
    scope_dir = os.path.dirname(first_file.replace("\\", "/"))

    # Look for evidence files in a sibling directory with similar extensions
    best_match = None
    best_content = None

    for ev_path, ev_content in evidence_context.items():
        if not ev_content:
            continue
        norm = ev_path.replace("\\", "/")

        # Skip files that are in scope (we're building those)
        if any(norm.endswith(f.replace("\\", "/")) for f in (file_scope if isinstance(file_scope, list) else [file_scope])):
            continue

        # Prefer .tsx/.ts files as pattern references
        if norm.endswith((".tsx", ".ts")):
            # Prefer files in a sibling component directory
            if "components" in norm:
                best_match = norm
                best_content = ev_content
                break
            if not best_match:
                best_match = norm
                best_content = ev_content

    return best_match, best_content


def generate_architecture_template(
    segment_spec: Dict[str, Any],
    skeleton: Dict[str, Any],
    all_skeletons: Optional[List[Dict[str, Any]]] = None,
    spec_id: str = "",
    spec_hash: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
    evidence_context: Optional[Dict[str, str]] = None,
    requirements: Optional[List[str]] = None,
) -> str:
    """Generate a partially-filled architecture template document.

    This is the main entry point. Call this before the LLM architecture
    generation. The returned document is prepended to the LLM's user
    content so it can see the deterministic scaffolding and fill in
    only the [LLM_FILL] sections.

    Args:
        segment_spec: This segment's spec dict (from spec.json).
        skeleton: This segment's skeleton contract dict.
        all_skeletons: All skeleton contracts in the job.
        spec_id: Spec ID for the header.
        spec_hash: Spec hash for the header.
        design_tokens: Design token registry dict (from token_registry).
        evidence_context: {path: content} of evidence files.
        requirements: List of requirement strings.

    Returns:
        Markdown string with deterministic sections filled and
        [LLM_FILL] markers for creative sections.
    """
    is_frontend = _detect_is_frontend_segment(segment_spec)

    # Apply orb-desktop prefix for frontend segments
    if is_frontend:
        prefixed_spec = dict(segment_spec)
        fs = prefixed_spec.get("file_scope", [])
        if isinstance(fs, list):
            prefixed_spec["file_scope"] = _apply_frontend_prefix(fs)
        elif isinstance(fs, str):
            parts = [f.strip() for f in fs.split(",") if f.strip()]
            prefixed_spec["file_scope"] = _apply_frontend_prefix(parts)

        # Also prefix grounding data targets
        gd = prefixed_spec.get("grounding_data", {})
        if gd:
            ct = gd.get("create_targets", [])
            for t in ct:
                if isinstance(t, dict) and "path" in t:
                    norm = t["path"].replace("\\", "/")
                    if norm.startswith("src/"):
                        t["path"] = f"{_FRONTEND_PREFIX}{norm}"
    else:
        prefixed_spec = segment_spec

    sections: List[str] = []

    # 0. Header
    if spec_id:
        sections.append(generate_header(spec_id, spec_hash))

    # 1. Architecture Vision [LLM_FILL] — placed first for LLM attention
    # (But we generate the template marker in the LLM fill sections below)

    # 2. File Inventory [DETERMINISTIC]
    sections.append(generate_file_inventory(prefixed_spec, skeleton))

    # 2B. Import Map [DETERMINISTIC]
    import_map = generate_import_map(skeleton, all_skeletons)
    if import_map:
        sections.append(import_map)

    # 2C. Dependency Context [DETERMINISTIC]
    dep_ctx = generate_dependency_context(skeleton)
    if dep_ctx:
        sections.append(dep_ctx)

    # 2D. Upstream Interface Contracts [DETERMINISTIC] — Fix 1
    # Read already-approved sibling architectures and inject their
    # actual export names and prop interfaces.
    try:
        from app.orchestrator.cross_segment_interfaces import (
            build_sibling_interface_section,
        )
        _seg_id = segment_spec.get("segment_id", "")
        _deps = skeleton.get("dependencies", [])
        if _deps and _seg_id:
            # Derive job_dir from segment_id or environment
            _parent_job = _seg_id.split("__")[0] if "__" in _seg_id else ""
            _job_dir = os.path.join(
                os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
                "jobs",
                _parent_job,
            ) if _parent_job else ""
            if not _job_dir or not os.path.isdir(_job_dir):
                # Fallback: try from all_skeletons metadata
                if all_skeletons:
                    _job_id = all_skeletons[0].get("job_id", "") if all_skeletons else ""
                    if _job_id:
                        _job_dir = os.path.join(
                            os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
                            "jobs", _job_id,
                        )
            if _job_dir and os.path.isdir(_job_dir):
                sibling_section = build_sibling_interface_section(
                    _seg_id, _deps, _job_dir,
                )
                if sibling_section:
                    sections.append(sibling_section)
                    logger.info(
                        "[arch_template] Injected sibling interfaces for %s",
                        _seg_id,
                    )
    except Exception as _sib_err:
        logger.debug("[arch_template] Sibling injection failed: %s", _sib_err)

    # 2D. Design Tokens [DETERMINISTIC]
    if design_tokens:
        token_section = generate_design_token_section(design_tokens)
        if token_section:
            sections.append(token_section)

    # 2E. Pattern Reference [DETERMINISTIC]
    pattern_file, pattern_content = _find_pattern_reference(
        segment_spec, evidence_context,
    )
    if pattern_file:
        sections.append(
            generate_pattern_reference(pattern_file, pattern_content)
        )

    # 3. LLM Fill sections (Vision, Detailed File Architecture, Critical Claims)
    reqs = requirements or segment_spec.get("requirements", [])
    sections.append(generate_llm_fill_sections(prefixed_spec, reqs))

    template = "\n\n".join(s for s in sections if s)

    # Stats
    deterministic_chars = sum(
        len(s) for s in sections
        if "[DETERMINISTIC]" in s or "FROZEN" in s.upper()
    )
    total_chars = len(template)
    det_pct = (deterministic_chars / total_chars * 100) if total_chars else 0

    logger.info(
        "[arch_template] Generated template: %d chars total, %d deterministic (%.0f%%)",
        total_chars, deterministic_chars, det_pct,
    )

    return template


def build_llm_instruction_prefix(template: str) -> str:
    """Build the instruction that tells the LLM how to use the template.

    This is prepended to the user content in the architecture prompt.
    """
    return (
        "## ARCHITECTURE TEMPLATE (partially pre-filled)\n\n"
        "The following architecture document has been partially generated "
        "from deterministic sources (skeleton contracts, design tokens, "
        "pattern references). Sections marked **[DETERMINISTIC]** are "
        "correct and complete — do NOT modify them.\n\n"
        "Your job is to fill in the sections marked **[LLM_FILL]**. "
        "These require your creative reasoning and codebase knowledge.\n\n"
        "**Rules:**\n"
        "1. Keep all [DETERMINISTIC] sections exactly as provided\n"
        "2. Replace each [LLM_FILL] marker with your content\n"
        "3. Do NOT add files to the File Inventory unless a size split is needed\n"
        "4. Do NOT change import paths — they are computed from skeleton contracts\n"
        "5. You MAY add additional sections if needed (e.g. error handling notes)\n\n"
        "---\n\n"
        f"{template}"
    )

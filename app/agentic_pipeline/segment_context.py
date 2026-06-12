# FILE: app/agentic_pipeline/segment_context.py
# Purpose: Single-Segment Context Builder.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Single-Segment Context Builder.

Builds focused context for ONE segment at a time, including:
- The segment's own template and spec
- Skeleton contract (just this segment's bindings)
- Pre-flight facts (just this segment's files)
- Evidence of what prior segments have already laid down
- Existing file content for MODIFY files

v2.0 (2026-03-06): Per-segment context for 3D-printer pipeline.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ARCH_MAP_PATH = os.path.join("D:\\Orb", ".architecture", "ARCHITECTURE_MAP.md")
_MAX_FILE_CONTENT_CHARS = 12_000
_MAX_ARCH_MAP_CHARS = 15_000


def build_segment_context(
    segment_id: str,
    segment_spec: Dict[str, Any],
    numbered_template: str,
    skeleton_contract: Dict[str, Any],
    preflight_result: Any,
    job_dir: str,
    laid_files: Dict[str, str],
    prior_arch_docs: Dict[str, str],
) -> str:
    """Build focused context for a single segment.

    Args:
        segment_id: The segment being built.
        segment_spec: This segment's spec from the manifest.
        numbered_template: The deterministic template with [LLM_FILL_N] markers.
        skeleton_contract: Full skeleton contract (we filter to relevant parts).
        preflight_result: Pre-flight file facts.
        job_dir: Job directory path.
        laid_files: Files already written by prior segments {path: summary}.
        prior_arch_docs: Arch docs from completed segments {seg_id: brief_summary}.
    """
    parts: List[str] = []

    # 1. Architecture map (truncated — just for reference)
    arch_map = _load_arch_map()
    if arch_map:
        parts.append("# ARCHITECTURE MAP (Reference)\n")
        if len(arch_map) > _MAX_ARCH_MAP_CHARS:
            arch_map = arch_map[:_MAX_ARCH_MAP_CHARS] + "\n... [truncated]"
        parts.append(arch_map)
        parts.append("\n---\n")

    # 2. Job spec (brief)
    spec_text = _load_spec_from_job(job_dir)
    if spec_text:
        parts.append("# JOB SPECIFICATION\n")
        parts.append(spec_text)
        parts.append("\n---\n")

    # 3. What's already been laid down (grounded truth)
    if laid_files:
        parts.append(f"# ALREADY LAID DOWN ({len(laid_files)} files from prior segments)\n")
        parts.append(
            "These files are REAL and exist in the sandbox. "
            "Import from them using their exact paths.\n\n"
        )
        for path, summary in laid_files.items():
            parts.append(f"- `{path}` — {summary}")
        parts.append("\n---\n")

    if prior_arch_docs:
        parts.append(f"# COMPLETED SEGMENTS ({len(prior_arch_docs)})\n")
        for seg_id_done, summary in prior_arch_docs.items():
            parts.append(f"### {seg_id_done}\n{summary}\n")
        parts.append("---\n")

    # 4. This segment's skeleton bindings
    skel_text = _format_segment_skeleton(skeleton_contract, segment_id)
    if skel_text:
        parts.append("# SKELETON CONTRACT (This Segment)\n")
        parts.append(skel_text)
        parts.append("\n---\n")

    # 5. Pre-flight facts for this segment's files
    file_scope = segment_spec.get("file_scope", [])
    preflight_text = _format_segment_preflight(preflight_result, file_scope)
    if preflight_text:
        parts.append("# PRE-FLIGHT FILE FACTS\n")
        parts.append(preflight_text)
        parts.append("\n---\n")

    # 6. Existing file content for MODIFY files
    modify_content = _format_modify_files(preflight_result, file_scope)
    if modify_content:
        parts.append("# EXISTING FILE CONTENT (MODIFY files)\n")
        parts.append(modify_content)
        parts.append("\n---\n")

    # 7. This segment's spec
    parts.append(f"# SEGMENT SPECIFICATION: {segment_id}\n")
    parts.append(json.dumps(segment_spec, indent=2))
    parts.append("\n---\n")

    # 8. The deterministic template (the thing to fill)
    parts.append(f"# DETERMINISTIC TEMPLATE: {segment_id}\n")
    parts.append("Fill ONLY the [LLM_FILL_N] markers below.\n\n")
    parts.append(numbered_template)
    parts.append("\n")

    context = "\n".join(parts)
    logger.info(
        "[segment_context] Built context for %s: %d chars",
        segment_id, len(context),
    )
    return context


def _load_arch_map() -> Optional[str]:
    try:
        if os.path.isfile(_ARCH_MAP_PATH):
            with open(_ARCH_MAP_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning("[segment_context] Failed to load arch map: %s", e)
    return None


def _load_spec_from_job(job_dir: str) -> Optional[str]:
    spec_path = os.path.join(job_dir, "spec.json")
    try:
        if os.path.isfile(spec_path):
            with open(spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            lines = []
            for key in ("summary", "objective", "key_requirements"):
                val = spec.get(key)
                if val:
                    lines.append(f"**{key}**: {json.dumps(val)}")
            return "\n".join(lines) if lines else None
    except Exception:
        pass
    return None


def _format_segment_skeleton(
    skeleton_contract: Dict[str, Any], segment_id: str,
) -> str:
    skeletons = skeleton_contract.get("skeletons", [])
    skel = next((s for s in skeletons if s.get("segment_id") == segment_id), None)
    if not skel:
        return ""
    lines = []
    deps = skel.get("dependencies", [])
    if deps:
        lines.append(f"Dependencies: {deps}")
    imports_from = skel.get("imports_from", {})
    if imports_from:
        for src, symbols in imports_from.items():
            lines.append(f"Imports from {src}: {symbols}")
    exposes = skel.get("exposes", [])
    if exposes:
        lines.append(f"Exposes: {exposes}")
    exports = skel.get("exports", [])
    if exports:
        for exp in exports:
            fp = exp.get("file_path", "")
            names = exp.get("names", [])
            consumed = exp.get("consumed_by", [])
            lines.append(f"Export: {fp} -> {names} (consumed by {consumed})")
    return "\n".join(lines)


def _format_segment_preflight(preflight_result: Any, file_scope: List[str]) -> str:
    if not preflight_result:
        return ""
    lines = []
    for f in file_scope:
        fact = preflight_result.get_fact(f) if hasattr(preflight_result, "get_fact") else None
        if fact:
            op = "MODIFY" if fact.exists else "CREATE"
            lines.append(f"- `{f}`: {op}")
        else:
            lines.append(f"- `{f}`: UNKNOWN")
    return "\n".join(lines)


def _format_modify_files(preflight_result: Any, file_scope: List[str]) -> str:
    if not preflight_result:
        return ""
    parts = []
    modify_files = preflight_result.get_modify_files() if hasattr(preflight_result, "get_modify_files") else []
    scope_set = {f.replace("\\", "/") for f in file_scope}
    for f in modify_files:
        if f.rel_path.replace("\\", "/") not in scope_set:
            continue
        if not f.content:
            continue
        content = f.content
        if len(content) > _MAX_FILE_CONTENT_CHARS:
            content = content[:_MAX_FILE_CONTENT_CHARS] + "\n... [truncated]"
        parts.append(f"### `{f.rel_path}`\n```\n{content}\n```\n")
    return "\n".join(parts)

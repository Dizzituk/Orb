# FILE: app/agentic_pipeline/tier_context.py
"""
Tier Context Builder — builds context for a dependency tier.

A tier is a group of segments that share the same dependency depth.
All segments in a tier can be built in parallel because they only
depend on segments from lower tiers (which are already laid down).

Includes grounded truth from prior tiers — real files that exist
in the sandbox, not theoretical plans.

v1.0 (2026-03-06): Initial implementation for tiered 3D printer.
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


def build_tier_context(
    tier_segments: List[Dict[str, Any]],
    numbered_templates: Dict[str, str],
    skeleton_contract: Dict[str, Any],
    preflight_result: Any,
    job_dir: str,
    laid_files: Dict[str, str],
    prior_arch_summaries: Dict[str, str],
) -> str:
    """Build context for an entire dependency tier.

    Args:
        tier_segments: All segment specs in this tier.
        numbered_templates: Templates with [LLM_FILL_N] markers per segment.
        skeleton_contract: Full skeleton contract.
        preflight_result: Pre-flight file facts.
        job_dir: Job directory path.
        laid_files: Files laid down by prior tiers {path: description}.
        prior_arch_summaries: Summaries of completed segments {seg_id: summary}.
    """
    parts: List[str] = []
    tier_seg_ids = [s.get("segment_id", "") for s in tier_segments]

    # 1. Architecture map (compact reference)
    arch_map = _load_arch_map()
    if arch_map:
        parts.append("# ARCHITECTURE MAP (Reference)\n")
        if len(arch_map) > _MAX_ARCH_MAP_CHARS:
            arch_map = arch_map[:_MAX_ARCH_MAP_CHARS] + "\n... [truncated]"
        parts.append(arch_map)
        parts.append("\n---\n")

    # 2. Job spec
    spec_text = _load_spec_from_job(job_dir)
    if spec_text:
        parts.append("# JOB SPECIFICATION\n")
        parts.append(spec_text)
        parts.append("\n---\n")

    # 3. Grounded truth — what prior tiers have laid down
    if laid_files:
        parts.append(f"# GROUNDED TRUTH — {len(laid_files)} files already in sandbox\n")
        parts.append(
            "These files EXIST and are REAL. Import from them using exact paths. "
            "Do NOT redefine anything they already export.\n\n"
        )
        for path, desc in laid_files.items():
            parts.append(f"- `{path}` — {desc}")
        parts.append("\n---\n")

    if prior_arch_summaries:
        parts.append(f"# COMPLETED SEGMENTS ({len(prior_arch_summaries)})\n")
        for seg_done, summary in prior_arch_summaries.items():
            parts.append(f"- **{seg_done}**: {summary}")
        parts.append("\n---\n")

    # 4. Skeleton contracts for segments in this tier
    skel_text = _format_tier_skeletons(skeleton_contract, tier_seg_ids)
    if skel_text:
        parts.append("# SKELETON CONTRACTS (This Tier)\n")
        parts.append(skel_text)
        parts.append("\n---\n")

    # 5. Pre-flight facts for this tier's files
    all_scope = []
    for seg in tier_segments:
        all_scope.extend(seg.get("file_scope", []))
    preflight_text = _format_preflight(preflight_result, all_scope)
    if preflight_text:
        parts.append("# PRE-FLIGHT FILE FACTS\n")
        parts.append(preflight_text)
        parts.append("\n---\n")

    # 6. Existing file content for MODIFY files in this tier
    modify_text = _format_modify_files(preflight_result, all_scope)
    if modify_text:
        parts.append("# EXISTING FILE CONTENT (MODIFY files)\n")
        parts.append(modify_text)
        parts.append("\n---\n")

    # 7. Segment specs + templates for this tier
    parts.append(f"# SEGMENTS IN THIS TIER ({len(tier_segments)})\n\n")
    for seg in tier_segments:
        seg_id = seg.get("segment_id", "")
        tmpl = numbered_templates.get(seg_id, "")
        file_scope = seg.get("file_scope", [])

        parts.append(f"## Segment: {seg_id}\n")
        parts.append(f"Files: {', '.join(f'`{f}`' for f in file_scope)}\n")

        # Segment spec details
        deps = seg.get("dependencies", [])
        if deps:
            parts.append(f"Dependencies: {deps}\n")
        reqs = seg.get("requirements", [])
        if reqs:
            parts.append(f"Requirements: {reqs}\n")

        if tmpl:
            parts.append(f"### Deterministic Template\n")
            parts.append(f"Fill ONLY the [LLM_FILL_N] markers.\n\n")
            parts.append(tmpl)
        parts.append("\n---\n")

    context = "\n".join(parts)
    logger.info(
        "[tier_context] Built context for tier (%d segments): %d chars",
        len(tier_segments), len(context),
    )
    return context


def _load_arch_map() -> Optional[str]:
    try:
        if os.path.isfile(_ARCH_MAP_PATH):
            with open(_ARCH_MAP_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning("[tier_context] Arch map load failed: %s", e)
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


def _format_tier_skeletons(
    skeleton_contract: Dict[str, Any], tier_seg_ids: List[str],
) -> str:
    skeletons = skeleton_contract.get("skeletons", [])
    tier_set = set(tier_seg_ids)
    relevant = [s for s in skeletons if s.get("segment_id") in tier_set]
    if not relevant:
        return ""
    lines = []
    for skel in relevant:
        sid = skel.get("segment_id", "")
        lines.append(f"### {sid}")
        deps = skel.get("dependencies", [])
        if deps:
            lines.append(f"  Dependencies: {deps}")
        imports_from = skel.get("imports_from", {})
        for src, symbols in imports_from.items():
            lines.append(f"  Imports from {src}: {symbols}")
        exports = skel.get("exports", [])
        for exp in exports:
            fp = exp.get("file_path", "")
            names = exp.get("names", [])
            lines.append(f"  Exports: {fp} -> {names}")
        lines.append("")
    return "\n".join(lines)


def _format_preflight(preflight_result: Any, file_scope: List[str]) -> str:
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
    modify_files = (
        preflight_result.get_modify_files()
        if hasattr(preflight_result, "get_modify_files") else []
    )
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

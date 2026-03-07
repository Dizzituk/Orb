# FILE: app/agentic_pipeline/loop_context.py
"""
Agentic Loop Context Builder.

Assembles the complete input context for the agentic architecture loop.
One context window contains everything the model needs to generate
ALL segment architectures simultaneously.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ARCH_MAP_PATH = os.path.join("D:\\Orb", ".architecture", "ARCHITECTURE_MAP.md")
_MAX_FILE_CONTENT_CHARS = 12_000


def build_loop_context(
    batch_segment_ids: List[str],
    manifest: Dict[str, Any],
    skeleton_contract: Dict[str, Any],
    preflight_result: Any,
    job_dir: str,
    experience_patterns: Optional[str] = None,
    arch_templates: Optional[Dict[str, str]] = None,
) -> str:
    """Build the complete context string for one agentic loop pass."""
    parts: List[str] = []

    arch_map = _load_arch_map()
    if arch_map:
        parts.append("# ARCHITECTURE MAP (Read-Only Reference)\n")
        if len(arch_map) > 20_000:
            arch_map = arch_map[:20_000] + "\n\n... [truncated]"
        parts.append(arch_map)
        parts.append("\n---\n")

    spec_text = _load_spec_from_job(job_dir)
    if spec_text:
        parts.append("# JOB SPECIFICATION\n")
        parts.append(spec_text)
        parts.append("\n---\n")

    skel_text = _format_skeleton_contracts(skeleton_contract, batch_segment_ids)
    if skel_text:
        parts.append("# SKELETON CONTRACTS (Cross-Segment Bindings)\n")
        parts.append(skel_text)
        parts.append("\n---\n")

    from app.agentic_pipeline.preflight_evidence import format_preflight_for_prompt
    preflight_text = format_preflight_for_prompt(preflight_result)
    if preflight_text:
        parts.append(preflight_text)
        parts.append("\n---\n")

    modify_content = _format_modify_file_content(preflight_result)
    if modify_content:
        parts.append("# EXISTING FILE CONTENT (for MODIFY files)\n")
        parts.append(modify_content)
        parts.append("\n---\n")

    if arch_templates:
        filled_count = sum(1 for t in arch_templates.values() if t)
        parts.append(f"# DETERMINISTIC TEMPLATES ({filled_count} segments)\n")
        parts.append("These templates are 90-98% complete. Fill ONLY the [LLM_FILL] markers.\n")
        parts.append("Keep ALL deterministic content intact. Do not regenerate it.\n\n")
        for seg_id in batch_segment_ids:
            tmpl = arch_templates.get(seg_id, "")
            if tmpl:
                parts.append(f"## Template: {seg_id}\n")
                parts.append(tmpl)
                parts.append("\n")
        parts.append("\n---\n")

    segment_specs = _load_segment_specs(job_dir, batch_segment_ids, manifest)
    if segment_specs:
        parts.append("# SEGMENT SPECIFICATIONS\n\n")
        for seg_id, spec in segment_specs.items():
            parts.append(f"## Segment: {seg_id}\n")
            parts.append(spec)
            parts.append("\n")
        parts.append("\n---\n")

    if experience_patterns:
        parts.append("# EXPERIENCE PATTERNS (from past jobs)\n")
        parts.append(experience_patterns)
        parts.append("\n---\n")

    context = "\n".join(parts)
    logger.info("[loop_context] Built context: %d chars, %d segments, %d sections",
                len(context), len(batch_segment_ids), len(parts))
    return context


def _load_arch_map() -> Optional[str]:
    try:
        if os.path.isfile(_ARCH_MAP_PATH):
            with open(_ARCH_MAP_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning("[loop_context] Failed to load arch map: %s", e)
    return None


def _load_spec_from_job(job_dir: str) -> Optional[str]:
    spec_path = os.path.join(job_dir, "spec.json")
    try:
        if os.path.isfile(spec_path):
            with open(spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            lines = []
            for key in ("summary", "objective", "key_requirements", "constraints", "design_preferences"):
                val = spec.get(key)
                if val:
                    lines.append(f"**{key}**: {json.dumps(val, indent=2)}")
            return "\n".join(lines) if lines else json.dumps(spec, indent=2)
    except Exception as e:
        logger.warning("[loop_context] Failed to load spec: %s", e)
    return None


def _load_segment_specs(job_dir: str, segment_ids: List[str], manifest: Dict[str, Any]) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    for seg_id in segment_ids:
        parent_job_id = os.path.basename(job_dir)
        sub_dir = os.path.join(os.path.dirname(job_dir), f"{parent_job_id}__{seg_id}")
        spec_path = os.path.join(sub_dir, "spec.json")
        if os.path.isfile(spec_path):
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    spec = json.load(f)
                specs[seg_id] = json.dumps(spec, indent=2)
            except Exception:
                pass
            continue
        for seg in manifest.get("segments", []):
            if seg.get("segment_id") == seg_id:
                specs[seg_id] = json.dumps(seg, indent=2)
                break
    return specs


def _format_skeleton_contracts(skeleton_contract: Dict[str, Any], batch_segment_ids: List[str]) -> str:
    if not skeleton_contract:
        return ""
    skeletons = skeleton_contract.get("skeletons", [])
    batch_set = set(batch_segment_ids)
    relevant = [s for s in skeletons if s.get("segment_id") in batch_set]
    if not relevant:
        return json.dumps(skeleton_contract, indent=2)
    lines = []
    for skel in relevant:
        seg_id = skel.get("segment_id", "unknown")
        lines.append(f"### {seg_id}")
        deps = skel.get("dependencies", [])
        if deps:
            lines.append(f"  Dependencies: {deps}")
        imports_from = skel.get("imports_from", {})
        if imports_from:
            for src, symbols in imports_from.items():
                lines.append(f"  Imports from {src}: {symbols}")
        exposes = skel.get("exposes", [])
        if exposes:
            lines.append(f"  Exposes: {exposes}")
        lines.append("")
    return "\n".join(lines)


def _format_modify_file_content(preflight_result: Any) -> str:
    if not preflight_result:
        return ""
    modify_files = preflight_result.get_modify_files()
    if not modify_files:
        return ""
    parts = []
    for f in modify_files:
        if not f.content:
            continue
        content = f.content
        if len(content) > _MAX_FILE_CONTENT_CHARS:
            content = content[:_MAX_FILE_CONTENT_CHARS] + f"\n\n... [truncated at {_MAX_FILE_CONTENT_CHARS} chars]"
        parts.append(f"### `{f.rel_path}` (current content)\n```\n{content}\n```\n")
    return "\n".join(parts)

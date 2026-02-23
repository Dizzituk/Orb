# FILE: app/orchestrator/seg_pipeline_step1.py
"""Step 1: Architecture generation (Critical Pipeline) + sanitisation + save."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.orchestrator.segment_state import get_job_dir
from app.orchestrator._segment_loop_utils_6 import _clear_stale_arch_versions

logger = logging.getLogger(__name__)


async def generate_architecture(
    seg_id: str,
    seg_job_id: str,
    segment_context: Dict[str, Any],
    project_id: int,
    db: Any,
    is_deterministic: bool,
    job_id: str,
    emit: Any,
) -> Optional[Dict[str, Any]]:
    """
    Generate architecture via Critical Pipeline or load pre-generated (deterministic).

    Returns dict with 'arch_text' and 'critique_passed', or None on failure.
    """
    if is_deterministic:
        return _load_deterministic_arch(seg_id, job_id, emit)

    return await _generate_llm_arch(
        seg_id, seg_job_id, segment_context, project_id, db, emit,
    )


def _load_deterministic_arch(
    seg_id: str, job_id: str, emit: Any,
) -> Optional[Dict[str, Any]]:
    """Load pre-generated architecture for deterministic refactor segments."""
    emit(f"  ⚡ Deterministic refactor path for {seg_id} — skipping LLM architecture")
    logger.info("[SEGMENT_LOOP] v6.1 Deterministic refactor bypass for %s", seg_id)

    pre_arch_path = os.path.join(
        get_job_dir(job_id), "segments", seg_id, "arch", "arch_v1.md",
    )
    if os.path.isfile(pre_arch_path):
        with open(pre_arch_path, "r", encoding="utf-8") as af:
            arch_text = af.read()
        emit(f"  ✅ Pre-generated architecture loaded ({len(arch_text)} chars)")
        logger.info(
            "[SEGMENT_LOOP] v6.1 Loaded deterministic arch for %s: %d chars",
            seg_id, len(arch_text),
        )
        return {"arch_text": arch_text, "critique_passed": True}

    logger.warning(
        "[SEGMENT_LOOP] v6.1 Deterministic flag set but no pre-generated arch at %s — falling back to LLM",
        pre_arch_path,
    )
    return None  # Caller should fall back to LLM path


async def _generate_llm_arch(
    seg_id: str,
    seg_job_id: str,
    segment_context: Dict[str, Any],
    project_id: int,
    db: Any,
    emit: Any,
) -> Optional[Dict[str, Any]]:
    """Generate architecture via LLM Critical Pipeline."""
    emit(f"  📝 Running Critical Pipeline for {seg_id}...")

    try:
        from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream
    except ImportError:
        return {"error": "Critical Pipeline not available"}

    arch_content_parts: List[str] = []
    done_metadata: Dict[str, Any] = {}

    try:
        async for event in generate_critical_pipeline_stream(
            project_id=project_id,
            message=json.dumps(segment_context.get("segment_spec", {})),
            db=db,
            job_id=seg_job_id,
            segment_context=segment_context,
        ):
            if not isinstance(event, str):
                continue
            for line in event.split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except (json.JSONDecodeError, ValueError):
                    continue
                evt_type = payload.get("type")
                if evt_type == "token":
                    arch_content_parts.append(payload.get("content", ""))
                elif evt_type == "done":
                    done_metadata = payload

        if not arch_content_parts:
            return {"error": f"Critical Pipeline produced no output for {seg_id}"}

        arch_text = "".join(arch_content_parts)
        critique_passed = done_metadata.get("critique_passed", False)
        arch_id = done_metadata.get("arch_id", "unknown")

        emit(
            f"  ✅ Architecture generated for {seg_id} "
            f"({len(arch_text)} chars, arch_id={arch_id})"
        )
        if not critique_passed:
            emit(f"  ⚠️ Critique did not fully pass — proceeding with caution")

        return {"arch_text": arch_text, "critique_passed": critique_passed}

    except Exception as e:
        logger.exception("[SEGMENT_LOOP] Critical Pipeline error for %s", seg_id)
        return {"error": f"Critical Pipeline failed for {seg_id}: {e}"}


def sanitise_architecture(
    arch_text: str,
    seg_id: str,
    file_scope: list,
    emit: Any,
) -> str:
    """
    v5.18: Architecture Sanitiser — deterministic post-generation cleanup.
    Catches known LLM hallucination patterns BEFORE architecture hits disk.
    Returns sanitised arch_text.
    """
    try:
        from app.orchestrator.architecture_sanitiser import sanitise_architecture as _sanitise
        arch_text, san_result = _sanitise(
            arch_text=arch_text,
            file_scope=file_scope,
            segment_id=seg_id,
        )
        if san_result.had_fixes:
            emit(f"  🧹 Architecture sanitiser: {san_result.fix_count} fix(es) applied")
            for fix in san_result.fixes_applied:
                emit(f"    🔧 [{fix['type']}] {fix['description'][:120]}")
            logger.info(
                "[SEGMENT_LOOP] v5.18 Sanitiser applied %d fix(es) for %s",
                san_result.fix_count, seg_id,
            )
            # Persist sanitiser result
            try:
                san_path = os.path.join(
                    get_job_dir("").rsplit("segments", 1)[0],  # approximate
                    "sanitiser_result.json",
                )
            except Exception:
                pass
        else:
            logger.debug("[SEGMENT_LOOP] v5.18 Sanitiser: no issues for %s", seg_id)
    except ImportError:
        logger.debug("[SEGMENT_LOOP] v5.18 Architecture sanitiser not available")
    except Exception as san_err:
        logger.warning("[SEGMENT_LOOP] v5.18 Sanitiser error (non-fatal): %s", san_err)
        emit(f"  ⚠️ Architecture sanitiser error (non-fatal): {san_err}")

    return arch_text


def save_architecture(
    arch_text: str,
    seg_id: str,
    job_id: str,
    emit: Any,
) -> str:
    """Save architecture to disk. Returns the path written."""
    seg_arch_dir = os.path.join(
        get_job_dir(job_id), "segments", seg_id, "arch",
    )
    os.makedirs(seg_arch_dir, exist_ok=True)

    # v5.8: Clear stale autofix versions
    seg_dir_for_clear = os.path.join(get_job_dir(job_id), "segments", seg_id)
    stale_removed = _clear_stale_arch_versions(seg_dir_for_clear)
    if stale_removed:
        emit(f"  🧹 Cleared {stale_removed} stale arch version(s)")
        logger.info(
            "[SEGMENT_LOOP] v5.8 Cleared %d stale arch version(s) for %s",
            stale_removed, seg_id,
        )

    seg_arch_path = os.path.join(seg_arch_dir, "arch_v1.md")
    try:
        with open(seg_arch_path, "w", encoding="utf-8") as f:
            f.write(arch_text)
        emit(f"  💾 Architecture saved: segments/{seg_id}/arch/arch_v1.md")
    except Exception as e:
        logger.warning("[SEGMENT_LOOP] Failed to save segment arch: %s", e)

    return seg_arch_path


def show_file_inventory(arch_text: str, emit: Any) -> None:
    """v3.0/v3.1: Extract and display File Inventory from architecture."""
    try:
        file_lines = []
        in_inventory = False
        past_header_row = False

        for line in arch_text.split("\n"):
            stripped = line.strip()
            if re.match(r'#{1,4}\s*.*[Ff]ile\s*[Ii]nventory', stripped):
                in_inventory = True
                past_header_row = False
                continue
            if in_inventory and (stripped.startswith('#') or stripped == '---'):
                if past_header_row:
                    in_inventory = False
                    continue
            if not in_inventory:
                continue
            if not stripped.startswith('|'):
                continue
            if re.match(r'\|[-\s|]+\|', stripped):
                past_header_row = True
                continue
            if 'File' in stripped and 'Purpose' in stripped:
                continue
            lower = stripped.lower()
            if '*(none' in lower or '_(none' in lower or '*(n/a' in lower or '_(n/a' in lower:
                continue
            m = re.search(r'\|\s*`([^`]+)`\s*\|\s*([^|]+)', stripped)
            if m:
                fp = m.group(1).strip()
                desc = m.group(2).strip()
                if fp and fp.lower() != 'file':
                    op = (
                        'CREATE' if any(kw in desc.lower() for kw in ('new', 'create', 'package'))
                        else 'MODIFY'
                    )
                    file_lines.append(f"    {op}: `{fp}` — {desc[:80]}")

        if file_lines:
            emit(f"  📂 File Inventory ({len(file_lines)} operations):")
            for fl in file_lines:
                emit(fl)
        else:
            emit(f"  📂 File Inventory: (could not parse — check arch_v1.md)")
    except Exception:
        pass

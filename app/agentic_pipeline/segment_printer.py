# FILE: app/agentic_pipeline/segment_printer.py
"""
Segment Printer — Tiered 3D Printer Pipeline.

Processes segments in dependency tiers:
  Tier 0: Segments with no dependencies (foundations)
  Tier 1: Segments depending only on Tier 0
  Tier 2: Segments depending on Tier 0 or 1
  ...etc.

Each tier gets ONE LLM call covering all segments in that tier.
After the call: deterministic extraction, write to sandbox, verify.
Next tier sees the real files from prior tiers as grounded truth.

Like a 3D printer: print a full layer, let it set, print the next.

v2.0 (2026-03-06): Tiered approach — one LLM call per dependency tier.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SegmentPrintResult:
    """Result of printing one segment."""
    segment_id: str
    tier: int = 0
    success: bool = False
    arch_doc: str = ""
    files_written: List[str] = field(default_factory=list)
    files_failed: List[str] = field(default_factory=list)
    error: str = ""


@dataclass
class TierResult:
    """Result of printing one tier."""
    tier_number: int
    segment_count: int = 0
    segments_ok: int = 0
    segments_failed: int = 0
    files_written: List[str] = field(default_factory=list)
    llm_calls: int = 0
    duration_seconds: float = 0.0


@dataclass
class PrinterResult:
    """Result of the full 3D printer pipeline."""
    success: bool = False
    segments_completed: int = 0
    segments_failed: int = 0
    total_segments: int = 0
    tier_count: int = 0
    arch_docs: Dict[str, str] = field(default_factory=dict)
    all_files_written: List[str] = field(default_factory=list)
    total_llm_calls: int = 0
    total_duration_seconds: float = 0.0
    tier_results: List[TierResult] = field(default_factory=list)
    segment_results: List[SegmentPrintResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dependency tier computation
# ---------------------------------------------------------------------------

def compute_dependency_tiers(
    segments: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Group segments into dependency tiers.

    Tier 0: no dependencies
    Tier 1: depends only on tier 0 segments
    Tier N: depends only on tier 0..N-1 segments

    Returns list of tiers, each tier is a list of segment specs.
    """
    seg_by_id = {s.get("segment_id", ""): s for s in segments}
    all_ids = set(seg_by_id.keys())

    assigned: Dict[str, int] = {}  # seg_id -> tier number
    tiers: List[List[Dict[str, Any]]] = []

    max_iterations = len(segments) + 1  # Safety valve
    for iteration in range(max_iterations):
        tier_segments = []
        for seg in segments:
            sid = seg.get("segment_id", "")
            if sid in assigned:
                continue
            deps = seg.get("dependencies", [])
            # Filter to deps that are actually in this job
            real_deps = [d for d in deps if d in all_ids]
            # All deps must be assigned to a prior tier
            if all(d in assigned for d in real_deps):
                tier_segments.append(seg)

        if not tier_segments:
            break

        for seg in tier_segments:
            assigned[seg.get("segment_id", "")] = iteration

        tiers.append(tier_segments)

    # Any unassigned segments (circular deps?) go in the last tier
    unassigned = [s for s in segments if s.get("segment_id", "") not in assigned]
    if unassigned:
        tiers.append(unassigned)
        logger.warning(
            "[printer] %d segment(s) with unresolvable dependencies placed in final tier",
            len(unassigned),
        )

    return tiers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_segment_printer(
    segments_in_order: List[Dict[str, Any]],
    skeleton_contract: Dict[str, Any],
    preflight_result: Any,
    arch_templates: Dict[str, str],
    job_dir: str,
    llm_call_fn: Callable,
    on_progress: Optional[Callable[[str], None]] = None,
    arch_provider: str = "openai",
    arch_model: str = "gpt-5.4",
) -> PrinterResult:
    """Run the tiered 3D printer pipeline.

    Groups segments into dependency tiers, processes one tier at a time.
    Each tier gets one LLM call, then deterministic extraction + verification.
    """
    result = PrinterResult(total_segments=len(segments_in_order))
    t_start = time.time()
    emit = on_progress or (lambda msg: None)

    # Compute dependency tiers
    tiers = compute_dependency_tiers(segments_in_order)
    result.tier_count = len(tiers)
    emit(f"\n🖨️ 3D Printer: {len(segments_in_order)} segments in {len(tiers)} tier(s)")
    for t_idx, tier in enumerate(tiers):
        tier_ids = [s.get("segment_id", "") for s in tier]
        emit(f"   Tier {t_idx}: {tier_ids}")

    # Grounded truth — accumulates as tiers complete
    laid_files: Dict[str, str] = {}
    prior_arch_summaries: Dict[str, str] = {}

    from app.agentic_pipeline.loop_controller import (
        build_system_prompt, parse_fills,
        splice_fills_into_templates, _extract_content,
    )
    from app.agentic_pipeline.tier_context import build_tier_context

    system_prompt = build_system_prompt()

    # --- Process each tier ---
    for tier_idx, tier_segments in enumerate(tiers):
        tier_seg_ids = [s.get("segment_id", "") for s in tier_segments]
        tier_file_count = sum(len(s.get("file_scope", [])) for s in tier_segments)

        emit(f"\n{'='*60}")
        emit(f"🖨️ TIER {tier_idx}: {len(tier_segments)} segment(s), {tier_file_count} file(s)")
        emit(f"{'='*60}")

        tier_result = TierResult(
            tier_number=tier_idx,
            segment_count=len(tier_segments),
        )
        t_tier = time.time()

        # --- Build tier context ---
        context = build_tier_context(
            tier_segments=tier_segments,
            numbered_templates=arch_templates,
            skeleton_contract=skeleton_contract,
            preflight_result=preflight_result,
            job_dir=job_dir,
            laid_files=laid_files,
            prior_arch_summaries=prior_arch_summaries,
        )

        # --- Count fills for this tier ---
        total_fills = 0
        fill_checklist_parts = []
        for seg in tier_segments:
            sid = seg.get("segment_id", "")
            tmpl = arch_templates.get(sid, "")
            seg_fills = tmpl.count("[LLM_FILL_")
            total_fills += seg_fills
            fill_checklist_parts.append(f"  {sid}: fills 1..{seg_fills}")
        fill_checklist = "\n".join(fill_checklist_parts)

        emit(f"   Fills to complete: {total_fills}")
        emit(f"   Context size: {len(context):,} chars")
        emit(f"   🤖 Calling {arch_provider}/{arch_model}...")

        # --- ONE LLM call for the entire tier ---
        user_prompt = (
            f"Fill ALL {total_fills} [LLM_FILL_N] markers for these {len(tier_segments)} segment(s).\n\n"
            f"CHECKLIST — output every one of these:\n{fill_checklist}\n\n"
            f"Use === FILL: <segment_id> / <N> === delimiters for each fill.\n"
            f"[LLM_FILL_2] is the most important: it must contain a fenced code block\n"
            f"for EVERY file in the segment, each starting with # FILE: or // FILE:.\n\n"
            f"{context}"
        )

        try:
            response = await llm_call_fn(
                provider_id=arch_provider,
                model_id=arch_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=64_000,
                timeout_seconds=600,
            )
        except Exception as e:
            emit(f"   ❌ Tier {tier_idx} LLM call failed: {e}")
            tier_result.duration_seconds = time.time() - t_tier
            result.tier_results.append(tier_result)
            result.errors.append(f"Tier {tier_idx} LLM failed: {e}")
            # Mark all segments in this tier as failed
            for seg in tier_segments:
                sr = SegmentPrintResult(
                    segment_id=seg.get("segment_id", ""),
                    tier=tier_idx, error=str(e),
                )
                result.segment_results.append(sr)
                result.segments_failed += 1
            continue

        raw_output = _extract_content(response)
        tier_result.llm_calls = 1
        result.total_llm_calls += 1
        emit(f"   📝 Got {len(raw_output)} chars from LLM")

        # --- Parse fills and splice into templates ---
        fills = parse_fills(raw_output)
        emit(f"   Parsed {len(fills)} fill(s) (expected {total_fills})")

        tier_templates = {
            seg.get("segment_id", ""): arch_templates.get(seg.get("segment_id", ""), "")
            for seg in tier_segments
        }

        if fills:
            arch_docs = splice_fills_into_templates(tier_templates, fills)
        else:
            # Fallback: try parsing as segment-delimited output
            from app.agentic_pipeline.loop_parser import parse_agentic_output
            parsed = parse_agentic_output(raw_output, tier_seg_ids)
            arch_docs = parsed.get_all_arch_docs()
            if not arch_docs:
                emit(f"   ⚠️ No fills AND no segment docs parsed — tier failed")
                arch_docs = {}

        emit(f"   Spliced into {len(arch_docs)} arch doc(s)")

        # --- Extract, write, and verify each segment ---
        for seg in tier_segments:
            seg_id = seg.get("segment_id", "")
            file_scope = seg.get("file_scope", [])
            arch_doc = arch_docs.get(seg_id, "")

            seg_result = SegmentPrintResult(segment_id=seg_id, tier=tier_idx)

            if not arch_doc:
                emit(f"   ⚠️ {seg_id}: No arch doc produced")
                seg_result.error = "No arch doc"
                result.segment_results.append(seg_result)
                result.segments_failed += 1
                continue

            # Save arch doc to disk
            seg_result.arch_doc = arch_doc
            result.arch_docs[seg_id] = arch_doc
            _save_segment_arch_doc(seg_id, arch_doc, job_dir)

            # Deterministic extraction
            emit(f"   📦 {seg_id}: Extracting {len(file_scope)} file(s)...")
            written, failed = await _extract_and_write_segment(
                arch_doc, file_scope, seg_id, emit,
            )
            seg_result.files_written = written
            seg_result.files_failed = failed
            tier_result.files_written.extend(written)
            result.all_files_written.extend(written)

            if written:
                emit(f"   ✅ {seg_id}: {len(written)} file(s) laid down")
            if failed:
                emit(f"   ⚠️ {seg_id}: {len(failed)} file(s) failed extraction: {failed}")

            seg_result.success = len(written) > 0
            if seg_result.success:
                result.segments_completed += 1
                tier_result.segments_ok += 1
            else:
                result.segments_failed += 1
                tier_result.segments_failed += 1
                result.errors.append(f"{seg_id}: No files extracted")

            result.segment_results.append(seg_result)

        # --- Verify the whole tier's output ---
        if tier_result.files_written:
            emit(f"   🔍 Verifying tier {tier_idx} ({len(tier_result.files_written)} files)...")
            await _verify_laid_files(tier_result.files_written, emit)

        # --- Update grounded truth for next tier ---
        for seg in tier_segments:
            seg_id = seg.get("segment_id", "")
            seg_r = next(
                (r for r in result.segment_results if r.segment_id == seg_id),
                None,
            )
            if seg_r and seg_r.files_written:
                for fp in seg_r.files_written:
                    laid_files[fp] = f"Laid by {seg_id} (tier {tier_idx})"
                file_list = ", ".join(f"`{f}`" for f in seg_r.files_written[:4])
                if len(seg_r.files_written) > 4:
                    file_list += f" (+{len(seg_r.files_written) - 4} more)"
                prior_arch_summaries[seg_id] = f"Laid {len(seg_r.files_written)} files: {file_list}"

        tier_result.duration_seconds = time.time() - t_tier
        result.tier_results.append(tier_result)
        emit(f"   ⏱️ Tier {tier_idx} complete: {tier_result.segments_ok}/{tier_result.segment_count} segments, "
             f"{len(tier_result.files_written)} files, {tier_result.duration_seconds:.1f}s")

    # --- Final summary ---
    result.total_duration_seconds = time.time() - t_start
    result.success = result.segments_completed > 0 and result.segments_failed == 0

    emit(f"\n{'='*60}")
    emit(f"🖨️ 3D Printer {'COMPLETE' if result.success else 'PARTIAL'}: "
         f"{result.segments_completed}/{result.total_segments} segments, "
         f"{len(result.all_files_written)} files, "
         f"{result.total_llm_calls} LLM calls ({result.tier_count} tiers), "
         f"{result.total_duration_seconds:.1f}s")
    emit(f"{'='*60}")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_segment_arch_doc(seg_id: str, arch_doc: str, job_dir: str) -> None:
    """Save arch doc to disk immediately after generation."""
    seg_arch_dir = os.path.join(job_dir, "segments", seg_id, "arch")
    os.makedirs(seg_arch_dir, exist_ok=True)
    arch_path = os.path.join(seg_arch_dir, "arch_v1.md")
    try:
        with open(arch_path, "w", encoding="utf-8") as f:
            f.write(arch_doc)
        logger.info("[printer] Saved arch doc: %s (%d chars)", arch_path, len(arch_doc))
    except Exception as e:
        logger.error("[printer] Failed to save arch doc %s: %s", arch_path, e)


async def _extract_and_write_segment(
    arch_doc: str,
    file_scope: List[str],
    seg_id: str,
    emit: Callable[[str], None],
) -> tuple:
    """Extract code from arch doc and write files to sandbox.

    Returns (written_paths, failed_paths).
    """
    written: List[str] = []
    failed: List[str] = []

    try:
        from app.overwatcher.architecture_executor.arch_code_extractor import (
            extract_code_for_files,
        )
    except ImportError:
        emit(f"      ❌ arch_code_extractor not available")
        return written, file_scope

    extraction = extract_code_for_files(arch_doc, file_scope)

    for file_path in file_scope:
        content = extraction.get_content_for_file(file_path)
        if not content:
            failed.append(file_path)
            continue

        if await _write_to_sandbox(file_path, content):
            written.append(file_path)
            emit(f"      ✅ {file_path}")
        else:
            failed.append(file_path)
            emit(f"      ❌ Write failed: {file_path}")

    return written, failed


async def _write_to_sandbox(file_path: str, content: str) -> bool:
    """Write a file via the sandbox bridge."""
    from app.pipeline_v2.sandbox_tools import write_file
    return await write_file(file_path, content)


async def _verify_laid_files(
    file_paths: List[str],
    emit: Callable[[str], None],
) -> bool:
    """Quick syntax check on laid files via sandbox."""
    if not file_paths:
        return True

    all_ok = True

    # Python syntax check
    py_files = [f for f in file_paths if f.endswith(".py")]
    for py_file in py_files:
        try:
            import httpx
            # Normalise path for sandbox
            sandbox_path = py_file.replace("/", "\\")
            if not sandbox_path.startswith("D:"):
                sandbox_path = f"D:\\Orb\\{sandbox_path}"
            cmd = (
                f'& "D:\\Orb\\.venv\\Scripts\\python.exe" -c '
                f'"import py_compile; py_compile.compile(r\'{sandbox_path}\', doraise=True); print(\'OK\')"'
            )
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "http://192.168.250.2:8765/shell/run",
                    json={"cmd": ["powershell", "-Command", cmd], "timeout_sec": 8},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "OK" not in data.get("stdout", ""):
                        stderr = data.get("stderr", "")[:200]
                        emit(f"      ⚠️ Syntax issue: {py_file} — {stderr}")
                        all_ok = False
                    else:
                        emit(f"      ✓ {py_file} syntax OK")
        except Exception:
            pass  # Non-fatal

    return all_ok

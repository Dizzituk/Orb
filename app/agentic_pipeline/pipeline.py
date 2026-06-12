# FILE: app/agentic_pipeline/pipeline.py
# Purpose: Agentic Pipeline Orchestrator — Top-Level Entry Point.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.agentic_pipeline, app.agentic_pipeline.loop_controller, app.agentic_pipeline.segment_printer, app.orchestrator (+3 more)
# Last-renovated: 2026-06-11
"""
Agentic Pipeline Orchestrator — Top-Level Entry Point.

Wires three stages: Agentic Loop -> Deterministic Extraction -> Phase Checkout.
Replaces: Critical Pipeline -> Critique/Revision -> Cohesion -> Implementer LLM.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool = False
    job_id: str = ""
    arch_docs: Dict[str, str] = field(default_factory=dict)
    files_written: List[str] = field(default_factory=list)
    boot_passed: bool = False
    build_passed: bool = False
    checkout_passed: bool = False
    total_llm_calls: int = 0
    total_duration_seconds: float = 0.0
    stage_durations: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


async def run_agentic_pipeline(
    job_id: str, manifest: Dict[str, Any], skeleton_contract: Dict[str, Any],
    job_dir: str, llm_call_fn: Callable, sandbox_client: Any = None,
    on_progress: Optional[Callable[[str], None]] = None,
    arch_provider: str = "openai", arch_model: str = "gpt-5.4",
    checkout_provider: str = "anthropic", checkout_model: str = "claude-sonnet-4-6",
) -> PipelineResult:
    """Run the complete three-stage agentic pipeline."""
    result = PipelineResult(job_id=job_id)
    pipeline_start = time.time()

    def _progress(msg):
        if on_progress:
            on_progress(msg)
        logger.info("[agentic_pipeline] %s", msg)

    _progress(f"Starting agentic pipeline for job {job_id}")

    segments_data = manifest.get("segments", [])
    segment_file_scopes = {s.get("segment_id", ""): s.get("file_scope", []) for s in segments_data}
    all_files = list(set(f for files in segment_file_scopes.values() for f in files))

    # === STAGE 0: Pre-Flight Evidence ===
    _progress("Stage 0: Gathering pre-flight evidence...")
    t0 = time.time()
    from app.agentic_pipeline.preflight_evidence import gather_preflight_evidence
    preflight = gather_preflight_evidence(all_files)
    result.stage_durations["preflight"] = time.time() - t0
    _progress(f"Pre-flight: {preflight.modify_count} MODIFY, {preflight.create_count} CREATE ({result.stage_durations['preflight']:.1f}s)")

    # === STAGE 0.4: Generate Skeleton Contracts (if missing) ===
    skel_path = os.path.join(job_dir, "segments", "skeleton_contract.json")
    if not skeleton_contract or not skeleton_contract.get("skeletons"):
        _progress("Stage 0.4: Generating skeleton contracts...")
        try:
            from app.orchestrator.skeleton_contracts import (
                generate_skeleton_contract, save_skeleton_contract,
            )
            _skel_contract = generate_skeleton_contract(manifest, job_id)
            if _skel_contract and _skel_contract.skeletons:
                save_skeleton_contract(_skel_contract, job_dir)
                skeleton_contract = {
                    "skeletons": [s.to_dict() if hasattr(s, "to_dict") else vars(s) for s in _skel_contract.skeletons]
                }
                _progress(f"  Generated {len(_skel_contract.skeletons)} skeleton contracts")
            else:
                _progress("  No skeleton contracts generated (empty manifest?)")
        except Exception as e:
            _progress(f"  Skeleton generation failed: {e}")
            logger.warning("[agentic_pipeline] Skeleton generation failed: %s", e, exc_info=True)

    # === STAGE 0.7: Generate Deterministic Templates ===
    _progress("Stage 0.7: Generating deterministic templates...")
    arch_templates: Dict[str, str] = {}
    try:
        from app.orchestrator.arch_template.engine import generate_architecture_template
        skeletons_list = skeleton_contract.get("skeletons", [])
        for seg_data in segments_data:
            seg_id = seg_data.get("segment_id", "")
            seg_skeleton = next((s for s in skeletons_list if s.get("segment_id") == seg_id), {})
            try:
                tmpl = generate_architecture_template(
                    segment_spec=seg_data,
                    skeleton=seg_skeleton,
                    all_skeletons=skeletons_list,
                    requirements=seg_data.get("requirements", []),
                )
                if tmpl:
                    arch_templates[seg_id] = tmpl
                    _progress(f"  Template for {seg_id}: {len(tmpl)} chars")
            except Exception as e:
                _progress(f"  Template failed for {seg_id}: {e}")
    except ImportError as e:
        _progress(f"Template engine not available: {e}")
    _progress(f"Templates generated: {len(arch_templates)}/{len(segments_data)}")

    # Number the [LLM_FILL] markers in all templates
    from app.agentic_pipeline.loop_controller import number_fill_markers
    numbered_templates: Dict[str, str] = {}
    total_fills = 0
    for seg_data in segments_data:
        sid = seg_data.get("segment_id", "")
        raw_tmpl = arch_templates.get(sid, "")
        if raw_tmpl:
            numbered, fill_count = number_fill_markers(raw_tmpl, sid)
            numbered_templates[sid] = numbered
            total_fills += fill_count
    _progress(f"Total fills to complete: {total_fills} across {len(numbered_templates)} templates")

    # === STAGE 1+2: 3D Printer — Architecture + Extraction per segment ===
    _progress("Stage 1+2: 3D Printer — laying segments one by one...")
    t1 = time.time()

    from app.agentic_pipeline.segment_printer import run_segment_printer
    printer_result = await run_segment_printer(
        segments_in_order=segments_data,
        skeleton_contract=skeleton_contract,
        preflight_result=preflight,
        arch_templates=numbered_templates,
        job_dir=job_dir,
        llm_call_fn=llm_call_fn,
        on_progress=on_progress,
        arch_provider=arch_provider,
        arch_model=arch_model,
    )

    result.total_llm_calls = printer_result.total_llm_calls
    result.arch_docs = printer_result.arch_docs
    result.files_written = printer_result.all_files_written
    result.errors.extend(printer_result.errors)
    result.stage_durations["printer"] = time.time() - t1
    _progress(
        f"Stage 1+2 complete: {printer_result.segments_completed}/{printer_result.total_segments} segments, "
        f"{len(result.files_written)} files ({result.stage_durations['printer']:.1f}s)"
    )

    # === STAGE 3: Phase Checkout ===
    _progress("Stage 3: Phase checkout verification...")
    t3 = time.time()

    boot_result = await _run_boot_check(sandbox_client)
    result.boot_passed = "PASS" in (boot_result or "")

    build_result = await _run_build_check(sandbox_client)
    result.build_passed = "PASS" in (build_result or "")

    written_contents = await _read_written_files(files_written, sandbox_client)
    spec_summary = _load_spec_summary(os.path.join(job_dir, "spec.json"))

    from app.agentic_pipeline.phase_checkout_model import run_phase_checkout
    verdict = await run_phase_checkout(
        spec_summary=spec_summary, written_files=written_contents,
        boot_result=boot_result, build_result=build_result,
        llm_call_fn=llm_call_fn, provider_id=checkout_provider, model_id=checkout_model,
        on_progress=on_progress,
    )

    result.total_llm_calls += 1
    result.checkout_passed = verdict.passed
    result.stage_durations["checkout"] = time.time() - t3
    _progress(f"Stage 3 complete: {'PASS' if verdict.passed else 'FAIL'} ({result.stage_durations['checkout']:.1f}s)")

    result.success = bool(result.arch_docs) and bool(result.files_written) and result.boot_passed and result.checkout_passed
    result.total_duration_seconds = time.time() - pipeline_start
    _progress(f"Pipeline {'SUCCESS' if result.success else 'FAILED'}: {result.total_llm_calls} LLM calls, {result.total_duration_seconds:.1f}s total")
    return result


def _save_arch_docs_to_disk(
    arch_docs: Dict[str, str], job_dir: str,
    on_progress=None,
) -> None:
    """Save arch docs to disk so the Implementer can load them later."""
    _progress = on_progress or (lambda msg: None)
    for seg_id, content in arch_docs.items():
        # Save to segments/<seg_id>/arch/arch_v1.md (standard location)
        seg_arch_dir = os.path.join(job_dir, "segments", seg_id, "arch")
        os.makedirs(seg_arch_dir, exist_ok=True)
        arch_path = os.path.join(seg_arch_dir, "arch_v1.md")
        try:
            with open(arch_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("[pipeline] Saved arch doc: %s (%d chars)", arch_path, len(content))
        except Exception as e:
            logger.error("[pipeline] Failed to save arch doc %s: %s", arch_path, e)
            _progress(f"  ⚠️ Failed to save arch doc for {seg_id}: {e}")


async def _run_deterministic_extraction(arch_docs, segment_file_scopes, sandbox_client=None, on_progress=None):
    """Deterministic extraction via arch_code_extractor. No LLM fallback."""
    files_written = []
    try:
        from app.overwatcher.architecture_executor.arch_code_extractor import extract_code_for_files
    except ImportError:
        logger.error("[pipeline] arch_code_extractor not available")
        return files_written

    for seg_id, arch_content in arch_docs.items():
        file_scope = segment_file_scopes.get(seg_id, [])
        if not file_scope:
            continue
        extraction = extract_code_for_files(arch_content, file_scope)
        for file_path in file_scope:
            content = extraction.get_content_for_file(file_path)
            if not content:
                logger.warning("[pipeline] No extractable code for %s in %s", file_path, seg_id)
                continue
            if await _write_to_sandbox(file_path, content, sandbox_client):
                files_written.append(file_path)
    return files_written


async def _write_to_sandbox(file_path, content, sandbox_client=None):
    # If no sandbox_client, skip writes entirely (comparison mode).
    # Do NOT fall back to direct httpx — that would contaminate production.
    if sandbox_client is None:
        logger.debug("[pipeline] Sandbox write skipped (no client): %s", file_path)
        return True  # Pretend success for comparison mode flow
    try:
        return bool(sandbox_client.write_file(file_path, content))
    except Exception as e:
        logger.error("[pipeline] Sandbox write failed for %s: %s", file_path, e)
        return False


async def _run_boot_check(sandbox_client=None):
    # In comparison mode (no client), skip boot check — we didn't write files
    if sandbox_client is None:
        return "SKIP: comparison mode (no sandbox writes)"
    try:
        import httpx
        boot_cmd = (
            'cd "D:\\Orb" ; '
            '& "D:\\Orb\\.venv\\Scripts\\python.exe" -c '
            '"import sys; sys.path.insert(0, r\'D:\\Orb\'); '
            'from app.db import init_db; init_db(); '
            'from main import app; print(\'BOOT_CHECK_PASS\')"'
        )
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(
                "http://192.168.250.2:8765/shell/run",
                json={"cmd": ["powershell", "-Command", boot_cmd], "timeout_sec": 30},
            )
            if resp.status_code == 200:
                data = resp.json()
                if "BOOT_CHECK_PASS" in data.get("stdout", ""):
                    return "PASS"
                return f"FAIL: {data.get('stdout', '')[:500]}"
            return f"FAIL: HTTP {resp.status_code}"
    except Exception as e:
        return f"FAIL: {e}"


async def _run_build_check(sandbox_client=None):
    # In comparison mode (no client), skip build check — we didn't write files
    if sandbox_client is None:
        return "SKIP: comparison mode (no sandbox writes)"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "http://192.168.250.2:8765/shell/run",
                json={"cmd": ["powershell", "-Command", 'cd "D:\\orb-desktop" ; npx tsc --noEmit 2>&1'], "timeout_sec": 55},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("returncode", -1) == 0:
                    return "PASS"
                return f"FAIL (rc={data.get('returncode')}): {data.get('stdout', '')[:500]}"
            return f"FAIL: HTTP {resp.status_code}"
    except Exception as e:
        return f"FAIL: {e}"


async def _read_written_files(file_paths, sandbox_client=None):
    try:
        from app.sandbox_fs import sandbox_read_text
    except ImportError:
        return {}
    return {path: content for path in file_paths if (content := sandbox_read_text(path))}


def _load_spec_summary(spec_path):
    try:
        if os.path.isfile(spec_path):
            with open(spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            parts = [f"{k}: {spec[k]}" for k in ("summary", "objective", "key_requirements") if spec.get(k)]
            return "\n".join(parts) if parts else json.dumps(spec)
    except Exception:
        pass
    return "(spec not available)"

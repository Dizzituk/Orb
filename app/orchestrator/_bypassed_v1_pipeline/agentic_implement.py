# FILE: app/orchestrator/agentic_implement.py
"""
Implement-only path for the agentic pipeline.

v8.1 (2026-03-06): Extracted from segment_loop.py.

When the Implementer button is pressed AFTER the Critical Pipeline
has already run the agentic pipeline, this module:
1. Loads the arch docs that were saved during Stage 1
2. Runs deterministic code extraction (Stage 2)
3. Runs final checkout — boot + build + surgical fix loop (Stage 3)

No LLM re-generation. Just extraction and verification.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from app.orchestrator.segment_state import JobState

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str], None]]


async def run_implement_only_from_agentic(
    job_id: str,
    manifest_path: str,
    on_progress: ProgressCallback = None,
) -> JobState:
    """
    Implement-only path: extract code from existing arch docs + final checkout.

    Called when implement_only=True and the agentic pipeline already ran.
    """
    emit = on_progress or (lambda msg: None)
    t_start = time.time()

    emit(f"🔧 **Implementer** (agentic pipeline — extraction + checkout)")
    emit(f"Job: {job_id}")

    # --- Locate the job directory ---
    job_dir = os.path.join("D:\\Orb", "jobs", "jobs", job_id)
    if not os.path.isdir(job_dir):
        logger.error("[IMPLEMENT_ONLY] Job dir not found: %s", job_dir)
        emit(f"❌ Job directory not found: {job_dir}")
        return JobState(job_id=job_id, overall_status="failed")

    # --- Load manifest for file scope info ---
    segment_file_scopes: Dict[str, List[str]] = {}
    total_segments = 0
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        segments_data = manifest_data.get("segments", [])
        total_segments = len(segments_data)
        segment_file_scopes = {
            s.get("segment_id", ""): s.get("file_scope", [])
            for s in segments_data
        }
    except Exception as e:
        logger.error("[IMPLEMENT_ONLY] Failed to load manifest: %s", e)
        emit(f"❌ Failed to load manifest: {e}")
        return JobState(job_id=job_id, overall_status="failed")

    # --- Load existing arch docs from job dir ---
    arch_docs = _load_arch_docs_from_job_dir(job_dir, emit)

    if not arch_docs:
        emit("❌ No architecture docs found — Critical Pipeline may not have completed.")
        emit("   Re-run the Critical Pipeline first.")
        return JobState(
            job_id=job_id,
            overall_status="failed",
            total_segments=total_segments,
        )

    emit(f"📄 Loaded {len(arch_docs)} architecture doc(s)")

    # --- Stage 2: Deterministic extraction ---
    emit("\n**Stage 2: Deterministic Code Extraction**")
    t2 = time.time()

    files_written = await _extract_and_write(
        arch_docs, segment_file_scopes, emit,
    )

    extraction_time = time.time() - t2
    emit(f"✅ Extraction complete: {len(files_written)} files written ({extraction_time:.1f}s)")

    if not files_written:
        emit("⚠️ No files were extracted — check arch doc code block format")
        return JobState(
            job_id=job_id,
            overall_status="failed",
            total_segments=total_segments,
        )

    # --- Stage 3: Final Checkout (boot + build + fix loop) ---
    emit("\n**Stage 3: Final Checkout**")
    t3 = time.time()

    boot_ok = await _run_boot_check(emit)
    build_ok = await _run_build_check(emit)

    # If boot or build failed, run surgical fix loop
    if not boot_ok or not build_ok:
        emit("🔧 Running surgical fix loop...")
        boot_ok, build_ok = await _run_fix_loop(
            files_written, boot_ok, build_ok, emit,
        )

    checkout_time = time.time() - t3
    total_time = time.time() - t_start

    success = boot_ok and build_ok
    status = "complete" if success else "failed"

    emit(f"\n{'✅' if success else '❌'} **Implementer {'PASSED' if success else 'FAILED'}**")
    emit(f"   Files: {len(files_written)} | Boot: {'✅' if boot_ok else '❌'} | Build: {'✅' if build_ok else '❌'}")
    emit(f"   Time: {total_time:.1f}s (extraction {extraction_time:.1f}s + checkout {checkout_time:.1f}s)")

    return JobState(
        job_id=job_id,
        overall_status=status,
        total_segments=total_segments,
    )


def _load_arch_docs_from_job_dir(
    job_dir: str, emit: Callable[[str], None],
) -> Dict[str, str]:
    """
    Load architecture docs from the job directory.

    Checks two locations:
    1. job_dir/segments/<seg_id>/arch/arch_v1.md (per-segment)
    2. job_dir/arch_docs/<seg_id>.md (agentic pipeline output)
    """
    arch_docs: Dict[str, str] = {}

    # Try per-segment arch dirs first
    segments_dir = os.path.join(job_dir, "segments")
    if os.path.isdir(segments_dir):
        for seg_dir_name in os.listdir(segments_dir):
            arch_path = os.path.join(
                segments_dir, seg_dir_name, "arch", "arch_v1.md",
            )
            if os.path.isfile(arch_path):
                try:
                    with open(arch_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content.strip():
                        arch_docs[seg_dir_name] = content
                        emit(f"  📄 {seg_dir_name}: {len(content)} chars")
                except Exception as e:
                    logger.warning(
                        "[IMPLEMENT_ONLY] Failed to read %s: %s",
                        arch_path, e,
                    )

    # Also check arch_docs dir (agentic pipeline may save here)
    arch_docs_dir = os.path.join(job_dir, "arch_docs")
    if os.path.isdir(arch_docs_dir):
        for arch_file in glob.glob(os.path.join(arch_docs_dir, "*.md")):
            seg_id = os.path.splitext(os.path.basename(arch_file))[0]
            if seg_id not in arch_docs:
                try:
                    with open(arch_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content.strip():
                        arch_docs[seg_id] = content
                        emit(f"  📄 {seg_id}: {len(content)} chars")
                except Exception as e:
                    logger.warning(
                        "[IMPLEMENT_ONLY] Failed to read %s: %s",
                        arch_file, e,
                    )

    return arch_docs


async def _extract_and_write(
    arch_docs: Dict[str, str],
    segment_file_scopes: Dict[str, List[str]],
    emit: Callable[[str], None],
) -> List[str]:
    """Extract code from arch docs and write to sandbox."""
    files_written: List[str] = []

    try:
        from app.overwatcher.architecture_executor.arch_code_extractor import (
            extract_code_for_files,
        )
    except ImportError:
        emit("❌ arch_code_extractor not available")
        logger.error("[IMPLEMENT_ONLY] arch_code_extractor import failed")
        return files_written

    for seg_id, arch_content in arch_docs.items():
        file_scope = segment_file_scopes.get(seg_id, [])
        if not file_scope:
            # If no scope in manifest, try extracting all files from the doc
            logger.info(
                "[IMPLEMENT_ONLY] No file_scope for %s — extracting all",
                seg_id,
            )

        extraction = extract_code_for_files(arch_content, file_scope)

        extract_paths = file_scope or list(extraction.extractions.keys())
        for file_path in extract_paths:
            content = extraction.get_content_for_file(file_path)
            if not content:
                emit(f"  ⚠️ No code extracted for {file_path}")
                continue

            if await _write_via_sandbox(file_path, content):
                files_written.append(file_path)
                emit(f"  ✅ {file_path}")
            else:
                emit(f"  ❌ Write failed: {file_path}")

    return files_written


async def _write_via_sandbox(file_path: str, content: str) -> bool:
    """Write a file via the sandbox bridge. Host writes are forbidden."""
    try:
        import httpx
        import base64

        encoded = base64.b64encode(
            content.encode("utf-8")
        ).decode("ascii")

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "http://192.168.250.2:8765/fs/write",
                json={"path": file_path, "content_base64": encoded},
            )
            return resp.status_code == 200
    except Exception as e:
        logger.error("[IMPLEMENT_ONLY] Sandbox write failed for %s: %s", file_path, e)
        return False


async def _run_boot_check(emit: Callable[[str], None]) -> bool:
    """Run backend boot check via sandbox."""
    emit("  🔄 Boot check...")
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
                json={
                    "cmd": ["powershell", "-Command", boot_cmd],
                    "timeout_sec": 30,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                stdout = data.get("stdout", "")
                if "BOOT_CHECK_PASS" in stdout:
                    emit("  ✅ Boot check PASSED")
                    return True
                else:
                    emit(f"  ❌ Boot check FAILED: {stdout[:300]}")
                    return False
            else:
                emit(f"  ❌ Boot check HTTP {resp.status_code}")
                return False
    except Exception as e:
        emit(f"  ❌ Boot check error: {e}")
        return False


async def _run_build_check(emit: Callable[[str], None]) -> bool:
    """Run frontend TypeScript build check via sandbox."""
    emit("  🔄 Build check (tsc)...")
    try:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "http://192.168.250.2:8765/shell/run",
                json={
                    "cmd": [
                        "powershell", "-Command",
                        'cd "D:\\orb-desktop" ; npx tsc --noEmit 2>&1',
                    ],
                    "timeout_sec": 55,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("returncode", -1) == 0:
                    emit("  ✅ Build check PASSED")
                    return True
                else:
                    stdout = data.get("stdout", "")
                    # Count errors
                    error_lines = [
                        l for l in stdout.split("\n") if ": error TS" in l
                    ]
                    emit(
                        f"  ❌ Build check FAILED: {len(error_lines)} TypeScript errors"
                    )
                    if error_lines[:5]:
                        for el in error_lines[:5]:
                            emit(f"     {el.strip()[:120]}")
                    return False
            else:
                emit(f"  ❌ Build check HTTP {resp.status_code}")
                return False
    except Exception as e:
        emit(f"  ❌ Build check error: {e}")
        return False


async def _run_fix_loop(
    files_written: List[str],
    boot_ok: bool,
    build_ok: bool,
    emit: Callable[[str], None],
    max_attempts: int = 3,
) -> tuple:
    """
    Surgical fix loop: re-check boot/build, attempt fixes if needed.

    Uses the existing final_checkout machinery if available, otherwise
    does simple re-check cycles.
    """
    for attempt in range(1, max_attempts + 1):
        if boot_ok and build_ok:
            break

        emit(f"  🔧 Fix attempt {attempt}/{max_attempts}")

        # Try using the final_checkout fix loop
        try:
            from app.orchestrator.final_checkout import run_final_checkout_fixes
            fix_result = await run_final_checkout_fixes(
                files_written=files_written,
                boot_passed=boot_ok,
                build_passed=build_ok,
            )
            boot_ok = fix_result.get("boot_passed", boot_ok)
            build_ok = fix_result.get("build_passed", build_ok)
            continue
        except (ImportError, Exception) as e:
            logger.debug(
                "[IMPLEMENT_ONLY] final_checkout fix loop not available: %s", e,
            )

        # Fallback: just re-check (no automatic fix)
        if not boot_ok:
            boot_ok = await _run_boot_check(emit)
        if not build_ok:
            build_ok = await _run_build_check(emit)

    return boot_ok, build_ok

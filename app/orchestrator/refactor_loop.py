# FILE: app/orchestrator/refactor_loop.py
"""
Refactor Loop — The scan-do-rescan orchestrator.

Runs the refactor cycle:
1. Scan → pick easiest oversized file
2. Extract → surgical extractor chips away at it
3. Boot check → verify system still starts
4. RAG update → rescan to update knowledge
5. Repeat until scan returns nothing

The loop handles both:
- Small files (20-50KB): may fully decompose in one pass
- Monoliths (50KB+): chips away, extracting what it can each pass

Boot check after every extraction. If boot fails, rollback and stop.

Usage:
    from app.orchestrator.refactor_loop import run_refactor_loop
    
    result = run_refactor_loop(max_passes=10)
    # result.passes_completed, result.files_resolved, result.status
"""

import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

# Boot check command
BOOT_CHECK_CMD = ["python", "main.py", "--boot-check"]
BOOT_CHECK_CWD = r"D:\Orb"
BOOT_SUCCESS_MARKER = "Phase 4 routers registered successfully"


@dataclass
class RefactorPassResult:
    """Result of a single refactor pass."""
    pass_number: int
    file_path: str
    file_size_before_kb: float
    file_size_after_kb: float
    symbols_extracted: int = 0
    new_module_path: Optional[str] = None
    boot_passed: bool = False
    rolled_back: bool = False
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class RefactorLoopResult:
    """Result of the full refactor loop."""
    job_id: str
    started_at: str
    completed_at: Optional[str] = None
    passes_completed: int = 0
    files_touched: int = 0
    total_kb_reduced: float = 0.0
    status: str = "running"  # running, complete, stopped, failed
    passes: List[RefactorPassResult] = field(default_factory=list)
    error: Optional[str] = None


def _boot_check() -> bool:
    """Run boot check. Returns True if system boots successfully."""
    try:
        result = subprocess.run(
            BOOT_CHECK_CMD,
            cwd=BOOT_CHECK_CWD,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return BOOT_SUCCESS_MARKER in result.stdout
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.error(f"[refactor_loop] Boot check failed: {e}")
        return False


def _run_extraction(file_path: str) -> dict:
    """
    Run one surgical extraction pass on a file.
    
    Returns:
        {
            "success": bool,
            "new_source": str or None,
            "new_module_source": str or None,
            "new_module_name": str,
            "symbols_extracted": int,
            "error": str or None,
        }
    """
    from app.orchestrator.surgical_extractor import extract_easiest

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        new_source, new_module, plan, result = extract_easiest(file_path, source)

        if result is None or new_source is None:
            return {
                "success": False,
                "new_source": None,
                "new_module_source": None,
                "new_module_name": "",
                "symbols_extracted": 0,
                "error": "No extractable symbols found",
            }

        return {
            "success": True,
            "new_source": new_source,
            "new_module_source": new_module,
            "new_module_name": plan.target_module if plan else "extract",
            "symbols_extracted": len(plan.symbols) if plan else 0,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "new_source": None,
            "new_module_source": None,
            "new_module_name": "",
            "symbols_extracted": 0,
            "error": str(e),
        }


def _apply_extraction(
    file_path: str,
    new_source: str,
    new_module_source: str,
    new_module_name: str,
) -> str:
    """
    Write the extraction results to disk.
    
    Returns path to the new module file.
    """
    # Write updated monolith
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_source)

    # Determine new module path
    # If new_module_name is already a full path (from surgical extractor),
    # use it directly. Otherwise build a path from the source file's directory.
    if os.path.isabs(new_module_name) or os.sep in new_module_name:
        new_module_path = new_module_name
    else:
        dir_path = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        module_filename = f"_{base_name}_{new_module_name}.py"
        new_module_path = os.path.join(dir_path, module_filename)

    # Avoid overwriting — increment if exists
    counter = 1
    base_path = os.path.splitext(new_module_path)[0]
    ext = os.path.splitext(new_module_path)[1] or ".py"
    while os.path.exists(new_module_path):
        new_module_path = f"{base_path}_{counter}{ext}"
        counter += 1

    # Write new module
    with open(new_module_path, "w", encoding="utf-8") as f:
        f.write(new_module_source)

    return new_module_path


def _rollback_extraction(
    file_path: str,
    backup_source: str,
    new_module_path: Optional[str],
) -> None:
    """Rollback an extraction: restore original file, delete new module."""
    # Restore original
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(backup_source)

    # Delete new module if it was created
    if new_module_path and os.path.exists(new_module_path):
        os.remove(new_module_path)


def run_refactor_pass(
    file_path: str,
    pass_number: int,
    job_id: str,
) -> RefactorPassResult:
    """
    Run a single refactor pass on a file.
    
    1. Read original source (backup)
    2. Run surgical extraction
    3. Write results to disk
    4. Boot check
    5. If boot fails, rollback
    6. Update RAG
    
    Returns RefactorPassResult
    """
    start = time.time()
    size_before = os.path.getsize(file_path) / 1024

    result = RefactorPassResult(
        pass_number=pass_number,
        file_path=file_path,
        file_size_before_kb=size_before,
        file_size_after_kb=size_before,
    )

    # Backup original source
    with open(file_path, "r", encoding="utf-8") as f:
        backup_source = f.read()

    # Run extraction
    extraction = _run_extraction(file_path)

    if not extraction["success"]:
        result.error = extraction["error"]
        result.duration_ms = (time.time() - start) * 1000
        logger.info(
            f"[refactor_loop] Pass {pass_number}: No extraction possible "
            f"for {file_path}: {extraction['error']}"
        )
        return result

    # Apply extraction to disk
    new_module_path = _apply_extraction(
        file_path,
        extraction["new_source"],
        extraction["new_module_source"],
        extraction["new_module_name"],
    )
    result.new_module_path = new_module_path
    result.symbols_extracted = extraction["symbols_extracted"]
    result.file_size_after_kb = os.path.getsize(file_path) / 1024

    # Boot check
    if _boot_check():
        result.boot_passed = True
        logger.info(
            f"[refactor_loop] Pass {pass_number}: {file_path} "
            f"{size_before:.1f}KB -> {result.file_size_after_kb:.1f}KB "
            f"({extraction['symbols_extracted']} symbols) BOOT PASS"
        )

        # Update RAG
        try:
            from app.db import get_db_session
            from app.rag.rescan import rescan_codebase

            db = get_db_session()
            rescan_codebase(db)
            db.close()
        except Exception as e:
            logger.warning(f"[refactor_loop] RAG update failed (non-fatal): {e}")

    else:
        # Boot failed — rollback
        _rollback_extraction(file_path, backup_source, new_module_path)
        result.boot_passed = False
        result.rolled_back = True
        result.file_size_after_kb = size_before
        result.error = "Boot check failed after extraction"
        logger.error(
            f"[refactor_loop] Pass {pass_number}: BOOT FAILED for {file_path}. "
            f"Rolled back."
        )

    result.duration_ms = (time.time() - start) * 1000
    return result


def run_refactor_loop(
    max_passes: int = 100,
    min_size_kb: float = 20.0,
) -> RefactorLoopResult:
    """
    Run the full scan-do-rescan refactor loop.
    
    Keeps going until:
    - Scanner finds no more oversized files
    - max_passes reached
    - A boot check fails (stops immediately)
    - An extraction returns no extractable symbols (file is at minimum viable)
    
    Args:
        max_passes: Safety limit on iterations
        min_size_kb: Minimum file size to consider for refactoring
        
    Returns:
        RefactorLoopResult with full pass history
    """
    from app.orchestrator.refactor_scanner import scan_for_refactor

    job_id = f"refactor-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    loop_result = RefactorLoopResult(
        job_id=job_id,
        started_at=datetime.utcnow().isoformat(),
    )

    logger.info(f"[refactor_loop] Starting refactor loop: {job_id}")

    files_touched = set()

    for pass_num in range(1, max_passes + 1):
        # SCAN — pick the next file
        scan = scan_for_refactor(min_size_kb=min_size_kb)

        if scan.next_file is None:
            loop_result.status = "complete"
            logger.info(
                f"[refactor_loop] COMPLETE — no more oversized files. "
                f"{pass_num - 1} passes, {len(files_touched)} files touched."
            )
            break

        target = scan.next_file
        logger.info(
            f"[refactor_loop] Pass {pass_num}/{max_passes}: "
            f"{target.path} ({target.size_kb:.1f}KB, score={target.extractability_score:.1f})"
        )

        # DO — run one extraction pass
        pass_result = run_refactor_pass(target.path, pass_num, job_id)
        loop_result.passes.append(pass_result)
        loop_result.passes_completed = pass_num

        if pass_result.boot_passed:
            files_touched.add(target.path)
            loop_result.total_kb_reduced += (
                pass_result.file_size_before_kb - pass_result.file_size_after_kb
            )
        elif pass_result.rolled_back:
            # Boot failed — stop the loop
            loop_result.status = "stopped"
            loop_result.error = f"Boot failed on pass {pass_num}: {target.path}"
            logger.error(f"[refactor_loop] STOPPED — boot failure on pass {pass_num}")
            break
        elif pass_result.error and "No extractable symbols" in pass_result.error:
            # This file can't be extracted further — skip it
            # In a real loop, we'd mark it and move to the next candidate
            logger.info(
                f"[refactor_loop] {target.path} at minimum viable size, skipping"
            )
            # Don't count as a real pass
            continue

    else:
        loop_result.status = "max_passes_reached"

    loop_result.files_touched = len(files_touched)
    loop_result.completed_at = datetime.utcnow().isoformat()

    logger.info(
        f"[refactor_loop] Finished: status={loop_result.status}, "
        f"passes={loop_result.passes_completed}, "
        f"files={loop_result.files_touched}, "
        f"reduced={loop_result.total_kb_reduced:.1f}KB"
    )

    return loop_result

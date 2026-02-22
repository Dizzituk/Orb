# FILE: app/orchestrator/refactor_stream.py
"""
Refactor Stream Handler — SSE stream for the refactor loop.

Yields Server-Sent Events as the refactor loop progresses.
Integrates with the ASTRA streaming infrastructure.
"""

import json
import logging
from typing import AsyncGenerator

from app.orchestrator.refactor_scanner import scan_for_refactor
from app.orchestrator.refactor_loop import (
    run_refactor_pass,
    RefactorLoopResult,
    _boot_check,
)

logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    """Format an SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _done_event(summary: str) -> str:
    """Build the 'done' SSE event the frontend requires to close the stream cleanly."""
    return _sse("done", {"provider": "local", "model": "refactor-loop", "summary": summary})


async def generate_refactor_stream(
    max_passes: int = 10000,
    min_size_kb: float = 20.0,
    **kwargs,
) -> AsyncGenerator[str, None]:
    import uuid
    from datetime import datetime

    job_id = f"refactor-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    yield _sse("refactor_start", {
        "job_id": job_id,
        "max_passes": max_passes,
        "min_size_kb": min_size_kb,
    })

    yield _sse("content", {
        "text": f"🔧 **Refactor Loop Started** (job: `{job_id}`)\n\n",
    })

    files_touched = set()
    files_errored = set()
    files_created = set()
    total_kb_reduced = 0.0
    passes_completed = 0
    starting_oversized = None

    for pass_num in range(1, max_passes + 1):
        scan = scan_for_refactor(min_size_kb=min_size_kb)
        if starting_oversized is None:
            starting_oversized = scan.oversized_files

        # Skip errored/created files — pick next viable candidate
        skip_set = files_errored | files_created
        if scan.next_file and scan.next_file.path in skip_set:
            found_viable = False
            if scan.all_candidates:
                for candidate in scan.all_candidates:
                    if candidate.path not in skip_set:
                        scan.next_file = candidate
                        found_viable = True
                        break
            if not found_viable:
                scan.next_file = None

        if scan.next_file is None:
            resolved = (starting_oversized or 0) - scan.oversized_files
            yield _sse("content", {
                "text": (
                    f"\n✅ **Refactor Complete!** No more viable candidates.\n"
                    f"- Started: {starting_oversized} oversized\n"
                    f"- Remaining: {scan.oversized_files}\n"
                    f"- Resolved: {resolved}\n"
                    f"- Passes: {passes_completed}\n"
                    f"- Files touched: {len(files_touched)}\n"
                    f"- Total reduced: {total_kb_reduced:.1f}KB\n"
                    f"- Skipped (errors/unfixable): {len(files_errored)}\n"
                ),
            })
            yield _sse("refactor_complete", {
                "job_id": job_id,
                "status": "complete",
                "passes_completed": passes_completed,
                "files_touched": len(files_touched),
                "total_kb_reduced": total_kb_reduced,
            })
            yield _done_event(f"Refactor complete: {passes_completed} passes, {total_kb_reduced:.1f}KB reduced")
            return

        target = scan.next_file
        short_path = target.path.replace("D:\\Orb\\", "")
        resolved = (starting_oversized or 0) - scan.oversized_files

        yield _sse("content", {
            "text": (
                f"**Pass {pass_num}:** `{short_path}` "
                f"({target.size_kb:.1f}KB) "
                f"[{scan.oversized_files} oversized, {resolved} resolved, "
                f"{total_kb_reduced:.0f}KB reduced]\n"
            ),
        })

        yield _sse("refactor_pass_start", {
            "pass_number": pass_num,
            "file": short_path,
            "size_kb": target.size_kb,
            "score": target.extractability_score,
        })

        # Run extraction
        try:
            result = run_refactor_pass(target.path, pass_num, job_id)
        except Exception as exc:
            files_errored.add(target.path)
            yield _sse("content", {
                "text": f"  → ❌ **Error:** {str(exc)[:100]}\n",
            })
            continue
        passes_completed = pass_num

        if result.boot_passed:
            kb_saved = result.file_size_before_kb - result.file_size_after_kb
            total_kb_reduced += kb_saved
            files_touched.add(target.path)
            if result.new_module_path:
                files_created.add(result.new_module_path)

            yield _sse("content", {
                "text": (
                    f"  → {result.file_size_before_kb:.1f}KB → "
                    f"{result.file_size_after_kb:.1f}KB "
                    f"(-{kb_saved:.1f}KB, {result.symbols_extracted} symbols) "
                    f"✅ Boot OK\n"
                ),
            })

        elif result.rolled_back:
            files_errored.add(target.path)
            yield _sse("content", {
                "text": f"  → ❌ **Boot Failed** — rolled back, skipping.\n",
            })
            continue

        elif result.error:
            files_errored.add(target.path)
            error_short = result.error[:80] if result.error else "unknown"
            yield _sse("content", {
                "text": f"  → ⏭️ Skipping: {error_short}\n",
            })
            continue

        yield _sse("refactor_progress", {
            "passes_completed": passes_completed,
            "files_touched": len(files_touched),
            "total_kb_reduced": total_kb_reduced,
            "remaining_oversized": scan.oversized_files,
            "starting_oversized": starting_oversized,
        })

    # Max passes reached
    yield _sse("content", {
        "text": (
            f"\n⚠️ **Max passes reached** ({max_passes}). "
            f"Reduced {total_kb_reduced:.1f}KB across {len(files_touched)} files.\n"
        ),
    })
    yield _sse("refactor_complete", {
        "job_id": job_id,
        "status": "max_passes_reached",
        "passes_completed": passes_completed,
        "files_touched": len(files_touched),
        "total_kb_reduced": total_kb_reduced,
    })
    yield _done_event(f"Refactor max passes: {passes_completed} passes, {total_kb_reduced:.1f}KB reduced")

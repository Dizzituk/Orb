# FILE: app/orchestrator/refactor_stream.py
"""
Refactor Stream Handler — SSE stream for the refactor loop.

Yields Server-Sent Events as the refactor loop progresses.
Uses the same SSE format as all other ASTRA streams:
  data: {"type": "token", "content": "..."}\n\n

v2.0 (2026-02-22): Fixed SSE format to match frontend expectations
v2.1 (2026-02-22): Added asyncio.sleep(0) after each yield to force flush.
v2.2 (2026-02-22): Three-strikes persistent state — files that fail 3 times
  are flagged for pipeline decomposition and skipped on subsequent runs.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

from app.orchestrator.refactor_scanner import scan_for_refactor
from app.orchestrator.refactor_loop import (
    run_refactor_pass,
    RefactorLoopResult,
    _boot_check,
)
from app.orchestrator.refactor_state import RefactorState

logger = logging.getLogger(__name__)


def _sse_token(text: str) -> str:
    """Send content that appears in the chat bubble."""
    return "data: " + json.dumps({"type": "token", "content": text}) + "\n\n"


def _sse_meta(event_type: str, data: dict) -> str:
    """Send a metadata/progress event (not rendered as text)."""
    return "data: " + json.dumps({"type": event_type, **data}) + "\n\n"


def _sse_done(summary: str) -> str:
    """Send the done event that closes the stream cleanly."""
    return "data: " + json.dumps({
        "type": "done",
        "provider": "local",
        "model": "refactor-loop",
        "summary": summary,
    }) + "\n\n"


async def generate_refactor_stream(
    max_passes: int = 10000,
    min_size_kb: float = 20.0,
    **kwargs,
) -> AsyncGenerator[str, None]:
    import uuid
    from datetime import datetime

    job_id = f"refactor-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # Load persistent state
    state = RefactorState.load()
    state.mark_run_start()
    stats = state.stats

    yield _sse_meta("metadata", {"provider": "local", "model": "refactor-loop"})
    yield _sse_token(f"🔧 **Refactor Loop Started** (job: `{job_id}`)\n")
    if stats["needs_pipeline"] > 0:
        yield _sse_token(
            f"📋 {stats['needs_pipeline']} files flagged for pipeline decomposition "
            f"(struck out in previous runs)\n"
        )
    yield _sse_token("\n")
    await asyncio.sleep(0)

    files_touched = set()
    files_errored = set()
    files_created = set()
    total_kb_reduced = 0.0
    passes_completed = 0
    starting_oversized = None

    for pass_num in range(1, max_passes + 1):
        await asyncio.sleep(0)

        scan = scan_for_refactor(min_size_kb=min_size_kb)
        if starting_oversized is None:
            starting_oversized = scan.oversized_files

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
            pipeline_count = len(state.pipeline_queue)
            yield _sse_token(
                f"\n✅ **Refactor Complete!** No more viable candidates.\n"
                f"- Started: {starting_oversized} oversized\n"
                f"- Remaining: {scan.oversized_files}\n"
                f"- Resolved: {resolved}\n"
                f"- Passes: {passes_completed}\n"
                f"- Files touched: {len(files_touched)}\n"
                f"- Total reduced: {total_kb_reduced:.1f}KB\n"
                f"- Skipped (errors): {len(files_errored)}\n"
                f"- Awaiting pipeline: {pipeline_count}\n"
            )
            await asyncio.sleep(0)

            # Save state before closing
            state.save()

            yield _sse_done(
                f"Refactor complete: {passes_completed} passes, "
                f"{total_kb_reduced:.1f}KB reduced, "
                f"{pipeline_count} awaiting pipeline"
            )
            return

        target = scan.next_file
        short_path = target.path.replace("D:\\Orb\\", "")
        resolved = (starting_oversized or 0) - scan.oversized_files

        # Show strike count if this file has previous failures
        rec = state.files.get(target.path)
        strike_info = f" ⚡{rec.strikes}/3" if rec and rec.strikes > 0 else ""

        yield _sse_token(
            f"**Pass {pass_num}:** `{short_path}` "
            f"({target.size_kb:.1f}KB){strike_info} "
            f"[{scan.oversized_files} oversized, {resolved} resolved, "
            f"{total_kb_reduced:.0f}KB reduced]\n"
        )
        await asyncio.sleep(0)

        try:
            result = run_refactor_pass(target.path, pass_num, job_id)
        except Exception as exc:
            files_errored.add(target.path)
            state.record_failure(target.path, target.size_kb, f"Exception: {str(exc)[:150]}")
            yield _sse_token(f"  → ❌ **Error:** {str(exc)[:100]}\n")
            await asyncio.sleep(0)
            continue
        passes_completed = pass_num

        if result.boot_passed:
            kb_saved = result.file_size_before_kb - result.file_size_after_kb
            total_kb_reduced += kb_saved
            files_touched.add(target.path)
            if result.new_module_path:
                files_created.add(result.new_module_path)

            state.record_success(target.path, result.file_size_before_kb, result.file_size_after_kb)

            yield _sse_token(
                f"  → {result.file_size_before_kb:.1f}KB → "
                f"{result.file_size_after_kb:.1f}KB "
                f"(-{kb_saved:.1f}KB, {result.symbols_extracted} symbols) "
                f"✅ Boot OK\n"
            )
            await asyncio.sleep(0)

        elif result.rolled_back:
            files_errored.add(target.path)
            reason = result.error or "Boot failed after extraction"
            state.record_failure(target.path, target.size_kb, reason[:200])

            rec = state.files.get(target.path)
            if rec and rec.needs_pipeline:
                yield _sse_token(
                    f"  → ❌ **Strike 3** — flagged for pipeline decomposition.\n"
                )
            else:
                strikes = rec.strikes if rec else 1
                yield _sse_token(f"  → ❌ **Rolled back** (strike {strikes}/3)\n")
            await asyncio.sleep(0)

        elif result.error:
            files_errored.add(target.path)
            state.record_failure(target.path, target.size_kb, result.error[:200])
            error_short = result.error[:80] if result.error else "unknown"

            rec = state.files.get(target.path)
            if rec and rec.needs_pipeline:
                yield _sse_token(
                    f"  → ⏭️ **Strike 3** — flagged for pipeline: {error_short}\n"
                )
            else:
                strikes = rec.strikes if rec else 1
                yield _sse_token(f"  → ⏭️ Skipping (strike {strikes}/3): {error_short}\n")
            await asyncio.sleep(0)

        # Progress: pass_num / starting_oversized, capped at 100%
        pct = min(round(100 * pass_num / max(starting_oversized or 1, 1)), 100)
        yield _sse_meta("refactor_progress", {
            "pct": pct,
            "pass_number": pass_num,
            "total_files": starting_oversized or 0,
            "files_done": len(files_touched),
            "files_errored": len(files_errored),
            "current_file": short_path,
            "status": "running",
        })

    # Max passes reached — save state
    state.save()

    yield _sse_token(
        f"\n⚠️ **Max passes reached** ({max_passes}). "
        f"Reduced {total_kb_reduced:.1f}KB across {len(files_touched)} files.\n"
    )
    await asyncio.sleep(0)
    yield _sse_done(
        f"Refactor max passes: {passes_completed} passes, {total_kb_reduced:.1f}KB reduced"
    )

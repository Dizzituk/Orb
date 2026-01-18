# FILE: app/llm/local_tools/zobie/streams/zobie_legacy.py
"""Legacy zobie map stream generator.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same behavior and SSE output format.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from typing import AsyncGenerator, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from app.llm.local_tools.archmap_helpers import (
    ZOBIE_MAPPER_SCRIPT,
    ZOBIE_MAPPER_TIMEOUT_SEC,
)

from ..config import (
    ZOBIE_CONTROLLER_URL,
    ZOBIE_MAPPER_OUT_DIR,
    ZOBIE_MAPPER_ARGS,
)
from ..sse import sse_token, sse_error, sse_done

logger = logging.getLogger(__name__)


async def _run_mapper() -> Tuple[str, str, List[str]]:
    """Run zobie_map.py and return (stdout, stderr, output_paths)."""
    cmd = [sys.executable, ZOBIE_MAPPER_SCRIPT, ZOBIE_CONTROLLER_URL, ZOBIE_MAPPER_OUT_DIR] + ZOBIE_MAPPER_ARGS
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=ZOBIE_MAPPER_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        raise RuntimeError(f"Mapper timed out after {ZOBIE_MAPPER_TIMEOUT_SEC}s")

    stdout = (stdout_b or b"").decode("utf-8", errors="replace")
    stderr = (stderr_b or b"").decode("utf-8", errors="replace")

    output_paths: List[str] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if os.path.isabs(s) and os.path.exists(s):
            output_paths.append(s)
            continue
        candidate = os.path.join(ZOBIE_MAPPER_OUT_DIR, s)
        if os.path.exists(candidate):
            output_paths.append(candidate)

    return stdout, stderr, output_paths


async def generate_local_zobie_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Legacy zobie map command - runs zobie_map.py directly.
    
    This outputs to the out folder (legacy behavior).
    For new architecture scans, use generate_update_architecture_stream instead.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)

    yield sse_token("🔧 Running legacy zobie_map.py...\n")
    yield sse_token(f"📡 Controller: {ZOBIE_CONTROLLER_URL}\n")
    yield sse_token(f"📂 Output: {ZOBIE_MAPPER_OUT_DIR}\n\n")

    try:
        stdout, stderr, output_paths = await _run_mapper()
        
        if stderr and "error" in stderr.lower():
            logger.warning(f"Mapper stderr: {stderr[:500]}")
            
    except Exception as e:
        logger.exception(f"Mapper failed: {e}")
        yield sse_error(f"Zobie map failed: {e}")
        yield sse_done(provider="local", model="zobie_mapper", success=False, error=str(e))
        return

    yield sse_token(f"📦 Generated {len(output_paths)} output files:\n")
    for p in output_paths[:10]:
        yield sse_token(f"   • {os.path.basename(p)}\n")
    if len(output_paths) > 10:
        yield sse_token(f"   ... and {len(output_paths) - 10} more\n")

    # Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[zobie_map] Generated {len(output_paths)} files in {ZOBIE_MAPPER_OUT_DIR}",
                provider="local",
                model="zobie_mapper",
            ),
        )
    except Exception:
        pass

    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "zobie_mapper", "zobie_map",
            0, 0, duration_ms, success=True, error=None,
        )

    summary = f"\n✅ Zobie map complete.\n📂 Output: {ZOBIE_MAPPER_OUT_DIR}\n⏱️ Duration: {duration_ms}ms\n"
    yield sse_token(summary)
    
    yield sse_done(
        provider="local",
        model="zobie_mapper",
        total_length=len(summary),
        meta={"outputs": output_paths, "out_dir": ZOBIE_MAPPER_OUT_DIR},
    )

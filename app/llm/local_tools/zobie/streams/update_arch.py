# FILE: app/llm/local_tools/zobie/streams/update_arch.py
"""UPDATE ARCHITECTURE stream generator (scope="code") - DB only.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same behavior and SSE output format.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from ..config import (
    SANDBOX_CONTROLLER_URL,
    CODE_SCAN_ROOTS,
)
from ..sse import sse_token, sse_error, sse_done
from ..sandbox_client import call_fs_tree
from ..db_ops import (
    ARCH_MODELS_AVAILABLE,
    save_scan_to_db,
    count_files_by_zone,
)

logger = logging.getLogger(__name__)


async def generate_update_architecture_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Scan D:\\Orb + D:\\orb-desktop and save to DB.
    
    This is the code-focused scan for architecture updates.
    Does NOT save to out folder - DB only.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield sse_token("🔍 Scanning code repositories...\n")
    yield sse_token(f"📡 Controller: {SANDBOX_CONTROLLER_URL}\n")
    yield sse_token(f"📂 Roots: {', '.join(CODE_SCAN_ROOTS)}\n\n")
    
    # Check if models available
    if not ARCH_MODELS_AVAILABLE:
        yield sse_error(
            "Architecture models not available. "
            "Run: alembic upgrade head to create tables."
        )
        yield sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error="models_not_available",
        )
        return
    
    # Call sandbox_controller
    status, data, error_msg = await loop.run_in_executor(
        None,
        lambda: call_fs_tree(CODE_SCAN_ROOTS, max_files=100000),
    )
    
    if status != 200 or data is None:
        logger.error(f"[update_arch] Failed: status={status}, error={error_msg}")
        
        if status == 404:
            yield sse_error(
                f"Sandbox controller /fs/tree not found at {SANDBOX_CONTROLLER_URL}\n"
                f"Please update sandbox_controller to v0.3.0 or later."
            )
        elif status is None:
            yield sse_error(
                f"Could not connect to sandbox controller at {SANDBOX_CONTROLLER_URL}\n"
                f"Error: {error_msg}\n"
                f"Is the sandbox running?"
            )
        else:
            yield sse_error(f"Scan failed (status={status}): {error_msg}")
        
        yield sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error=f"status={status}",
        )
        return
    
    # Extract file data
    files_data = data.get("files", [])
    roots_scanned = data.get("roots_scanned", CODE_SCAN_ROOTS)
    scan_time_ms = data.get("scan_time_ms", 0)
    
    yield sse_token(f"📊 Found {len(files_data)} files in {scan_time_ms}ms\n")
    
    # Save to DB
    yield sse_token("💾 Saving to database...\n")
    
    try:
        scan_id = save_scan_to_db(
            db=db,
            scope="code",
            files_data=files_data,
            roots_scanned=roots_scanned,
            scan_time_ms=scan_time_ms,
        )
        
        if scan_id:
            zone_counts = count_files_by_zone(db, scan_id) if count_files_by_zone else {}
            
            yield sse_token(f"\n✅ Architecture updated (scan_id={scan_id})\n")
            yield sse_token(f"📁 Total files: {len(files_data)}\n")
            
            if zone_counts:
                yield sse_token("📊 By zone:\n")
                for zone, count in sorted(zone_counts.items()):
                    yield sse_token(f"   • {zone}: {count}\n")
        else:
            yield sse_token("⚠️ Could not save to DB (models not available)\n")
            
    except Exception as e:
        logger.exception(f"[update_arch] DB save failed: {e}")
        yield sse_error(f"Failed to save to DB: {e}")
        yield sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error=str(e),
        )
        return
    
    # Record in memory service
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[architecture] Updated: {len(files_data)} files (scan_id={scan_id})",
                provider="local",
                model="architecture_scanner",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "architecture_scanner", "update_architecture",
            0, 0, duration_ms, success=True, error=None,
        )
    
    yield sse_done(
        provider="local",
        model="architecture_scanner",
        total_length=len(files_data),
        meta={
            "scan_id": scan_id,
            "files": len(files_data),
            "roots": roots_scanned,
            "scope": "code",
        },
    )

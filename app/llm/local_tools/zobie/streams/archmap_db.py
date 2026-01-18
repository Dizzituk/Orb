# FILE: app/llm/local_tools/zobie/streams/archmap_db.py
"""CREATE ARCHITECTURE MAP (from DB) stream generator - Opus generates map.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same behavior and SSE output format.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from app.llm.local_tools.archmap_helpers import (
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
    ARCHMAP_SYSTEM_PROMPT,
)

from ..config import (
    FULL_ARCHMAP_OUTPUT_DIR,
    FULL_ARCHMAP_OUTPUT_FILE,
)
from ..sse import sse_token, sse_error, sse_done
from ..db_ops import (
    ARCH_MODELS_AVAILABLE,
    ArchitectureFileIndex,
    get_latest_scan,
    count_files_by_zone,
)

# Import update_architecture stream for auto-scan
from .update_arch import generate_update_architecture_stream

logger = logging.getLogger(__name__)


def _build_db_archmap_prompt(files: List[Dict], zone_counts: Dict[str, int]) -> str:
    """Build architecture map prompt from file data."""
    
    # Group files by root then by relative directory
    by_root: Dict[str, Dict[str, List[Dict]]] = {}
    
    for f in files:
        root = f.get("root", "")
        path = f.get("path", "")
        name = f.get("name", "")
        ext = f.get("ext", "")
        size = f.get("size_bytes") or f.get("size") or 0
        
        if not root or not path:
            continue
        
        # Get relative path from root
        try:
            rel_path = path.replace(root, "").lstrip("\\/")
            rel_dir = os.path.dirname(rel_path).replace("\\", "/")
        except:
            rel_dir = ""
        
        if root not in by_root:
            by_root[root] = {}
        if rel_dir not in by_root[root]:
            by_root[root][rel_dir] = []
        
        by_root[root][rel_dir].append({
            "name": name,
            "ext": ext,
            "size": size,
        })
    
    # Build prompt with full tree
    lines = [
        "# Architecture Map Request",
        "",
        "Analyze this codebase and generate a comprehensive architecture map.",
        "",
        f"## Summary: {len(files)} files across {len(by_root)} root(s)",
        "",
    ]
    
    # Full file tree
    for root, dirs in sorted(by_root.items()):
        root_name = os.path.basename(root) or root
        lines.append(f"## {root_name}/")
        lines.append("```")
        
        # Sort directories for consistent output
        for dir_path in sorted(dirs.keys()):
            files_in_dir = dirs[dir_path]
            
            if dir_path:
                lines.append(f"{dir_path}/")
                prefix = "  "
            else:
                prefix = ""
            
            # Sort files by name
            for f in sorted(files_in_dir, key=lambda x: x["name"]):
                size_kb = f["size"] / 1024 if f["size"] else 0
                if size_kb > 10:
                    lines.append(f"{prefix}{f['name']} ({size_kb:.1f}KB)")
                else:
                    lines.append(f"{prefix}{f['name']}")
        
        lines.append("```")
        lines.append("")
    
    lines.extend([
        "## Instructions",
        "",
        "Create an architecture map that includes:",
        "",
        "### 1. System Overview",
        "- What is this system? (infer from file structure)",
        "- Main technology stack",
        "",
        "### 2. Component Breakdown",
        "For each major directory/module:",
        "- Purpose and responsibility",
        "- Key files and what they do",
        "",
        "### 3. Data Flow",
        "- How do requests flow through the system?",
        "- Key entry points (main.py, App.tsx, etc.)",
        "",
        "### 4. Integration Points",
        "- How do backend and frontend communicate?",
        "- External dependencies or services",
        "",
        "### 5. Observations",
        "- Architectural patterns used",
        "- Potential areas of concern (large files, complex directories)",
        "",
        "Be specific and reference actual file names from the tree above.",
    ])
    
    return "\n".join(lines)


async def generate_local_architecture_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Load architecture data from DB and generate human-readable map with Claude Opus.
    
    This is the lowercase "Create architecture map" command.
    Does NOT scan - just reads from DB and generates map.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield sse_token("📖 Loading architecture from database...\n")
    
    # Check if models available
    if not ARCH_MODELS_AVAILABLE:
        yield sse_error(
            "Architecture models not available. "
            "Run: alembic upgrade head to create tables."
        )
        yield sse_done(
            provider="local",
            model="architecture_mapper",
            success=False,
            error="models_not_available",
        )
        return
    
    # Get latest code scan
    latest_scan = get_latest_scan(db, scope="code")
    
    if not latest_scan:
        yield sse_token("⚠️ No code scan found in DB. Running scan first...\n\n")
        async for chunk in generate_update_architecture_stream(project_id, message, db, trace):
            yield chunk
        yield sse_token("\n")
        
        # Try again
        latest_scan = get_latest_scan(db, scope="code")
        if not latest_scan:
            yield sse_error("Failed to create architecture scan")
            yield sse_done(
                provider="local",
                model="architecture_mapper",
                success=False,
                error="no_scan_data",
            )
            return
    
    # Load files from scan
    files = db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == latest_scan.id
    ).all()
    
    yield sse_token(f"📊 Found {len(files)} files from scan {latest_scan.id}\n")
    yield sse_token(f"🕐 Scan timestamp: {latest_scan.finished_at}\n\n")
    
    # Group by zone
    zone_counts = count_files_by_zone(db, latest_scan.id)
    
    yield sse_token("📊 By zone:\n")
    for zone, count in sorted(zone_counts.items()):
        yield sse_token(f"   • {zone}: {count}\n")
    yield sse_token("\n")
    
    # Build prompt for Opus
    yield sse_token("🤖 Generating architecture map with Claude Opus...\n\n")
    
    # Prepare file list for prompt
    file_list = []
    for f in files:
        file_list.append({
            "path": f.path,
            "name": f.name,
            "ext": f.ext,
            "zone": f.zone,
            "size": f.size_bytes,
        })
    
    # Build structured prompt
    prompt = _build_db_archmap_prompt(file_list, zone_counts)
    
    # Call Claude Opus
    try:
        from app.llm.streaming import stream_llm
        
        messages = [{"role": "user", "content": prompt}]
        map_content = ""
        
        async for event in stream_llm(
            messages=messages,
            system_prompt=ARCHMAP_SYSTEM_PROMPT,
            provider=ARCHMAP_PROVIDER,
            model=ARCHMAP_MODEL,
        ):
            event_type = event.get("type")
            if event_type == "token":
                text = event.get("text", "")
                map_content += text
                yield sse_token(text)
            elif event_type == "error":
                yield sse_error(event.get("message", "Unknown error"))
                yield sse_done(
                    provider=ARCHMAP_PROVIDER,
                    model=ARCHMAP_MODEL,
                    success=False,
                    error=event.get("message"),
                )
                return
            elif event_type == "done":
                break
        
        yield sse_token("\n")
        
        # Save map to disk (same location as full map)
        if map_content:
            try:
                output_dir = Path(FULL_ARCHMAP_OUTPUT_DIR).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                map_path = output_dir / FULL_ARCHMAP_OUTPUT_FILE
                with open(map_path, "w", encoding="utf-8") as f:
                    f.write(map_content)
                yield sse_token(f"\n💾 Saved: {map_path}\n")
            except Exception as save_err:
                logger.warning(f"[archmap] Failed to save map to disk: {save_err}")
                yield sse_token(f"\n⚠️ Could not save to disk: {save_err}\n")
        
    except Exception as e:
        logger.exception(f"[archmap] Opus call failed: {e}")
        yield sse_error(f"Failed to generate map: {e}")
        yield sse_done(
            provider=ARCHMAP_PROVIDER,
            model=ARCHMAP_MODEL,
            success=False,
            error=str(e),
        )
        return
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", ARCHMAP_PROVIDER, ARCHMAP_MODEL, "create_architecture_map",
            len(prompt), len(map_content), duration_ms, success=True, error=None,
        )
    
    yield sse_done(
        provider=ARCHMAP_PROVIDER,
        model=ARCHMAP_MODEL,
        total_length=len(map_content),
        meta={
            "scan_id": latest_scan.id,
            "files": len(files),
            "zones": zone_counts,
        },
    )

# FILE: app/project_registry/scanner.py
"""
Multi-project architecture scanner.

Scans a registered project's root paths via the sandbox controller,
saves results to the architecture DB scoped to the project, and
generates an architecture map file within the project folder.

Reuses the existing zobie infrastructure (sandbox_client, db_ops,
rag_helpers) but scopes everything by a project-specific scope tag.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.llm.local_tools.zobie.sse import sse_token, sse_error, sse_done
from app.llm.local_tools.zobie.sandbox_client import call_fs_tree
from app.llm.local_tools.zobie.db_ops import (
    ARCH_MODELS_AVAILABLE,
    save_scan_to_db,
    count_files_by_zone,
)
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from . import service as registry_service
from .models import RegisteredProject

logger = logging.getLogger(__name__)


def _project_scope(slug: str) -> str:
    """Generate the DB scope tag for a project's architecture scan."""
    return f"project:{slug}"


async def generate_project_scan_stream(
    project: RegisteredProject,
    chat_project_id: int,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Scan a registered project and save architecture to DB.

    Follows the same pattern as update_arch.py but scoped to
    the project's root paths with a project-specific scope tag.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    root_paths = registry_service.get_root_paths(project)

    yield sse_token(f"🔍 Scanning project: **{project.name}**\n")
    yield sse_token(f"📂 Roots: {', '.join(root_paths)}\n\n")

    if not root_paths:
        yield sse_error("No root paths configured for this project.")
        yield sse_done(
            provider="local", model="project_scanner",
            success=False, error="no_root_paths",
        )
        return

    if not ARCH_MODELS_AVAILABLE:
        yield sse_error(
            "Architecture models not available. "
            "Run: alembic upgrade head to create tables."
        )
        yield sse_done(
            provider="local", model="project_scanner",
            success=False, error="models_not_available",
        )
        return

    # Call sandbox controller for each root
    all_files = []
    total_scan_ms = 0

    for root in root_paths:
        yield sse_token(f"📡 Scanning {root}...\n")
        status, data, error_msg = await loop.run_in_executor(
            None,
            lambda r=root: call_fs_tree([r], max_files=50000),
        )

        if status != 200 or data is None:
            yield sse_error(f"Scan failed for {root}: {error_msg}")
            continue

        files = data.get("files", [])
        scan_ms = data.get("scan_time_ms", 0)
        all_files.extend(files)
        total_scan_ms += scan_ms
        yield sse_token(f"   Found {len(files)} files ({scan_ms}ms)\n")

    if not all_files:
        yield sse_error("No files found across any root paths.")
        yield sse_done(
            provider="local", model="project_scanner",
            success=False, error="no_files_found",
        )
        return

    yield sse_token(f"\n📊 Total: {len(all_files)} files in {total_scan_ms}ms\n")

    # Save to DB with project-scoped tag
    scope = _project_scope(project.slug)
    yield sse_token("💾 Saving to database...\n")

    try:
        scan_id = save_scan_to_db(
            db=db,
            scope=scope,
            files_data=all_files,
            roots_scanned=root_paths,
            scan_time_ms=total_scan_ms,
        )

        if scan_id:
            zone_counts = count_files_by_zone(db, scan_id) or {}

            yield sse_token(f"\n✅ Architecture scan saved (scan_id={scan_id})\n")
            yield sse_token(f"📁 Total files: {len(all_files)}\n")

            if zone_counts:
                yield sse_token("📊 By zone:\n")
                for zone, count in sorted(zone_counts.items()):
                    yield sse_token(f"   • {zone}: {count}\n")

            # Update project scan metadata
            registry_service.update_scan_metadata(
                db=db,
                project_id=project.id,
                scan_id=scan_id,
                file_count=len(all_files),
            )

            # Generate architecture map file in project folder
            arch_map_path = await _generate_arch_map_file(
                project, all_files, scan_id, db,
            )
            if arch_map_path:
                yield sse_token(f"📄 Architecture map: {arch_map_path}\n")
        else:
            yield sse_token("⚠️ Could not save to DB\n")

    except Exception as e:
        logger.exception("[project_scanner] DB save failed: %s", e)
        yield sse_error(f"Failed to save: {e}")
        yield sse_done(
            provider="local", model="project_scanner",
            success=False, error=str(e),
        )
        return

    # Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=chat_project_id,
                role="assistant",
                content=(
                    f"[project_scan] Scanned {project.name}: "
                    f"{len(all_files)} files (scan_id={scan_id})"
                ),
                provider="local",
                model="project_scanner",
            ),
        )
    except Exception:
        pass

    duration_ms = int(loop.time() * 1000) - started_ms

    if trace:
        trace.log_model_call(
            "local_tool", "local", "project_scanner",
            f"scan_project:{project.slug}",
            0, 0, duration_ms, success=True, error=None,
        )

    yield sse_done(
        provider="local",
        model="project_scanner",
        total_length=len(all_files),
        meta={
            "scan_id": scan_id,
            "project": project.name,
            "slug": project.slug,
            "files": len(all_files),
            "roots": root_paths,
            "scope": scope,
        },
    )


async def _generate_arch_map_file(
    project: RegisteredProject,
    files_data: list,
    scan_id: int,
    db: Session,
) -> Optional[str]:
    """
    Generate an ARCHITECTURE_MAP.md file in the project's root folder.

    This provides a human-readable overview that also serves as a
    quick-load context document for ASTRA.
    """
    root_paths = registry_service.get_root_paths(project)
    if not root_paths:
        return None

    # Use the first root path as the output location
    output_dir = root_paths[0]
    arch_dir = os.path.join(output_dir, ".architecture")
    os.makedirs(arch_dir, exist_ok=True)
    output_path = os.path.join(arch_dir, "ARCHITECTURE_MAP.md")

    # Group files by zone and extension
    by_zone: dict = {}
    by_ext: dict = {}
    total_size = 0
    for f in files_data:
        zone = f.get("zone", "other")
        ext = f.get("ext", "")
        by_zone.setdefault(zone, []).append(f)
        if ext:
            by_ext.setdefault(ext, []).append(f)
        total_size += f.get("size_bytes", 0)

    # Build the map
    lines = [
        f"# Architecture Map: {project.name}",
        f"",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Scan ID: {scan_id}",
        f"Total files: {len(files_data)}",
        f"Total size: {total_size / 1024:.0f} KB",
        f"",
        f"## Project Info",
        f"- Type: {project.project_type or 'unknown'}",
        f"- Language: {project.primary_language or 'unknown'}",
        f"- Roots: {', '.join(root_paths)}",
        f"",
        f"## Files by Zone",
        f"",
    ]

    for zone in sorted(by_zone.keys()):
        zone_files = by_zone[zone]
        lines.append(f"### {zone} ({len(zone_files)} files)")
        lines.append("")
        # Show up to 50 files per zone
        for f in sorted(zone_files, key=lambda x: x.get("path", ""))[:50]:
            size_kb = (f.get("size_bytes", 0) or 0) / 1024
            lines.append(f"- `{f.get('path', '')}` ({size_kb:.1f} KB)")
        if len(zone_files) > 50:
            lines.append(f"- ... and {len(zone_files) - 50} more")
        lines.append("")

    lines.append("## Extensions Summary")
    lines.append("")
    for ext in sorted(by_ext.keys(), key=lambda e: len(by_ext[e]), reverse=True)[:15]:
        lines.append(f"- `{ext}`: {len(by_ext[ext])} files")
    lines.append("")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Update project with arch map path
        registry_service.update_scan_metadata(
            db=db,
            project_id=project.id,
            scan_id=scan_id,
            file_count=len(files_data),
            arch_map_path=output_path,
        )
        logger.info("[project_scanner] Wrote architecture map: %s", output_path)
        return output_path
    except Exception as e:
        logger.warning("[project_scanner] Failed to write arch map: %s", e)
        return None
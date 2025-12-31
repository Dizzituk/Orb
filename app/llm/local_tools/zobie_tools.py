# FILE: app/llm/local_tools/zobie_tools.py
"""Streaming local-tool generators for architecture commands.

Commands:
- UPDATE ARCHITECTURE: Scan repo → store in Orb/.architecture/ (no LLM)
- CREATE ARCHITECTURE MAP: Load .architecture/ → Claude Opus 4.5 → ARCHITECTURE_MAP.md
- ZOBIE MAP: Raw repo scan (legacy, for debugging)

v2.0 (2025-12): Split architecture update from map generation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import shutil
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from app.llm.local_tools.archmap_helpers import (
    # Triggers
    _UPDATE_ARCH_TRIGGER_SET,
    _ARCHMAP_TRIGGER_SET,
    # Paths
    ARCHITECTURE_DIR,
    ARCHMAP_OUTPUT_DIR,
    ARCHMAP_OUTPUT_FILE,
    # Model config
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
    ARCHMAP_FALLBACK_PROVIDER,
    ARCHMAP_FALLBACK_MODEL,
    ARCHMAP_MAX_TOKENS,
    ARCHMAP_TEMPERATURE,
    # Scan config
    ZOBIE_MAPPER_SCRIPT,
    ZOBIE_MAPPER_TIMEOUT_SEC,
    # Functions
    get_architecture_dir,
    get_architecture_file,
    architecture_exists,
    load_architecture_manifest,
    load_architecture_files,
    load_architecture_enums,
    load_architecture_routes,
    load_architecture_imports,
    build_archmap_prompt,
    ARCHMAP_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# =============================================================================
# ZOBIE MAPPER SETTINGS (for raw scan)
# =============================================================================

ZOBIE_CONTROLLER_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL", "http://192.168.250.2:8765")
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR", r"D:\tools\zobie_mapper\out")
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "200000 0 60 120000").strip()
ZOBIE_MAPPER_ARGS: List[str] = [a for a in ZOBIE_MAPPER_ARGS_RAW.split() if a]


# =============================================================================
# SSE HELPERS
# =============================================================================

def _sse_token(content: str) -> str:
    return "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"


def _sse_error(error: str) -> str:
    return "data: " + json.dumps({"type": "error", "error": error}) + "\n\n"


def _sse_done(
    *,
    provider: str,
    model: str,
    total_length: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "type": "done",
        "provider": provider,
        "model": model,
        "total_length": int(total_length or 0),
        "success": bool(success),
    }
    if error:
        payload["error"] = str(error)
    if meta:
        payload["meta"] = meta
    return "data: " + json.dumps(payload) + "\n\n"


# =============================================================================
# MAPPER EXECUTION
# =============================================================================

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


def _find_latest_matching(out_dir: str, pattern: str) -> str:
    """Find the most recently modified file matching pattern."""
    try:
        best = ""
        best_mtime = -1.0
        for name in os.listdir(out_dir):
            if re.match(pattern, name):
                p = os.path.join(out_dir, name)
                try:
                    mt = os.path.getmtime(p)
                except Exception:
                    mt = -1
                if mt > best_mtime:
                    best_mtime = mt
                    best = p
        return best
    except Exception:
        return ""


# =============================================================================
# CONVERT ZOBIE OUTPUT TO ARCHITECTURE FORMAT
# =============================================================================

def _convert_to_architecture_format(index_data: Dict[str, Any], out_dir: str) -> Dict[str, Dict[str, Any]]:
    """Convert zobie_map INDEX output to new architecture format.
    
    Transforms:
    - scanned_files[] with symbols -> files.json with classes/functions/signatures
    """
    files: Dict[str, Any] = {}
    
    scanned = index_data.get("scanned_files", [])
    for sf in scanned:
        path = sf.get("path", "")
        if not path:
            continue
        
        # Build file entry
        entry: Dict[str, Any] = {
            "path": path,
            "language": sf.get("language", ""),
            "bytes": sf.get("bytes", 0),
            "imports": [],
            "classes": [],
            "functions": [],
            "constants": [],
            "exports": [],
        }
        
        # Convert imports
        raw_imports = sf.get("imports", [])
        for imp in raw_imports:
            if isinstance(imp, str):
                entry["imports"].append({"module": imp, "names": None})
            elif isinstance(imp, dict):
                entry["imports"].append(imp)
        
        # Convert symbols to classes and functions
        symbols = sf.get("symbols", [])
        for sym in symbols:
            kind = sym.get("kind", "")
            name = sym.get("name", "")
            line = sym.get("line", 0)
            
            if kind == "class":
                entry["classes"].append({
                    "name": name,
                    "line": line,
                    "bases": [],
                    "decorators": [],
                    "docstring": "",
                    "fields": [],
                    "methods": [],
                })
            elif kind in ("function", "function_like"):
                entry["functions"].append({
                    "name": name,
                    "line": line,
                    "signature": "()",  # Basic - zobie doesn't extract full signatures
                    "docstring": "",
                    "decorators": [],
                })
        
        # Add enum info if present
        enums = sf.get("enums", [])
        for enum in enums:
            entry["classes"].append({
                "name": enum.get("name", ""),
                "line": enum.get("line", 0),
                "bases": [enum.get("base", "Enum")],
                "decorators": [],
                "docstring": "",
                "members": enum.get("members", []),
            })
        
        # Add routes as metadata
        routes = sf.get("routes", [])
        if routes:
            entry["routes"] = routes
        
        files[path] = entry
    
    return files


def _extract_enums_index(index_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract enum index from zobie data."""
    enums: Dict[str, Any] = {}
    
    scanned = index_data.get("scanned_files", [])
    for sf in scanned:
        path = sf.get("path", "")
        file_enums = sf.get("enums", [])
        
        for enum in file_enums:
            enum_name = enum.get("name", "")
            key = f"{path}::{enum_name}"
            enums[key] = {
                "file": path,
                "line": enum.get("line", 0),
                "bases": [enum.get("base", "Enum")],
                "members": [{"name": m, "value": m.lower()} for m in enum.get("members", [])],
                "member_count": enum.get("member_count", len(enum.get("members", []))),
            }
    
    return enums


def _extract_routes_index(index_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract routes index from zobie data."""
    routes: Dict[str, Any] = {}
    
    scanned = index_data.get("scanned_files", [])
    for sf in scanned:
        path = sf.get("path", "")
        file_routes = sf.get("routes", [])
        
        for route in file_routes:
            method = route.get("method", "GET").upper()
            route_path = route.get("path", "/")
            key = f"{method} {route_path}"
            
            routes[key] = {
                "file": path,
                "line": route.get("line", 0),
                "function": route.get("handler", ""),
                "method": method,
                "path": route_path,
                "decorator_target": route.get("decorator_target", ""),
            }
    
    return routes


def _extract_imports_graph(index_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract import graph from zobie data."""
    imports: Dict[str, Any] = {}
    
    scanned = index_data.get("scanned_files", [])
    for sf in scanned:
        path = sf.get("path", "")
        file_imports = sf.get("imports", [])
        
        imports[path] = {
            "imports_from": file_imports if isinstance(file_imports, list) else [],
            "imported_by": [],  # Would need reverse lookup
        }
    
    return imports


# =============================================================================
# UPDATE ARCHITECTURE (Command 1)
# =============================================================================

async def generate_update_architecture_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """Scan repo and store in .architecture/ (no LLM, no versioning)."""
    
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield _sse_token("🔍 Scanning repository...\n")
    
    # 1) Run mapper
    try:
        stdout, stderr, output_paths = await _run_mapper()
    except Exception as e:
        logger.exception(f"Mapper failed: {e}")
        yield _sse_error(f"Scan failed: {e}")
        yield _sse_done(provider="local", model="architecture_scanner", success=False, error=str(e))
        return
    
    # 2) Find INDEX file
    index_path = next(
        (p for p in output_paths if os.path.basename(p).startswith("INDEX_") and p.lower().endswith(".json")),
        None
    )
    if not index_path:
        index_path = _find_latest_matching(ZOBIE_MAPPER_OUT_DIR, r"^INDEX_.*\.json$")
    
    if not index_path or not os.path.exists(index_path):
        yield _sse_error("Scan completed but INDEX file not found")
        yield _sse_done(provider="local", model="architecture_scanner", success=False, error="index_not_found")
        return
    
    yield _sse_token("📦 Processing scan results...\n")
    
    # 3) Load index
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    except Exception as e:
        yield _sse_error(f"Failed to load index: {e}")
        yield _sse_done(provider="local", model="architecture_scanner", success=False, error=str(e))
        return
    
    # 4) Convert to architecture format
    arch_dir = get_architecture_dir()
    now = datetime.now(timezone.utc).isoformat()
    
    # Manifest
    manifest = {
        "version": "1.0",
        "last_scan": now,
        "repo_root": index_data.get("repo_root", ""),
        "scan_repo_root": index_data.get("scan_repo_root", ""),
        "files_count": len(index_data.get("scanned_files", [])),
        "source_index": os.path.basename(index_path),
    }
    
    # Files
    files = _convert_to_architecture_format(index_data, ZOBIE_MAPPER_OUT_DIR)
    
    # Enums
    enums = _extract_enums_index(index_data)
    
    # Routes
    routes = _extract_routes_index(index_data)
    
    # Imports
    imports = _extract_imports_graph(index_data)
    
    yield _sse_token("💾 Saving architecture data...\n")
    
    # 5) Write files (fixed names, always overwrite)
    try:
        with open(arch_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        
        with open(arch_dir / "files.json", "w", encoding="utf-8") as f:
            json.dump(files, f, indent=2)
        
        with open(arch_dir / "enums.json", "w", encoding="utf-8") as f:
            json.dump(enums, f, indent=2)
        
        with open(arch_dir / "routes.json", "w", encoding="utf-8") as f:
            json.dump(routes, f, indent=2)
        
        with open(arch_dir / "imports.json", "w", encoding="utf-8") as f:
            json.dump(imports, f, indent=2)
        
        # Copy additional useful files if they exist
        for pattern, dest_name in [
            (r"^CALLGRAPH_EDGES_.*\.json$", "callgraph.json"),
            (r"^INVARIANTS_.*\.json$", "invariants.json"),
            (r"^SYMBOL_INDEX_.*\.json$", "symbols.json"),
        ]:
            src = _find_latest_matching(ZOBIE_MAPPER_OUT_DIR, pattern)
            if src and os.path.exists(src):
                shutil.copy2(src, arch_dir / dest_name)
        
    except Exception as e:
        logger.exception(f"Failed to save architecture: {e}")
        yield _sse_error(f"Failed to save: {e}")
        yield _sse_done(provider="local", model="architecture_scanner", success=False, error=str(e))
        return
    
    # 6) Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[architecture] Updated: {len(files)} files, {len(enums)} enums, {len(routes)} routes",
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
    
    summary = (
        f"✅ Architecture updated.\n\n"
        f"📁 Location: {arch_dir}\n"
        f"📊 Files: {len(files)}\n"
        f"🔢 Enums: {len(enums)}\n"
        f"🛣️ Routes: {len(routes)}\n"
        f"⏱️ Duration: {duration_ms}ms\n"
    )
    
    yield _sse_token(summary)
    yield _sse_done(
        provider="local",
        model="architecture_scanner",
        total_length=len(summary),
        meta={"arch_dir": str(arch_dir), "files": len(files), "enums": len(enums), "routes": len(routes)},
    )


# =============================================================================
# CREATE ARCHITECTURE MAP (Command 2)
# =============================================================================

async def generate_local_architecture_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """Load .architecture/ and generate human-readable map with Claude Opus 4.5."""
    
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    # 1) Check architecture exists
    if not architecture_exists():
        yield _sse_token("⚠️ No architecture data found. Running scan first...\n\n")
        async for chunk in generate_update_architecture_stream(project_id, message, db, trace):
            yield chunk
        yield _sse_token("\n")
    
    yield _sse_token("📖 Loading architecture data...\n")
    
    # 2) Load architecture
    manifest = load_architecture_manifest()
    files = load_architecture_files()
    enums = load_architecture_enums()
    routes = load_architecture_routes()
    imports = load_architecture_imports()
    
    if not files:
        yield _sse_error("Architecture data is empty. Run 'update architecture' first.")
        yield _sse_done(provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, success=False, error="empty_architecture")
        return
    
    yield _sse_token(f"📊 Loaded {len(files)} files, {len(enums)} enums, {len(routes)} routes\n")
    yield _sse_token("🤖 Generating architecture map with Claude Opus 4.5... (this may take a minute)\n")
    
    # 3) Build prompt
    user_prompt = build_archmap_prompt(manifest, files, enums, routes, imports)
    
    # 4) Call LLM (collect silently, don't stream to chat)
    try:
        from app.llm.streaming import stream_llm
        
        provider = ARCHMAP_PROVIDER
        model = ARCHMAP_MODEL
        
        messages = [
            {"role": "user", "content": user_prompt},
        ]
        
        full_response = ""
        
        async for chunk in stream_llm(
            messages=messages,
            system_prompt=ARCHMAP_SYSTEM_PROMPT,
            provider=provider,
            model=model,
        ):
            if isinstance(chunk, dict):
                if chunk.get("type") == "error":
                    raise RuntimeError(chunk.get("message", "Unknown error"))
                content = chunk.get("text", "") or chunk.get("content", "")
            else:
                content = str(chunk)
            
            if content:
                full_response += content
                # Don't yield content to chat - collect silently
        
    except Exception as e:
        logger.exception(f"LLM call failed: {e}")
        
        # Try fallback
        yield _sse_token(f"⚠️ Opus failed, trying fallback...\n")
        
        try:
            from app.llm.streaming import stream_llm
            
            provider = ARCHMAP_FALLBACK_PROVIDER
            model = ARCHMAP_FALLBACK_MODEL
            
            full_response = ""
            async for chunk in stream_llm(
                messages=messages,
                system_prompt=ARCHMAP_SYSTEM_PROMPT,
                provider=provider,
                model=model,
            ):
                if isinstance(chunk, dict):
                    if chunk.get("type") == "error":
                        raise RuntimeError(chunk.get("message", "Unknown error"))
                    content = chunk.get("text", "") or chunk.get("content", "")
                else:
                    content = str(chunk)
                
                if content:
                    full_response += content
                    # Don't yield content to chat - collect silently
                    
        except Exception as e2:
            yield _sse_error(f"Both providers failed: {e2}")
            yield _sse_done(provider=provider, model=model, success=False, error=str(e2))
            return
    
    # 5) Save output
    output_dir = Path(ARCHMAP_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ARCHMAP_OUTPUT_FILE
    
    try:
        header = (
            f"# Orb/ASTRA Architecture Map\n\n"
            f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
            f"Model: {provider}/{model}\n"
            f"Source: {get_architecture_dir()}\n\n"
            f"---\n\n"
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + full_response)
        
    except Exception as e:
        logger.exception(f"Failed to save: {e}")
        yield _sse_error(f"Failed to save file: {e}")
        yield _sse_done(provider=provider, model=model, success=False, error=str(e))
        return
    
    # 6) Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[archmap] Generated: {output_path}",
                provider=provider,
                model=model,
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    if trace:
        trace.log_model_call(
            "local_tool", provider, model, "create_architecture_map",
            0, 0, duration_ms, success=True, error=None,
        )
    
    # Final status message
    yield _sse_token(
        f"✅ Architecture map generated.\n\n"
        f"📁 Output: {output_path}\n"
        f"📊 Size: {len(full_response):,} characters\n"
        f"⏱️ Duration: {duration_ms}ms\n"
    )
    
    yield _sse_done(
        provider=provider,
        model=model,
        total_length=len(full_response),
        meta={"output": str(output_path)},
    )


# =============================================================================
# ZOBIE MAP (Legacy - raw scan)
# =============================================================================

async def generate_local_zobie_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """Run raw repo scanner (legacy command for debugging)."""
    
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield _sse_token("🧟 Running raw repo scanner...\n")
    
    try:
        stdout, stderr, output_paths = await _run_mapper()
        
        if trace:
            duration_ms = int(loop.time() * 1000) - started_ms
            trace.log_model_call(
                "local_tool", "local", "zobie_mapper", "zobie_mapper",
                0, 0, duration_ms, success=True, error=None,
            )
    except Exception as e:
        if trace:
            trace.log_error(f"Zobie mapper failed: {e}")
        yield _sse_error(f"ZOBIE MAP failed: {e}")
        yield _sse_done(provider="local", model="zobie_mapper", success=False, error=str(e))
        return
    
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content="[zobie_map] Raw scan complete.",
                provider="local",
                model="zobie_mapper",
            ),
        )
    except Exception:
        pass
    
    if not output_paths:
        guess = _find_latest_matching(ZOBIE_MAPPER_OUT_DIR, r"^(ARCH_MAP_|INDEX_|MANIFEST_|REPO_TREE_).*\.(md|json|txt)$")
        if guess:
            output_paths = [guess]
    
    summary = "Raw scan complete.\n\nOutputs:\n" + "\n".join(f"- {p}" for p in output_paths) + "\n"
    yield _sse_token(summary)
    yield _sse_done(
        provider="local",
        model="zobie_mapper",
        total_length=len(summary),
        meta={"outputs": output_paths, "out_dir": ZOBIE_MAPPER_OUT_DIR},
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "generate_update_architecture_stream",
    "generate_local_architecture_map_stream",
    "generate_local_zobie_map_stream",
]

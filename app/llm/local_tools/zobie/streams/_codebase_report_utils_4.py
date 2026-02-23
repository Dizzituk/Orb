from __future__ import annotations
import asyncio
import json
import logging
from ..sse import sse_done, sse_error, sse_token
from app.llm.audit_logger import RoutingTrace
from app.llm.local_tools.zobie.streams._codebase_report_utils_2 import CODEBASE_REPORT_OUTPUT_DIR, _detect_duplicate_filenames, _detect_floating_files
from app.llm.local_tools.zobie.streams._codebase_report_utils_3 import _compute_incremental, _format_size, _load_state, _save_state, _scan_directory, _scan_file_for_absolute_paths, _scan_file_for_blocked_refs
from app.llm.local_tools.zobie.streams._codebase_report_structural import run_structural_scan
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from typing import AsyncGenerator, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
FILE_SIZE_WARN_BYTES = 250 * 1024      # 250 KB
FILE_SIZE_HIGH_BYTES = 1 * 1024 * 1024  # 1 MB
FILE_SIZE_CRITICAL_BYTES = 5 * 1024 * 1024  # 5 MB
FILE_LINES_HIGH = 1200
FILE_LINES_CRITICAL = 2500


CODEBASE_REPORT_STATE_FILE = Path(r"D:\Orb\data\codebase_report_state.json")

FILE_LINES_WARN = 600

FULL_MAX_BYTES_PER_FILE = 64 * 1024  # 64 KB

TOP_N_LONGEST = 30

def _is_text_file(path: Path) -> bool:
    """Check if file is text-like (for line counting)."""
    from .codebase_report import TEXT_EXTENSIONS
    ext = path.suffix.lower()
    name = path.name.lower()
    # Handle extensionless files by name
    if ext == "" and name in {"dockerfile", "makefile", "jenkinsfile", "vagrantfile"}:
        return True
    return ext in TEXT_EXTENSIONS

def _categorize_by_size(size_bytes: int) -> str:
    """Categorize file by size threshold."""
    if size_bytes >= FILE_SIZE_CRITICAL_BYTES:
        return "CRITICAL"
    if size_bytes >= FILE_SIZE_HIGH_BYTES:
        return "HIGH"
    if size_bytes >= FILE_SIZE_WARN_BYTES:
        return "WARN"
    return "OK"

def _categorize_by_lines(line_count: Optional[int]) -> str:
    """Categorize file by line count threshold."""
    if line_count is None:
        return "UNKNOWN"
    if line_count >= FILE_LINES_CRITICAL:
        return "CRITICAL"
    if line_count >= FILE_LINES_HIGH:
        return "HIGH"
    if line_count >= FILE_LINES_WARN:
        return "WARN"
    return "OK"

async def generate_codebase_report_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a codebase hygiene/bloat/drift report.
    
    Commands:
    - "codebase report fast" - Quick metadata scan
    - "codebase report full" - Deep content scan
    """
    from .codebase_report import AbsolutePathFinding, CODEBASE_REPORT_ROOTS, EXPECTED_ROOT_ITEMS_DESKTOP, EXPECTED_ROOT_ITEMS_ORB, FileEntry, _generate_json_report, _generate_markdown_report
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    # Parse mode from message
    msg_lower = message.lower()
    if "full" in msg_lower:
        mode = "FULL"
    else:
        mode = "FAST"
    
    yield sse_token(f"📊 [CODEBASE_REPORT] mode={mode} roots=2\n\n")
    logger.info(f"[CODEBASE_REPORT] mode={mode} roots=2")
    
    # Ensure output directory exists
    try:
        CODEBASE_REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        yield sse_token(f"✓ Output dir: {CODEBASE_REPORT_OUTPUT_DIR}\n")
    except Exception as e:
        yield sse_error(f"Failed to create output dir: {e}")
        yield sse_done(provider="local", model="codebase_report", success=False, error=str(e))
        return
    
    # Load previous state
    old_state = _load_state()
    
    # Scan both roots
    all_files: List[FileEntry] = []
    all_suspect_folders: List[str] = []
    all_extension_counts: Dict[str, int] = {}
    
    for root in CODEBASE_REPORT_ROOTS:
        if not root.exists():
            yield sse_token(f"⚠️ Root not found: {root}\n")
            continue
        
        root_label = root.name  # "Orb" or "orb-desktop"
        yield sse_token(f"📁 Scanning {root}...\n")
        
        # FAST always counts lines for text files; FULL does deeper analysis
        files, suspect_folders, ext_counts = _scan_directory(
            root, root_label, count_lines=True
        )
        
        all_files.extend(files)
        all_suspect_folders.extend(suspect_folders)
        for ext, count in ext_counts.items():
            all_extension_counts[ext] = all_extension_counts.get(ext, 0) + count
        
        yield sse_token(f"   ✓ {len(files)} files found\n")
    
    yield sse_token(f"\n📊 Total: {len(all_files)} files scanned\n\n")
    
    # Detect floating files
    floating_orb = _detect_floating_files(CODEBASE_REPORT_ROOTS[0], EXPECTED_ROOT_ITEMS_ORB)
    floating_desktop = _detect_floating_files(CODEBASE_REPORT_ROOTS[1], EXPECTED_ROOT_ITEMS_DESKTOP) if len(CODEBASE_REPORT_ROOTS) > 1 else []
    
    # Build state dict for incremental comparison
    # Key = root_label + "/" + relative_path
    current_state: Dict[str, FileEntry] = {}
    for f in all_files:
        key = f"{f.root_label}/{f.relative_path}"
        # Exclude the state file itself
        if CODEBASE_REPORT_STATE_FILE.name in f.relative_path:
            continue
        current_state[key] = f
    
    # Compute incremental changes
    new_files, changed_files, deleted_files = _compute_incremental(current_state, old_state)
    yield sse_token(f"📈 Incremental: new={len(new_files)} changed={len(changed_files)} deleted={len(deleted_files)}\n\n")
    logger.info(f"[CODEBASE_REPORT] incremental new={len(new_files)} changed={len(changed_files)} deleted={len(deleted_files)}")
    
    # FULL mode: Deep content scanning
    abs_path_findings: List[AbsolutePathFinding] = []
    blocked_refs: List[Tuple[str, int, str]] = []
    duplicate_names: Dict[str, List[str]] = {}
    
    if mode == "FULL":
        yield sse_token("🔍 FULL mode: Scanning file contents...\n")
        
        # Scan for absolute paths
        for i, f in enumerate(all_files):
            if i % 100 == 0 and i > 0:
                yield sse_token(f"   ... {i}/{len(all_files)} files checked\n")
            
            findings = _scan_file_for_absolute_paths(f.path)
            abs_path_findings.extend(findings)
            
            refs = _scan_file_for_blocked_refs(f.path)
            blocked_refs.extend(refs)
        
        yield sse_token(f"   ✓ Found {len(abs_path_findings)} absolute path references\n")
        yield sse_token(f"   ✓ Found {len(blocked_refs)} blocked folder references\n")
        
        # Detect duplicate filenames
        duplicate_names = _detect_duplicate_filenames(all_files)
        yield sse_token(f"   ✓ Found {len(duplicate_names)} duplicate filename patterns\n\n")
    
    # =================================================================
    # STRUCTURAL ANALYSIS — run smart scanner on qualifying Python files
    # =================================================================
    yield sse_token("🔬 Running structural analysis on Python files...\n")
    structural_report = run_structural_scan(all_files)
    yield sse_token(
        f"   ✓ Scanned {structural_report.files_scanned} files, "
        f"found {len(structural_report.findings)} structural concerns "
        f"({structural_report.warning_count} warnings, {structural_report.info_count} info)\n"
    )
    if structural_report.files_skipped_small:
        yield sse_token(f"   ℹ️  Skipped {structural_report.files_skipped_small} small files (<200 lines)\n")
    yield sse_token("\n")

    # Generate timestamp for filenames
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H%M")
    timestamp_iso = now.isoformat(timespec="seconds")
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    # Generate reports
    yield sse_token("📝 Generating reports...\n")
    
    md_content = _generate_markdown_report(
        mode=mode,
        timestamp=timestamp_iso,
        duration_ms=duration_ms,
        files=all_files,
        suspect_folders=all_suspect_folders,
        extension_counts=all_extension_counts,
        floating_orb=floating_orb,
        floating_desktop=floating_desktop,
        new_files=new_files,
        changed_files=changed_files,
        deleted_files=deleted_files,
        abs_path_findings=abs_path_findings if mode == "FULL" else None,
        blocked_refs=blocked_refs if mode == "FULL" else None,
        duplicate_names=duplicate_names if mode == "FULL" else None,
        structural_report=structural_report,
    )
    
    json_content = _generate_json_report(
        mode=mode,
        timestamp=timestamp_iso,
        duration_ms=duration_ms,
        files=all_files,
        suspect_folders=all_suspect_folders,
        extension_counts=all_extension_counts,
        floating_orb=floating_orb,
        floating_desktop=floating_desktop,
        new_files=new_files,
        changed_files=changed_files,
        deleted_files=deleted_files,
        abs_path_findings=abs_path_findings if mode == "FULL" else None,
        blocked_refs=blocked_refs if mode == "FULL" else None,
        duplicate_names=duplicate_names if mode == "FULL" else None,
    )
    
    # Write files
    md_filename = f"CODEBASE_REPORT_{mode}_{timestamp_str}.md"
    json_filename = f"CODEBASE_REPORT_{mode}_{timestamp_str}.json"
    
    md_path = CODEBASE_REPORT_OUTPUT_DIR / md_filename
    json_path = CODEBASE_REPORT_OUTPUT_DIR / json_filename
    
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        yield sse_token(f"✓ Wrote: {md_path}\n")
        logger.info(f"[CODEBASE_REPORT] wrote report: {md_path}")
    except Exception as e:
        yield sse_error(f"Failed to write MD: {e}")
    
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=2)
        yield sse_token(f"✓ Wrote: {json_path}\n")
        logger.info(f"[CODEBASE_REPORT] wrote report: {json_path}")
    except Exception as e:
        yield sse_error(f"Failed to write JSON: {e}")
    
    # Save state for next incremental run
    new_state = {
        key: {"size_bytes": entry.size_bytes, "mtime": entry.mtime}
        for key, entry in current_state.items()
    }
    _save_state(new_state)
    yield sse_token("✓ State saved for incremental tracking\n\n")
    
    # Final summary
    total_bytes = sum(f.size_bytes for f in all_files)
    yield sse_token(f"📊 **CODEBASE REPORT COMPLETE**\n")
    yield sse_token(f"   Mode: {mode}\n")
    yield sse_token(f"   Files: {len(all_files)}\n")
    yield sse_token(f"   Total size: {_format_size(total_bytes)}\n")
    yield sse_token(f"   Duration: {duration_ms} ms\n\n")
    yield sse_token(f"📁 Reports written to:\n")
    yield sse_token(f"   • {md_path}\n")
    yield sse_token(f"   • {json_path}\n")
    
    logger.info(f"[CODEBASE_REPORT] files_scanned={len(all_files)} bytes={total_bytes} duration_ms={duration_ms}")
    
    yield sse_done(
        provider="local",
        model="codebase_report",
        total_length=len(all_files),
        success=True,
        meta={
            "mode": mode,
            "files_scanned": len(all_files),
            "total_bytes": total_bytes,
            "duration_ms": duration_ms,
            "md_path": str(md_path),
            "json_path": str(json_path),
        },
    )

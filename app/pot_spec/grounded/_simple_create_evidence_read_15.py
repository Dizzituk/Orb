# Purpose: simple create utils 15 — evidence path-resolve + file read (split from _simple_create_utils_15.py).
# Called-by: app.pot_spec.grounded._simple_create_evidence, app.pot_spec.grounded._simple_create_utils_15, app.pot_spec.grounded._simple_create_utils_16
# Depends-on: app.pot_spec.grounded._sbx_fs, app.pot_spec.grounded._simple_create_utils_14
# Last-renovated: 2026-06-21
from __future__ import annotations
import logging
import os
from typing import List, Optional, Tuple
from app.pot_spec.grounded._simple_create_utils_14 import _EVIDENCE_MAX_FILE_CHARS
from app.pot_spec.grounded._sbx_fs import _sbx_isfile, _sbx_exists, _sbx_read
logger = logging.getLogger(__name__)


def _read_text_any_encoding(file_path: str) -> str:
    """
    v10.0: Read a text file from the sandbox.

    The sandbox controller handles encoding internally.
    No host fallbacks. No multi-encoding retry.
    """
    content = _sbx_read(file_path)
    return content if content is not None else ""


def _resolve_evidence_path(
    file_path: str,
    project_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve a bare or relative file path to an absolute sandbox path.

    v10.0: Uses architecture INDEX.json first (fast, no sandbox round-trips),
    then tries common subdirectory patterns, then direct root+path.
    All existence checks go through the sandbox. No host fallbacks.

    Returns the resolved absolute path, or None if not found.
    """
    basename = os.path.basename(file_path)

    # Strategy 1: Architecture INDEX.json lookup (host operational data)
    try:
        import json as _json
        _idx_path = os.path.join("D:\\Orb", ".architecture", "INDEX.json")
        if os.path.isfile(_idx_path):
            with open(_idx_path, "r", encoding="utf-8") as _f:
                _idx = _json.load(_f)
            for _entry in _idx.get("files", []):
                _name = _entry.get("name", "")
                _path = _entry.get("path", "")
                if _name == basename and _path:
                    if _sbx_isfile(_path):
                        return _path
                # Also try matching the relative path suffix
                if _path and _path.replace("\\", "/").endswith(
                    file_path.replace("\\", "/")
                ):
                    if _sbx_isfile(_path):
                        return _path
    except Exception:
        pass

    # Strategy 2: Try common subdirectory patterns via sandbox
    _subdirs = [
        "src", "app", "src/components", "src/services", "src/types",
        "src/components/chat-panel", "src/components/debug",
        "src/components/builds",
    ]
    if project_paths:
        for root in project_paths:
            for subdir in _subdirs:
                candidate = os.path.join(root, subdir, file_path)
                if _sbx_isfile(candidate):
                    return candidate
            # Direct root + path
            candidate = os.path.join(root, file_path)
            if _sbx_isfile(candidate):
                return candidate

    return None


def _host_read_file(file_path: str, max_chars: int = 0, project_paths: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Read a file from the host filesystem for evidence fulfilment.

    v4.1: Added project_paths parameter for resolving relative paths.
    If file_path is not absolute or doesn't exist, tries resolving against
    each project root (e.g. 'app/llm/stream_router.py' → 'D:\\Orb\\app\\llm\\stream_router.py').

    Returns (success, content_or_error_message).
    Uses _read_text_any_encoding for robust encoding handling.
    """
    if not max_chars:
        max_chars = _EVIDENCE_MAX_FILE_CHARS

    # Normalise path separators for Windows
    file_path = file_path.replace('/', os.sep).replace('\\', os.sep)

    # v10.0: Resolve relative/bare paths using INDEX.json + sandbox checks
    if not _sbx_exists(file_path):
        resolved = _resolve_evidence_path(file_path, project_paths)
        if resolved:
            logger.info("[SPEC_GATE_EVIDENCE] Resolved path: %s → %s", file_path, resolved)
            file_path = resolved

    # v4.2: TypeScript barrel export fallback.
    # If 'foo.ts' not found, try 'foo/index.ts' and 'foo/index.tsx'.
    # Common TS pattern: `import { X } from './types'` resolves to types/index.ts
    if not _sbx_exists(file_path):
        _tried_barrel = False
        if file_path.endswith(('.ts', '.tsx')):
            _stem = file_path.rsplit('.', 1)[0]
            for _barrel_ext in ('/index.ts', '/index.tsx'):
                _barrel = _stem + _barrel_ext
                if _sbx_exists(_barrel):
                    logger.info(
                        "[SPEC_GATE_EVIDENCE] v4.2 Barrel fallback: %s → %s",
                        file_path, _barrel,
                    )
                    file_path = _barrel
                    _tried_barrel = True
                    break
        if not _tried_barrel:
            logger.info("[SPEC_GATE_EVIDENCE] File not found: %s", file_path)
            return False, f"File not found: {file_path}"

    if not _sbx_isfile(file_path):
        logger.info("[SPEC_GATE_EVIDENCE] Not a file: %s", file_path)
        return False, f"Path is not a file: {file_path}"

    try:
        content = _read_text_any_encoding(file_path)
        if not content:
            return False, f"File is empty or unreadable: {file_path}"
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... [truncated at {max_chars} chars, file has {len(content)} total]"
        logger.info("[SPEC_GATE_EVIDENCE] Read %d chars from %s", min(len(content), max_chars), file_path)
        return True, content
    except Exception as exc:
        logger.warning("[SPEC_GATE_EVIDENCE] Failed to read %s: %s", file_path, exc)
        return False, f"Read error: {exc}"

import logging
import os
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_SANDBOX_CLIENT_AVAILABLE = True
call_fs_tree = None
_CONFIG_AVAILABLE = True
FILESYSTEM_QUERY_ALLOWED_ROOTS = []
FILESYSTEM_QUERY_BLOCKED_PATHS = []


EVIDENCE_GATHERING_BUILD_ID = "2026-02-12-v2.2-index-json-path-fallback"

EVIDENCE_ALLOWED_ROOTS = FILESYSTEM_QUERY_ALLOWED_ROOTS if _CONFIG_AVAILABLE else [
    "D:\\",
    "D:\\Orb",
    "D:\\orb-desktop",
    "C:\\Users\\dizzi\\OneDrive",
]

EVIDENCE_FORBIDDEN_PATHS = FILESYSTEM_QUERY_BLOCKED_PATHS if _CONFIG_AVAILABLE else [
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\Users\\dizzi\\AppData",
]

def sandbox_list_directory(path: str) -> Tuple[bool, List[Dict]]:
    """
    v1.26: List directory contents from SANDBOX filesystem.
    
    Returns:
        (success: bool, files: List[Dict])
        Each file dict contains 'path', 'name', 'size', 'is_dir'
    """
    if not _SANDBOX_CLIENT_AVAILABLE or not call_fs_tree:
        logger.warning("[evidence_gathering] v1.26 sandbox_list_directory: sandbox client not available")
        return False, []
    
    try:
        logger.info("[evidence_gathering] v1.26 sandbox_list_directory: listing %s", path)
        
        status, data, error = call_fs_tree([path], max_files=100)
        
        if status != 200 or not data:
            logger.warning(
                "[evidence_gathering] v1.26 sandbox_list_directory: failed for %s (status=%s, error=%s)",
                path, status, error
            )
            return False, []
        
        files = data.get("files", [])
        
        result = []
        for f in files:
            if isinstance(f, dict):
                result.append({
                    "path": f.get("path", ""),
                    "name": os.path.basename(f.get("path", "")),
                    "size": f.get("size"),
                    "is_dir": f.get("is_dir", False),
                })
            else:
                result.append({
                    "path": str(f),
                    "name": os.path.basename(str(f)),
                    "size": None,
                    "is_dir": False,
                })
        
        logger.info(
            "[evidence_gathering] v1.26 sandbox_list_directory: found %d items in %s",
            len(result), path
        )
        return True, result
        
    except Exception as e:
        logger.warning(
            "[evidence_gathering] v1.26 sandbox_list_directory: exception listing %s: %s",
            path, e
        )
        return False, []

def read_interface_signatures(
    path: str,
    max_chars: int = 2000,
) -> Tuple[bool, List[str], Optional[str]]:
    """
    v2.2: Read class/function/export signatures from a source file.
    
    Host-direct read — does NOT use the sandbox client. This is used by
    the file verifier during segmentation, which runs before any sandbox
    session is active.
    
    Extracts structural signatures (class names, function definitions,
    exports) from the first `max_chars` of a file. Supports Python (.py)
    and TypeScript/React (.ts, .tsx, .js, .jsx).
    
    Args:
        path: Absolute file path on the host filesystem
        max_chars: Maximum characters to read for signature extraction
    
    Returns:
        (success: bool, signatures: List[str], excerpt: Optional[str])
        - success: True if the file was readable
        - signatures: Extracted class/function/export signature lines
        - excerpt: First ~500 chars of file content
    """
    import re as _re
    
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(max_chars)
    except OSError as e:
        logger.warning("[evidence_gathering] v2.2 read_interface_signatures: cannot read %s: %s", path, e)
        return False, [], None
    
    excerpt = content[:500]
    signatures: list = []
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.py':
        # Python: class and public function definitions
        for match in _re.finditer(r'^class\s+(\w+)\s*[\(:]', content, _re.MULTILINE):
            line_start = content.rfind('\n', 0, match.start()) + 1
            line_end = content.find('\n', match.start())
            if line_end == -1:
                line_end = len(content)
            sig = content[line_start:line_end].strip()
            signatures.append(sig[:200])
        
        for match in _re.finditer(r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)', content, _re.MULTILINE):
            name = match.group(1)
            if name.startswith('_') and name != '__init__':
                continue
            line_start = content.rfind('\n', 0, match.start()) + 1
            line_end = content.find('\n', match.start())
            if line_end == -1:
                line_end = len(content)
            sig = content[line_start:line_end].strip()
            signatures.append(sig[:200])
    
    elif ext in ('.ts', '.tsx', '.js', '.jsx'):
        # TypeScript/JS: exported definitions
        for match in _re.finditer(
            r'^export\s+(?:default\s+)?(?:function|const|class|interface|type|enum)\s+(\w+)',
            content, _re.MULTILINE,
        ):
            line_start = content.rfind('\n', 0, match.start()) + 1
            line_end = content.find('\n', match.start())
            if line_end == -1:
                line_end = len(content)
            sig = content[line_start:line_end].strip()
            signatures.append(sig[:200])
    
    logger.info(
        "[evidence_gathering] v2.2 read_interface_signatures: %s — %d signatures extracted",
        path, len(signatures),
    )
    return True, signatures, excerpt

def _resolve_via_index_json(ref_normalized: str) -> Optional[str]:
    """
    v2.2: Search INDEX.json for a file matching the bare filename or relative path.

    INDEX.json lives at D:\\Orb\\.architecture\\INDEX.json and contains an entry
    for every project file with its absolute path.  This function builds a
    basename → [absolute_path, ...] lookup on first call, then matches:
      1. Exact basename match  (e.g. "architecture_executor.py")
      2. Suffix match          (e.g. "app\\overwatcher\\architecture_executor.py")

    If multiple matches exist, prefer the one under EVIDENCE_ALLOWED_ROOTS.
    Returns the absolute path or None.
    """
    global _INDEX_JSON_CACHE, _INDEX_JSON_MTIME

    index_path = os.path.join("D:\\Orb", ".architecture", "INDEX.json")

    # Load / refresh cache
    try:
        current_mtime = os.path.getmtime(index_path) if os.path.isfile(index_path) else 0.0
    except OSError:
        current_mtime = 0.0

    if _INDEX_JSON_CACHE is None or current_mtime != _INDEX_JSON_MTIME:
        _INDEX_JSON_CACHE = {}
        _INDEX_JSON_MTIME = current_mtime
        try:
            import json as _json
            with open(index_path, "r", encoding="utf-8") as fh:
                index_data = _json.load(fh)
            for entry in index_data.get("files", []):
                abs_path = entry.get("path", "")
                if abs_path:
                    basename = os.path.basename(abs_path).lower()
                    _INDEX_JSON_CACHE.setdefault(basename, []).append(abs_path)
            logger.info(
                "[evidence_gathering] v2.2 INDEX.json loaded: %d unique basenames, %d files",
                len(_INDEX_JSON_CACHE),
                sum(len(v) for v in _INDEX_JSON_CACHE.values()),
            )
        except Exception as exc:
            logger.warning("[evidence_gathering] v2.2 Failed to load INDEX.json: %s", exc)
            return None

    if not _INDEX_JSON_CACHE:
        return None

    # Normalise the reference for matching
    ref_lower = ref_normalized.replace("/", "\\").lower()
    ref_basename = os.path.basename(ref_lower)

    candidates = _INDEX_JSON_CACHE.get(ref_basename, [])
    if not candidates:
        return None

    # If only one match, return it
    if len(candidates) == 1:
        return candidates[0]

    # Multiple matches — try suffix match (e.g. ref="app\overwatcher\foo.py")
    if "\\" in ref_lower:
        suffix = "\\" + ref_lower  # prepend separator for safe endswith
        for c in candidates:
            if c.lower().endswith(suffix) or c.lower().endswith(ref_lower):
                return c

    # Prefer paths under EVIDENCE_ALLOWED_ROOTS (project files over random matches)
    for c in candidates:
        c_lower = c.lower()
        for root in EVIDENCE_ALLOWED_ROOTS:
            if c_lower.startswith(root.lower()):
                return c

    # Last resort: first candidate
    return candidates[0]

def format_evidence_for_prompt(package: EvidencePackage) -> str:
    """
    v1.27: Format evidence package for injection into LLM prompt.
    
    Now handles multi-target read packages with full content.
    """
    lines = [
        "## Filesystem Evidence (Ground Truth)",
        f"Timestamp: {package.ground_truth_timestamp}",
        f"Task Type: {package.task_type}",
    ]
    
    if package.is_multi_target:
        lines.append("Mode: MULTI-TARGET READ")
    
    lines.append("")
    lines.append("### Target Files")
    
    if package.target_files:
        for fe in package.target_files:
            lines.append(fe.to_evidence_line())
            
            # v1.27: Include full content for multi-target reads
            if package.is_multi_target and fe.full_content:
                lines.append("")
                lines.append(f"**Content of {fe.resolved_path}:**")
                lines.append("```")
                lines.append(fe.full_content)
                lines.append("```")
                lines.append("")
    else:
        lines.append("  (No files resolved)")
    
    if package.validation_errors:
        lines.append("")
        lines.append("### Validation Errors")
        for err in package.validation_errors:
            lines.append(f"  - ⚠️ {err}")
    
    if package.warnings:
        lines.append("")
        lines.append("### Warnings")
        for warn in package.warnings:
            lines.append(f"  - {warn}")
    
    return "\n".join(lines)

async def format_multi_target_reply(
    package: EvidencePackage,
    provider_id: str = "openai",
    model_id: str = "gpt-5-mini",
    llm_call_func = None,
    user_request: Optional[str] = None,
) -> str:
    """
    v1.30: Format a SYNTHESIZED reply for multi-target read operations.
    
    CRITICAL FIX v1.30: Now generates ONE SYNTHESIZED reply from ALL files!
    Previous versions generated separate replies per file, missing the conceptual link.
    
    Example:
        File 1: "My name is Astra"
        File 2: "I'm going to be an assistant"
        File 3: "I'm going to help you every day"
        File 4: "Where shall we begin?"
        
        Synthesized output: "The files together describe an AI assistant named Astra
                            that will help the user daily and is ready to start."
    
    Args:
        package: EvidencePackage with multi-target files
        provider_id: LLM provider for reply generation
        model_id: LLM model for reply generation
        llm_call_func: Optional LLM call function
        user_request: The user's original request for context
    """
    if not package.is_multi_target:
        return ""
    
    valid_targets = package.get_all_valid_targets()
    
    if not valid_targets:
        lines = ["I couldn't find any of the requested files."]
        for result in package.multi_target_results:
            lines.append(f"- {result['name']}: Not found")
        return "\n".join(lines)
    
    # Build list of files for synthesis
    total = len(package.target_files)
    found = len(valid_targets)
    
    # v1.30: Collect all file contents for synthesis
    file_contents = []
    for fe in valid_targets:
        content = fe.full_content if fe.full_content else fe.content_preview
        if content:
            file_contents.append({
                'path': fe.resolved_path,
                'name': os.path.basename(fe.resolved_path) if fe.resolved_path else fe.original_reference,
                'content': content,
                'content_type': fe.detected_structure,
            })
    
    if not file_contents:
        return "Found files but all were empty or unreadable."
    
    # v1.30: Generate SYNTHESIZED reply using qa_processing
    try:
        from .qa_processing import generate_synthesized_reply_from_files
        _SYNTHESIS_AVAILABLE = True
    except ImportError:
        logger.warning("[evidence_gathering] v1.30 generate_synthesized_reply_from_files not available")
        _SYNTHESIS_AVAILABLE = False
    
    # Build header with file status
    header_lines = []
    if found < total:
        missing = [r["name"] for r in package.multi_target_results if not r["found"]]
        header_lines.append(f"Found {found} of {total} files. Missing: {', '.join(missing)}")
        header_lines.append("")
    
    # v1.30: CRITICAL - Call synthesis function for COMBINED understanding
    if _SYNTHESIS_AVAILABLE:
        logger.info(
            "[evidence_gathering] v1.30 MULTI-FILE SYNTHESIS: Calling generate_synthesized_reply_from_files for %d files",
            len(file_contents)
        )
        try:
            synthesis = await generate_synthesized_reply_from_files(
                file_contents=file_contents,
                provider_id=provider_id,
                model_id=model_id,
                llm_call_func=llm_call_func,
                user_request=user_request,
            )
            if synthesis:
                header_lines.append("**Synthesized Understanding:**")
                header_lines.append(synthesis)
                logger.info(
                    "[evidence_gathering] v1.30 MULTI-FILE SYNTHESIS SUCCESS: %d chars",
                    len(synthesis)
                )
            else:
                header_lines.append("(Synthesis returned empty - see file contents below)")
                logger.warning("[evidence_gathering] v1.30 Synthesis returned empty")
        except Exception as e:
            logger.error(
                "[evidence_gathering] v1.30 MULTI-FILE SYNTHESIS FAILED: %s", e
            )
            header_lines.append(f"(Synthesis error: {str(e)[:100]})")
    else:
        # Fallback: just list file contents
        logger.warning("[evidence_gathering] v1.30 Synthesis unavailable, using fallback")
        header_lines.append("**Files read:**")
        for i, fc in enumerate(file_contents, 1):
            name = fc.get('name', f'File {i}')
            content = fc['content'][:200] if len(fc['content']) > 200 else fc['content']
            header_lines.append(f"")
            header_lines.append(f"File {i} ({name}):")
            header_lines.append(content)
    
    return "\n".join(header_lines)

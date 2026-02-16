from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from .constants import SOURCE_CONTEXT_MAX_CHARS, INTERFACE_SUMMARY_MAX_CHARS
from .path_resolution import _resolve_multi_root_path
from ..sandbox_client import SandboxClient

logger = logging.getLogger(__name__)

__all__ = [
    "_read_source_context",
    "_format_job_context",
    "_extract_file_interfaces",
    "_extract_existing_imports",
    "_extract_router_registrations",
    "_build_resolved_endpoints",
]


async def _read_existing_file(
    sandbox: SandboxClient,
    file_path: str,
    max_chars: int = SOURCE_CONTEXT_MAX_CHARS
) -> Optional[str]:
    """
    Read an existing file from the sandbox, truncating if necessary.
    
    Args:
        sandbox: SandboxClient instance
        file_path: Path to file to read
        max_chars: Maximum characters to read
        
    Returns:
        File content (truncated if necessary) or None if file doesn't exist
    """
    try:
        content = await sandbox.read_file(file_path)
        if content is None:
            return None
        if len(content) > max_chars:
            logger.warning(f"Truncating {file_path} from {len(content)} to {max_chars} chars")
            content = content[:max_chars] + "\n... [TRUNCATED] ..."
        return content
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return None


async def _read_source_context(
    sandbox: SandboxClient,
    source_files: List[str],
    project_roots: List[Path]
) -> str:
    """
    Read content from source files being decomposed.
    
    Args:
        sandbox: SandboxClient instance
        source_files: List of source file paths (relative or absolute)
        project_roots: List of project root directories for path resolution
        
    Returns:
        Formatted source context string
    """
    if not source_files:
        return ""
    
    lines = ["## SOURCE FILES (v3.0 CRITICAL)\n"]
    lines.append("The following source files contain the REAL implementation being extracted/decomposed.")
    lines.append("You MUST copy function bodies, class definitions, constants, and imports VERBATIM from these files.\n")
    
    for sf in source_files:
        resolved = _resolve_multi_root_path(sf, project_roots)
        content = await _read_existing_file(sandbox, resolved, SOURCE_CONTEXT_MAX_CHARS)
        if content:
            lines.append(f"### Source: `{sf}`\n```python\n{content}\n```\n")
        else:
            lines.append(f"### Source: `{sf}` (NOT FOUND)\n")
    
    return "\n".join(lines)


def _format_job_context(
    job: Dict,
    files_created: List[str],
    available_modules: Dict[str, List[str]],
    source_context: str,
    interface_summary: str,
    endpoint_context: str
) -> str:
    """
    Format the complete context for a file generation job.
    
    Args:
        job: Job dictionary containing file path, architecture, etc.
        files_created: List of file paths already created in this job
        available_modules: Dict mapping import categories to module lists
        source_context: Pre-formatted source file context
        interface_summary: Pre-formatted interface summary
        endpoint_context: Pre-formatted endpoint context
        
    Returns:
        Formatted context string
    """
    lines = []
    
    # Files already created
    if files_created:
        lines.append("## Files Already Created in This Job\n")
        lines.append("These files exist on disk. Use the EXACT class names, method signatures, and import paths listed here.\n")
        for fp in files_created:
            lines.append(f"  - `{fp}`")
        lines.append("")
    
    # Available modules
    lines.append("## Available Modules (DO NOT invent imports outside this list)\n")
    
    if available_modules.get("sibling"):
        lines.append("### Sibling modules (use `from .module import ...`)")
        lines.append("These are in the same package. Import with a single dot.\n")
        for mod in available_modules["sibling"]:
            lines.append(f"  - `{mod}`")
        lines.append("")
    
    if available_modules.get("parent"):
        lines.append("### Parent modules (use `from ..module import ...`)")
        lines.append("These are in the parent package directory. Import with double dot `.`.")
        lines.append("Do NOT use absolute imports like `from app.x.y import Z`. Use RELATIVE imports: `from ..module_name import ClassName`.\n")
        for mod in available_modules["parent"]:
            lines.append(f"  - `{mod}`")
        lines.append("")
    
    lines.append("**CRITICAL**: Do NOT invent imports to files not listed above. Do NOT use absolute imports (e.g. `from app.models.X`) when a relative import from the parent package exists (e.g. `from ..X import Y`). Every import MUST resolve to a file in one of these lists.")
    
    # Source context (if any)
    if source_context:
        lines.append("\n" + source_context)
    
    # Interface summary (if any)
    if interface_summary:
        lines.append("\n" + interface_summary)
    
    # Endpoint context (if any)
    if endpoint_context:
        lines.append("\n" + endpoint_context)
    
    return "\n".join(lines)


def _extract_file_interfaces(content: str, file_path: str) -> str:
    """
    Extract class/function signatures from a file's content.
    
    Args:
        content: File content
        file_path: Path to the file (for error messages)
        
    Returns:
        Interface summary string
    """
    if not content:
        return f"### `{file_path}` (empty or not found)\n"
    
    lines = [f"### `{file_path}`\n"]
    
    # Extract class definitions
    class_pattern = re.compile(r'^class\s+(\w+).*?:', re.MULTILINE)
    for match in class_pattern.finditer(content):
        lines.append(f"  - class `{match.group(1)}`")
    
    # Extract function/method definitions (public only)
    func_pattern = re.compile(r'^(?:async\s+)?def\s+([a-zA-Z_]\w*)\s*\([^)]*\)', re.MULTILINE)
    for match in func_pattern.finditer(content):
        func_name = match.group(1)
        if not func_name.startswith('_'):
            lines.append(f"  - function `{func_name}(...)`")
    
    # Extract __all__ exports
    all_pattern = re.compile(r'__all__\s*=\s*\[(.*?)\]', re.DOTALL)
    all_match = all_pattern.search(content)
    if all_match:
        exports = re.findall(r'["\']([^"\']+)["\']', all_match.group(1))
        if exports:
            lines.append(f"  - exports: {', '.join(exports)}")
    
    lines.append("")
    return "\n".join(lines)


def _extract_existing_imports(content: str) -> List[str]:
    """
    Extract all import statements from file content.
    
    Args:
        content: File content
        
    Returns:
        List of import statement strings
    """
    if not content:
        return []
    
    imports = []
    
    # Match "from X import Y" and "import X"
    import_pattern = re.compile(
        r'^(?:from\s+[\w.]+\s+import\s+[^\n]+|import\s+[^\n]+)',
        re.MULTILINE
    )
    
    for match in import_pattern.finditer(content):
        imports.append(match.group(0).strip())
    
    return imports


def _extract_router_registrations(content: str) -> List[Dict[str, str]]:
    """
    Extract router registration statements from file content.
    
    Looks for patterns like:
    - app.include_router(some_router, prefix="/path")
    - router.include_router(sub_router, prefix="/sub")
    
    Args:
        content: File content
        
    Returns:
        List of dicts with keys: router_var, prefix
    """
    if not content:
        return []
    
    registrations = []
    
    # Pattern: .include_router(router_var, prefix="/path", ...)
    pattern = re.compile(
        r'\.include_router\s*\(\s*(\w+)\s*,\s*prefix\s*=\s*["\']([^"\']+)["\']',
        re.MULTILINE
    )
    
    for match in pattern.finditer(content):
        registrations.append({
            "router_var": match.group(1),
            "prefix": match.group(2)
        })
    
    return registrations


def _build_resolved_endpoints(
    router_file: str,
    router_var: str,
    prefix: str,
    router_content: str
) -> List[str]:
    """
    Build a list of resolved endpoint strings from a router file.
    
    Args:
        router_file: Path to the router file
        router_var: Router variable name (e.g., "router", "health_router")
        prefix: URL prefix for this router
        router_content: Content of the router file
        
    Returns:
        List of endpoint strings like "GET /api/health/status"
    """
    if not router_content:
        return []
    
    endpoints = []
    
    # Pattern: @router.get("/path"), @router.post("/path"), etc.
    pattern = re.compile(
        rf'@{re.escape(router_var)}\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
        re.MULTILINE
    )
    
    for match in pattern.finditer(router_content):
        method = match.group(1).upper()
        path = match.group(2)
        
        # Combine prefix and path
        full_path = prefix.rstrip('/') + '/' + path.lstrip('/')
        endpoints.append(f"{method} {full_path}")
    
    return endpoints
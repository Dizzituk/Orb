"""
Context building utilities for architecture execution.

Provides cross-file context accumulation, interface extraction,
import pattern extraction, router registration tracking, and
resolved endpoint building.

RESTORED from original monolith — refactored version had wrong signatures.
BUILD_ID: 2026-02-16-v1.1-restored-from-monolith
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import SOURCE_CONTEXT_MAX_CHARS, INTERFACE_SUMMARY_MAX_CHARS
from .path_resolution import _resolve_multi_root_path
from ..sandbox_client import SandboxClient

logger = logging.getLogger(__name__)


# =============================================================================
# File Reading (READ-ONLY — Overwatcher verification)
# =============================================================================

async def _read_existing_file(client: SandboxClient, path: str) -> Optional[str]:
    """Read existing file from sandbox for modification context (READ-ONLY)."""
    try:
        cmd = f'Get-Content -Path "{path}" -Raw -Encoding UTF8'
        result = client.shell_run(cmd, timeout_seconds=30)
        if result.stdout is not None:
            return result.stdout
        return None
    except Exception as e:
        logger.error("[arch_exec] Read exception for %s: %s", path, e)
        return None


# =============================================================================
# v3.0: Source Context for CREATE Extractions
# =============================================================================

async def _read_source_context(
    client: SandboxClient,
    source_files: List[str],
    sandbox_base: str,
) -> str:
    """v3.0: Read source files and format as context for the Implementer.

    Returns formatted source context string, or empty string if no sources found.
    """
    if not source_files:
        return ""

    context_parts = []
    total_chars = 0

    for src_path in source_files:
        abs_path = _resolve_multi_root_path(src_path, sandbox_base)
        content = await _read_existing_file(client, abs_path)
        if content and len(content.strip()) > 10:
            # Truncate if too large
            if total_chars + len(content) > SOURCE_CONTEXT_MAX_CHARS:
                remaining = SOURCE_CONTEXT_MAX_CHARS - total_chars
                if remaining > 500:
                    content = content[:remaining] + "\n# ... [TRUNCATED — file too large for full context]"
                else:
                    continue
            context_parts.append(
                f"## Source File: `{src_path}`\n"
                f"The following is the ACTUAL content of the source file. "
                f"When extracting functions/classes, copy the REAL code from here — "
                f"do NOT rewrite or reimagine the implementation.\n"
                f"```\n{content}\n```"
            )
            total_chars += len(content)
            logger.info("[arch_exec] v3.0 Loaded source context: %s (%d chars)", src_path, len(content))

    if not context_parts:
        return ""

    return (
        "## SOURCE FILES (v3.0 — COPY REAL CODE, DO NOT REWRITE)\n\n"
        "The architecture is extracting/decomposing code from the source file(s) below. "
        "You MUST copy the actual function/class implementations verbatim from these sources. "
        "Do NOT reimagine, rewrite, or hallucinate alternative implementations. "
        "Use the exact same imports, function signatures, variable names, and logic.\n\n"
        + "\n\n".join(context_parts)
    )


# =============================================================================
# v2.3: Cross-File Context — Interface Extraction & Job Context
# =============================================================================

def _extract_file_interfaces(file_path: str, content: str) -> str:
    """Extract key interfaces from generated file content.

    v2.4: Enhanced extraction with canonical import paths, actual exported names,
    and TypeScript interface property extraction.

    v2.3: Lightweight extraction of class names, function signatures,
    exported constants, and import paths from Python and TypeScript files.

    Args:
        file_path: Relative path of the file (used to determine language)
        content: The generated file content

    Returns:
        A concise summary of the file's key interfaces with import paths
    """
    lines = content.split('\n')
    interfaces: List[str] = []

    is_python = file_path.endswith('.py')
    is_typescript = file_path.endswith(('.ts', '.tsx'))

    if is_python:
        # v2.4: Collect exported names to build canonical import statement
        exported_names: List[str] = []
        # v2.5: Collect router endpoint information for URL path resolution
        router_prefix = ""  # APIRouter(prefix="/xxx")
        endpoint_paths: List[str] = []  # [("GET", "/status"), ...]

        for line in lines:
            stripped = line.rstrip()
            # Class definitions
            class_match = re.match(r'^class\s+(\w+)', stripped)
            if class_match:
                interfaces.append(stripped.rstrip(':'))
                exported_names.append(class_match.group(1))
                continue
            # Top-level function definitions (not indented = module-level)
            func_match = re.match(r'^(?:async\s+)?def\s+(\w+)', stripped)
            if func_match:
                interfaces.append(stripped.rstrip(':'))
                # Only export public functions (no leading underscore)
                if not func_match.group(1).startswith('_'):
                    exported_names.append(func_match.group(1))
                continue
            # Module-level constants (ALL_CAPS = ...)
            const_match = re.match(r'^([A-Z][A-Z_0-9]+)\s*=', stripped)
            if const_match:
                if len(stripped) > 80:
                    interfaces.append(stripped[:80] + '...')
                else:
                    interfaces.append(stripped)
                exported_names.append(const_match.group(1))
                continue
            # Router instances — v2.5: also capture prefix
            router_match = re.match(r'^router\s*=\s*APIRouter\((.*)\)', stripped)
            if router_match:
                interfaces.append(stripped)
                exported_names.append('router')
                # Extract prefix if present
                prefix_match = re.search(r'prefix\s*=\s*["\']([^"\']*)["\']', router_match.group(1))
                if prefix_match:
                    router_prefix = prefix_match.group(1)
                continue
            if re.match(r'^router\s*=\s*APIRouter', stripped):
                interfaces.append(stripped)
                exported_names.append('router')
                continue
            # v2.5: Endpoint decorators — @router.get("/path"), @router.post("/path"), etc.
            ep_match = re.match(r'^@router\.(get|post|put|patch|delete|websocket)\s*\(\s*["\']([^"\']+)', stripped)
            if ep_match:
                method = ep_match.group(1).upper()
                path = ep_match.group(2)
                endpoint_paths.append(f"{method} {path}")

        # v2.4: Build canonical import path with actual exported names
        module_path = file_path.replace('/', '.').replace('\\', '.')
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        # Strip orb-desktop. prefix if present (shouldn't be for Python, but defensive)
        if module_path.startswith('orb-desktop.'):
            module_path = module_path[len('orb-desktop.'):]

        if exported_names:
            # Show actual importable names (limit to keep concise)
            names_str = ', '.join(exported_names[:8])
            if len(exported_names) > 8:
                names_str += ', ...'
            interfaces.insert(0, f"Import: from {module_path} import {names_str}")
        else:
            interfaces.insert(0, f"Import: from {module_path} import ...")

        # v2.5: Append endpoint summary if this is a router file
        if endpoint_paths:
            interfaces.append("")
            interfaces.append(f"Router prefix: '{router_prefix}' (empty = no prefix)")
            interfaces.append("Endpoints (before registration prefix):")
            for ep in endpoint_paths:
                interfaces.append(f"  {ep}")

    elif is_typescript:
        # v2.4: Also extract interface properties and compute import path
        in_interface = False
        interface_name = ""
        interface_props: List[str] = []
        brace_depth = 0

        for line in lines:
            stripped = line.rstrip()

            # Track interface blocks for property extraction
            if in_interface:
                brace_depth += stripped.count('{') - stripped.count('}')
                # Extract property definitions inside interface
                prop_match = re.match(r'^\s+(\w+)(\??):(.+)', stripped)
                if prop_match:
                    prop_name = prop_match.group(1)
                    optional = prop_match.group(2)
                    prop_type = prop_match.group(3).strip().rstrip(';').strip()
                    interface_props.append(f"  {prop_name}{optional}: {prop_type}")
                if brace_depth <= 0:
                    # Interface block closed — emit summary
                    if interface_props:
                        interfaces.append(f"interface {interface_name} {{")
                        for prop in interface_props:
                            interfaces.append(prop)
                        interfaces.append("}")
                    in_interface = False
                    interface_props = []
                continue

            # Named exports (including interface starts)
            iface_match = re.match(r'^export\s+(?:default\s+)?interface\s+(\w+)', stripped)
            if iface_match:
                interface_name = iface_match.group(1)
                in_interface = True
                brace_depth = stripped.count('{') - stripped.count('}')
                # If single-line interface, don't enter block mode
                if brace_depth <= 0 and '{' in stripped:
                    in_interface = False
                    interfaces.append(stripped[:120] + ('...' if len(stripped) > 120 else ''))
                continue

            if re.match(r'^export\s+(default\s+)?(function|const|class|type|enum)\s+', stripped):
                sig = stripped[:120] + ('...' if len(stripped) > 120 else '')
                interfaces.append(sig)
                continue

            # Default export at end of file
            if re.match(r'^export\s+default\s+\w+', stripped):
                interfaces.append(stripped)

        # v2.4: Compute TypeScript import path using @/ alias convention
        ts_path = file_path.replace('\\', '/')
        # Strip orb-desktop/ prefix and src/ to get @/ path
        if ts_path.startswith('orb-desktop/src/'):
            import_path = '@/' + ts_path[len('orb-desktop/src/'):]
        elif ts_path.startswith('src/'):
            import_path = '@/' + ts_path[len('src/'):]
        else:
            import_path = './' + ts_path
        # Remove extension for import
        for ext in ('.tsx', '.ts'):
            if import_path.endswith(ext):
                import_path = import_path[:-len(ext)]
                break
        interfaces.insert(0, f"Import: import {{ ... }} from '{import_path}'")

    else:
        # For other file types, just note it was created
        return f"File created: {file_path}"

    if not interfaces:
        # Fallback: show first 30 lines as context
        preview = '\n'.join(lines[:30])
        if len(preview) > INTERFACE_SUMMARY_MAX_CHARS:
            preview = preview[:INTERFACE_SUMMARY_MAX_CHARS] + '\n...'
        return f"File: {file_path}\n{preview}"

    summary = f"File: {file_path}\n" + '\n'.join(interfaces)
    if len(summary) > INTERFACE_SUMMARY_MAX_CHARS:
        summary = summary[:INTERFACE_SUMMARY_MAX_CHARS] + '\n...'
    return summary


def _extract_existing_imports(file_content: str, file_path: str) -> str:
    """Extract existing import statements from a file being modified.

    v2.4: Scans the current file content to find all import statements.
    These are injected into the MODIFY prompt so the Implementer follows
    established import patterns rather than inventing new module paths.

    Args:
        file_content: The current content of the file being modified
        file_path: The file path (used to determine language)

    Returns:
        Formatted string of existing imports, or empty string if none found
    """
    if not file_content:
        return ""

    imports: List[str] = []
    is_python = file_path.endswith('.py')
    is_typescript = file_path.endswith(('.ts', '.tsx'))

    for line in file_content.split('\n'):
        stripped = line.strip()

        if is_python:
            # from X import Y  or  import X
            if re.match(r'^(?:from\s+\S+\s+import\s|import\s+\S)', stripped):
                imports.append(stripped)
        elif is_typescript:
            # import { X } from 'Y'  or  import X from 'Y'
            if re.match(r'^import\s+', stripped):
                imports.append(stripped)

    if not imports:
        return ""

    # Limit to avoid bloating the prompt (most files have <30 imports)
    if len(imports) > 40:
        imports = imports[:40]
        imports.append(f"... ({len(imports)} total imports, showing first 40)")

    return '\n'.join(imports)


def _extract_router_registrations(file_content: str) -> Dict[str, str]:
    """Extract include_router registration prefixes from a Python file (e.g. main.py).

    v2.5: Scans for patterns like:
        app.include_router(voice_router, prefix="/voice")
        app.include_router(transcribe.router, prefix="/transcription")

    Returns dict mapping router variable names to their registration prefix.
    E.g. {"voice_router": "/voice", "transcribe.router": "/transcription"}
    """
    registrations: Dict[str, str] = {}
    if not file_content:
        return registrations

    for match in re.finditer(
        r'app\.include_router\(\s*([\w.]+)'
        r'(?:.*?prefix\s*=\s*["\']([^"\']*)["\'])?',
        file_content,
    ):
        router_name = match.group(1)
        prefix = match.group(2) or ""
        registrations[router_name] = prefix

    return registrations


def _build_resolved_endpoints(job_context: Dict[str, str], router_registrations: Dict[str, str]) -> str:
    """Build a resolved API endpoints section from router interfaces + registration prefixes.

    v2.5: Combines:
    - Router endpoint info from _extract_file_interfaces (router prefix + endpoints)
    - Registration prefix from _extract_router_registrations (include_router prefix)

    Returns formatted string showing actual resolved URLs, or empty if no endpoints.
    """
    resolved: List[str] = []

    for rel_path, summary in job_context.items():
        if 'Endpoints (before registration prefix):' not in summary:
            continue

        # Parse router prefix from summary
        router_prefix = ""
        prefix_match = re.search(r"Router prefix: '([^']*)'", summary)
        if prefix_match:
            router_prefix = prefix_match.group(1)

        # Parse endpoints from summary
        endpoints: List[str] = []
        in_endpoints = False
        for line in summary.split('\n'):
            if line.strip() == 'Endpoints (before registration prefix):':
                in_endpoints = True
                continue
            if in_endpoints and line.strip().startswith(('GET ', 'POST ', 'PUT ', 'PATCH ', 'DELETE ', 'WEBSOCKET ')):
                endpoints.append(line.strip())
            elif in_endpoints and not line.strip().startswith(' '):
                break

        if not endpoints:
            continue

        # Find registration prefix: check if this router's variable name is registered
        reg_prefix = ""
        filename = Path(rel_path).stem  # e.g. "transcribe" from "app/routers/transcribe.py"
        for router_name, prefix in router_registrations.items():
            # Match "transcribe_router", "transcribe.router", or the module name
            if filename in router_name or router_name.startswith(filename):
                reg_prefix = prefix
                break

        # Combine: registration prefix + router prefix + endpoint path
        combined_prefix = reg_prefix.rstrip('/') + router_prefix.rstrip('/')

        for ep in endpoints:
            # e.g. "GET /status" -> "GET /transcription/status" (with combined prefix)
            parts = ep.split(' ', 1)
            if len(parts) == 2:
                method, path = parts
                full_path = combined_prefix + path if path.startswith('/') else combined_prefix + '/' + path
                resolved.append(f"  {method:8s} {full_path}  (from {Path(rel_path).name})")

    if not resolved:
        return ""

    lines = [
        "",
        "## Resolved API Endpoints",
        "",
        "These are the ACTUAL backend URL paths. Use these exact paths in frontend API calls.",
        "Do NOT add prefixes that aren't shown here.",
        "",
    ]
    lines.extend(resolved)
    lines.append("")
    return '\n'.join(lines)


def _format_job_context(
    job_context: Dict[str, str],
    router_registrations: Optional[Dict[str, str]] = None,
) -> str:
    """Format accumulated job context into a prompt section.

    v2.5: Now includes resolved API endpoints section.
    v2.3: Builds a structured context block that tells the Implementer LLM
    what files have already been created/modified in this job, including
    their key interfaces and import paths.

    Args:
        job_context: Dict mapping relative file paths to interface summaries
        router_registrations: Dict of router name -> registration prefix from main.py

    Returns:
        Formatted string for inclusion in the Implementer prompt,
        or empty string if no context yet
    """
    if not job_context:
        return ""

    sections = []
    sections.append("## Files Already Created in This Job")
    sections.append("")
    sections.append(
        "The following files have already been created or modified in this job. "
        "Use the EXACT import paths, class names, and method signatures shown below. "
        "Do NOT invent alternative names or paths — these are the ground truth."
    )
    sections.append("")

    for rel_path, summary in job_context.items():
        sections.append(f"### `{rel_path}`")
        sections.append("```")
        sections.append(summary)
        sections.append("```")
        sections.append("")

    # v2.5: Append resolved API endpoints
    endpoints_section = _build_resolved_endpoints(
        job_context, router_registrations or {}
    )
    if endpoints_section:
        sections.append(endpoints_section)

    return '\n'.join(sections)

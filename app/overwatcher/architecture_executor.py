# FILE: app/overwatcher/architecture_executor.py
"""Architecture Executor: Supervises architecture-level spec execution.

The Architecture Executor is part of the OVERWATCHER domain. It:
    - Parses architecture documents (READ-ONLY)
    - Calls the Implementer LLM (Sonnet) to generate file content
    - Delegates ALL writes to the Implementer via run_implementer_task()
    - Reads from sandbox to verify results independently
    - Implements three-strike error handling

CRITICAL RULE: Overwatcher (this module) NEVER writes to the sandbox.
    All writes go through the Implementer (implementer.py).

v2.6 (2026-02-07): AUTO __init__.py FOR NEW PYTHON PACKAGES
    - New Python files in directories without __init__.py get auto-created init files
    - _ensure_python_init_files() scans manifest + sandbox before the task loop
    - Checks both the file manifest AND the sandbox (existing dirs may already have init)
    - Only targets backend Python paths (not frontend orb-desktop/ paths)
    - Injected init files are empty (just a comment) and don't need LLM generation
    - Fixes: new app/services/ and app/routers/ dirs being non-importable packages

v2.5 (2026-02-07): CROSS-FILE COHERENCE — FACTORY FUNCTIONS + URL PATHS
    - Bug 1 fix: System prompt now requires factory/singleton functions when consuming code expects them
    - Bug 1 fix: Two-pass context — after all CREATEs, re-extract interfaces before MODIFY phase
    - Bug 2 fix: _extract_file_interfaces() now captures router prefixes and endpoint paths
    - Bug 2 fix: _extract_router_registrations() captures include_router prefix from main.py
    - Bug 2 fix: _format_job_context() now includes resolved API endpoints section
    - System prompt: MODIFY rule 12 — frontend API calls must use exact endpoint paths from context
    - System prompt: CREATE rule 9 — implement ALL module-level accessors (factories, singletons)

v2.4 (2026-02-07): CROSS-FILE CONTEXT ACCURACY IMPROVEMENTS
    - _extract_file_interfaces() now includes canonical import paths with actual exported names
    - Python: 'from app.services.transcription_service import TranscriptionService' (not generic)
    - TypeScript: import paths use @/ alias convention; interface props extracted with types
    - Added _extract_existing_imports() for MODIFY operations (existing import pattern awareness)
    - MODIFY prompt now includes existing imports so Sonnet follows established patterns
    - IMPLEMENTER_MODIFY_FILE_SYSTEM updated: import pattern matching, .gitignore safety rules

v2.3 (2026-02-07): CROSS-FILE CONTEXT DURING IMPLEMENTATION
    - Implementer LLM now receives accumulated interface context from earlier files in same job
    - After each successful file create/modify, key interfaces extracted and accumulated
    - MODIFY operations include actual import paths from job-created files
    - System prompts updated to reference cross-file context when present
    - Addresses: hallucinated imports, mismatched interfaces, README overwrites
    - Added _extract_file_interfaces() for lightweight interface extraction
    - Added _format_job_context() to build context section for prompts

v2.2 (2026-02-07): MULTI-ROOT PATH RESOLUTION
    - Frontend files (orb-desktop/ prefix) resolve to D:\orb-desktop
    - Backend files resolve to D:\Orb (sandbox_base) as before
    - Added _resolve_multi_root_path() for split project root handling
    - Fixes: frontend files (src/components/*.tsx) written to D:\Orb\src\ instead of D:\orb-desktop\src\

v2.1 (2026-02-06): Proper separation of concerns.
    - Removed all direct sandbox writes (was violating Overwatcher read-only rule)
    - Delegates writes to Implementer via run_implementer_task()
    - Overwatcher reads from sandbox only for independent verification
    - Three-strike error handling per file task
    - Implementer LLM (Sonnet) generates code, Implementer module writes it

v2.0 (2026-02-06): Full implementation (SUPERSEDED — had write violations)
v1.0 (2026-02-06): Stub (returned success without doing anything)

BUILD_ID: 2026-02-06-v2.1-proper-separation

SAFETY INVARIANT:
    - This module is READ-ONLY for sandbox access
    - All writes go through implementer.run_implementer_task()
    - Architecture document is read-only context
    - Each file operation verified after write via independent read
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .spec_resolution import ResolvedSpec
from .sandbox_client import (
    SandboxClient,
    get_sandbox_client,
)

logger = logging.getLogger(__name__)

ARCHITECTURE_EXECUTOR_BUILD_ID = "2026-02-07-v2.6-auto-init-py"
print(f"[ARCHITECTURE_EXECUTOR_LOADED] BUILD_ID={ARCHITECTURE_EXECUTOR_BUILD_ID}")


# =============================================================================
# Constants
# =============================================================================

MAX_STRIKES_PER_TASK = 3   # Three-strike error handling
IMPLEMENTER_MAX_TOKENS = 16000
VERIFY_READ_TIMEOUT = 30


# =============================================================================
# Sandbox READ-ONLY Helpers (Overwatcher is allowed to read for verification)
# =============================================================================

def _verify_file_via_sandbox(client: SandboxClient, path: str, expected_min_chars: int = 10) -> Dict[str, Any]:
    """Read a file from sandbox for independent verification.
    
    Overwatcher is allowed to READ from sandbox — this is verification only.
    """
    try:
        cmd = f'Get-Content -Path "{path}" -Raw -Encoding UTF8'
        result = client.shell_run(cmd, timeout_seconds=VERIFY_READ_TIMEOUT)
        
        if result.stdout is not None:
            content = result.stdout
            return {
                "exists": True,
                "chars": len(content),
                "valid": len(content.strip()) >= expected_min_chars,
                "error": None,
            }
        
        return {
            "exists": False,
            "chars": 0,
            "valid": False,
            "error": f"Read returned None: {(result.stderr or '')[:100]}",
        }
    except Exception as e:
        return {
            "exists": False,
            "chars": 0,
            "valid": False,
            "error": str(e),
        }


# =============================================================================
# Architecture Document Parsing (unchanged from v2.0 — this is read-only)
# =============================================================================

def parse_file_inventory(architecture: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Parse the File Inventory section to extract new and modified files.
    
    Returns:
        Tuple of (new_files, modified_files)
        Each is a list of dicts with 'path' and 'description'
    """
    new_files: List[Dict[str, str]] = []
    modified_files: List[Dict[str, str]] = []
    
    # Look for "New Files" section
    new_section = re.search(
        r'(?:New Files|Files to Create)(.*?)(?=\n##|\n###.*Modified|\Z)',
        architecture, re.DOTALL | re.IGNORECASE,
    )
    if new_section:
        for match in re.finditer(
            r'\|\s*`([^`]+)`\s*\|\s*([^|]+)',
            new_section.group(1),
        ):
            path = match.group(1).strip()
            desc = match.group(2).strip()
            if path and not path.startswith('---') and not path.lower() == 'file':
                new_files.append({"path": path, "description": desc})
    
    # Look for "Modified Files" section  
    mod_section = re.search(
        r'(?:Modified Files|Files to Modify)(.*?)(?=\n## [^#]|\n---\s*$|\Z)',
        architecture, re.DOTALL | re.IGNORECASE,
    )
    if mod_section:
        for match in re.finditer(
            r'\|\s*`([^`]+)`\s*\|\s*([^|]+)',
            mod_section.group(1),
        ):
            path = match.group(1).strip()
            desc = match.group(2).strip()
            if path and not path.startswith('---') and not path.lower() == 'file':
                modified_files.append({"path": path, "description": desc})
    
    # Fallback: look for "New File:" headers
    if not new_files:
        for match in re.finditer(
            r'#+\s+[\d.]*\s*New File[:\s]+`([^`]+)`',
            architecture,
        ):
            path = match.group(1).strip()
            if path not in [f["path"] for f in new_files]:
                new_files.append({"path": path, "description": "From architecture design"})
    
    # Fallback: look for "Modifications to" headers
    if not modified_files:
        for match in re.finditer(
            r'#+\s+[\d.]*\s*Modifications? to\s+`([^`]+)`',
            architecture,
        ):
            path = match.group(1).strip()
            if path not in [f["path"] for f in modified_files]:
                modified_files.append({"path": path, "description": "From architecture design"})
    
    return new_files, modified_files


def extract_section_for_file(architecture: str, file_path: str) -> str:
    """Extract the architecture sections relevant to a specific file."""
    sections: List[str] = []
    lines = architecture.split('\n')
    
    filename = Path(file_path).name
    path_variants = [file_path, file_path.replace('/', '\\'), filename]
    
    in_relevant_section = False
    section_lines: List[str] = []
    section_depth = 0
    
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+', line)
        
        if header_match:
            depth = len(header_match.group(1))
            
            if in_relevant_section and section_lines:
                sections.append('\n'.join(section_lines))
                section_lines = []
            
            is_relevant = any(v in line for v in path_variants)
            
            if is_relevant:
                in_relevant_section = True
                section_depth = depth
                section_lines = [line]
            elif in_relevant_section and depth <= section_depth:
                in_relevant_section = False
                section_lines = []
            elif in_relevant_section:
                section_lines.append(line)
        elif in_relevant_section:
            section_lines.append(line)
    
    if in_relevant_section and section_lines:
        sections.append('\n'.join(section_lines))
    
    if sections:
        return '\n\n---\n\n'.join(sections)
    
    # Fallback: paragraphs mentioning the file
    paragraphs = architecture.split('\n\n')
    relevant = [p for p in paragraphs if any(v in p for v in path_variants)]
    if relevant:
        return '\n\n'.join(relevant[:5])
    
    return ""


# =============================================================================
# Implementer LLM Prompts
# =============================================================================

IMPLEMENTER_NEW_FILE_SYSTEM = """You are a code implementation agent. You receive an architecture specification for a single file and you generate the COMPLETE file content.

RULES:
1. Output ONLY the file content — no markdown fences, no explanations, no preamble.
2. The file must be complete and syntactically valid.
3. Follow the architecture specification exactly — use the same imports, class names, function signatures, and patterns described.
4. Include all code blocks from the specification, properly integrated.
5. Add appropriate docstrings and type hints as shown in the specification.
6. Do NOT add features not specified in the architecture.
7. If the architecture shows code blocks, use them as the implementation — they are the ground truth.
8. CROSS-FILE REFERENCES: If a "Files Already Created in This Job" section is provided, use the EXACT class names, method signatures, and import paths listed there. Do NOT invent alternative names or paths — these files already exist on disk.
9. COMPLETENESS: If the architecture specification or consuming code (in cross-file context) references factory functions (e.g. get_model_manager()), singleton accessors (e.g. TranscriptionService.get_instance()), or module-level convenience functions, you MUST implement them. Do NOT create only classes when the architecture or other files expect callable module-level functions. Every symbol that another file imports must actually exist.
"""

IMPLEMENTER_MODIFY_FILE_SYSTEM = """You are a code implementation agent. You receive an existing file and modification instructions from an architecture specification. You output the COMPLETE modified file.

RULES:
1. Output ONLY the complete modified file content — no markdown fences, no explanations.
2. Apply ALL modifications described in the architecture specification.
3. Preserve all existing code that is not explicitly being changed.
4. The output must be the COMPLETE file, not a diff or partial update.
5. Maintain existing code style, indentation, and patterns.
6. If the architecture shows specific code to add (imports, functions, etc.), include them exactly.
7. CROSS-FILE REFERENCES: If a "Files Already Created in This Job" section is provided, use the EXACT import paths and class/function names from those files. Do NOT hallucinate module paths — use only paths that match files listed in the job context or that already exist in the current file.
8. MODIFY SCOPE: For documentation files (README, CHANGELOG, etc.), ADD or UPDATE only the relevant section — do NOT rewrite the entire file. Preserve all existing content not related to the modification.
9. ROUTER REGISTRATION: When adding routers, check the existing file's pattern for how routers are registered (prefix on APIRouter vs prefix on include_router). Follow the established pattern.
10. IMPORT PATTERNS: When adding new imports, check the "Existing Imports" section if provided. Use the same module paths and import patterns as the file already uses. Do NOT invent new module paths — follow what already works in this file.
11. GITIGNORE SAFETY: For .gitignore modifications, be conservative. NEVER add broad glob patterns like *.json, *.md, *.txt, *.yaml, *.yml that would exclude tracked project files. Only add specific paths or narrow patterns (e.g. dist/, node_modules/, *.pyc).
12. API URL PATHS: When adding or modifying frontend API calls (fetch, axios, etc.), use the EXACT endpoint paths from the "Resolved API Endpoints" section in the cross-file context if provided. Do NOT invent URL prefixes — the resolved paths show the actual backend URLs including any router prefix.
"""


# =============================================================================
# Helpers
# =============================================================================

def _extract_llm_content(llm_result: Any) -> str:
    """Extract text content from an LLM response."""
    if isinstance(llm_result, str):
        return llm_result
    if hasattr(llm_result, 'content'):
        content = llm_result.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for block in content:
                if hasattr(block, 'text'):
                    texts.append(block.text)
                elif isinstance(block, dict) and 'text' in block:
                    texts.append(block['text'])
            return '\n'.join(texts)
    if isinstance(llm_result, dict):
        return llm_result.get('content', '') or llm_result.get('text', '')
    return str(llm_result)


def _strip_markdown_fences(content: str) -> str:
    """Strip markdown code fences if the LLM wrapped its output."""
    stripped = content.strip()
    if stripped.startswith('```'):
        first_newline = stripped.find('\n')
        if first_newline > 0:
            stripped = stripped[first_newline + 1:]
        if stripped.rstrip().endswith('```'):
            stripped = stripped.rstrip()[:-3].rstrip()
    return stripped


async def _resolve_sandbox_base(client: SandboxClient) -> str:
    """Determine the sandbox base path for the project (READ-ONLY check)."""
    candidates = [
        r"C:\Orb\Orb",
        r"C:\Orb",
        r"D:\Orb",
    ]
    
    for candidate in candidates:
        try:
            cmd = f'Test-Path -Path "{candidate}\\main.py"'
            result = client.shell_run(cmd, timeout_seconds=10)
            if result.stdout and "True" in result.stdout:
                logger.info("[arch_exec] Found project at %s", candidate)
                return candidate
        except Exception:
            continue
    
    logger.warning("[arch_exec] Could not resolve sandbox base — using D:\\Orb")
    return r"D:\Orb"


# v2.2: Multi-root path resolution for split backend/frontend projects
# Known project roots on this machine:
#   D:\Orb         — Backend (Python/FastAPI)
#   D:\orb-desktop — Frontend (Electron/React/TypeScript)
FRONTEND_PREFIX = "orb-desktop/"
FRONTEND_ROOT = r"D:\orb-desktop"

def _resolve_multi_root_path(rel_path: str, sandbox_base: str) -> str:
    """Resolve a relative path to its correct absolute path.
    
    v2.2: The project has two separate root directories:
    - Backend (D:\Orb): paths like app/routers/voice.py, main.py
    - Frontend (D:\orb-desktop): paths like orb-desktop/src/components/VoiceInput.tsx
    
    Architecture map and prompt both use orb-desktop/ prefix for frontend files.
    This function strips the prefix and resolves to the correct root.
    
    Args:
        rel_path: Relative path from architecture document
        sandbox_base: Resolved backend base (e.g. D:\Orb)
        
    Returns:
        Absolute path with correct root
    """
    normalized = rel_path.replace("\\", "/")
    
    if normalized.startswith(FRONTEND_PREFIX):
        # Strip the orb-desktop/ prefix and resolve against frontend root
        frontend_rel = normalized[len(FRONTEND_PREFIX):]
        abs_path = f"{FRONTEND_ROOT}\\{frontend_rel.replace('/', '\\')}"
        logger.info("[arch_exec] v2.2 Frontend path: %s -> %s", rel_path, abs_path)
        return abs_path
    else:
        # Backend path — resolve against sandbox_base as before
        abs_path = f"{sandbox_base}\\{normalized.replace('/', '\\')}"
        return abs_path


def _ensure_python_init_files(
    new_files: List[Dict[str, str]],
    modified_files: List[Dict[str, str]],
    sandbox_base: str,
    client: SandboxClient,
) -> List[Dict[str, str]]:
    """Auto-create __init__.py files for new Python package directories.
    
    v2.6: When the architecture creates Python files in new directories
    (e.g. app/services/transcription_service.py), those directories need
    __init__.py to be importable as Python packages. The architecture
    rarely includes these, and the Implementer doesn't know to create them.
    
    This function:
    1. Collects all directories that will contain new .py files
    2. For each directory, walks up to the project root checking for __init__.py
    3. Skips directories that already have __init__.py (in manifest or on disk)
    4. Returns a list of __init__.py file entries to prepend to new_files
    
    Only applies to backend Python paths (not orb-desktop/ frontend paths).
    
    Args:
        new_files: List of new file dicts from parse_file_inventory
        modified_files: List of modified file dicts (for manifest awareness)
        sandbox_base: Resolved backend root (e.g. D:\Orb)
        client: SandboxClient for checking existing files on disk
    
    Returns:
        List of __init__.py file dicts to prepend to new_files
    """
    # Collect all paths already in the manifest (new + modified)
    manifest_paths = set()
    for f in new_files:
        manifest_paths.add(f["path"].replace("\\", "/"))
    for f in modified_files:
        manifest_paths.add(f["path"].replace("\\", "/"))
    
    # Collect directories that need __init__.py checking
    dirs_needing_init: set = set()
    
    for f in new_files:
        rel_path = f["path"].replace("\\", "/")
        
        # Skip non-Python files
        if not rel_path.endswith(".py"):
            continue
        
        # Skip frontend paths
        if rel_path.startswith(FRONTEND_PREFIX):
            continue
        
        # Skip if this IS an __init__.py (already being created)
        if rel_path.endswith("__init__.py"):
            continue
        
        # Walk up directory tree from the file's parent to the project root
        parts = rel_path.split("/")
        for depth in range(1, len(parts)):  # depth=1 is immediate parent dir
            dir_path = "/".join(parts[:depth])
            init_path = f"{dir_path}/__init__.py"
            
            # Skip if __init__.py already in manifest
            if init_path in manifest_paths:
                continue
            
            # Skip top-level (no __init__.py needed at project root)
            if "/" not in dir_path:
                # e.g. "app" — this IS a package dir, check it
                # But if dir_path is just a filename component, skip
                pass
            
            dirs_needing_init.add(init_path)
    
    if not dirs_needing_init:
        return []
    
    # Check which of these __init__.py files already exist on disk
    init_files_to_create: List[Dict[str, str]] = []
    
    for init_path in sorted(dirs_needing_init):
        abs_path = f"{sandbox_base}\\{init_path.replace('/', '\\')}"
        
        # Check if file exists in sandbox
        try:
            cmd = f'Test-Path -Path "{abs_path}" -PathType Leaf'
            result = client.shell_run(cmd, timeout_seconds=10)
            if result.stdout and result.stdout.strip().lower() == "true":
                logger.info(
                    "[arch_exec] v2.6 __init__.py already exists: %s",
                    init_path,
                )
                continue
        except Exception as e:
            logger.warning(
                "[arch_exec] v2.6 Could not check %s: %s — will create anyway",
                init_path, e,
            )
        
        init_files_to_create.append({
            "path": init_path,
            "description": f"v2.6 auto-created: Python package init for {init_path.rsplit('/', 1)[0]}/",
        })
        logger.info(
            "[arch_exec] v2.6 Auto-creating __init__.py: %s",
            init_path,
        )
    
    if init_files_to_create:
        print(
            f"[ARCH_EXEC] v2.6 Auto-creating {len(init_files_to_create)} __init__.py file(s): "
            + ", ".join(f["path"] for f in init_files_to_create)
        )
    
    return init_files_to_create


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
# v2.3: Cross-File Context — Interface Extraction & Job Context
# =============================================================================

# Maximum chars of interface summary per file (keeps prompt size manageable)
INTERFACE_SUMMARY_MAX_CHARS = 1500


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
        prefix_match = re.search(r"Router prefix: '([^']*)'" , summary)
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
        sections.append(f"```")
        sections.append(summary)
        sections.append(f"```")
        sections.append("")
    
    # v2.5: Append resolved API endpoints
    endpoints_section = _build_resolved_endpoints(
        job_context, router_registrations or {}
    )
    if endpoints_section:
        sections.append(endpoints_section)
    
    return '\n'.join(sections)


# =============================================================================
# Main Executor
# =============================================================================

async def run_architecture_execution(
    *,
    spec: ResolvedSpec,
    architecture_content: str,
    architecture_path: str,
    job_id: str,
    llm_call_fn: Optional[Callable] = None,
    artifact_root: str = "D:/Orb/jobs",
) -> Dict[str, Any]:
    """Supervise architecture-level spec execution.
    
    The Overwatcher (this function) is the supervisor. It:
    1. Parses the architecture document to find file operations
    2. For each file, calls the Implementer LLM (Sonnet) to generate content
    3. Delegates each write to the Implementer via run_implementer_task()
    4. Reads back from sandbox to independently verify
    5. Implements three-strike error handling per task
    
    The Implementer LLM (Sonnet) generates the code.
    The Implementer module (implementer.py) writes it to the sandbox.
    The Overwatcher (this module) only reads for verification.
    """
    start_time = time.time()
    trace: List[Dict[str, Any]] = []
    artifacts_written: List[str] = []
    
    def elapsed_ms() -> int:
        return int((time.time() - start_time) * 1000)
    
    def add_trace(stage: str, status: str, details: Optional[Dict] = None):
        trace.append({
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        })
    
    add_trace("ARCHITECTURE_EXECUTION_START", "started", {
        "spec_id": spec.spec_id,
        "architecture_path": architecture_path,
        "architecture_chars": len(architecture_content),
        "job_id": job_id,
    })
    
    logger.info(
        "[arch_exec] v2.1 Starting architecture execution for spec %s (%d chars)",
        spec.spec_id, len(architecture_content),
    )
    print(f"[ARCH_EXEC] Starting: spec={spec.spec_id}, arch={len(architecture_content)} chars")
    
    # =========================================================================
    # Step 1: Parse file inventory
    # =========================================================================
    new_files, modified_files = parse_file_inventory(architecture_content)
    total_operations = len(new_files) + len(modified_files)
    
    logger.info("[arch_exec] Files: %d new, %d modified", len(new_files), len(modified_files))
    print(f"[ARCH_EXEC] Files: {len(new_files)} new, {len(modified_files)} modified")
    
    add_trace("ARCHITECTURE_PARSE", "success", {
        "new_files": [f["path"] for f in new_files],
        "modified_files": [f["path"] for f in modified_files],
        "total_operations": total_operations,
    })
    
    if total_operations == 0:
        error_msg = "No file operations found in architecture document."
        add_trace("ARCHITECTURE_PARSE", "failed", {"error": error_msg})
        return {"success": False, "decision": "FAIL", "error": error_msg, "trace": trace, "artifacts_written": []}
    
    # =========================================================================
    # Step 2: Validate prerequisites
    # =========================================================================
    if llm_call_fn is None:
        error_msg = "LLM function required for architecture execution"
        add_trace("ARCHITECTURE_EXECUTION", "failed", {"error": error_msg})
        return {"success": False, "decision": "FAIL", "error": error_msg, "trace": trace, "artifacts_written": []}
    
    # Get sandbox client (READ-ONLY for Overwatcher — verification only)
    client = get_sandbox_client()
    if not client.is_connected():
        error_msg = "SAFETY: Sandbox not available"
        add_trace("ARCHITECTURE_EXECUTION", "failed", {"error": error_msg})
        return {"success": False, "decision": "FAIL", "error": error_msg, "trace": trace, "artifacts_written": []}
    
    add_trace("SANDBOX_CONNECTED", "success")
    
    # Get Implementer LLM config
    try:
        from app.llm.stage_models import get_implementer_config
        impl_config = get_implementer_config()
        impl_provider = impl_config.provider
        impl_model = impl_config.model
        impl_max_tokens = min(impl_config.max_output_tokens or IMPLEMENTER_MAX_TOKENS, IMPLEMENTER_MAX_TOKENS)
    except Exception as e:
        logger.warning("[arch_exec] Could not load implementer config: %s — using defaults", e)
        impl_provider = "anthropic"
        impl_model = "claude-sonnet-4-5-20250929"
        impl_max_tokens = IMPLEMENTER_MAX_TOKENS
    
    # =========================================================================
    # Step 3: Resolve sandbox base path (READ-ONLY check)
    # =========================================================================
    sandbox_base = await _resolve_sandbox_base(client)
    logger.info("[arch_exec] Sandbox base: %s", sandbox_base)
    add_trace("SANDBOX_BASE_RESOLVED", "success", {"base_path": sandbox_base})
    
    # =========================================================================
    # Step 3b: v2.6 Auto-create __init__.py for new Python packages
    # =========================================================================
    try:
        init_files = _ensure_python_init_files(
            new_files, modified_files, sandbox_base, client
        )
        if init_files:
            # Prepend to new_files so they're created BEFORE the files that need them
            new_files = init_files + new_files
            total_operations = len(new_files) + len(modified_files)
            add_trace("AUTO_INIT_PY", "success", {
                "init_files_added": [f["path"] for f in init_files],
                "new_total_operations": total_operations,
            })
            logger.info(
                "[arch_exec] v2.6 Added %d __init__.py files, total ops now %d",
                len(init_files), total_operations,
            )
    except Exception as e:
        # Non-fatal — continue without auto-init if it fails
        logger.warning("[arch_exec] v2.6 _ensure_python_init_files failed: %s", e)
        add_trace("AUTO_INIT_PY", "failed", {"error": str(e)})
    
    # =========================================================================
    # Step 4: Process all file tasks
    # =========================================================================
    # Import the Implementer's atomic task interface
    from .implementer import run_implementer_task
    
    files_created = 0
    files_modified_count = 0
    files_failed = 0
    
    all_tasks = (
        [{"info": f, "action": "create"} for f in new_files] +
        [{"info": f, "action": "modify"} for f in modified_files]
    )
    
    # v2.3: Cross-file context accumulator
    # After each successful file operation, we capture key interfaces
    # and inject them as context for subsequent Implementer calls.
    job_context: Dict[str, str] = {}
    # v2.5: Router registration prefix tracker (from main.py include_router calls)
    router_registrations: Dict[str, str] = {}
    # v2.5: Track file contents for two-pass re-extraction
    created_file_contents: Dict[str, str] = {}  # rel_path -> content
    
    # v2.5: Identify boundary between CREATE and MODIFY tasks for two-pass
    create_count = len(new_files)
    
    for i, task in enumerate(all_tasks, 1):
        file_info = task["info"]
        action = task["action"]
        rel_path = file_info["path"]
        # v2.2: Multi-root path resolution
        # Frontend files (orb-desktop/ prefix) resolve to D:\orb-desktop
        # Backend files resolve to sandbox_base (D:\Orb)
        abs_path = _resolve_multi_root_path(rel_path, sandbox_base)
        
        logger.info("[arch_exec] [%d/%d] %s: %s", i, total_operations, action.upper(), rel_path)
        print(f"[ARCH_EXEC] [{i}/{total_operations}] {action.upper()}: {rel_path}")
        
        add_trace("FILE_TASK_START", "processing", {
            "operation": action,
            "relative_path": rel_path,
            "absolute_path": abs_path,
            "task_number": i,
        })
        
        # =====================================================================
        # Three-strike error loop
        # =====================================================================
        task_success = False
        last_error = None
        
        for strike in range(1, MAX_STRIKES_PER_TASK + 1):
            logger.info("[arch_exec] %s strike %d/%d", rel_path, strike, MAX_STRIKES_PER_TASK)
            
            # --- v2.6: Skip LLM for auto-generated __init__.py files ---
            if rel_path.endswith("__init__.py") and file_info.get("description", "").startswith("v2.6 auto-created"):
                file_content = "# Auto-generated by architecture executor v2.6\n"
                logger.info("[arch_exec] v2.6 Direct-writing __init__.py: %s", rel_path)
                
                try:
                    impl_result = await run_implementer_task(
                        path=abs_path,
                        content=file_content,
                        action="create",
                        ensure_parents=True,
                        client=client,
                    )
                    if impl_result.success:
                        task_success = True
                    else:
                        last_error = f"Implementer write failed for __init__.py: {impl_result.error}"
                except Exception as e:
                    last_error = f"__init__.py write exception: {e}"
                break  # No retries needed for __init__.py
            
            # --- Generate content via Implementer LLM ---
            try:
                file_context = extract_section_for_file(architecture_content, rel_path)
                
                if not file_context:
                    last_error = f"No architecture context found for {rel_path}"
                    logger.warning("[arch_exec] %s", last_error)
                    break  # No point retrying — architecture doesn't mention this file
                
                # v2.3/v2.5: Build cross-file context section (with resolved endpoints)
                job_context_section = _format_job_context(job_context, router_registrations)
                
                if action == "create":
                    user_prompt = (
                        f"Generate the complete content for a new file: `{rel_path}`\n\n"
                        f"## Architecture Specification\n\n{file_context}\n\n"
                    )
                    if job_context_section:
                        user_prompt += f"{job_context_section}\n\n"
                    user_prompt += "Output ONLY the file content. No markdown fences, no explanations."
                    system_prompt = IMPLEMENTER_NEW_FILE_SYSTEM
                else:
                    # Modify: read existing file first (Overwatcher is allowed to read)
                    existing_content = await _read_existing_file(client, abs_path)
                    if existing_content is None:
                        last_error = f"Cannot read existing file for modification: {abs_path}"
                        logger.error("[arch_exec] %s", last_error)
                        break  # File doesn't exist — can't modify
                    
                    user_prompt = (
                        f"Apply the following modifications to `{rel_path}`.\n\n"
                        f"## Current File Content\n```\n{existing_content}\n```\n\n"
                    )
                    
                    # v2.4: Extract and inject existing import patterns
                    existing_imports = _extract_existing_imports(existing_content, rel_path)
                    if existing_imports:
                        user_prompt += (
                            f"## Existing Imports\n"
                            f"The file currently uses these imports. Follow the same "
                            f"import patterns and module paths for any new imports you add.\n"
                            f"```\n{existing_imports}\n```\n\n"
                        )
                    
                    user_prompt += f"## Modification Instructions\n\n{file_context}\n\n"
                    if job_context_section:
                        user_prompt += f"{job_context_section}\n\n"
                    user_prompt += "Output the COMPLETE modified file. No markdown fences."
                    system_prompt = IMPLEMENTER_MODIFY_FILE_SYSTEM
                
                # Add error context for retry strikes
                if strike > 1 and last_error:
                    user_prompt += (
                        f"\n\n## Previous Attempt Failed\n"
                        f"Error from previous attempt: {last_error}\n"
                        f"Please fix the issue and try again."
                    )
                
                llm_result = await llm_call_fn(
                    provider_id=impl_provider,
                    model_id=impl_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=impl_max_tokens,
                )
                
                file_content = _extract_llm_content(llm_result)
                file_content = _strip_markdown_fences(file_content)
                
                if not file_content or len(file_content.strip()) < 10:
                    last_error = "LLM returned empty/minimal content"
                    logger.warning("[arch_exec] Strike %d: %s for %s", strike, last_error, rel_path)
                    add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                        "path": rel_path, "error": last_error,
                    })
                    continue
                
                logger.info(
                    "[arch_exec] LLM generated %d chars for %s (strike %d)",
                    len(file_content), rel_path, strike,
                )
                
            except Exception as e:
                last_error = f"LLM call failed: {e}"
                logger.exception("[arch_exec] Strike %d: %s", strike, last_error)
                add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                    "path": rel_path, "error": last_error,
                })
                continue
            
            # --- Delegate write to Implementer ---
            try:
                impl_result = await run_implementer_task(
                    path=abs_path,
                    content=file_content,
                    action=action,
                    ensure_parents=True,
                    client=client,
                )
                
                if not impl_result.success:
                    last_error = f"Implementer write failed: {impl_result.error}"
                    logger.warning("[arch_exec] Strike %d: %s", strike, last_error)
                    add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                        "path": rel_path, "error": last_error,
                    })
                    continue
                
                logger.info(
                    "[arch_exec] Implementer wrote %s: %d chars, verified=%s",
                    rel_path, impl_result.chars_written, impl_result.verified,
                )
                
            except Exception as e:
                last_error = f"Implementer exception: {e}"
                logger.exception("[arch_exec] Strike %d: %s", strike, last_error)
                add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                    "path": rel_path, "error": last_error,
                })
                continue
            
            # --- Independent verification (Overwatcher reads to verify) ---
            verify = _verify_file_via_sandbox(client, abs_path, expected_min_chars=10)
            
            if not verify["valid"]:
                last_error = f"Overwatcher verification failed: {verify['error'] or 'file too short/missing'}"
                logger.warning("[arch_exec] Strike %d: %s", strike, last_error)
                add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {
                    "path": rel_path, "error": last_error,
                })
                continue
            
            # SUCCESS — all three checks passed
            task_success = True
            break
        
        # --- Record task result ---
        if task_success:
            if action == "create":
                files_created += 1
                # v2.5: Store content for two-pass re-extraction
                created_file_contents[rel_path] = file_content
            else:
                files_modified_count += 1
                # v2.5: Capture router registrations from modified files (e.g. main.py)
                if rel_path.endswith('.py'):
                    try:
                        regs = _extract_router_registrations(file_content)
                        if regs:
                            router_registrations.update(regs)
                            logger.info(
                                "[arch_exec] v2.5 Captured router registrations from %s: %s",
                                rel_path, regs,
                            )
                    except Exception as e:
                        logger.warning("[arch_exec] v2.5 Router registration extraction failed: %s", e)
            artifacts_written.append(abs_path)
            
            # v2.3: Capture interfaces for cross-file context
            try:
                interface_summary = _extract_file_interfaces(rel_path, file_content)
                job_context[rel_path] = interface_summary
                logger.info(
                    "[arch_exec] v2.3 Captured interfaces for %s (%d chars)",
                    rel_path, len(interface_summary),
                )
            except Exception as e:
                # Non-fatal — we still succeeded, just couldn't extract interfaces
                logger.warning(
                    "[arch_exec] v2.3 Interface extraction failed for %s: %s",
                    rel_path, e,
                )
            
            logger.info("[arch_exec] ✓ %s %s", action.upper(), rel_path)
            print(f"[ARCH_EXEC] ✓ {action.upper()} {rel_path}")
            
            add_trace("FILE_TASK_SUCCESS", action, {
                "path": rel_path,
                "absolute_path": abs_path,
                "job_context_files": list(job_context.keys()),  # v2.3
            })
        
        # =====================================================================
        # v2.5: Two-pass boundary — after all CREATEs, re-extract interfaces
        # This ensures MODIFY tasks get the FULL cross-file context from all
        # created files, not just the ones that happened to be created earlier.
        # =====================================================================
        if i == create_count and created_file_contents:
            logger.info(
                "[arch_exec] v2.5 Two-pass: re-extracting interfaces from %d created files",
                len(created_file_contents),
            )
            print(f"[ARCH_EXEC] v2.5 Two-pass: refreshing context from {len(created_file_contents)} created files")
            for created_path, created_content in created_file_contents.items():
                try:
                    refreshed = _extract_file_interfaces(created_path, created_content)
                    job_context[created_path] = refreshed
                except Exception as e:
                    logger.warning("[arch_exec] v2.5 Two-pass extraction failed for %s: %s", created_path, e)
            add_trace("TWO_PASS_CONTEXT_REFRESH", "success", {
                "files_refreshed": list(created_file_contents.keys()),
            })
        else:
            files_failed += 1
            logger.error(
                "[arch_exec] ✗ %s %s FAILED after %d strikes: %s",
                action.upper(), rel_path, MAX_STRIKES_PER_TASK, last_error,
            )
            print(f"[ARCH_EXEC] ✗ {action.upper()} {rel_path} FAILED: {last_error}")
            
            add_trace("FILE_TASK_FAILED", "exhausted_strikes", {
                "path": rel_path,
                "strikes": MAX_STRIKES_PER_TASK,
                "last_error": last_error,
            })
    
    # =========================================================================
    # Step 5: Summary
    # =========================================================================
    total_succeeded = files_created + files_modified_count
    success = total_succeeded > 0 and files_failed == 0
    
    summary = {
        "total_operations": total_operations,
        "files_created": files_created,
        "files_modified": files_modified_count,
        "files_failed": files_failed,
        "total_succeeded": total_succeeded,
        "elapsed_ms": elapsed_ms(),
    }
    
    if success:
        status_label = "✓ SUCCESS"
    elif total_succeeded > 0:
        status_label = f"⚠ PARTIAL ({total_succeeded}/{total_operations})"
    else:
        status_label = "✗ FAILED"
    
    logger.info(
        "[arch_exec] %s: %d created, %d modified, %d failed (%dms)",
        status_label, files_created, files_modified_count, files_failed, elapsed_ms(),
    )
    print(
        f"[ARCH_EXEC] {status_label}: "
        f"{files_created} created, {files_modified_count} modified, "
        f"{files_failed} failed ({elapsed_ms()}ms)"
    )
    
    add_trace(
        "ARCHITECTURE_EXECUTION_COMPLETE",
        "success" if success else "partial" if total_succeeded > 0 else "failed",
        summary,
    )
    
    error_msg = None
    if not success:
        if total_succeeded == 0:
            error_msg = f"Architecture execution failed: 0/{total_operations} file operations succeeded"
        else:
            error_msg = (
                f"Architecture execution partial: {total_succeeded}/{total_operations} "
                f"succeeded, {files_failed} failed"
            )
    
    return {
        "success": success,
        "decision": "PASS" if success else "FAIL",
        "error": error_msg,
        "trace": trace,
        "artifacts_written": artifacts_written,
        "summary": summary,
    }


__all__ = [
    "run_architecture_execution",
    "parse_file_inventory",
    "extract_section_for_file",
    "_extract_file_interfaces",
    "_extract_existing_imports",
    "_extract_router_registrations",
    "_build_resolved_endpoints",
    "_format_job_context",
    "_ensure_python_init_files",
    "ARCHITECTURE_EXECUTOR_BUILD_ID",
]

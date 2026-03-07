# FILE: app/agentic_pipeline/preflight_evidence.py
"""
Pre-Flight Evidence Gatherer for the Agentic Loop.

Runs before the agentic loop starts. For each file in scope:
  - sandbox_isfile() → classify as CREATE or MODIFY
  - sandbox_read_text() → AST parse → exports, imports, function sigs
  - Produce structured facts for the loop's input context

Returns both structured data (for the batch sizer and check runner)
and formatted text (for injection into the LLM context window).

All sandbox reads go through app.sandbox_fs. No host fallbacks.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileExport:
    """A single exported symbol from a file."""
    name: str
    kind: str  # "function", "class", "const", "interface", "type", "enum"
    is_default: bool = False


@dataclass
class FileFacts:
    """Pre-flight facts about a single file."""
    rel_path: str
    abs_path: Optional[str] = None
    exists: bool = False
    size_bytes: int = 0
    line_count: int = 0
    action: str = "CREATE"  # "CREATE" or "MODIFY"
    exports: List[FileExport] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    content: Optional[str] = None  # Full content for MODIFY files
    error: Optional[str] = None

    @property
    def export_names(self) -> List[str]:
        return [e.name for e in self.exports]


@dataclass
class PreflightResult:
    """Complete pre-flight evidence for all files in scope."""
    file_facts: Dict[str, FileFacts] = field(default_factory=dict)
    modify_count: int = 0
    create_count: int = 0
    total_files: int = 0
    sandbox_reachable: bool = True
    error: Optional[str] = None

    def get_facts(self, rel_path: str) -> Optional[FileFacts]:
        norm = rel_path.replace("\\", "/")
        return self.file_facts.get(norm)

    def get_modify_files(self) -> List[FileFacts]:
        return [f for f in self.file_facts.values() if f.action == "MODIFY"]

    def get_create_files(self) -> List[FileFacts]:
        return [f for f in self.file_facts.values() if f.action == "CREATE"]

    def get_export_map(self) -> Dict[str, List[str]]:
        """Map of rel_path -> list of export names, for cross-segment checks."""
        return {
            path: facts.export_names
            for path, facts in self.file_facts.items()
            if facts.exports
        }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOTS = [r"D:\Orb", r"D:\orb-desktop"]


def _resolve_to_absolute(
    rel_path: str,
    project_roots: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve a relative path to an absolute path for sandbox lookup."""
    if len(rel_path) > 2 and rel_path[1] == ":":
        return rel_path

    roots = project_roots or _PROJECT_ROOTS
    norm = rel_path.replace("/", os.sep).replace("\\", os.sep)

    if norm.startswith("orb-desktop" + os.sep):
        stripped = norm[len("orb-desktop" + os.sep):]
        return os.path.join(r"D:\orb-desktop", stripped)

    if norm.startswith("src" + os.sep):
        return os.path.join(r"D:\orb-desktop", norm)

    if norm.startswith("app" + os.sep):
        return os.path.join(r"D:\Orb", norm)

    # Default to first root (D:\Orb) — sandbox_isfile will verify existence
    if roots:
        return os.path.join(roots[0], norm)

    return None


# ---------------------------------------------------------------------------
# Export extraction (Python)
# ---------------------------------------------------------------------------

def _extract_python_exports(content: str) -> List[FileExport]:
    """Extract public function/class names from Python source via AST."""
    exports: List[FileExport] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return exports

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                exports.append(FileExport(name=node.name, kind="function"))
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                exports.append(FileExport(name=node.name, kind="class"))
    return exports


def _extract_python_imports(content: str) -> List[str]:
    """Extract import targets from Python source via AST."""
    imports: List[str] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


# ---------------------------------------------------------------------------
# Export extraction (TypeScript / JavaScript)
# ---------------------------------------------------------------------------

_TS_EXPORT_RE = re.compile(
    r"export\s+(?:(default)\s+)?"
    r"(?:function|const|let|var|class|interface|type|enum)\s+"
    r"(\w+)",
)

_TS_IMPORT_RE = re.compile(
    r"import\s+(?:.*?)\s+from\s+['\"]([^'\"]+)['\"]",
)


def _extract_ts_exports(content: str) -> List[FileExport]:
    """Extract exported symbols from TypeScript/JS source via regex."""
    exports: List[FileExport] = []
    for m in _TS_EXPORT_RE.finditer(content):
        is_default = m.group(1) is not None
        name = m.group(2)
        line = content[m.start():m.end()]
        kind = "const"
        for kw in ("function", "class", "interface", "type", "enum"):
            if kw in line:
                kind = kw
                break
        exports.append(FileExport(name=name, kind=kind, is_default=is_default))
    return exports


def _extract_ts_imports(content: str) -> List[str]:
    """Extract import source paths from TypeScript/JS source."""
    return [m.group(1) for m in _TS_IMPORT_RE.finditer(content)]


# ---------------------------------------------------------------------------
# Main gathering function
# ---------------------------------------------------------------------------

def gather_preflight_evidence(
    file_scope: List[str],
    project_roots: Optional[List[str]] = None,
) -> PreflightResult:
    """Gather pre-flight facts for all files in scope from the sandbox."""
    try:
        from app.sandbox_fs import sandbox_isfile, sandbox_read_text
    except ImportError:
        logger.error("[preflight_evidence] sandbox_fs not available")
        return PreflightResult(sandbox_reachable=False, error="sandbox_fs not available")

    result = PreflightResult(total_files=len(file_scope))
    roots = project_roots or _PROJECT_ROOTS

    for rel_path in file_scope:
        norm_path = rel_path.replace("\\", "/")
        abs_path = _resolve_to_absolute(rel_path, roots)
        facts = FileFacts(rel_path=norm_path, abs_path=abs_path)

        if not abs_path:
            facts.action = "CREATE"
            facts.error = "could not resolve path"
            result.file_facts[norm_path] = facts
            result.create_count += 1
            continue

        try:
            exists = sandbox_isfile(abs_path)
        except Exception as e:
            facts.error = f"sandbox check failed: {e}"
            result.file_facts[norm_path] = facts
            result.create_count += 1
            continue

        if not exists:
            facts.action = "CREATE"
            result.file_facts[norm_path] = facts
            result.create_count += 1
            continue

        facts.exists = True
        facts.action = "MODIFY"
        result.modify_count += 1

        content = sandbox_read_text(abs_path)
        if content is None:
            facts.error = "exists but unreadable"
            result.file_facts[norm_path] = facts
            continue

        facts.content = content
        facts.size_bytes = len(content.encode("utf-8", errors="replace"))
        facts.line_count = content.count("\n") + 1

        ext = os.path.splitext(abs_path)[1].lower()
        if ext == ".py":
            facts.exports = _extract_python_exports(content)
            facts.imports = _extract_python_imports(content)
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            facts.exports = _extract_ts_exports(content)
            facts.imports = _extract_ts_imports(content)

        result.file_facts[norm_path] = facts

    logger.info(
        "[preflight_evidence] Gathered facts: %d MODIFY, %d CREATE, %d total",
        result.modify_count, result.create_count, result.total_files,
    )
    return result


# ---------------------------------------------------------------------------
# Formatting for context injection
# ---------------------------------------------------------------------------

def format_preflight_for_prompt(result: PreflightResult) -> str:
    """Format pre-flight evidence as markdown for LLM prompt injection."""
    if not result.file_facts:
        return ""

    lines = [
        "## PRE-FLIGHT FILE FACTS (Deterministic — from sandbox)",
        "",
        "These facts are ground truth. Do NOT contradict them.",
        "",
    ]

    modify_files = result.get_modify_files()
    if modify_files:
        lines.append("### Existing Files (MODIFY only)")
        for f in modify_files:
            size_kb = round(f.size_bytes / 1024, 1)
            exports_str = ", ".join(f.export_names[:12]) if f.exports else "(none)"
            if len(f.exports) > 12:
                exports_str += f" +{len(f.exports) - 12} more"
            lines.append(f"- **`{f.rel_path}`** — {size_kb}KB, {f.line_count} lines")
            lines.append(f"  Exports: {exports_str}")
        lines.append("")

    create_files = result.get_create_files()
    if create_files:
        lines.append("### New Files (CREATE)")
        for f in create_files:
            lines.append(f"- **`{f.rel_path}`** — does not exist yet")
        lines.append("")

    lines.append(
        f"**Summary**: {result.modify_count} MODIFY, "
        f"{result.create_count} CREATE, {result.total_files} total"
    )
    lines.append("")
    return "\n".join(lines)

from __future__ import annotations
import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DETERMINISTIC_CHECKER_BUILD_ID = "2026-02-21-v3.0-facade-gate-and-divergence-check"

@dataclass
class DetCheckIssue:
    severity: str           # "blocking" or "warning"
    category: str           # e.g. "missing_export", "syntax_error", "import_error"
    description: str
    line_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "line_hint": self.line_hint,
        }

@dataclass
class DetCheckResult:
    passed: bool = True
    issues: List[DetCheckIssue] = field(default_factory=list)
    reasoning: str = ""
    skipped: bool = False
    skip_reason: str = ""

    @property
    def blocking_issues(self) -> List[DetCheckIssue]:
        return [i for i in self.issues if i.severity == "blocking"]

    @property
    def warning_issues(self) -> List[DetCheckIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "blocking_count": len(self.blocking_issues),
            "warning_count": len(self.warning_issues),
            "reasoning": self.reasoning,
            "skipped": self.skipped,
        }

def extract_required_exports(
    interface_contract: str,
    file_path: str,
) -> List[str]:
    """
    Extract symbol names the contract says this file MUST export.

    Parses the 'MUST DEFINE AND EXPORT' section of skeleton contract markdown.
    Uses the same multi-occurrence scanning approach as signature_checker v1.3.

    Returns list of bare names (not full signatures).
    """
    file_path_norm = file_path.replace("\\", "/").strip()
    required: List[str] = []

    lines = interface_contract.split("\n")
    in_file_section = False
    in_exports = False

    for line in lines:
        stripped = line.strip()
        stripped_norm = stripped.replace("\\", "/")

        # Detect file path reference (backtick-wrapped)
        if f"`{file_path_norm}`" in stripped_norm:
            in_file_section = True
            in_exports = False
            continue

        if in_file_section:
            # Detect MUST EXPORT header (handles MUST DEFINE AND EXPORT too)
            if "MUST" in stripped and "EXPORT" in stripped:
                in_exports = True
                continue

            # Section boundary — stop
            if stripped.startswith("###") or stripped.startswith("## "):
                # v1.3: Don't give up — keep scanning for more occurrences
                in_file_section = False
                in_exports = False
                continue

            # New file entry — check if it's a DIFFERENT file
            if stripped.startswith("- `") and "`" in stripped[3:]:
                match = re.match(r'^-\s*`([^`]+)`', stripped)
                if match:
                    candidate = match.group(1).strip().replace("\\", "/")
                    # Is this a file path (not a signature)?
                    is_file = ("/" in candidate or candidate.endswith(".py"))
                    is_sig = candidate.startswith("def ") or candidate.startswith("async def ")
                    if is_file and not is_sig and candidate != file_path_norm:
                        in_file_section = False
                        in_exports = False
                        continue

            # Collect export names from indented bullet items
            if in_exports and stripped.startswith("- `"):
                match = re.match(r'^-\s*`([^`]+)`', stripped)
                if match:
                    symbol = match.group(1).strip()
                    # Extract just the function name from full signature
                    if symbol.startswith("def ") or symbol.startswith("async def "):
                        name_match = re.match(r'(?:async\s+)?def\s+(\w+)\s*\(', symbol)
                        if name_match:
                            name = name_match.group(1)
                            if name not in required:
                                required.append(name)
                    else:
                        # Bare name (e.g. run_segmented_job)
                        if re.match(r'^\w+$', symbol) and symbol not in required:
                            required.append(symbol)

    return required

def extract_expected_exports_from_arch(
    architecture_section: str,
) -> List[str]:
    """
    Extract the expected exports from an architecture file section.

    Looks for:
    1. An `__all__` block in the architecture's code blocks
    2. A `### Re-exports` section with `from .module import (...)` blocks
    3. A `### Exports` section listing symbol names

    Returns a list of symbol names that this file is expected to define or re-export.
    """
    if not architecture_section:
        return []

    exports: List[str] = []

    # Strategy 1: Parse __all__ from code blocks in the architecture
    all_match = re.search(
        r'__all__\s*=\s*\[([^\]]+)\]',
        architecture_section,
        re.DOTALL,
    )
    if all_match:
        content = all_match.group(1)
        for name_match in re.finditer(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']', content):
            exports.append(name_match.group(1))
        if exports:
            return exports

    # Strategy 2: Parse function/class/constant headers from architecture
    # Matches: #### `symbol_name` (function, ...) or ### `symbol_name` (...)
    for match in re.finditer(
        r'^#{3,4}\s+`([a-zA-Z_][a-zA-Z0-9_]*)`\s+\(',
        architecture_section,
        re.MULTILINE,
    ):
        exports.append(match.group(1))

    return exports

def extract_segment_interface(
    file_path: str,
    file_content: str,
) -> Dict[str, Any]:
    """
    Deterministic extraction of a file's public interface using AST.

    Returns a structured dict with:
    - exports: list of exported symbol names
    - functions: dict of func_name -> {async, params, return_type, line}
    - classes: dict of class_name -> {bases, methods, line}
    - type_aliases: dict of name -> annotation_str
    - imports_from: list of {module, names}

    This is injected as hard evidence into seg-06's prompt so the LLM
    has zero room to hallucinate sibling interfaces.
    """
    result: Dict[str, Any] = {
        "file_path": file_path,
        "exports": [],
        "functions": {},
        "classes": {},
        "type_aliases": {},
        "imports_from": [],
    }

    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        result["error"] = "SyntaxError — cannot parse"
        return result

    # __all__
    dunder_all = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        dunder_all = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                dunder_all.append(elt.value)

    # Top-level definitions
    all_names = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_names.append(node.name)
            func_info = {
                "async": isinstance(node, ast.AsyncFunctionDef),
                "line": node.lineno,
                "params": [],
                "return_type": _unparse_safe(node.returns),
            }
            for arg in node.args.args:
                param = {"name": arg.arg, "type": _unparse_safe(arg.annotation)}
                func_info["params"].append(param)
            # Keyword-only args
            for arg in node.args.kwonlyargs:
                param = {"name": arg.arg, "type": _unparse_safe(arg.annotation), "keyword_only": True}
                func_info["params"].append(param)
            result["functions"][node.name] = func_info

        elif isinstance(node, ast.ClassDef):
            all_names.append(node.name)
            cls_info = {
                "line": node.lineno,
                "bases": [_unparse_safe(b) for b in node.bases],
                "methods": [],
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cls_info["methods"].append(item.name)
            result["classes"][node.name] = cls_info

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    all_names.append(target.id)
                    # Check if it's a type alias (assigned from typing construct)
                    val_str = _unparse_safe(node.value)
                    if val_str and ("Optional" in val_str or "Callable" in val_str
                                    or "List" in val_str or "Dict" in val_str
                                    or "Union" in val_str or "Tuple" in val_str):
                        result["type_aliases"][target.id] = val_str

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            all_names.append(node.target.id)

    # Imports from siblings
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level and node.level > 0:
                names = [alias.name for alias in (node.names or [])]
                result["imports_from"].append({
                    "module": "." * node.level + node.module,
                    "names": names,
                })

    result["exports"] = dunder_all if dunder_all is not None else all_names
    return result

def format_segment_interfaces(
    interfaces: List[Dict[str, Any]],
) -> str:
    """
    Format extracted interfaces into evidence text for injection into prompts.

    Produces a structured, unambiguous representation that leaves no room
    for LLM guessing about sibling module contents.
    """
    lines = []
    lines.append("## Deterministic Sibling Interface Evidence (GROUND TRUTH)")
    lines.append("Extracted by AST from actual implemented files on disk.")
    lines.append("DO NOT invent, guess, or assume any interface not listed here.")
    lines.append("")

    for iface in interfaces:
        fp = iface.get("file_path", "?")
        lines.append(f"### {fp}")

        if iface.get("error"):
            lines.append(f"  ERROR: {iface['error']}")
            lines.append("")
            continue

        exports = iface.get("exports", [])
        if exports:
            lines.append(f"  EXPORTS: {', '.join(exports)}")

        for fname, finfo in iface.get("functions", {}).items():
            prefix = "async " if finfo.get("async") else ""
            params = []
            for p in finfo.get("params", []):
                pstr = p["name"]
                if p.get("type"):
                    pstr += f": {p['type']}"
                params.append(pstr)
            ret = f" -> {finfo['return_type']}" if finfo.get("return_type") else ""
            lines.append(f"  {prefix}def {fname}({', '.join(params)}){ret}")

        for cname, cinfo in iface.get("classes", {}).items():
            bases = f"({', '.join(cinfo['bases'])})" if cinfo.get("bases") else ""
            lines.append(f"  class {cname}{bases}: methods={cinfo.get('methods', [])}")

        for tname, tval in iface.get("type_aliases", {}).items():
            lines.append(f"  {tname} = {tval}")

        if iface.get("imports_from"):
            lines.append("  IMPORTS:")
            for imp in iface["imports_from"]:
                lines.append(f"    from {imp['module']} import {', '.join(imp['names'])}")

        lines.append("")

    return "\n".join(lines)

def _unparse_safe(node) -> Optional[str]:
    """Safely unparse an AST node to string. Returns None if not possible."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "deterministic_check": "_deterministic_checker_utils_3",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.overwatcher.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

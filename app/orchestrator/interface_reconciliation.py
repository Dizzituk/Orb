# FILE: app/orchestrator/interface_reconciliation.py
"""
Interface Reconciliation — Option A: Prevent naming drift.

When a segment depends on other segments, this module reads the ACTUAL
output files from those completed segments off the sandbox, extracts
the real interfaces (function names, class names, exports), and produces
a reconciliation block that gets injected into the architecture content
BEFORE the Implementer writes code.

This prevents the #1 cause of cross-segment failures: the Implementer
guessing function names from the spec when the actual completed files
used different names.

Example:
    Spec says: "source context extraction module"
    Implementer guesses: `from source_context import extract_source_files`
    Actual file exports: `_detect_source_files_from_architecture`

    With reconciliation, the Implementer gets:
    ```
    ## DEPENDENCY REALITY — Actual interfaces from completed segments
    ### source_context.py (from seg-08)
    - _detect_source_files_from_architecture(arch_content: str) -> List[str]
    - _read_source_context(source_files, sandbox_base) -> str
    ```

v1.0 (2026-02-15): Initial implementation
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

INTERFACE_RECONCILIATION_BUILD_ID = "2026-02-15-v1.0-initial"
print(f"[INTERFACE_RECONCILIATION_LOADED] BUILD_ID={INTERFACE_RECONCILIATION_BUILD_ID}")


def _extract_python_interfaces(file_content: str, file_path: str) -> Dict[str, Any]:
    """
    Extract public interface from a Python file using AST parsing.

    Returns:
        Dict with:
            - functions: list of {name, args, returns, is_async}
            - classes: list of {name, methods: [{name, args}]}
            - constants: list of names (ALL_CAPS variables)
            - exports: list of names from __all__ if defined
            - imports_from: list of {module, names} for relative imports
    """
    result = {
        "file_path": file_path,
        "functions": [],
        "classes": [],
        "constants": [],
        "exports": [],
        "imports_from": [],
    }

    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        logger.warning("[interface_recon] SyntaxError parsing %s: %s", file_path, e)
        return result

    for node in ast.iter_child_nodes(tree):
        # Functions (top-level)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args_list = []
            for arg in node.args.args:
                arg_name = arg.arg
                # Try to get type annotation
                if arg.annotation:
                    try:
                        ann = ast.unparse(arg.annotation)
                        args_list.append(f"{arg_name}: {ann}")
                    except Exception:
                        args_list.append(arg_name)
                else:
                    args_list.append(arg_name)

            returns = None
            if node.returns:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:
                    pass

            result["functions"].append({
                "name": node.name,
                "args": args_list,
                "returns": returns,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            })

        # Classes (top-level)
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_args = [a.arg for a in item.args.args if a.arg != "self"]
                    methods.append({"name": item.name, "args": m_args})
            result["classes"].append({
                "name": node.name,
                "methods": methods,
            })

        # Constants (ALL_CAPS top-level assignments)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    result["constants"].append(target.id)

        # __all__ export list
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                result["exports"].append(elt.value)

    # Also check for __all__ in a second pass (handles cases where it's
    # assigned after other code)
    if not result["exports"]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    result["exports"].append(elt.value)

    return result


def _format_interface_block(interfaces: List[Dict[str, Any]], segment_map: Dict[str, str]) -> str:
    """
    Format extracted interfaces into a markdown block for injection into
    architecture content.

    Args:
        interfaces: List of interface dicts from _extract_python_interfaces
        segment_map: Dict mapping file_path -> segment_id

    Returns:
        Markdown string ready for injection.
    """
    if not interfaces:
        return ""

    lines = [
        "",
        "---",
        "",
        "## DEPENDENCY REALITY — Actual interfaces from completed segments",
        "",
        "**CRITICAL**: The interfaces below are extracted from the ACTUAL files already",
        "written to the sandbox by previous segments. When importing from these modules,",
        "use EXACTLY these function/class names. Do NOT guess or invent names — use what",
        "is listed here.",
        "",
    ]

    for iface in interfaces:
        fp = iface["file_path"]
        seg_id = segment_map.get(fp, "unknown")
        module_name = fp.replace("\\", "/").replace("/", ".").rstrip(".py")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        lines.append(f"### `{fp}` (from {seg_id})")
        lines.append("")

        # Show __all__ exports if available
        if iface["exports"]:
            lines.append(f"**Exports (`__all__`)**: {', '.join(f'`{e}`' for e in iface['exports'])}")
            lines.append("")

        # Functions
        if iface["functions"]:
            lines.append("**Functions:**")
            for fn in iface["functions"]:
                async_prefix = "async " if fn.get("is_async") else ""
                args_str = ", ".join(fn["args"][:6])  # Cap at 6 args for readability
                if len(fn["args"]) > 6:
                    args_str += ", ..."
                ret = f" -> {fn['returns']}" if fn.get("returns") else ""
                lines.append(f"- `{async_prefix}def {fn['name']}({args_str}){ret}`")
            lines.append("")

        # Classes
        if iface["classes"]:
            lines.append("**Classes:**")
            for cls in iface["classes"]:
                method_names = [m["name"] for m in cls["methods"] if not m["name"].startswith("_")]
                if method_names:
                    lines.append(f"- `class {cls['name']}` — methods: {', '.join(f'`{m}`' for m in method_names[:8])}")
                else:
                    lines.append(f"- `class {cls['name']}`")
            lines.append("")

        # Constants
        if iface["constants"]:
            lines.append(f"**Constants:** {', '.join(f'`{c}`' for c in iface['constants'][:10])}")
            lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def read_dependency_interfaces_from_sandbox(
    segment: Any,
    completed_segments: Dict[str, Any],
    manifest: Any,
    sandbox_base: str = "D:\\Orb",
) -> str:
    """
    Read actual output files from completed dependency segments and extract
    their interfaces.

    This is the main entry point for Option A reconciliation.

    Args:
        segment: The SegmentSpec about to be executed
        completed_segments: Dict of {seg_id: SegmentState} for completed deps
        manifest: The full SegmentManifest
        sandbox_base: Root of the sandbox filesystem

    Returns:
        Markdown string to append to architecture_content, or "" if no
        dependencies or no interfaces found.
    """
    if not segment.dependencies:
        return ""

    all_interfaces: List[Dict[str, Any]] = []
    segment_map: Dict[str, str] = {}  # file_path -> segment_id

    try:
        from app.services.sandbox import get_sandbox_client
        client = get_sandbox_client()
    except Exception as e:
        logger.warning("[interface_recon] Cannot get sandbox client: %s", e)
        client = None

    for dep_id in segment.dependencies:
        dep_state = completed_segments.get(dep_id)
        if dep_state is None:
            continue

        dep_spec = manifest.get_segment(dep_id) if manifest else None
        if dep_spec is None:
            continue

        # Get the files this dependency was supposed to create
        for rel_path in dep_spec.file_scope:
            # Only process Python files for now
            if not rel_path.endswith(".py"):
                continue

            # Read the file from sandbox
            abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
            file_content = None

            # Try reading via sandbox client first (remote sandbox)
            if client is not None:
                try:
                    result = client.read_file(abs_path)
                    if result and result.get("content"):
                        file_content = result["content"]
                except Exception:
                    pass

            # Fallback: read directly (local/same-machine sandbox)
            if file_content is None and os.path.isfile(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        file_content = f.read()
                except Exception as e:
                    logger.warning("[interface_recon] Cannot read %s: %s", abs_path, e)
                    continue

            if not file_content:
                logger.debug("[interface_recon] No content for %s (may not exist yet)", rel_path)
                continue

            # Extract interfaces
            iface = _extract_python_interfaces(file_content, rel_path)
            if iface["functions"] or iface["classes"] or iface["exports"]:
                all_interfaces.append(iface)
                segment_map[rel_path] = dep_id
                logger.info(
                    "[interface_recon] Extracted %d functions, %d classes from %s (%s)",
                    len(iface["functions"]), len(iface["classes"]),
                    rel_path, dep_id,
                )

    if not all_interfaces:
        logger.info("[interface_recon] No interfaces found from %d dependencies", len(segment.dependencies))
        return ""

    block = _format_interface_block(all_interfaces, segment_map)
    logger.info(
        "[interface_recon] Generated reconciliation block: %d files, %d chars",
        len(all_interfaces), len(block),
    )
    return block


def inject_reconciliation_into_architecture(
    architecture_content: str,
    reconciliation_block: str,
) -> str:
    """
    Inject the reconciliation block into architecture content.

    Appends the block at the end of the architecture, just before the
    file inventory section (if found) or at the very end.

    This ensures the Implementer sees the real interfaces when generating
    code, without modifying any other part of the architecture.
    """
    if not reconciliation_block:
        return architecture_content

    # Try to insert before File Inventory section
    file_inv_pattern = re.compile(r'^(#{1,4}\s*.*[Ff]ile\s*[Ii]nventory)', re.MULTILINE)
    match = file_inv_pattern.search(architecture_content)

    if match:
        insert_pos = match.start()
        return (
            architecture_content[:insert_pos]
            + reconciliation_block
            + "\n"
            + architecture_content[insert_pos:]
        )

    # No File Inventory found — append at end
    return architecture_content + "\n" + reconciliation_block

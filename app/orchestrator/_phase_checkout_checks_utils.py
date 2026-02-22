from __future__ import annotations
import ast
import re
from typing import Any, Optional, Tuple


def _find_largest_function(source_code: str) -> Tuple[int, str]:
    """Find the largest function body in a Python file."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return (0, "")

    max_lines = 0
    max_name = ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "end_lineno") and node.end_lineno:
                fn_lines = node.end_lineno - node.lineno + 1
                if fn_lines > max_lines:
                    max_lines = fn_lines
                    max_name = node.name
    return (max_lines, max_name)

_SILENT_IMPORT_PATTERNS = [
    re.compile(r"No module named '([^']+)'"),
    re.compile(r"cannot import name '([^']+)' from '([^']+)'"),
    re.compile(r"ImportError: ([^\n]+)"),
    re.compile(r"ModuleNotFoundError: ([^\n]+)"),
]

_KNOWN_PREEXISTING_FAILURES = {
    "numpy",
    "scipy",
    "pandas",
    "cv2",
    "PIL",
    "torch",
    "tensorflow",
}

def _try_deterministic_import_fix(
    content: str,
    error_summary: str,
    full_stderr: str,
) -> Optional[Tuple[str, str]]:
    """
    Try to fix import errors without LLM -- pure string manipulation.

    Handles:
    - "No module named 'app.models'" -> comment out or remove the import line
    - "cannot import name 'X' from 'Y'" -> comment out the import line

    Returns (fixed_content, description) or None.
    """
    # Extract the problematic module/name from the error
    no_module_match = re.search(r"No module named '([^']+)'", error_summary)
    cannot_import_match = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_summary)

    if no_module_match:
        bad_module = no_module_match.group(1)
        # Find and comment out import lines referencing this module
        lines = content.split("\n")
        fixed_lines = []
        changes = 0
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith("from ") or stripped.startswith("import ")) and bad_module in stripped:
                fixed_lines.append(f"# PHASE_CHECKOUT_FIX: removed bad import: {line.strip()}")
                changes += 1
            else:
                fixed_lines.append(line)

        if changes > 0:
            return ("\n".join(fixed_lines), f"Commented out {changes} import(s) of '{bad_module}'")

    if cannot_import_match:
        bad_name = cannot_import_match.group(1)
        bad_source = cannot_import_match.group(2)
        lines = content.split("\n")
        fixed_lines = []
        changes = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("from ") and bad_source in stripped and bad_name in stripped:
                fixed_lines.append(f"# PHASE_CHECKOUT_FIX: removed bad import: {line.strip()}")
                changes += 1
            else:
                fixed_lines.append(line)

        if changes > 0:
            return ("\n".join(fixed_lines), f"Commented out import of '{bad_name}' from '{bad_source}'")

    return None

def _build_fix_prompt(
    failing_file: str,
    broken_content: str,
    error_summary: str,
    full_stderr: str,
    error_type: str,
) -> str:
    """Build the LLM prompt for a targeted boot fix."""
    # Truncate content/stderr to fit context
    max_content = 15000
    max_stderr = 3000
    content_display = broken_content[:max_content]
    if len(broken_content) > max_content:
        content_display += f"\n\n... [truncated, {len(broken_content)} chars total]"
    stderr_display = full_stderr[:max_stderr]

    # v3.0: Check experience memory for matching boot fix patterns
    _memory_section = ""
    try:
        from app.experience.retrieval import retrieve_for_stage, format_injection
        from app.db import get_db_session
        from app.orchestrator.strike_tracker import _error_signature
        _mem_db = get_db_session()
        _boot_patterns = retrieve_for_stage(
            _mem_db, stage="phase_checkout",
            context=f"Boot fix for {failing_file}: {error_summary[:100]}",
            error_signature=_error_signature(error_summary),
            max_results=3,
        )
        if _boot_patterns:
            _memory_section = "\n\n" + format_injection(_boot_patterns, stage="phase_checkout")
        _mem_db.close()
    except Exception:
        pass

    return (
        f"## BOOT FIX REQUIRED\n\n"
        f"**Error type:** {error_type}\n"
        f"**Failing file:** {failing_file}\n\n"
        f"### Error Summary\n```\n{error_summary}\n```\n\n"
        f"### Full Traceback\n```\n{stderr_display}\n```\n\n"
        f"### Current File Content (broken)\n```python\n{content_display}\n```\n\n"
        f"{_memory_section}\n\n"
        f"Output the complete fixed file now. Nothing else."
    )

def _discover_sandbox_base(client: Any, default_base: str) -> str:
    """Find the actual repo base path inside the sandbox."""
    for candidate in [r"C:\Orb\Orb", r"C:\Orb", r"D:\Orb"]:
        try:
            test = client.shell_run(
                f'Test-Path -Path "{candidate}\\main.py"',
                timeout_seconds=10,
            )
            if (test.stdout or "").strip().lower() == "true":
                return candidate
        except Exception:
            continue
    return default_base

def _file_exists_in_sandbox(
    client: Any,
    rel_path: str,
    sandbox_base: str,
) -> bool:
    """Check if a file exists in the sandbox."""
    normed = rel_path.replace("/", "\\")
    if not (normed.startswith("C:") or normed.startswith("D:")):
        abs_path = f"{sandbox_base}\\{normed}"
    else:
        abs_path = normed

    try:
        result = client.shell_run(
            f'Test-Path -Path "{abs_path}" -PathType Leaf',
            timeout_seconds=10,
        )
        return (result.stdout or "").strip().lower() == "true"
    except Exception:
        return False

def map_file_to_segment(
    file_path: Optional[str],
    state: Any,
) -> Optional[str]:
    """Map a failing file path to the segment that produced it."""
    from .phase_checkout_checks import _norm
    if not file_path:
        return None
    target = _norm(file_path)
    for seg_id, seg_state in state.segments.items():
        for out_file in (seg_state.output_files or []):
            if _norm(out_file) == target:
                return seg_id
    return None

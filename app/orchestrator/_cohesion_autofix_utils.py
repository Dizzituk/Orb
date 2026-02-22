import logging
import os
import re
from typing import List, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


COHESION_AUTOFIX_BUILD_ID = "2026-02-13-v1.0-three-tier-autofix"

def _is_import_depth_issue(issue) -> bool:
    """Check if an import_mismatch is a simple depth correction (. → ..)."""
    desc = issue.description.lower()
    fix = (issue.suggested_fix or "").lower()

    # Look for patterns indicating import depth
    depth_indicators = [
        "from ..",       # fix mentions double-dot
        "'..' prefix",
        "two levels up",
        "parent package",
        "does not exist",  # .X resolves to wrong path
    ]
    return any(ind in desc or ind in fix for ind in depth_indicators)

def _fix_import_depth(issue, arch_text: str) -> Tuple[str, bool, str]:
    """
    Fix import depth: from .module → from ..module.

    Extracts the module name from the issue and replaces single-dot
    relative imports with double-dot in the architecture text.
    """
    desc = issue.description
    fix = issue.suggested_fix or ""

    # Extract module names from the issue
    # Patterns like: "from .implementer import" or "'from .implementer'"
    modules_to_fix = set()

    # Pattern 1: Extract from suggested_fix "from .X" → "from ..X"
    fix_patterns = re.findall(r"from\s+\.(\w+)", fix)
    modules_to_fix.update(fix_patterns)

    # Pattern 2: Extract from description
    desc_patterns = re.findall(r"'from\s+\.(\w+)\s+import", desc)
    modules_to_fix.update(desc_patterns)

    # Pattern 3: Look for '.X' resolves to / does not exist
    resolve_patterns = re.findall(r"['\"]\.(\w+)['\"]", desc)
    modules_to_fix.update(resolve_patterns)

    if not modules_to_fix:
        return arch_text, False, "Could not extract module name from issue"

    changes = []
    fixed_text = arch_text
    for module in modules_to_fix:
        # Replace `from .{module}` with `from ..{module}` 
        # But NOT `from ..{module}` (already correct) or `from .{other_module}`
        # Match in both code blocks and prose
        old_pattern = f"from .{module}"
        new_pattern = f"from ..{module}"

        # Don't replace if it's already double-dot
        # Use word boundary after module name to avoid partial matches
        count = 0
        lines = fixed_text.split("\n")
        new_lines = []
        for line in lines:
            if old_pattern in line and f"from ..{module}" not in line:
                new_line = line.replace(old_pattern, new_pattern)
                if new_line != line:
                    count += 1
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        fixed_text = "\n".join(new_lines)

        if count > 0:
            changes.append(f"'{old_pattern}' → '{new_pattern}' ({count} occurrence(s))")

    if changes:
        return fixed_text, True, "; ".join(changes)
    return arch_text, False, f"Pattern 'from .{list(modules_to_fix)[0]}' not found in architecture"

def _fix_missing_import(issue, arch_text: str) -> Tuple[str, bool, str]:
    """
    Add missing import statement to architecture code blocks.

    For 'import logging', adds both the import and logger setup.
    """
    desc = issue.description.lower()

    if "import logging" not in desc and "import logging" not in (issue.suggested_fix or "").lower():
        return arch_text, False, "Not a logging import issue"

    # Find code blocks in the architecture that belong to the affected file
    # Architecture files have code blocks like:
    #   ```python
    #   import os
    #   ...
    #   ```
    # We need to add `import logging` after the last existing import

    import_line = "import logging"
    logger_line = 'logger = logging.getLogger(__name__)'

    # Check if already present
    if import_line in arch_text and logger_line in arch_text:
        return arch_text, False, "Already contains import logging"

    # Strategy: Find python code blocks and add logging import after the last
    # import/from line in the first code block that has imports
    fixed_text = arch_text
    changes = []

    # Find all ```python ... ``` blocks
    code_block_pattern = re.compile(r'(```python\n)(.*?)(```)', re.DOTALL)
    
    def _add_logging_to_block(match):
        prefix = match.group(1)
        code = match.group(2)
        suffix = match.group(3)

        if import_line in code:
            return match.group(0)  # Already has it

        # Find the last import line
        lines = code.split("\n")
        last_import_idx = -1
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import_idx = idx

        if last_import_idx >= 0:
            # Insert after last import
            lines.insert(last_import_idx + 1, import_line)
            lines.insert(last_import_idx + 2, logger_line)
            changes.append("Added import logging + logger setup after imports")
            return prefix + "\n".join(lines) + suffix

        return match.group(0)  # No imports found, leave as-is

    # Only fix the first code block that has imports (main module block)
    fixed_text = code_block_pattern.sub(_add_logging_to_block, fixed_text, count=1)

    if changes:
        return fixed_text, True, "; ".join(changes)
    return arch_text, False, "Could not find suitable code block to add logging import"

def _fix_naming_mismatch(issue, arch_text: str) -> Tuple[str, bool, str]:
    """
    Fix function/variable naming mismatch by replacing actual with expected.
    """
    expected = issue.expected
    actual = issue.actual

    if not expected or not actual:
        return arch_text, False, "Missing expected/actual values"

    # Use word-boundary replacement to avoid partial matches
    pattern = re.compile(r'\b' + re.escape(actual) + r'\b')
    fixed_text, count = pattern.subn(expected, arch_text)

    if count > 0:
        return fixed_text, True, f"Renamed '{actual}' → '{expected}' ({count} occurrence(s))"
    return arch_text, False, f"Pattern '{actual}' not found in architecture"

def _build_micro_patch_prompt(
    issue, arch_text: str, segment_id: str, job_dir: str = "",
) -> str:
    """Build a focused prompt for micro LLM patching.

    v3.8 (Fix 10): Now includes sibling export map and skeleton context
    so the fix LLM has evidence to make correct decisions about which
    segment owns which symbol.
    """
    parts = [
        f"# Fix Required for Segment: {segment_id}\n",
        f"## Issue: {issue.issue_id} [{issue.category}]\n",
        f"**Problem:** {issue.description}\n",
    ]
    if issue.suggested_fix:
        parts.append(f"**Required Fix:** {issue.suggested_fix}\n")
    if issue.expected:
        parts.append(f"**Expected:** {issue.expected}")
    if issue.actual:
        parts.append(f"**Actual:** {issue.actual}")

    # ── v3.8 (Fix 10): Inject sibling export map evidence ──────────
    # Gives the fix LLM visibility into what symbols exist across all
    # segments, so it can resolve duplicates and phantoms correctly.
    if job_dir:
        try:
            _export_lines = _build_sibling_export_context(job_dir, segment_id)
            if _export_lines:
                parts.append("\n---\n")
                parts.append("## Sibling Export Map (ground truth)\n")
                parts.append(
                    "These are the REAL function/constant names exported by "
                    "each sibling segment. Use this to determine which segment "
                    "owns which symbol. If resolving a duplicate, KEEP the "
                    "function in the segment listed here and REMOVE it from "
                    "the other. If resolving a missing import, check this map "
                    "to find where the symbol actually lives.\n"
                )
                parts.append("\n".join(_export_lines))
        except Exception as _e:
            logger.debug("[cohesion_autofix] v3.8 Export map injection failed: %s", _e)

    parts.append("\n---\n")
    parts.append("## Architecture Document (apply the fix to this)\n")
    parts.append(arch_text)
    parts.append("\n---\n")
    parts.append(
        "Apply ONLY the fix described above. Do not change anything else. "
        "Return the COMPLETE architecture document with the fix applied."
    )

    return "\n".join(parts)

def _build_sibling_export_context(
    job_dir: str, current_segment_id: str,
) -> List[str]:
    """Build export context lines from sibling enrichment files.

    v3.8 (Fix 10): Provides the auto-fix LLM with ground truth about
    which segment exports which symbols.
    """
    import json as _json

    # Derive parent job dir (strip __seg-* suffix)
    _parent_jid = os.path.basename(job_dir)
    if "__" in _parent_jid:
        _parent_jid = _parent_jid.split("__")[0]
        _parent_dir = os.path.join(os.path.dirname(job_dir), _parent_jid)
    else:
        _parent_dir = job_dir

    _segments_dir = os.path.join(_parent_dir, "segments")
    if not os.path.isdir(_segments_dir):
        return []

    lines: List[str] = []
    for _seg_name in sorted(os.listdir(_segments_dir)):
        _enrich_path = os.path.join(_segments_dir, _seg_name, "enrichment.json")
        if not os.path.isfile(_enrich_path):
            continue

        try:
            with open(_enrich_path, "r", encoding="utf-8") as _f:
                _enrich = _json.load(_f)
        except Exception:
            continue

        _symbols: list = []
        for _exp in _enrich.get("exports", []):
            if isinstance(_exp, str):
                _symbols.append(_exp)
            elif isinstance(_exp, dict) and _exp.get("name"):
                _symbols.append(_exp["name"])
        for _func in _enrich.get("functions", []):
            _name = _func.get("name", "") if isinstance(_func, dict) else str(_func)
            if _name and _name not in _symbols:
                _symbols.append(_name)
        for _const in _enrich.get("constants", []):
            _name = _const.get("name", "") if isinstance(_const, dict) else str(_const)
            if _name and _name not in _symbols:
                _symbols.append(_name)

        if _symbols:
            _is_self = " **(THIS SEGMENT)**" if _seg_name == current_segment_id else ""
            _sym_str = ", ".join(f"`{s}`" for s in _symbols)
            lines.append(f"- **{_seg_name}**{_is_self} exports: {_sym_str}")

    return lines

def _save_patched_architecture(seg_id: str, arch_text: str, job_dir: str):
    """
    Save a patched architecture to disk.

    Writes to arch_v{next}.md so the original is preserved.
    Also updates the 'latest' symlink logic by using the highest version number.
    """
    arch_dir = os.path.join(job_dir, "segments", seg_id, "arch")
    os.makedirs(arch_dir, exist_ok=True)

    # Find the next available version number
    existing = [f for f in os.listdir(arch_dir) if f.startswith("arch_v") and f.endswith(".md")]
    max_version = 0
    for fname in existing:
        try:
            v = int(fname.replace("arch_v", "").replace(".md", ""))
            max_version = max(max_version, v)
        except ValueError:
            pass

    next_version = max_version + 1
    new_path = os.path.join(arch_dir, f"arch_v{next_version}.md")

    # Add autofix header comment
    header = (
        f"<!-- COHESION AUTOFIX: Patched from arch_v{max_version}.md by "
        f"cohesion_autofix v1.0 -->\n\n"
    )

    with open(new_path, "w", encoding="utf-8") as f:
        f.write(header + arch_text)

    logger.info("[cohesion_autofix] Saved %s (%d chars)", new_path, len(arch_text))

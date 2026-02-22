from __future__ import annotations
import logging
import os
import re
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


COHESION_CHECK_BUILD_ID = "2026-02-18-v3.7-sectional-tier2-fix"

def _extract_arch_file_paths(arch_content: str) -> List[str]:
    """Extract file paths from architecture document File Inventory tables only.
    
    IMPORTANT: Only extracts from the File Inventory section to avoid false
    positives from paths mentioned in prose, docstrings, and import examples.
    """
    paths = []
    seen = set()

    # Find the File Inventory section
    inv_match = re.search(r'(?:^|\n)#+\s*File Inventory', arch_content)
    if not inv_match:
        return paths
    
    inv_start = inv_match.start()
    # Find the end of the inventory section (next ## heading or ---)
    inv_end_match = re.search(r'\n(?:##[^#]|---)', arch_content[inv_start + 20:])
    if inv_end_match:
        inv_section = arch_content[inv_start:inv_start + 20 + inv_end_match.start()]
    else:
        # Take a reasonable chunk
        inv_section = arch_content[inv_start:inv_start + 3000]

    # Extract paths from table rows in the inventory section.
    # Only match the FIRST backtick-wrapped path in each table row
    # to avoid picking up filenames from description columns.
    for line in inv_section.split("\n"):
        # Must be a table row (starts with |) and not a header separator
        if not line.strip().startswith("|") or line.strip().startswith("|---"):
            continue
        # Skip rows with "none" or "N/A" markers (in either first cell or description)
        line_lower = line.lower()
        if "*(none" in line_lower or "*(n/a" in line_lower or "_(none" in line_lower or "_(n/a" in line_lower:
            continue
        # Find FIRST backtick-wrapped path in this row
        match = re.search(
            r'`((?:app|src|tests|config|orb-desktop)[/\\][\w/\\._-]+\.[a-z]+)`',
            line
        )
        if not match:
            # Try root-level file (e.g. main.py)
            match = re.search(
                r'`([\w_-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|md|css))`',
                line
            )
            # Only accept if it's truly the first cell content
            if match and ("/" in match.group(1) or "\\" in match.group(1)):
                match = None
        if match:
            p = match.group(1)
            key = p.replace("\\", "/").lower()
            if key not in seen:
                seen.add(key)
                paths.append(p)

    return paths

def _extract_segment_references(arch_content: str) -> List[int]:
    """Extract segment number references from architecture content."""
    refs = set()
    for match in re.finditer(r'[Ss]egment[\s_-]*(\d+)', arch_content):
        refs.add(int(match.group(1)))
    return sorted(refs)

def _build_cohesion_prompt(
    architectures: Dict[str, str],
    contract_json: Optional[str] = None,
    source_file_evidence: Optional[Dict[str, str]] = None,
) -> str:
    """Build the prompt for the LLM cohesion check."""
    parts = []
    parts.append("# Cross-Segment Architecture Cohesion Check\n")
    parts.append("You are reviewing multiple segment architectures for a single job.")
    parts.append("Check for cross-segment compatibility issues:\n")
    parts.append("1. **Import resolution**: Do imports between segments resolve correctly?")
    parts.append("   - Consider the DIRECTORY STRUCTURE: files in a sub-package use `..` to import from parent package")
    parts.append("2. **Naming matches**: Do function/class/variable names match across boundaries?")
    parts.append("3. **Signature compatibility**: Do function signatures match what callers expect?")
    parts.append("4. **Data shape compatibility**: Do data structures match across segment boundaries?")
    parts.append("5. **Contract compliance**: Do segments fulfil their skeleton contract obligations?")
    parts.append("6. **Endpoint consistency**: Do API endpoints and router prefixes align?")
    parts.append("")
    parts.append("Severity rules:")
    parts.append("- 'blocking': Would cause import errors, type errors, or runtime crashes")
    parts.append("- 'warning': Might cause issues or indicates suboptimal design")
    parts.append("- If the SOURCE FILE EVIDENCE below confirms a claim in the architecture, it is NOT blocking.")
    parts.append("  Only flag as blocking if the architecture CONTRADICTS the source evidence or would cause runtime errors.")
    parts.append("")

    if contract_json:
        parts.append("## Skeleton Contract\n")
        parts.append("```json")
        # Truncate if very large
        if len(contract_json) > 8000:
            parts.append(contract_json[:8000] + "\n... (truncated)")
        else:
            parts.append(contract_json)
        parts.append("```\n")

    # v2.2: Project structure context for import validation
    if source_file_evidence:
        # Derive directory structure from file paths
        _dirs = set()
        for _sf_path in source_file_evidence.keys():
            _path_parts = _sf_path.replace("\\", "/").split("/")
            for _depth in range(1, len(_path_parts)):
                _dirs.add("/".join(_path_parts[:_depth]))
        if _dirs:
            parts.append("## Project Directory Structure\n")
            parts.append(
                "The following directories exist in the project. Use this to determine "
                "correct relative import paths (e.g. files in `app/overwatcher/architecture_executor/` "
                "must use `from ..spec_resolution import ...` to reach `app/overwatcher/spec_resolution.py`, "
                "NOT `from .spec_resolution import ...`).\n"
            )
            for _d in sorted(_dirs):
                parts.append(f"- `{_d}/`")
            parts.append("")

    # v2.2: Source file evidence for verification
    if source_file_evidence:
        parts.append("## Source File Evidence (GROUND TRUTH)\n")
        parts.append(
            "The following file(s) are the ORIGINAL source code being refactored. "
            "Use these to VERIFY claims in the architectures. If an architecture "
            "states a function signature or constant value, check it against this evidence. "
            "Only flag mismatches between architecture and THIS evidence as issues.\n"
        )
        for _sf_path, _sf_content in source_file_evidence.items():
            # Cap at 60K per file for cohesion check (less than critical pipeline)
            _sf_inject = _sf_content[:60_000]
            parts.append(f"**`{_sf_path}`** ({len(_sf_content):,} chars)")
            parts.append(f"```python\n{_sf_inject}\n```\n")
            if len(_sf_content) > 60_000:
                parts.append(f"... (truncated from {len(_sf_content):,} chars)\n")

    for seg_id, arch in architectures.items():
        parts.append(f"## Architecture: {seg_id}\n")
        # Truncate each architecture to avoid context overflow
        if len(arch) > 15000:
            parts.append(arch[:15000])
            parts.append(f"\n... (truncated from {len(arch)} chars)")
        else:
            parts.append(arch)
        parts.append("")

    parts.append("## Response Format\n")
    parts.append("Respond with a JSON object:")
    parts.append("```json")
    parts.append("""{
  "status": "pass" | "fail",
  "issues": [
    {
      "issue_id": "COH-001",
      "severity": "blocking" | "warning",
      "category": "import_mismatch|naming_mismatch|shape_mismatch|missing_export|contract_violation|endpoint_mismatch",
      "description": "What the issue is",
      "source_segment": "seg-01-...",
      "related_segment": "seg-02-...",
      "file_path": "app/foo/bar.py",
      "suggested_fix": "How to fix it"
    }
  ],
  "notes": "Optional overall notes"
}""")
    parts.append("```")
    parts.append("")
    parts.append("If all segments are compatible, return status 'pass' with an empty issues array.")
    parts.append("Only report REAL issues — do not invent problems.")

    return "\n".join(parts)

def _classify_fix_tier(issue: CohesionIssue) -> int:
    """
    Classify an issue into auto-fix tier based on category and content.

    Returns:
        1 = Deterministic fix (regex/string replacement, zero cost)
        2 = Micro-LLM fix (small targeted call, ~500 tokens)
        3 = Full regeneration (existing pipeline, expensive)
    """
    from .cohesion_check import CohesionIssue
    desc_lower = issue.description.lower()
    fix_lower = issue.suggested_fix.lower()
    cat = issue.category

    # ----- TIER 1: Deterministic -----

    # Import depth: from .X → from ..X
    # v3.6: Tightened — only Tier 1 if SAME module name at different depths.
    # Previously matched whenever description had 'from .' and fix had 'from ..'
    # which caused false positives when they referred to DIFFERENT modules
    # (e.g. description mentions 'from ._utils' and fix mentions 'from ..segment_state').
    # v3.7: Removed unconditional "relative import" early return — it bypassed
    # the module-name safety check and allowed different-module replacements
    # (e.g. ._utils → ..segment_state) to be classified as safe Tier 1.
    if cat == "import_mismatch":
        # Only Tier 1 if we can confirm same module name at different depths
        if "from ." in desc_lower and "from .." in fix_lower:
            _old_mod = re.search(r"from\s+\.(\w+)\s+import", issue.description)
            _new_mod = re.search(r"from\s+\.{2,}(\w+)\s+import", issue.suggested_fix)
            if _old_mod and _new_mod and _old_mod.group(1) == _new_mod.group(1):
                return 1  # Same module, different depth -> safe Tier 1
            if _old_mod and _new_mod and _old_mod.group(1) != _new_mod.group(1):
                return 2  # Different modules -> needs LLM judgement
        # v3.3: Import name mismatch with both names known
        if issue.expected and issue.actual:
            return 1

    # Missing stdlib imports (logging, os, etc.)
    if cat == "missing_import":
        if "import logging" in desc_lower or "import logging" in fix_lower:
            return 1

    # Naming mismatch with both names known
    if cat == "naming_mismatch":
        if issue.expected and issue.actual:
            return 1

    # ----- TIER 2: Micro-LLM -----

    # Missing exports that need context-aware insertion
    if cat == "missing_export" and issue.suggested_fix:
        return 2

    # v2.7: Missing symbol (cross-segment import of undefined name)
    # Tier 2: LLM adds the missing definition to the target architecture
    if cat == "missing_symbol" and issue.suggested_fix:
        return 2

    # Contract violations with clear fix description
    if cat == "contract_violation" and issue.suggested_fix and \
       len(issue.suggested_fix) > 20:
        return 2

    # Import mismatch that isn't simple depth (needs LLM judgement)
    if cat == "import_mismatch" and issue.suggested_fix:
        return 2

    # ----- TIER 3: Full regen (default) -----
    return 3

def _extract_import_replacements(issue: CohesionIssue) -> List[tuple]:
    """
    Extract (old_pattern, new_pattern) pairs from an import_mismatch issue.

    Parses the description and suggested_fix for patterns like:
      "from .implementer import" → "from ..implementer import"
    """
    from .cohesion_check import CohesionIssue
    replacements = []
    combined = issue.description + " " + issue.suggested_fix

    # Pattern 1: 'from .X import' → 'from ..X import'
    # Matches: "Change 'from .implementer import ...' to 'from ..implementer import ...'"
    # v3.7: Added module-name guard — only match if the module name is the
    # same at both depths.  Without this, a suggested_fix like
    # "Change 'from ._utils import' to 'from ..segment_state import'"
    # would produce a cross-module replacement that corrupts the architecture.
    pairs = re.findall(
        r"['\"]from\s+(\.+)(\w+)\s+import[^'\"]*['\"]\s*(?:to|→|->)\s*['\"]from\s+(\.{2,})(\w+)\s+import",
        combined,
    )
    for old_dots, old_name, new_dots, new_name in pairs:
        if old_name == new_name:  # v3.7: Same module at different depth — safe
            replacements.append((f"from {old_dots}{old_name} import", f"from {new_dots}{new_name} import"))

    # Pattern 2: Explicit "from .X" / "from ..X" in suggested_fix
    # v3.6: Only pair if the module name matches (different depth of SAME module).
    # Previously paired any single-dot module with any double-dot module,
    # which corrupted architectures when description and fix mentioned
    # different modules (e.g. ._utils in desc, ..segment_state in fix).
    if not replacements:
        old_match = re.search(r"from\s+(\.)(\w+)\s+import", issue.description)
        new_match = re.search(r"from\s+(\.{2,})(\w+)\s+import", issue.suggested_fix)
        if old_match and new_match and old_match.group(2) == new_match.group(2):
            replacements.append((
                f"from {old_match.group(1)}{old_match.group(2)} import",
                f"from {new_match.group(1)}{new_match.group(2)} import",
            ))

    # Pattern 3: General ".module" → "..module" mentioned anywhere
    if not replacements:
        singles = re.findall(r"'\.([a-zA-Z_]\w*)'", issue.description)
        doubles = re.findall(r"'\.\.([a-zA-Z_]\w*)'", issue.suggested_fix)
        for mod in set(singles) & set(doubles):
            replacements.append((f"from .{mod} import", f"from ..{mod} import"))

    return replacements

def _inject_logging_import(arch_text: str) -> Optional[str]:
    """
    Inject 'import logging' + logger line into architecture text.

    Finds the imports section (```python block with import statements)
    and adds logging if missing.
    """
    if "import logging" in arch_text:
        return None  # Already present

    logging_block = "import logging\nlogger = logging.getLogger(__name__)"

    # Strategy 1: Find a python code block with imports and inject after last import
    code_blocks = list(re.finditer(
        r"```python\n(.*?)```",
        arch_text,
        re.DOTALL,
    ))

    for block_match in code_blocks:
        block_content = block_match.group(1)
        # Check if this block has import statements
        if not re.search(r"^(?:import |from )", block_content, re.MULTILINE):
            continue

        # Find the last import line in this block
        import_lines = list(re.finditer(
            r"^(?:import |from )[^\n]+",
            block_content,
            re.MULTILINE,
        ))
        if import_lines:
            last_import = import_lines[-1]
            insert_pos = block_match.start(1) + last_import.end()
            return (
                arch_text[:insert_pos]
                + "\n" + logging_block
                + arch_text[insert_pos:]
            )

    # Strategy 2: Find any "## Imports" or similar heading and inject after
    imports_heading = re.search(
        r"^#+\s*(?:Imports|Dependencies|Module Imports)[^\n]*\n",
        arch_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if imports_heading:
        insert_pos = imports_heading.end()
        # If there's a code fence right after, inject inside it
        after = arch_text[insert_pos:insert_pos + 20]
        if after.strip().startswith("```python"):
            fence_end = arch_text.index("\n", insert_pos + arch_text[insert_pos:].index("```python")) + 1
            return (
                arch_text[:fence_end]
                + logging_block + "\n"
                + arch_text[fence_end:]
            )

    return None  # Couldn't find safe injection point

def _save_patched_architecture(
    job_dir: str,
    seg_id: str,
    patched_text: str,
    fix_notes: List[str],
) -> str:
    """
    Save a patched architecture as the next version number.

    If current is arch_v1.md, saves as arch_v2.md. Preserves history.
    Returns the path of the saved file.
    """
    arch_dir = os.path.join(job_dir, "segments", seg_id, "arch")

    # Find current highest version
    existing = []
    if os.path.isdir(arch_dir):
        for f in os.listdir(arch_dir):
            m = re.match(r"arch_v(\d+)\.md$", f)
            if m:
                existing.append(int(m.group(1)))

    next_ver = max(existing, default=0) + 1
    new_filename = f"arch_v{next_ver}.md"
    new_path = os.path.join(arch_dir, new_filename)

    # Prepend auto-fix header
    header = (
        f"<!-- AUTO-FIX v3.0: {len(fix_notes)} fix(es) applied by cohesion auto-fixer -->\n"
        + "".join(f"<!-- FIX: {note} -->\n" for note in fix_notes)
        + "\n"
    )

    os.makedirs(arch_dir, exist_ok=True)
    with open(new_path, "w", encoding="utf-8") as f:
        f.write(header + patched_text)

    logger.info(
        "[cohesion_auto_fix] Saved patched architecture: %s (%d chars, %d fixes)",
        new_path, len(patched_text), len(fix_notes),
    )
    return new_path


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "_apply_tier1_fix": "_cohesion_check_utils_5",
    "_apply_tier2_fix": "_cohesion_check_utils_5",
    "_parse_cohesion_response": "_cohesion_check_utils_5",
    "load_cohesion_result": "_cohesion_check_utils_5",
    "save_cohesion_result": "_cohesion_check_utils_5",
    "CohesionIssue": "_cohesion_check_utils_6",
    "CohesionResult": "_cohesion_check_utils_6",
    "load_segment_architectures": "_cohesion_check_utils_6",
    "run_cohesion_check": "_cohesion_check_utils_6",
    "attempt_auto_fixes": "_cohesion_check_utils_7",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.orchestrator.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

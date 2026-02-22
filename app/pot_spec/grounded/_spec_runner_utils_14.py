from __future__ import annotations
import logging
import os
import re
from app.pot_spec.grounded._spec_runner_utils_13 import _discover_project_roots, _extract_acceptance_from_spec
from typing import List
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _reconcile_ac_names_against_source(
    spec_markdown: str,
    file_scope: List[str],
) -> List[str]:
    """
    v5.5: Reconcile function/class names in acceptance criteria against source files.

    Scans acceptance criteria for backtick-wrapped identifiers (e.g. `execute_architecture`)
    and checks if they exist in the actual source files' __all__, def, or class declarations.

    Returns a list of warning strings for names that don't match any source definition.
    These warnings are appended to the spec as a reconciliation note.
    """
    warnings: List[str] = []
    if not spec_markdown or not file_scope:
        return warnings

    # Step 1: Extract identifiers from acceptance criteria
    ac_section = _extract_acceptance_from_spec(spec_markdown)
    ac_text = ' '.join(ac_section)
    # Find backtick-wrapped identifiers that look like Python names
    ac_names = set(re.findall(r'`([a-zA-Z_][a-zA-Z0-9_]*)`', ac_text))
    if not ac_names:
        return warnings

    # Filter to names that look like function/class names (not constants, not short)
    ac_names = {n for n in ac_names if len(n) > 3 and not n.isupper()}
    if not ac_names:
        return warnings

    # Step 2: Gather all defined names from source files
    source_names: set = set()
    discovery = _discover_project_roots()
    roots = discovery.get('roots', [])

    for rel_path in file_scope:
        # Try to find the actual file
        abs_paths_to_try = []
        for root in roots:
            candidate = os.path.join(root, rel_path.replace('/', os.sep))
            abs_paths_to_try.append(candidate)
        # Also try the path as-is (might already be absolute)
        abs_paths_to_try.append(rel_path)

        for abs_path in abs_paths_to_try:
            if not os.path.isfile(abs_path):
                continue
            try:
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                # Extract __all__ entries
                all_match = re.search(r'__all__\s*=\s*\[([^\]]+)\]', content, re.DOTALL)
                if all_match:
                    all_entries = re.findall(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']', all_match.group(1))
                    source_names.update(all_entries)
                # Extract function defs
                for m in re.finditer(r'^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content, re.MULTILINE):
                    source_names.add(m.group(1))
                # Extract class defs
                for m in re.finditer(r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:(]', content, re.MULTILINE):
                    source_names.add(m.group(1))
            except Exception:
                continue
            break  # Found the file, no need to try other roots

    if not source_names:
        return warnings  # Couldn't read any source files

    # Step 3: Flag names in AC that don't exist in source
    for ac_name in sorted(ac_names):
        if ac_name not in source_names:
            # Check for close matches (e.g. execute_architecture vs run_architecture_execution)
            close = [s for s in source_names if ac_name.replace('_', '') in s.replace('_', '') or
                     s.replace('_', '') in ac_name.replace('_', '')]
            if close:
                warnings.append(
                    f"AC references `{ac_name}` but source defines `{close[0]}`. "
                    f"Possible name mismatch — verify which is correct."
                )
            else:
                warnings.append(
                    f"AC references `{ac_name}` which was not found in any source file's "
                    f"__all__, def, or class declarations."
                )

    if warnings:
        logger.warning(
            "[spec_runner] v5.5 AC name reconciliation: %d warning(s): %s",
            len(warnings), '; '.join(warnings),
        )
        print(f"[spec_runner] v5.5 AC NAME RECONCILIATION: {len(warnings)} warning(s)")

    return warnings

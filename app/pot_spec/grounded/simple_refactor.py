# FILE: app/pot_spec/grounded/simple_refactor.py
"""
SpecGate v3.8 - Direct Spec Builder

NO GATES. REAL EVIDENCE. STRICT VALIDATION.

v3.8 (2026-02-02): 
  - HARD validation with explicit substring check
  - Fixed className regex (case sensitivity bug)
  - Context-aware ternary detection (only skip if in className context)

v3.7 (2026-02-02): Added strict validation, smart truncation
v3.6 (2026-02-02): Added skip detection for credentials, storage keys
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SIMPLE_REFACTOR_BUILD_ID = "2026-02-02-v3.9-word-boundary-validation"
print(f"[SIMPLE_REFACTOR_LOADED] BUILD_ID={SIMPLE_REFACTOR_BUILD_ID}")


# =============================================================================
# MATCH VALIDATION - WORD BOUNDARY
# =============================================================================

def _validate_match(line_content: str, search_term: str) -> bool:
    """
    Validate that search term appears as a WHOLE WORD, not partial match.
    
    This prevents false positives like:
    - 'waitForBackendReady' matching 'orb' (f-orB-ackend)
    - 'showErrorBox' matching 'orb' (Err-orB-ox)
    
    Word boundary = start/end of string OR non-alphanumeric character.
    """
    if not line_content or not search_term:
        return False
    
    # Use word boundary regex for proper matching
    # \b matches word boundaries (start/end of word)
    pattern = rf'\b{re.escape(search_term)}\b'
    return bool(re.search(pattern, line_content, re.IGNORECASE))


# =============================================================================
# SKIP DETECTION
# =============================================================================

def _should_skip(line_content: str, file_path: str, search_term: str) -> Tuple[bool, str]:
    """
    Determine if a match should be SKIPPED (not changed).
    
    For UI text rebrand:
    - CHANGE: User-visible text (titles, labels, messages, placeholders)
    - SKIP: Code identifiers, CSS classes, storage keys, paths, etc.
    """
    line_lower = line_content.lower()
    path_lower = file_path.lower()
    search_lower = search_term.lower()
    
    # === PACKAGE.JSON IDENTIFIERS ===
    if 'package.json' in path_lower or 'package-lock.json' in path_lower:
        if '"name"' in line_lower:
            return True, "package name"
        if '"appid"' in line_lower:
            return True, "app id"
        if '"productname"' in line_lower:
            return True, "product name"
    
    # === STORAGE KEYS ===
    if re.search(r'(STORAGE|SESSION|CACHE)_KEY\s*[=:]\s*[\'"]', line_content, re.IGNORECASE):
        return True, "storage key"
    if re.search(r'(localStorage|sessionStorage)\s*[\.\[]', line_content):
        return True, "storage access"
    
    # === CREDENTIAL SERVICE ===
    if re.search(r'CREDENTIAL[_\s]*(SERVICE|NAME|KEY)\s*[=:]', line_content, re.IGNORECASE):
        return True, "credential service"
    if re.search(r'serviceName\s*[=:]\s*[\'"]', line_content):
        return True, "credential service"
    
    # === ENVIRONMENT VARIABLES ===
    if re.search(rf'\b{search_term.upper()}_[A-Z_]+\b', line_content):
        return True, "env var name"
    if re.search(r'process\.env\.[A-Z_]+', line_content):
        return True, "env var access"
    
    # === CSS CLASS NAMES (in .css files) ===
    if '.css' in path_lower:
        # .orb, .message-row.orb, etc.
        if re.search(rf'\.{search_lower}\b', line_lower):
            return True, "CSS class"
    
    # === JSX className ATTRIBUTE ===
    # v3.8: Fixed case - use line_lower and lowercase pattern
    
    # Pattern 1: className="... orb ..."
    if re.search(rf'classname\s*=\s*"[^"]*\b{search_lower}\b[^"]*"', line_lower):
        return True, "JSX className"
    
    # Pattern 2: className={`... orb ...`} (template literal)
    if re.search(rf'classname\s*=\s*\{{[^}}]*{search_lower}', line_lower):
        return True, "JSX className"
    
    # Pattern 3: className={condition ? 'x' : 'orb'} - BUT only if className is present
    # v3.8: Context-aware - only flag as "class ternary" if it's actually in a className context
    if 'classname' in line_lower or 'class=' in line_lower:
        # There's a className on this line - check if search term is in a ternary
        if re.search(rf"['\"`]{search_lower}['\"`]", line_lower):
            if re.search(r'\?[^:]+:[^;]+' + search_lower, line_lower):
                return True, "className ternary"
    
    # Pattern 4: HTML class attribute
    if re.search(rf'\bclass\s*=\s*["\'][^"\']*\b{search_lower}\b', line_lower):
        return True, "HTML class"
    
    # === FILE/FOLDER PATHS ===
    if re.search(rf'[\'"][A-Za-z]:\\\\[^\']*{search_term}', line_content, re.IGNORECASE):
        return True, "file path"
    if re.search(rf'\b(cd|dir|ls|cat|type)\s+[^\s]*{search_lower}', line_lower):
        return True, "command path"
    
    # Paths in <code> tags (documentation)
    if re.search(rf'<code>[^<]*[\\\/][^<]*{search_term}[^<]*</code>', line_content, re.IGNORECASE):
        return True, "doc path"
    
    # === IMPORT/REQUIRE PATHS ===
    if re.search(r'(from|import)\s+[\'"]', line_content):
        return True, "import path"
    if re.search(r'require\s*\([\'"]', line_content):
        return True, "require path"
    
    # === CONSTANT DEFINITIONS ===
    if re.search(rf'\b(const|let|var)\s+{search_term.upper()}[A-Z_]*\s*=', line_content):
        return True, "constant name"
    
    # === TOKEN PREFIX HINTS ===
    if re.search(rf'\({search_lower}_[^)]*\)', line_lower):
        return True, "token prefix hint"
    
    # === CODE COMMENTS ===
    # Single-line comments: // ... or /* ... */
    if re.search(r'^\s*//\s*', line_content):
        return True, "code comment"
    # Block comment continuation: * ...
    if re.search(r'^\s*\*\s+', line_content):
        return True, "doc comment"
    # CSS comment
    if re.search(r'^\s*/\*', line_content):
        return True, "code comment"
    
    return False, ""


def _truncate_line(line: str, max_len: int = 80, search_term: str = "") -> str:
    """
    Truncate line for display, keeping the search term visible.
    """
    line = line.strip()
    if len(line) <= max_len:
        return line
    
    # If we have a search term, center around it
    if search_term:
        idx = line.lower().find(search_term.lower())
        if idx >= 0:
            half = (max_len - len(search_term)) // 2
            start = max(0, idx - half)
            end = min(len(line), start + max_len - 6)  # Room for "..." on both ends
            
            result = line[start:end]
            if start > 0:
                result = "..." + result
            if end < len(line):
                result = result + "..."
            return result
    
    return line[:max_len-3] + "..."


# =============================================================================
# DIRECT SPEC BUILDER
# =============================================================================

def build_direct_spec(
    search_term: str,
    replace_term: str,
    raw_matches: List[Any],
    goal: str,
    total_files: int,
) -> str:
    """
    Build POT spec with STRICT VALIDATION.
    
    Every line MUST contain the search term to be included.
    """
    logger.info(
        "[simple_refactor] v3.8 DIRECT BUILD: %s → %s (%d raw matches)",
        search_term, replace_term, len(raw_matches)
    )
    print(f"[simple_refactor] v3.9 DIRECT BUILD: {search_term} → {replace_term}")
    
    # === VALIDATE AND CLASSIFY ===
    changes: List[Tuple[Any, str]] = []
    skips: List[Tuple[Any, str, str]] = []
    invalid_count = 0
    
    for m in raw_matches:
        file_path = getattr(m, 'file_path', '') or ''
        line_content = getattr(m, 'line_content', '') or ''
        line_num = getattr(m, 'line_number', 0)
        
        # v3.9: WORD BOUNDARY validation - reject partial matches in camelCase
        short_path = file_path.replace('\\', '/').split('/')[-1]
        
        # Use word boundary regex
        pattern = rf'\b{re.escape(search_term)}\b'
        is_valid = bool(re.search(pattern, line_content, re.IGNORECASE))
        
        if not is_valid:
            invalid_count += 1
            logger.warning(
                "[simple_refactor] v3.9 PARTIAL MATCH REJECTED: %s L%d (no word boundary)",
                short_path, line_num
            )
            print(f"[simple_refactor] v3.9 REJECTED {short_path} L{line_num}: '{search_term}' not a whole word")
            continue
        
        # Check skip rules
        should_skip, reason = _should_skip(line_content, file_path, search_term)
        if should_skip:
            skips.append((m, file_path, reason))
        else:
            changes.append((m, file_path))
    
    if invalid_count > 0:
        logger.warning("[simple_refactor] v3.8 FILTERED %d false positives", invalid_count)
        print(f"[simple_refactor] v3.8 TOTAL FILTERED: {invalid_count} false positives")
    
    # === EXTRACT GOAL ===
    actual_goal = ""
    if goal:
        for line in goal.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith(('**', '#', '-', '*', 'Known', 'Target', 'Execution', 'Search', 'Replace', 'Scope', 'Constraints', 'Intent')):
                continue
            if line.startswith("What is being built:"):
                actual_goal = line[20:].strip()
                break
            if not actual_goal:
                actual_goal = line
    
    if not actual_goal:
        actual_goal = f"Replace '{search_term}' with '{replace_term}' in user-visible UI text"
    
    # === BUILD SPEC ===
    lines = []
    
    lines.append(f"# SPoT Spec — {search_term} → {replace_term}")
    lines.append("")
    
    lines.append("## Goal")
    lines.append("")
    lines.append(actual_goal)
    lines.append("")
    
    lines.append("## Replace (case-preserving)")
    lines.append("")
    lines.append(f"- `{search_term}` → `{replace_term}`")
    if search_term.upper() != search_term:
        lines.append(f"- `{search_term.upper()}` → `{replace_term.upper()}`")
    if search_term.lower() != search_term:
        lines.append(f"- `{search_term.lower()}` → `{replace_term.lower()}`")
    lines.append("")
    
    # === CHANGES ===
    lines.append(f"## Change ({len(changes)} matches)")
    lines.append("")
    
    if not changes:
        lines.append("*No user-visible text changes identified*")
        lines.append("")
    else:
        changes_by_file: Dict[str, List[Any]] = {}
        for m, file_path in changes:
            if file_path not in changes_by_file:
                changes_by_file[file_path] = []
            changes_by_file[file_path].append(m)
        
        for file_path in sorted(changes_by_file.keys()):
            matches = changes_by_file[file_path]
            lines.append(f"### `{file_path}`")
            for m in matches:
                line_num = getattr(m, 'line_number', '?')
                line_content = getattr(m, 'line_content', '')
                truncated = _truncate_line(line_content, 80, search_term)
                lines.append(f"- L{line_num}: `{truncated}`")
            lines.append("")
    
    # === SKIPS ===
    if skips:
        lines.append(f"## Skip ({len(skips)} matches)")
        lines.append("")
        
        skips_by_reason: Dict[str, List[Tuple[Any, str]]] = {}
        for m, file_path, reason in skips:
            if reason not in skips_by_reason:
                skips_by_reason[reason] = []
            skips_by_reason[reason].append((m, file_path))
        
        for reason in sorted(skips_by_reason.keys()):
            items = skips_by_reason[reason]
            lines.append(f"### {reason} ({len(items)} matches)")
            for m, file_path in items:
                line_num = getattr(m, 'line_number', '?')
                line_content = getattr(m, 'line_content', '')
                truncated = _truncate_line(line_content, 60, search_term)
                short_path = file_path.replace('\\', '/').split('/')[-1]
                lines.append(f"- `{short_path}` L{line_num}: `{truncated}`")
            lines.append("")
    
    # === SUMMARY ===
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Raw matches:** {len(raw_matches)}")
    if invalid_count > 0:
        lines.append(f"- **False positives filtered:** {invalid_count}")
    lines.append(f"- **To change:** {len(changes)}")
    lines.append(f"- **To skip:** {len(skips)}")
    
    change_files = set(fp for _, fp in changes)
    lines.append(f"- **Files to modify:** {len(change_files)}")
    lines.append("")
    
    # === ACCEPTANCE ===
    lines.append("## Acceptance")
    lines.append("")
    lines.append("- [ ] App boots without errors")
    lines.append(f"- [ ] UI text shows '{replace_term}' instead of '{search_term}'")
    lines.append("- [ ] No console errors")
    lines.append("- [ ] Skipped items remain unchanged")
    lines.append("")
    
    spec = "\n".join(lines)
    
    logger.info(
        "[simple_refactor] v3.8 Spec: %d chars (%d change, %d skip, %d filtered)",
        len(spec), len(changes), len(skips), invalid_count
    )
    print(f"[simple_refactor] v3.9 SPEC READY: {len(spec)} chars ({len(changes)} change, {len(skips)} skip, {invalid_count} rejected)")
    
    return spec


# =============================================================================
# ENTRY POINTS
# =============================================================================

def is_simple_refactor(operation_type: str, combined_text: str, questions_none: bool) -> bool:
    """Always use direct path for refactors."""
    if operation_type == "refactor":
        logger.info("[simple_refactor] v3.8 Using direct path")
        return True
    return False


def run_simple_refactor(
    raw_matches: List[Any],
    search_term: str,
    replace_term: str,
    goal: str,
    total_files: int,
) -> str:
    """Main entry point."""
    return build_direct_spec(
        search_term=search_term,
        replace_term=replace_term,
        raw_matches=raw_matches,
        goal=goal,
        total_files=total_files,
    )


__all__ = [
    "is_simple_refactor",
    "run_simple_refactor",
    "build_direct_spec",
    "SIMPLE_REFACTOR_BUILD_ID",
]

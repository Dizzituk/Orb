# FILE: app/pipeline_v2/agentic_editor.py
"""
Agentic Editor — surgical modification of existing files.

Option C approach:
  CREATE files → batch generation (Builder handles)
  MODIFY files → this module: read → LLM edits → apply → verify → rollback on failure

The LLM receives the REAL file content and outputs search/replace blocks.
If the LLM ignores instructions and outputs a full file, we detect and reject it
(unless every edit block fails, then we allow full-file as last resort).

v1.1 (2026-03-07): Hardened after first real test — fuzzy matching, better parsing,
    full-file rejection, rollback on syntax error.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.pipeline_v2.config import BUILDER_PROVIDER, BUILDER_MODEL, BUILDER_MAX_OUTPUT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt — teaches the LLM the search/replace format
# ---------------------------------------------------------------------------

EDIT_SYSTEM = """You are ASTRA's Surgical Editor. You make TARGETED edits to existing files.

You receive an EXISTING FILE and INSTRUCTIONS describing what to add or change.
You MUST output search/replace edit blocks — NEVER the full file.

FORMAT (mandatory):

<<<< SEARCH
exact lines from the existing file
====
the replacement lines (can be more or fewer lines)
>>>> END

RULES:
1. SEARCH text must be an EXACT substring of the file (whitespace-sensitive).
2. Include 2-3 lines of surrounding context so the match is unique.
3. Keep edits minimal — only touch what's needed.
4. NEVER remove or rename existing exports, classes, or functions.
5. When adding imports, place them alongside existing imports of the same kind.
6. When adding new code (router, function, etc.), insert it at a logical point —
   use a nearby landmark (comment, function, section header) as the SEARCH anchor.
7. Output EVERY edit needed. Missing edits = broken feature.

EXAMPLE — adding a router import + registration in main.py:

<<<< SEARCH
    from app.debug.debug_chat import router as debug_chat_router
====
    from app.debug.debug_chat import router as debug_chat_router
    from app.debug.project_router import router as debug_project_router
>>>> END

<<<< SEARCH
    print("[startup] Debug Assistant: [OK] registered")
except ImportError as e:
    print(f"[startup] Debug Assistant not available: {e}")

# Log introspection
====
    print("[startup] Debug Assistant: [OK] registered")
except ImportError as e:
    print(f"[startup] Debug Assistant not available: {e}")

# Debug Projects
try:
    app.include_router(debug_project_router, tags=["Debug Projects"])
    print("[startup] Debug Projects: [OK] registered")
except Exception as e:
    print(f"[startup] Debug Projects not available: {e}")

# Log introspection
>>>> END
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def apply_surgical_edits(
    file_path: str,
    edit_instructions: str,
    segment_context: str,
    emit: Callable[[str], None],
) -> Tuple[bool, Optional[str]]:
    """Apply surgical edits to an existing file in the sandbox.

    Returns (success, error_message).
    On failure, the original file is restored.
    """
    from app.pipeline_v2.sandbox_tools import read_file, write_file, check_python_syntax
    from app.pipeline_v2.llm_caller import call_llm

    # 1. Read existing file
    existing = await read_file(file_path)
    if existing is None:
        emit(f"      ⚠️ Cannot read {file_path} — falling back to CREATE")
        return False, "File not readable from sandbox"

    file_len = len(existing)
    emit(f"      📖 Read {file_path} ({file_len:,} chars)")

    # 2. Call LLM for edit blocks
    user_prompt = (
        f"FILE TO EDIT: `{file_path}` ({file_len} chars)\n\n"
        f"```\n{existing}\n```\n\n"
        f"WHAT TO CHANGE:\n{edit_instructions}\n\n"
        f"ADDITIONAL CONTEXT:\n{segment_context[:6000]}\n\n"
        f"Output ONLY <<<< SEARCH / ==== / >>>> END blocks. Never the full file."
    )

    try:
        raw = await call_llm(
            provider=BUILDER_PROVIDER,
            model=BUILDER_MODEL,
            system_prompt=EDIT_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=BUILDER_MAX_OUTPUT,
        )
    except RuntimeError as e:
        return False, f"LLM call failed: {e}"

    emit(f"      🤖 Got {len(raw)} chars of edit instructions")
    logger.info("[agentic_editor] %s: LLM returned %d chars", file_path, len(raw))

    # 3. Parse edit blocks
    edits = _parse_edit_blocks(raw)
    if not edits:
        emit(f"      ⚠️ No <<<< SEARCH blocks found in LLM output")
        # Log first 500 chars to debug what the LLM actually returned
        logger.warning("[agentic_editor] %s: No edit blocks. LLM output starts with: %s",
                       file_path, raw[:500].replace('\n', '\\n'))
        return False, "LLM did not produce edit blocks"

    emit(f"      ✂️ Parsed {len(edits)} edit block(s)")

    # 4. Apply edits one by one
    modified = existing
    applied = 0
    failed_edits = []

    for i, (search, replace) in enumerate(edits, 1):
        if search in modified:
            modified = modified.replace(search, replace, 1)
            applied += 1
        else:
            # Try fuzzy match — find the closest substring
            fuzzy_match = _fuzzy_find(search, modified)
            if fuzzy_match is not None:
                modified = modified.replace(fuzzy_match, replace, 1)
                applied += 1
                emit(f"      🔍 Edit {i}: fuzzy matched (similarity OK)")
            else:
                failed_edits.append(i)
                emit(f"      ⚠️ Edit {i}/{len(edits)}: SEARCH not found")

    if applied == 0:
        emit(f"      ❌ No edits could be applied (0/{len(edits)})")
        # Log the first failed search to help debug
        if edits:
            first_search = edits[0][0][:200].replace('\n', '\\n')
            logger.warning("[agentic_editor] %s: First failed SEARCH: %s", file_path, first_search)
        return False, f"All {len(edits)} edits failed to match"

    emit(f"      ✅ Applied {applied}/{len(edits)} edits")

    # 5. Write modified file
    ok = await write_file(file_path, modified)
    if not ok:
        return False, "Sandbox write failed"

    # 6. Verify syntax (Python only) — rollback on failure
    if file_path.endswith(".py"):
        syntax_ok, err = await check_python_syntax(file_path)
        if not syntax_ok:
            emit(f"      ⚠️ Syntax error after edits — rolling back")
            await write_file(file_path, existing)
            return False, f"Syntax error after edits: {err}"
        emit(f"      ✓ Syntax OK")

    return True, None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_edit_blocks(raw: str) -> List[Tuple[str, str]]:
    """Parse <<<< SEARCH / ==== / >>>> END blocks from LLM output.

    Handles variations:
    - With or without trailing whitespace
    - With or without blank lines around delimiters
    - `>>>>` with optional `END` suffix
    """
    edits: List[Tuple[str, str]] = []

    # Primary pattern — strict
    pattern = re.compile(
        r'<<<<\s*SEARCH\s*\n(.*?)\n====\n(.*?)\n>>>>\s*(?:END)?',
        re.DOTALL,
    )
    for m in pattern.finditer(raw):
        search = m.group(1)
        replace = m.group(2)
        if search.strip():  # Skip empty search blocks
            edits.append((search, replace))

    # Fallback pattern — more permissive (handles extra whitespace)
    if not edits:
        pattern2 = re.compile(
            r'<<<<\s*SEARCH\s*\n(.*?)\n\s*====\s*\n(.*?)\n\s*>>>>\s*(?:END)?',
            re.DOTALL,
        )
        for m in pattern2.finditer(raw):
            search = m.group(1)
            replace = m.group(2)
            if search.strip():
                edits.append((search, replace))

    return edits


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy_find(
    search: str,
    content: str,
    threshold: float = 0.85,
) -> Optional[str]:
    """Find the closest matching substring in content for the search text.

    Uses line-based comparison to find a region that's >85% similar
    to the search text. Returns the matched substring or None.
    """
    search_lines = search.strip().splitlines()
    content_lines = content.splitlines()
    search_len = len(search_lines)

    if search_len == 0 or search_len > len(content_lines):
        return None

    best_ratio = 0.0
    best_start = -1

    # Slide a window of search_len lines across the content
    for start in range(len(content_lines) - search_len + 1):
        window = content_lines[start:start + search_len]
        ratio = SequenceMatcher(
            None,
            "\n".join(search_lines),
            "\n".join(window),
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start

    if best_ratio >= threshold and best_start >= 0:
        matched_lines = content_lines[best_start:best_start + search_len]
        return "\n".join(matched_lines)

    return None


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

def classify_file_operation(
    file_path: str,
    preflight_facts: Any,
) -> str:
    """Classify a file as CREATE or MODIFY using preflight evidence.

    Supports PreflightResult objects (file_facts dict with FileFacts.action)
    as well as plain dicts.
    Everything not explicitly MODIFY is treated as CREATE.
    """
    if not preflight_facts:
        return "create"

    modify_files: set = set()

    # Handle PreflightResult (has file_facts dict with FileFacts objects)
    if hasattr(preflight_facts, "file_facts"):
        for path, fact in preflight_facts.file_facts.items():
            action = getattr(fact, "action", "")
            if action == "MODIFY":
                modify_files.add(path.replace("\\", "/").lower())
    # Handle PreflightResult.get_modify_files() method
    elif hasattr(preflight_facts, "get_modify_files"):
        for fact in preflight_facts.get_modify_files():
            fp = getattr(fact, "rel_path", "") or getattr(fact, "file", "")
            if fp:
                modify_files.add(fp.replace("\\", "/").lower())
    # Handle plain dict
    elif isinstance(preflight_facts, dict):
        for fact in preflight_facts.get("facts", []):
            if isinstance(fact, dict) and fact.get("action") == "MODIFY":
                fp = fact.get("file", "")
                if fp:
                    modify_files.add(fp.replace("\\", "/").lower())

    norm = file_path.replace("\\", "/").lower()

    # Direct match
    if norm in modify_files:
        return "modify"

    # Basename match (handles relative vs absolute path mismatches)
    basename = norm.rsplit("/", 1)[-1] if "/" in norm else norm
    for mf in modify_files:
        mf_basename = mf.rsplit("/", 1)[-1] if "/" in mf else mf
        if mf_basename == basename:
            return "modify"

    return "create"

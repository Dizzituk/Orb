from __future__ import annotations
import json
import logging
import os
import re
from app.orchestrator._cohesion_check_utils_8 import _extract_import_replacements, _inject_logging_import
from typing import Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _parse_cohesion_response(llm_output: str) -> CohesionResult:
    """Parse the LLM's cohesion check response."""
    from ._cohesion_check_utils_6 import CohesionIssue, CohesionResult
    # Strip markdown fences
    cleaned = llm_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # Clean trailing commas (common LLM output issue)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("[cohesion_check] Failed to parse LLM response: %s", e)
        return CohesionResult(
            status="error",
            notes=f"Failed to parse LLM response: {e}",
        )

    result = CohesionResult(
        status=data.get("status", "pass"),
        notes=data.get("notes", ""),
    )

    for issue_data in data.get("issues", []):
        result.issues.append(CohesionIssue(
            issue_id=issue_data.get("issue_id", ""),
            severity=issue_data.get("severity", "warning"),
            category=issue_data.get("category", ""),
            description=issue_data.get("description", ""),
            source_segment=issue_data.get("source_segment", ""),
            related_segment=issue_data.get("related_segment", ""),
            file_path=issue_data.get("file_path", ""),
            suggested_fix=issue_data.get("suggested_fix", ""),
        ))

    # Ensure status reflects issues
    if result.blocking_issues and result.status == "pass":
        result.status = "fail"

    return result

def _apply_tier1_fix(issue: CohesionIssue, arch_text: str) -> Optional[str]:
    """
    Apply a deterministic Tier 1 fix to architecture text.

    Returns patched text or None if fix couldn't be applied.
    """
    from ._cohesion_check_utils_6 import CohesionIssue
    cat = issue.category

    # --- Import depth fixes ---
    if cat == "import_mismatch":
        replacements = _extract_import_replacements(issue)
        if not replacements:
            return None
        patched = arch_text
        applied = []
        for old_pat, new_pat in replacements:
            if old_pat in patched:
                patched = patched.replace(old_pat, new_pat)
                applied.append(f"{old_pat} → {new_pat}")
        if applied:
            issue.auto_fix_note = f"Tier 1: Replaced {'; '.join(applied)}"
            return patched
        return None

    # --- Missing logging import ---
    if cat == "missing_import" and "logging" in issue.description.lower():
        patched = _inject_logging_import(arch_text)
        if patched:
            issue.auto_fix_note = "Tier 1: Injected 'import logging' + logger init"
            return patched
        return None

    # --- Naming mismatch ---
    if cat == "naming_mismatch" and issue.expected and issue.actual:
        if issue.actual in arch_text:
            patched = arch_text.replace(issue.actual, issue.expected)
            issue.auto_fix_note = f"Tier 1: Renamed '{issue.actual}' \u2192 '{issue.expected}'"
            return patched
        return None

    # --- Import name mismatch (v3.3: post-execution cohesion) ---
    # When the cohesion checker identifies a specific wrong→correct name pair
    # in an import statement, fix it deterministically.
    if cat == "import_mismatch" and issue.expected and issue.actual:
        # issue.actual = wrong import name, issue.expected = correct name
        if issue.actual in arch_text:
            patched = arch_text.replace(issue.actual, issue.expected)
            issue.auto_fix_note = f"Tier 1: Fixed import '{issue.actual}' \u2192 '{issue.expected}'"
            return patched
        return None

    return None

def _get_tier2_category_guidance(issue) -> str:
    """Return category-specific editing instructions for Tier 2 LLM fixes.

    v1.1 (2026-03-01): Concrete examples per category to guide the LLM
    toward structural fixes rather than superficial text edits.
    """
    cat = issue.category

    if cat == "import_mismatch":
        return (
            "This is an IMPORT MISMATCH. The architecture references an import \n"
            "path or symbol name that doesn't match what the source segment exports.\n"
            "\n"
            "HOW TO FIX:\n"
            f"- Find every occurrence of the ACTUAL value: '{issue.actual or 'unknown'}'\n"
            f"- Replace with the EXPECTED value: '{issue.expected or 'unknown'}'\n"
            "- Check import statements, function calls, type references, and descriptions.\n"
            "- If the fix changes a module path (e.g. 'from app.foo' to 'from app.bar'),\n"
            "  update ALL import statements referencing the old path.\n"
            "\n"
            "EXAMPLE:\n"
            "Before: from app.education.scraper import scrape_url\n"
            "After:  from app.education.scraper import scrape_content_from_url"
        )

    if cat == "missing_export":
        return (
            "This is a MISSING EXPORT. Another segment imports a symbol that this \n"
            "segment's architecture doesn't define.\n"
            "\n"
            "HOW TO FIX:\n"
            f"- Find the code block for the file that should export the symbol.\n"
            f"- Add the missing function/class/constant definition.\n"
            f"- Ensure it matches the signature the importing segment expects.\n"
            "- If the expected value contains a function signature, use it exactly.\n"
        )

    if cat == "naming_mismatch":
        return (
            "This is a NAMING MISMATCH. A symbol is defined with one name but \n"
            "referenced with a different name in the architecture.\n"
            "\n"
            "HOW TO FIX:\n"
            f"- Find: '{issue.actual or 'unknown'}'\n"
            f"- Replace with: '{issue.expected or 'unknown'}'\n"
            "- Do this EVERYWHERE in the document (definitions, imports, calls).\n"
        )

    if cat == "shape_mismatch":
        return (
            "This is a SHAPE MISMATCH. A function's parameters or return type \n"
            "don't match what the calling segment expects.\n"
            "\n"
            "HOW TO FIX:\n"
            f"- Find the function definition in the code block.\n"
            f"- Expected signature: {issue.expected or 'see description'}\n"
            f"- Actual signature: {issue.actual or 'see description'}\n"
            "- Change the ACTUAL to match the EXPECTED.\n"
            "- Also update any return type annotations if mentioned.\n"
        )

    if cat == "contract_violation":
        return (
            "This is a CONTRACT VIOLATION. The architecture breaks a skeleton \n"
            "contract from the spec.\n"
            "\n"
            "HOW TO FIX:\n"
            f"- The expected contract: {issue.expected or 'see description'}\n"
            "- Ensure the architecture's code block matches this contract exactly.\n"
            "- Pay attention to function names, parameter types, and return types.\n"
        )

    # Generic fallback
    return (
        f"Category: {cat}\n"
        f"Apply the suggested fix as precisely as possible.\n"
        f"Only change what is needed to resolve this specific issue.\n"
    )


async def _apply_tier2_fix(
    issue: CohesionIssue,
    arch_text: str,
    seg_id: str,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5-20250929",
) -> Optional[str]:
    """
    Apply a micro-LLM Tier 2 fix to architecture text.

    v3.7: Sends only the relevant SECTION of the architecture (not the
    full document) to minimise tokens and avoid API timeouts. The LLM
    returns just the patched section, which is spliced back in.

    Returns patched full text or None if fix couldn't be applied.
    """
    from ._cohesion_check_utils_6 import CohesionIssue
    try:
        from app.providers.registry import llm_call
    except ImportError:
        logger.warning("[cohesion_auto_fix] LLM not available for Tier 2 fix")
        return None

    # =====================================================================
    # v3.7: Extract relevant section instead of sending full architecture.
    # Find section boundaries using markdown headers (## or ###).
    # The issue description usually references a section name or keyword.
    # =====================================================================
    arch_lines = arch_text.split("\n")
    _issue_keywords = set()
    for _word in (issue.description + " " + issue.suggested_fix).lower().split():
        _clean = _word.strip("(),.:'\"`")
        if len(_clean) > 3 and _clean not in {"from", "import", "this", "that", "with", "should", "must", "the", "and", "for"}:
            _issue_keywords.add(_clean)

    # Find section headers and score them by keyword overlap
    _sections: list = []  # [(start_line, end_line, header_text, score)]
    _header_lines: list = []
    for i, line in enumerate(arch_lines):
        if line.strip().startswith("#"):
            _header_lines.append(i)

    for idx, hdr_line in enumerate(_header_lines):
        _end = _header_lines[idx + 1] if idx + 1 < len(_header_lines) else len(arch_lines)
        _section_text = "\n".join(arch_lines[hdr_line:_end]).lower()
        _score = sum(1 for kw in _issue_keywords if kw in _section_text)
        _sections.append((hdr_line, _end, arch_lines[hdr_line].strip(), _score))

    # Pick the best-matching section(s). Include sections with score > 0.
    _sections.sort(key=lambda s: s[3], reverse=True)
    _best_sections = [s for s in _sections if s[3] > 0]

    if _best_sections and len(arch_text) > 4000:
        # Use sectional approach — extract top 1-2 sections
        _selected = _best_sections[:2]
        _selected.sort(key=lambda s: s[0])  # Keep original order

        _extract_start = max(0, _selected[0][0] - 2)
        _extract_end = min(len(arch_lines), _selected[-1][1] + 2)
        _section_text = "\n".join(arch_lines[_extract_start:_extract_end])
        _prefix = "\n".join(arch_lines[:_extract_start])
        _suffix = "\n".join(arch_lines[_extract_end:])
        _using_section = True

        logger.info(
            "[cohesion_auto_fix] Tier 2 sectional: lines %d-%d of %d (%.0f%% of doc)",
            _extract_start, _extract_end, len(arch_lines),
            100 * (_extract_end - _extract_start) / max(1, len(arch_lines)),
        )
    else:
        # Small document or no good section match — send the whole thing
        _section_text = arch_text
        _prefix = ""
        _suffix = ""
        _using_section = False

    # v1.1: Improved Tier 2 prompt with structured fix instructions.
    # Previous prompt was too vague ("fix one issue") and had 0% success.
    # New prompt gives category-specific instructions with concrete examples.
    _category_guidance = _get_tier2_category_guidance(issue)

    prompt = f"""You are editing an architecture document to fix a cohesion issue.

ISSUE DETAILS:
- ID: {issue.issue_id}
- Category: {issue.category}
- Severity: {issue.severity}
- Description: {issue.description}
- Expected: {issue.expected or 'N/A'}
- Actual: {issue.actual or 'N/A'}
- Suggested fix: {issue.suggested_fix}
- Segment: {seg_id}

CATEGORY-SPECIFIC GUIDANCE:
{_category_guidance}

{'ARCHITECTURE SECTION' if _using_section else 'ARCHITECTURE DOCUMENT'} TO EDIT:
{_section_text}

RULES:
1. Apply ONLY the specific fix described above. Change nothing else.
2. Return the {'COMPLETE SECTION' if _using_section else 'COMPLETE DOCUMENT'} with the fix applied.
3. Do NOT add commentary, explanations, or markdown fences.
4. Preserve ALL existing content, formatting, headers, and structure.
5. If the fix involves changing a function signature, update ALL references
   to that function in the document (imports, calls, descriptions).
6. If the fix involves renaming a symbol, do a global find-replace.
"""

    try:
        _system = (
            "You are a precise architecture editor. Apply the requested fix "
            "and return the complete " + ("section" if _using_section else "document") + ". No commentary."
        )
        _out_budget = min(len(_section_text) // 2 + 2000, 8000)
        response = await llm_call(
            provider_id=provider,
            model_id=model,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_system,
            max_tokens=_out_budget,
            timeout_seconds=300,
        )

        patched_section = response.content if response else None

        if patched_section and len(patched_section) > len(_section_text) * 0.3:
            # Strip any wrapping markdown fences the LLM might add
            patched_section = patched_section.strip()
            if patched_section.startswith("```") and patched_section.endswith("```"):
                first_nl = patched_section.index("\n") + 1
                patched_section = patched_section[first_nl:-3].strip()

            # Reassemble full document if we used sectional approach
            if _using_section:
                parts = []
                if _prefix:
                    parts.append(_prefix)
                parts.append(patched_section)
                if _suffix:
                    parts.append(_suffix)
                patched = "\n".join(parts)
            else:
                patched = patched_section

            issue.auto_fix_note = f"Tier 2: LLM micro-patch ({provider}/{model})"
            logger.info(
                "[cohesion_auto_fix] Tier 2 fix applied for %s in %s (%d→%d chars, sectional=%s)",
                issue.issue_id, seg_id, len(arch_text), len(patched), _using_section,
            )
            return patched
        else:
            logger.warning(
                "[cohesion_auto_fix] Tier 2 LLM response too short/empty for %s",
                issue.issue_id,
            )
            return None

    except Exception as e:
        logger.warning("[cohesion_auto_fix] Tier 2 LLM call failed: %s", e)
        return None

def save_cohesion_result(result: CohesionResult, job_dir: str) -> str:
    """Save cohesion result to disk alongside the manifest."""
    from ._cohesion_check_utils_6 import CohesionResult
    segments_dir = os.path.join(job_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    path = os.path.join(segments_dir, "cohesion_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("[cohesion_check] Saved: %s", path)
    return path

def load_cohesion_result(job_dir: str) -> Optional[CohesionResult]:
    """Load cohesion result from disk. Returns None if not found."""
    from ._cohesion_check_utils_6 import CohesionResult
    path = os.path.join(job_dir, "segments", "cohesion_check.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return CohesionResult.from_dict(json.load(f))
    except Exception as e:
        logger.warning("[cohesion_check] Failed to load: %s", e)
        return None

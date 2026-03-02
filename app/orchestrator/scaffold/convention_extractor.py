# FILE: app/orchestrator/scaffold/convention_extractor.py
"""
Convention Extractor for the Scaffold Engine.

Reads an existing codebase file (the convention reference) and extracts
reusable patterns using Python AST parsing and TypeScript regex analysis.

Extracted once per job, cached, shared across all files in the job.
Falls back to hardcoded minimal conventions derived from
foundation_templates.py when no reference file is available.

v1.0 (2026-03-01): Initial implementation.
"""
from __future__ import annotations

import ast
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from app.orchestrator.scaffold.models import (
    ConventionProfile,
    FileLanguage,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def extract_conventions(
    reference_content: str,
    reference_path: str,
    language: FileLanguage,
) -> ConventionProfile:
    """Extract conventions from a codebase reference file.

    Args:
        reference_content: Full text content of the reference file.
        reference_path: Path to the reference file (for identification).
        language: Language of the reference file.

    Returns:
        A populated ConventionProfile.
    """
    profile = ConventionProfile(
        source_file=reference_path,
        language=language,
    )

    if language == FileLanguage.PYTHON:
        _extract_python_conventions(reference_content, profile)
    elif language == FileLanguage.TYPESCRIPT:
        _extract_typescript_conventions(reference_content, profile)
    else:
        logger.warning(
            "[convention_extractor] Unknown language for %s — using defaults",
            reference_path,
        )
        return get_fallback_conventions(language)

    if not profile.is_valid():
        logger.warning(
            "[convention_extractor] Extracted profile is empty for %s — using fallback",
            reference_path,
        )
        return get_fallback_conventions(language)

    logger.info(
        "[convention_extractor] Extracted conventions from %s (lang=%s)",
        reference_path, language.value,
    )
    return profile


# =============================================================================
# PYTHON CONVENTION EXTRACTION (AST-based)
# =============================================================================


def _extract_python_conventions(content: str, profile: ConventionProfile) -> None:
    """Extract Python conventions using AST parsing + regex."""
    lines = content.splitlines()

    # --- Import block extraction ---
    profile.import_order = _extract_python_import_blocks(content)

    # --- SPEC header style ---
    for line in lines[:5]:
        if line.startswith("# SPEC_ID:") or line.startswith("# FILE:"):
            profile.spec_header_style = line.split(":")[0] + ":"
            break

    # --- Logger pattern ---
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("logger") and "getLogger" in stripped:
            profile.logger_pattern = stripped
            break

    # --- Router/APIRouter pattern ---
    _extract_router_pattern(lines, profile)

    # --- Auth pattern ---
    for line in lines:
        if "Depends(require_auth)" in line:
            profile.auth_pattern = "dependencies=[Depends(require_auth)]"
            break

    # --- DB session pattern ---
    for line in lines:
        if "Depends(get_db)" in line:
            profile.db_pattern = "db: Session = Depends(get_db)"
            break

    # --- Error handling pattern ---
    _extract_error_pattern(lines, profile)

    # --- Pydantic Config pattern ---
    _extract_pydantic_config(content, profile)

    # --- Docstring style ---
    profile.docstring_style = _detect_docstring_style(content)

    # --- Section separator ---
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#\s*[═=─-]{10,}", stripped):
            profile.section_separator = stripped
            break


def _extract_python_import_blocks(content: str) -> List[str]:
    """Extract grouped import blocks preserving ordering.

    Returns a list of import block strings, where each block is a
    group of imports separated by blank lines. This preserves the
    stdlib → third_party → local ordering convention.
    """
    lines = content.splitlines()
    blocks: List[str] = []
    current_block: List[str] = []
    in_imports = False

    in_docstring = False
    for line in lines:
        stripped = line.strip()

        # Skip multi-line docstrings (triple-quoted blocks)
        if not in_imports:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Toggle docstring state. Single-line docstrings
                # like '"""Short doc."""' open and close on same line.
                quote = stripped[:3]
                if stripped.count(quote) >= 2:
                    continue  # Single-line docstring
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue

        # Skip file header comments
        if stripped.startswith("#") and not in_imports:
            continue

        is_import = stripped.startswith("import ") or stripped.startswith("from ")

        if is_import:
            in_imports = True
            current_block.append(stripped)
        elif in_imports and not stripped:
            # Blank line between import groups
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
        elif in_imports and stripped and not is_import:
            # Non-import line after imports → end of import section
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            break

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


def _extract_router_pattern(lines: List[str], profile: ConventionProfile) -> None:
    """Extract the APIRouter instantiation pattern."""
    collecting = False
    router_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if "APIRouter(" in stripped or "router = APIRouter(" in stripped:
            collecting = True
            router_lines.append(stripped)
            if ")" in stripped:
                collecting = False
                break
            continue
        if collecting:
            router_lines.append(stripped)
            if ")" in stripped:
                break

    if router_lines:
        profile.router_pattern = "\n".join(router_lines)


def _extract_error_pattern(lines: List[str], profile: ConventionProfile) -> None:
    """Extract the first HTTPException usage as the error pattern."""
    for line in lines:
        stripped = line.strip()
        if "raise HTTPException(" in stripped:
            profile.error_pattern = stripped
            break


def _extract_pydantic_config(content: str, profile: ConventionProfile) -> None:
    """Extract Pydantic Config class pattern if present."""
    m = re.search(
        r"class\s+Config\s*:\s*\n((?:\s+.+\n)*)",
        content,
    )
    if m:
        config_body = m.group(0).strip()
        profile.pydantic_config = config_body


def _detect_docstring_style(content: str) -> str:
    """Detect docstring style: google, numpy, or simple."""
    if "Args:" in content and "Returns:" in content:
        return "google"
    if "Parameters\n----------" in content:
        return "numpy"
    return "simple"


# =============================================================================
# TYPESCRIPT CONVENTION EXTRACTION (Regex-based)
# =============================================================================


def _extract_typescript_conventions(
    content: str,
    profile: ConventionProfile,
) -> None:
    """Extract TypeScript conventions using regex analysis."""
    lines = content.splitlines()

    # --- Import blocks ---
    profile.ts_import_order = _extract_ts_import_blocks(content)

    # --- SPEC header style ---
    for line in lines[:3]:
        if line.startswith("// SPEC_ID:") or line.startswith("// FILE:"):
            profile.spec_header_style = line.split(":")[0] + ":"
            break

    # --- Component pattern ---
    _extract_component_pattern(content, profile)

    # --- State pattern (useState) ---
    _extract_state_pattern(lines, profile)

    # --- Effect pattern (useEffect) ---
    _extract_effect_pattern(content, profile)

    # --- Fetch/API pattern ---
    _extract_fetch_pattern(lines, profile)

    # --- Loading pattern ---
    _extract_loading_pattern(content, profile)

    # --- Error pattern ---
    for line in lines:
        if "catch" in line or "error" in line.lower():
            stripped = line.strip()
            if stripped.startswith("} catch") or stripped.startswith("catch"):
                profile.ts_error_pattern = stripped
                break


def _extract_ts_import_blocks(content: str) -> List[str]:
    """Extract TypeScript import blocks preserving grouping."""
    lines = content.splitlines()
    blocks: List[str] = []
    current_block: List[str] = []
    in_imports = False
    in_block_comment = False

    for line in lines:
        stripped = line.strip()

        # Skip block comments (/** ... */)
        if not in_imports:
            if stripped.startswith("/**") or stripped.startswith("/*"):
                if "*/" not in stripped[2:]:
                    in_block_comment = True
                continue
            if in_block_comment:
                if "*/" in stripped:
                    in_block_comment = False
                continue

        # Skip single-line comments
        if stripped.startswith("//") and not in_imports:
            continue
        if stripped.startswith("*") and not in_imports:
            continue

        is_import = stripped.startswith("import ")

        if is_import:
            in_imports = True
            current_block.append(stripped)
        elif in_imports and not stripped:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
        elif in_imports and stripped and not is_import:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            break

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


def _extract_component_pattern(content: str, profile: ConventionProfile) -> None:
    """Extract the component export/declaration pattern."""
    # export function ComponentName() {
    m = re.search(r"export\s+(?:default\s+)?function\s+(\w+)", content)
    if m:
        profile.ts_component_pattern = f"export function {m.group(1)}"
        return

    # export default function ComponentName() {
    m = re.search(r"export\s+default\s+function\s+(\w+)", content)
    if m:
        profile.ts_component_pattern = f"export default function {m.group(1)}"
        return

    # const ComponentName = () => { ... }; export default ComponentName;
    m = re.search(r"const\s+(\w+)\s*=\s*\(", content)
    if m:
        profile.ts_component_pattern = f"const {m.group(1)}"


def _extract_state_pattern(lines: List[str], profile: ConventionProfile) -> None:
    """Extract the first useState declaration as the state pattern.

    Requires '=' in the line to skip import lines like:
        import { useState } from 'react';
    and match actual usage like:
        const [data, setData] = useState<Type>(initial);
    """
    for line in lines:
        stripped = line.strip()
        if "useState" in stripped and "=" in stripped and not stripped.startswith("import "):
            profile.ts_state_pattern = stripped
            break


def _extract_effect_pattern(content: str, profile: ConventionProfile) -> None:
    """Extract the useEffect structure."""
    m = re.search(r"useEffect\(\s*\(\)\s*=>\s*\{", content)
    if m:
        # Grab up to 3 lines of the effect body
        start = m.start()
        snippet = content[start:start + 200]
        effect_lines = snippet.split("\n")[:4]
        profile.ts_effect_pattern = "\n".join(effect_lines)


def _extract_fetch_pattern(lines: List[str], profile: ConventionProfile) -> None:
    """Detect the API fetch style used."""
    for line in lines:
        stripped = line.strip()
        if "await fetch(" in stripped:
            profile.ts_fetch_pattern = "fetch"
            return
        if "axios." in stripped:
            profile.ts_fetch_pattern = "axios"
            return
        if "Api." in stripped or "api." in stripped:
            profile.ts_fetch_pattern = "custom_api"
            return


def _extract_loading_pattern(content: str, profile: ConventionProfile) -> None:
    """Extract the loading/error conditional rendering pattern."""
    # Look for loading state conditional
    m = re.search(
        r"if\s*\(\s*(?:loading|isLoading)\s*\)\s*(?:return|{)",
        content,
    )
    if m:
        start = m.start()
        snippet = content[start:start + 150]
        profile.ts_loading_pattern = snippet.split("\n")[0].strip()
        return

    # JSX conditional: {loading && <...>} or {loading ? <...> : <...>}
    m = re.search(r"\{(?:loading|isLoading)\s*[?&]", content)
    if m:
        profile.ts_loading_pattern = "jsx_conditional"


# =============================================================================
# FALLBACK CONVENTIONS
# =============================================================================


def get_fallback_conventions(language: FileLanguage) -> ConventionProfile:
    """Return hardcoded minimal conventions when no reference is available.

    Derived from ASTRA's existing codebase patterns and the
    foundation_templates.py patterns.
    """
    if language == FileLanguage.PYTHON:
        return _get_python_fallback()
    elif language == FileLanguage.TYPESCRIPT:
        return _get_typescript_fallback()

    return ConventionProfile(language=language)


def _get_python_fallback() -> ConventionProfile:
    """Minimal Python conventions matching ASTRA's existing codebase."""
    return ConventionProfile(
        source_file="<fallback>",
        language=FileLanguage.PYTHON,
        import_order=[
            "from __future__ import annotations\nimport logging",
            "from fastapi import APIRouter, Depends, HTTPException, status\n"
            "from pydantic import BaseModel, Field\n"
            "from sqlalchemy.orm import Session",
        ],
        logger_pattern="logger = logging.getLogger(__name__)",
        router_pattern=(
            "router = APIRouter(\n"
            '    prefix="/{module}",\n'
            '    tags=["{Module}"],\n'
            "    dependencies=[Depends(require_auth)],\n"
            ")"
        ),
        auth_pattern="dependencies=[Depends(require_auth)]",
        db_pattern="db: Session = Depends(get_db)",
        error_pattern='raise HTTPException(status_code=404, detail=str(e))',
        docstring_style="simple",
        pydantic_config="class Config:\n    from_attributes = True",
        spec_header_style="# SPEC_ID:",
        section_separator="# " + "=" * 50,
    )


def _get_typescript_fallback() -> ConventionProfile:
    """Minimal TypeScript conventions matching ASTRA's orb-desktop patterns."""
    return ConventionProfile(
        source_file="<fallback>",
        language=FileLanguage.TYPESCRIPT,
        ts_import_order=[
            "import { useState, useCallback, useEffect } from 'react'",
        ],
        ts_component_pattern="export function ComponentName",
        ts_state_pattern="const [data, setData] = useState<Type | null>(null)",
        ts_effect_pattern="useEffect(() => {\n    // fetch data\n  }, [])",
        ts_fetch_pattern="custom_api",
        ts_loading_pattern="jsx_conditional",
        spec_header_style="// SPEC_ID:",
        section_separator="// " + "-" * 50,
    )


# =============================================================================
# CONVENTION CACHE
# =============================================================================

# Job-level cache: {job_id: {language: ConventionProfile}}
_convention_cache: Dict[str, Dict[FileLanguage, ConventionProfile]] = {}


def get_or_extract_conventions(
    job_id: str,
    reference_content: Optional[str],
    reference_path: str,
    language: FileLanguage,
) -> ConventionProfile:
    """Get cached conventions or extract + cache them.

    Conventions are extracted once per job per language and reused
    for all files in that job.
    """
    if job_id in _convention_cache:
        cached = _convention_cache[job_id].get(language)
        if cached:
            logger.debug(
                "[convention_extractor] Cache hit for job=%s lang=%s",
                job_id, language.value,
            )
            return cached

    if reference_content:
        profile = extract_conventions(reference_content, reference_path, language)
    else:
        logger.info(
            "[convention_extractor] No reference content for job=%s lang=%s — using fallback",
            job_id, language.value,
        )
        profile = get_fallback_conventions(language)

    # Cache it
    if job_id not in _convention_cache:
        _convention_cache[job_id] = {}
    _convention_cache[job_id][language] = profile

    return profile


def clear_convention_cache(job_id: Optional[str] = None) -> None:
    """Clear the convention cache (for testing or job cleanup)."""
    if job_id:
        _convention_cache.pop(job_id, None)
    else:
        _convention_cache.clear()

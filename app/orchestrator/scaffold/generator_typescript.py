# FILE: app/orchestrator/scaffold/generator_typescript.py
"""
TypeScript/TSX Scaffold Generator.

Consumes a ParsedFileSpec and produces a ScaffoldFile for TS/TSX files.
Handles COMPONENT, TYPES, and SERVICE roles.

v1.0 (2026-03-01): Batch 2 — initial TypeScript scaffold generation.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .models import (
    FillDifficulty,
    FillManifest,
    FillMarker,
    FileLanguage,
    FileRole,
    LockedRegion,
    ParsedClass,
    ParsedFileSpec,
    ParsedFunction,
    ParsedImport,
    ParsedTypeAlias,
    ScaffoldFile,
)

logger = logging.getLogger(__name__)


def generate_typescript_scaffold(spec: ParsedFileSpec) -> Optional[ScaffoldFile]:
    """Generate a TypeScript scaffold from a parsed file spec.

    Returns ScaffoldFile or None if not applicable.
    """
    if spec.language != FileLanguage.TYPESCRIPT:
        return None
    if not spec.is_scaffold_eligible():
        return None

    if spec.role == FileRole.TYPES:
        return _generate_types_file(spec)
    elif spec.role == FileRole.COMPONENT:
        return _generate_component_file(spec)
    elif spec.role == FileRole.SERVICE:
        return _generate_service_file(spec)
    else:
        logger.debug(
            "[scaffold_ts] Unsupported role %s for %s", spec.role, spec.file_path,
        )
        return None


# =============================================================================
# TYPES FILES (100% deterministic — zero fills)
# =============================================================================

def _generate_types_file(spec: ParsedFileSpec) -> ScaffoldFile:
    """Generate a TypeScript types/interfaces file — no fills needed."""
    lines: List[str] = []
    locked: List[LockedRegion] = []

    # Imports
    import_start = len(lines) + 1
    for imp in spec.imports:
        lines.append(imp.statement)
    if spec.imports:
        lines.append('')
    import_end = len(lines)
    if spec.imports:
        locked.append(LockedRegion(
            line_start=import_start, line_end=import_end,
            region_type="imports",
        ))

    # Type aliases
    for ta in spec.type_aliases:
        lines.append(f'export type {ta.name} = {ta.definition};')
        lines.append('')

    # Interfaces / classes (rendered as interfaces in TS)
    for cls in spec.classes:
        lines.extend(_render_ts_interface(cls))
        lines.append('')

    # Enums
    for enum in spec.enums:
        lines.append(f'export enum {enum.name} {{')
        for name, val in enum.members:
            lines.append(f'  {name} = {val!r},')
        lines.append('}')
        lines.append('')

    content = '\n'.join(lines)
    manifest = FillManifest(
        file_path=spec.file_path, fills=[], locked_regions=locked,
    )
    return ScaffoldFile(
        file_path=spec.file_path, content=content, manifest=manifest,
        language=FileLanguage.TYPESCRIPT, role=FileRole.TYPES,
    )


# =============================================================================
# COMPONENT FILES (fills at component body/render logic)
# =============================================================================

def _generate_component_file(spec: ParsedFileSpec) -> ScaffoldFile:
    """Generate a React component scaffold with fills at component logic."""
    lines: List[str] = []
    fills: List[FillMarker] = []
    locked: List[LockedRegion] = []
    fill_counter = 0

    # Imports (locked)
    import_start = len(lines) + 1
    for imp in spec.imports:
        lines.append(imp.statement)
    if spec.imports:
        lines.append('')
    import_end = len(lines)
    if spec.imports:
        locked.append(LockedRegion(
            line_start=import_start, line_end=import_end,
            region_type="imports",
        ))

    # Type definitions (locked — interfaces/props)
    for cls in spec.classes:
        type_start = len(lines) + 1
        lines.extend(_render_ts_interface(cls))
        lines.append('')
        type_end = len(lines)
        locked.append(LockedRegion(
            line_start=type_start, line_end=type_end,
            region_type="type_definition",
        ))

    # Component functions
    for func in spec.functions:
        fill_counter += 1
        fill_id = f"FILL_{fill_counter:03d}"

        # Signature (locked)
        sig_start = len(lines) + 1
        sig = _render_ts_function_signature(func, export=True)
        lines.append(sig)
        locked.append(LockedRegion(
            line_start=sig_start, line_end=sig_start,
            region_type="function_signature",
        ))

        # Fill for component body
        fill_start = len(lines) + 1
        lines.append(f'  // LLM_FILL: {fill_id}')
        lines.append(f'  // Implement: {func.body_hint or func.name}')
        lines.append(f'  // Props: {", ".join(func.params)}')
        lines.append(f'  return <div>TODO</div>;')
        lines.append(f'  // FILL_END: {fill_id}')
        fill_end = len(lines)
        lines.append('}')
        lines.append('')

        fills.append(FillMarker(
            id=fill_id,
            location=f"{func.name}:body",
            line_start=fill_start, line_end=fill_end,
            context=func.body_hint or f"Implement {func.name} component",
            max_lines=30,
            inputs_available=func.params,
            return_type="JSX.Element",
            difficulty=FillDifficulty.STANDARD,
        ))

    content = '\n'.join(lines)
    manifest = FillManifest(
        file_path=spec.file_path, fills=fills, locked_regions=locked,
    )
    return ScaffoldFile(
        file_path=spec.file_path, content=content, manifest=manifest,
        language=FileLanguage.TYPESCRIPT, role=FileRole.COMPONENT,
    )


# =============================================================================
# SERVICE FILES (fills at method bodies)
# =============================================================================

def _generate_service_file(spec: ParsedFileSpec) -> ScaffoldFile:
    """Generate a TS service scaffold with fills at function bodies."""
    lines: List[str] = []
    fills: List[FillMarker] = []
    locked: List[LockedRegion] = []
    fill_counter = 0

    # Imports (locked)
    import_start = len(lines) + 1
    for imp in spec.imports:
        lines.append(imp.statement)
    if spec.imports:
        lines.append('')
    import_end = len(lines)
    if spec.imports:
        locked.append(LockedRegion(
            line_start=import_start, line_end=import_end,
            region_type="imports",
        ))

    # Type definitions
    for cls in spec.classes:
        lines.extend(_render_ts_interface(cls))
        lines.append('')

    # Functions
    for func in spec.functions:
        fill_counter += 1
        fill_id = f"FILL_{fill_counter:03d}"

        sig_start = len(lines) + 1
        sig = _render_ts_function_signature(func, export=True)
        lines.append(sig)
        locked.append(LockedRegion(
            line_start=sig_start, line_end=sig_start,
            region_type="function_signature",
        ))

        fill_start = len(lines) + 1
        lines.append(f'  // LLM_FILL: {fill_id}')
        lines.append(f'  // Implement: {func.body_hint or func.name}')
        if func.return_type:
            lines.append(f'  // Returns: {func.return_type}')
        lines.append(f'  throw new Error("Not implemented");')
        lines.append(f'  // FILL_END: {fill_id}')
        fill_end = len(lines)
        lines.append('}')
        lines.append('')

        fills.append(FillMarker(
            id=fill_id,
            location=f"{func.name}:body",
            line_start=fill_start, line_end=fill_end,
            context=func.body_hint or f"Implement {func.name}",
            max_lines=20,
            inputs_available=func.params,
            return_type=func.return_type,
            difficulty=FillDifficulty.STANDARD,
        ))

    content = '\n'.join(lines)
    manifest = FillManifest(
        file_path=spec.file_path, fills=fills, locked_regions=locked,
    )
    return ScaffoldFile(
        file_path=spec.file_path, content=content, manifest=manifest,
        language=FileLanguage.TYPESCRIPT, role=FileRole.SERVICE,
    )


# =============================================================================
# RENDER HELPERS
# =============================================================================

def _render_ts_interface(cls: ParsedClass) -> List[str]:
    """Render a TypeScript interface from a ParsedClass."""
    lines: List[str] = []
    extends = f' extends {", ".join([cls.parent] if cls.parent else "")}' if cls.parent else ''
    lines.append(f'export interface {cls.name}{extends} {{')
    for f in cls.fields:
        optional = '?' if f.is_optional else ''
        lines.append(f'  {f.name}{optional}: {f.type_str};')
    lines.append('}')
    return lines


def _render_ts_function_signature(func: ParsedFunction, export: bool = False) -> str:
    """Render a TypeScript function signature opening line."""
    prefix = "export " if export else ""
    async_prefix = "async " if func.is_async else ""
    params = ", ".join(func.params) if func.params else ""
    ret = f": {func.return_type}" if func.return_type else ""
    return f'{prefix}{async_prefix}function {func.name}({params}){ret} {{'

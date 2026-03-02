# FILE: app/orchestrator/scaffold/generator_python.py
"""
Python Scaffold Generator.

Consumes a ParsedFileSpec and produces a ScaffoldFile with:
- Deterministic imports, class stubs, function signatures
- LLM_FILL markers at function bodies where creative logic is needed
- LockedRegions marking sections the Implementer must not modify

v1.0 (2026-03-01): Batch 2 — handles MODEL, ROUTER, SERVICE roles.
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
    ParsedEnum,
    ParsedField,
    ParsedFileSpec,
    ParsedFunction,
    ParsedImport,
    ScaffoldFile,
)

logger = logging.getLogger(__name__)


def generate_python_scaffold(spec: ParsedFileSpec) -> Optional[ScaffoldFile]:
    """Generate a Python scaffold from a parsed file spec.

    Dispatches to role-specific generators based on spec.role.

    Returns ScaffoldFile or None if scaffolding not applicable.
    """
    if spec.language != FileLanguage.PYTHON:
        return None
    if not spec.is_scaffold_eligible():
        return None

    if spec.role == FileRole.MODEL:
        return _generate_model_file(spec)
    elif spec.role == FileRole.ROUTER:
        return _generate_router_file(spec)
    elif spec.role == FileRole.SERVICE:
        return _generate_service_file(spec)
    elif spec.role == FileRole.CONFIG:
        return _generate_config_file(spec)
    else:
        logger.debug(
            "[scaffold_py] Unsupported role %s for %s", spec.role, spec.file_path,
        )
        return None


# =============================================================================
# MODEL FILES (100% deterministic — zero fills)
# =============================================================================

def _generate_model_file(spec: ParsedFileSpec) -> ScaffoldFile:
    """Generate a complete model file — no LLM fills needed."""
    lines: List[str] = []
    fills: List[FillMarker] = []
    locked: List[LockedRegion] = []

    # Docstring
    lines.append(f'"""')
    lines.append(f'{spec.purpose}')
    lines.append(f'"""')
    lines.append('')

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

    # Constants
    for const in spec.constants:
        lines.append(const)
    if spec.constants:
        lines.append('')

    # Enums
    for enum in spec.enums:
        lines.extend(_render_enum(enum))
        lines.append('')

    # Classes (models)
    for cls in spec.classes:
        lines.extend(_render_model_class(cls))
        lines.append('')
        lines.append('')

    content = '\n'.join(lines)
    manifest = FillManifest(
        file_path=spec.file_path,
        fills=fills,
        locked_regions=locked,
    )

    return ScaffoldFile(
        file_path=spec.file_path,
        content=content,
        manifest=manifest,
        language=FileLanguage.PYTHON,
        role=FileRole.MODEL,
    )


# =============================================================================
# ROUTER FILES (fills at endpoint handler bodies)
# =============================================================================

def _generate_router_file(spec: ParsedFileSpec) -> ScaffoldFile:
    """Generate a router scaffold with fills at endpoint handler bodies."""
    lines: List[str] = []
    fills: List[FillMarker] = []
    locked: List[LockedRegion] = []
    fill_counter = 0

    # Docstring
    lines.append(f'"""')
    lines.append(f'{spec.purpose}')
    lines.append(f'"""')
    lines.append('')

    # Imports (locked)
    import_start = len(lines) + 1
    for imp in spec.imports:
        lines.append(imp.statement)
    lines.append('')
    import_end = len(lines)
    locked.append(LockedRegion(
        line_start=import_start, line_end=import_end,
        region_type="imports",
    ))

    # Logger
    lines.append('logger = logging.getLogger(__name__)')
    lines.append('')

    # Router instantiation
    lines.append('router = APIRouter()')
    lines.append('')
    lines.append('')

    # Endpoint functions
    for func in spec.functions:
        fill_counter += 1
        fill_id = f"FILL_{fill_counter:03d}"

        # Decorator (locked)
        decorator_start = len(lines) + 1
        if func.decorators:
            for dec in func.decorators:
                lines.append(dec)
        decorator_end = len(lines)
        if func.decorators:
            locked.append(LockedRegion(
                line_start=decorator_start, line_end=decorator_end,
                region_type="endpoint_decorator",
            ))

        # Signature (locked)
        sig_line = len(lines) + 1
        sig = _render_function_signature(func)
        lines.append(sig)
        locked.append(LockedRegion(
            line_start=sig_line, line_end=sig_line,
            region_type="function_signature",
        ))

        # Docstring
        if func.docstring:
            lines.append(f'    """{func.docstring}"""')

        # Fill marker
        fill_start = len(lines) + 1
        lines.append(f'    # LLM_FILL: {fill_id}')
        lines.append(f'    # Implement: {func.body_hint or func.name}')
        lines.append(f'    # Inputs: {", ".join(func.params)}')
        if func.return_type:
            lines.append(f'    # Returns: {func.return_type}')
        lines.append(f'    pass  # FILL_END: {fill_id}')
        fill_end = len(lines)

        fills.append(FillMarker(
            id=fill_id,
            location=f"{func.name}:body",
            line_start=fill_start,
            line_end=fill_end,
            context=func.body_hint or f"Implement {func.name}",
            max_lines=15,
            inputs_available=func.params,
            return_type=func.return_type,
            difficulty=_estimate_difficulty(func),
        ))
        lines.append('')
        lines.append('')

    content = '\n'.join(lines)
    manifest = FillManifest(
        file_path=spec.file_path, fills=fills, locked_regions=locked,
    )
    return ScaffoldFile(
        file_path=spec.file_path, content=content, manifest=manifest,
        language=FileLanguage.PYTHON, role=FileRole.ROUTER,
    )


# =============================================================================
# SERVICE FILES (fills at method bodies)
# =============================================================================

def _generate_service_file(spec: ParsedFileSpec) -> ScaffoldFile:
    """Generate a service scaffold with fills at method bodies."""
    lines: List[str] = []
    fills: List[FillMarker] = []
    locked: List[LockedRegion] = []
    fill_counter = 0

    # Docstring + imports (same pattern as router)
    lines.append(f'"""')
    lines.append(f'{spec.purpose}')
    lines.append(f'"""')
    lines.append('')

    import_start = len(lines) + 1
    for imp in spec.imports:
        lines.append(imp.statement)
    lines.append('')
    import_end = len(lines)
    locked.append(LockedRegion(
        line_start=import_start, line_end=import_end,
        region_type="imports",
    ))

    lines.append('logger = logging.getLogger(__name__)')
    lines.append('')
    lines.append('')

    # Classes with method fills
    for cls in spec.classes:
        class_start = len(lines) + 1
        lines.append(f'class {cls.name}({", ".join([cls.parent] if cls.parent else [])}):')
        # ParsedClass has no docstring field

        lines.append('')

        for method in cls.methods:
            fill_counter += 1
            fill_id = f"FILL_{fill_counter:03d}"

            sig = _render_method_signature(method, is_first=(method == cls.methods[0]))
            lines.append(f'    {sig}')

            if method.docstring:
                lines.append(f'        """{method.docstring}"""')

            fill_start = len(lines) + 1
            lines.append(f'        # LLM_FILL: {fill_id}')
            lines.append(f'        # Implement: {method.body_hint or method.name}')
            lines.append(f'        pass  # FILL_END: {fill_id}')
            fill_end = len(lines)

            fills.append(FillMarker(
                id=fill_id,
                location=f"{cls.name}.{method.name}:body",
                line_start=fill_start, line_end=fill_end,
                context=method.body_hint or f"Implement {cls.name}.{method.name}",
                max_lines=15,
                inputs_available=method.params,
                return_type=method.return_type,
                difficulty=_estimate_difficulty(method),
            ))
            lines.append('')

        lines.append('')

    # Standalone functions
    for func in spec.functions:
        fill_counter += 1
        fill_id = f"FILL_{fill_counter:03d}"

        sig = _render_function_signature(func)
        lines.append(sig)
        if func.docstring:
            lines.append(f'    """{func.docstring}"""')

        fill_start = len(lines) + 1
        lines.append(f'    # LLM_FILL: {fill_id}')
        lines.append(f'    # Implement: {func.body_hint or func.name}')
        lines.append(f'    pass  # FILL_END: {fill_id}')
        fill_end = len(lines)

        fills.append(FillMarker(
            id=fill_id,
            location=f"{func.name}:body",
            line_start=fill_start, line_end=fill_end,
            context=func.body_hint or f"Implement {func.name}",
            max_lines=15,
            inputs_available=func.params,
            return_type=func.return_type,
            difficulty=_estimate_difficulty(func),
        ))
        lines.append('')
        lines.append('')

    content = '\n'.join(lines)
    manifest = FillManifest(
        file_path=spec.file_path, fills=fills, locked_regions=locked,
    )
    return ScaffoldFile(
        file_path=spec.file_path, content=content, manifest=manifest,
        language=FileLanguage.PYTHON, role=FileRole.SERVICE,
    )


# =============================================================================
# CONFIG FILES (100% deterministic)
# =============================================================================

def _generate_config_file(spec: ParsedFileSpec) -> ScaffoldFile:
    """Generate config file — same as model, zero fills."""
    return _generate_model_file(spec)


# =============================================================================
# RENDER HELPERS
# =============================================================================

def _render_function_signature(func: ParsedFunction) -> str:
    """Render a Python function signature line."""
    prefix = "async " if func.is_async else ""
    params = ", ".join(func.params) if func.params else ""
    ret = f" -> {func.return_type}" if func.return_type else ""
    return f"{prefix}def {func.name}({params}){ret}:"


def _render_method_signature(func: ParsedFunction, is_first: bool = False) -> str:
    """Render a Python method signature line (indented)."""
    prefix = "async " if func.is_async else ""
    params = ", ".join(func.params) if func.params else "self"
    if params and not params.startswith("self") and not params.startswith("cls"):
        params = "self, " + params
    ret = f" -> {func.return_type}" if func.return_type else ""
    return f"{prefix}def {func.name}({params}){ret}:"


def _render_model_class(cls: ParsedClass) -> List[str]:
    """Render a model class (SQLAlchemy or Pydantic dataclass)."""
    lines: List[str] = []
    bases = ", ".join([cls.parent] if cls.parent else [])
    lines.append(f'class {cls.name}({bases}):')
    # ParsedClass has no docstring field

    for f in cls.fields:
        line = _render_field(f)
        lines.append(f'    {line}')

    if not cls.fields and not cls.methods:
        lines.append('    pass')

    return lines


def _render_field(f: ParsedField) -> str:
    """Render a single field definition."""
    if f.default:
        return f'{f.name}: {f.type_str} = {f.default}'
    return f'{f.name}: {f.type_str}'


def _render_enum(enum: ParsedEnum) -> List[str]:
    """Render an enum class."""
    lines: List[str] = []
    lines.append(f'class {enum.name}({enum.parent or "str, Enum"}):')
    for name, val in enum.members:
        lines.append(f'    {name} = {val!r}')
    if not enum.members:
        lines.append('    pass')
    return lines


def _estimate_difficulty(func: ParsedFunction) -> FillDifficulty:
    """Estimate fill difficulty from function metadata."""
    purpose = (func.body_hint or "").lower()
    if any(w in purpose for w in ["simple", "crud", "get", "list", "delete"]):
        return FillDifficulty.TRIVIAL
    if any(w in purpose for w in ["transform", "algorithm", "calculate", "complex"]):
        return FillDifficulty.COMPLEX
    return FillDifficulty.STANDARD

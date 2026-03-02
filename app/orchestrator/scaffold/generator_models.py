# FILE: app/orchestrator/scaffold/generator_models.py
"""
Model-Specific Generator Enhancements.

Extends the base Python generator with SQLAlchemy and Pydantic-specific
patterns for deterministic model generation (zero LLM fills).

v1.0 (2026-03-01): Batch 2 — SQLAlchemy Column rendering, Pydantic
Field rendering, relationship rendering.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .models import (
    ParsedClass,
    ParsedField,
    ParsedFileSpec,
    FileRole,
)

logger = logging.getLogger(__name__)


def render_sqlalchemy_class(cls: ParsedClass) -> List[str]:
    """Render a SQLAlchemy model class with proper Column definitions."""
    lines: List[str] = []
    bases = cls.parent or "Base"
    lines.append(f'class {cls.name}({bases}):')
    if getattr(cls, "docstring", ""):
        lines.append(f'    """{getattr(cls, "docstring", "")}"""')

    # Table name
    table_name = _class_to_table_name(cls.name)
    lines.append(f'    __tablename__ = "{table_name}"')
    lines.append('')

    # Columns
    for f in cls.fields:
        col_line = _render_sqlalchemy_column(f)
        lines.append(f'    {col_line}')

    # Relationships (from class.design_notes or parsed metadata)
    for method in cls.methods:
        if method.name.startswith('_') and 'relationship' in method.body_hint.lower():
            lines.append(f'    {method.name} = relationship("{method.return_type}")')

    if not cls.fields and not cls.methods:
        lines.append('    pass')

    return lines


def render_pydantic_class(cls: ParsedClass) -> List[str]:
    """Render a Pydantic model/schema class."""
    lines: List[str] = []
    bases = cls.parent or "BaseModel"
    lines.append(f'class {cls.name}({bases}):')
    if getattr(cls, "docstring", ""):
        lines.append(f'    """{getattr(cls, "docstring", "")}"""')

    for f in cls.fields:
        line = _render_pydantic_field(f)
        lines.append(f'    {line}')

    # Config class if needed
    if cls.parent and 'BaseModel' in cls.parent:
        lines.append('')
        lines.append('    class Config:')
        lines.append('        from_attributes = True')

    if not cls.fields:
        lines.append('    pass')

    return lines


def detect_model_framework(spec: ParsedFileSpec) -> str:
    """Detect whether a model file uses SQLAlchemy or Pydantic.

    Returns 'sqlalchemy', 'pydantic', or 'unknown'.
    """
    for imp in spec.imports:
        if 'sqlalchemy' in imp.module:
            return 'sqlalchemy'
        if 'pydantic' in imp.module:
            return 'pydantic'

    # Check class bases
    for cls in spec.classes:
        for base in ([cls.parent] if cls.parent else []):
            if base in ('Base', 'DeclarativeBase'):
                return 'sqlalchemy'
            if base in ('BaseModel', 'BaseSchema'):
                return 'pydantic'

    return 'unknown'


# =============================================================================
# HELPERS
# =============================================================================

def _class_to_table_name(class_name: str) -> str:
    """Convert CamelCase class name to snake_case table name.

    EducationCourse -> education_courses
    """
    import re
    # Insert underscores before uppercase letters
    s = re.sub(r'([A-Z])', r'_\1', class_name).strip('_').lower()
    # Pluralise (simple rule: add 's')
    if not s.endswith('s'):
        s += 's'
    return s


def _render_sqlalchemy_column(f: ParsedField) -> str:
    """Render a SQLAlchemy Column definition from a ParsedField."""
    type_map = {
        'str': 'String',
        'int': 'Integer',
        'float': 'Float',
        'bool': 'Boolean',
        'datetime': 'DateTime',
        'date': 'Date',
        'text': 'Text',
        'uuid': 'String(36)',
    }

    sa_type = type_map.get(f.type_str.lower().split('[')[0].strip('optional'), 'String')

    parts = [sa_type]

    if f.is_primary_key:
        parts.append('primary_key=True')
        parts.append('autoincrement=True')
    if f.is_foreign_key and f.default:
        parts.append(f'ForeignKey("{f.default}")')
    if not f.is_optional and not f.is_primary_key:
        parts.append('nullable=False')
    elif f.is_optional:
        parts.append('nullable=True')

    if f.default and not f.is_foreign_key:
        parts.append(f'default={f.default}')

    return f'{f.name} = Column({", ".join(parts)})'


def _render_pydantic_field(f: ParsedField) -> str:
    """Render a Pydantic field definition from a ParsedField."""
    type_str = f.type_str
    if f.is_optional and not type_str.startswith('Optional'):
        type_str = f'Optional[{type_str}]'

    if f.default:
        return f'{f.name}: {type_str} = {f.default}'
    elif f.is_optional:
        return f'{f.name}: {type_str} = None'
    return f'{f.name}: {type_str}'

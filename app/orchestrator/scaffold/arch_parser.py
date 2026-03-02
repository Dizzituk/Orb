# FILE: app/orchestrator/scaffold/arch_parser.py
"""
Architecture Document Parser for the Scaffold Engine.

Extracts structured per-file specifications from approved architecture
markdown. This is deterministic regex/pattern parsing — not LLM
interpretation.

Reuses existing parsing utilities from architecture_executor/parsing.py
for file inventory extraction. Adds deeper extraction of imports,
classes, functions, and types from each file's architecture section.

v1.0 (2026-03-01): Initial implementation.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from app.orchestrator.scaffold.models import (
    FileLanguage,
    FileRole,
    ParsedClass,
    ParsedEnum,
    ParsedField,
    ParsedFileSpec,
    ParsedFunction,
    ParsedImport,
    ParsedTypeAlias,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LANGUAGE & ROLE DETECTION
# =============================================================================

_PYTHON_EXTENSIONS = {".py"}
_TS_EXTENSIONS = {".ts", ".tsx", ".jsx"}

_ROLE_PATTERNS: List[Tuple[str, FileRole]] = [
    # Python roles
    (r"models?\.py$", FileRole.MODEL),
    (r"schemas?\.py$", FileRole.MODEL),
    (r"router\.py$", FileRole.ROUTER),
    (r"api\.py$", FileRole.ROUTER),
    (r"service\.py$", FileRole.SERVICE),
    (r"config\.py$", FileRole.CONFIG),
    (r"constants\.py$", FileRole.CONFIG),
    # TypeScript roles
    (r"types?\.tsx?$", FileRole.TYPES),
    (r"interfaces?\.tsx?$", FileRole.TYPES),
    (r"View\.tsx$", FileRole.COMPONENT),
    (r"Tab\.tsx$", FileRole.COMPONENT),
    (r"Card\.tsx$", FileRole.COMPONENT),
    (r"Panel\.tsx$", FileRole.COMPONENT),
    (r"Modal\.tsx$", FileRole.COMPONENT),
    (r"Dashboard\.tsx$", FileRole.COMPONENT),
    (r"\.tsx$", FileRole.COMPONENT),  # Default .tsx → component
    (r"Api\.ts$", FileRole.SERVICE),
    (r"Service\.ts$", FileRole.SERVICE),
    (r"config\.ts$", FileRole.CONFIG),
]


def detect_language(file_path: str) -> FileLanguage:
    """Detect language from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _PYTHON_EXTENSIONS:
        return FileLanguage.PYTHON
    if ext in _TS_EXTENSIONS:
        return FileLanguage.TYPESCRIPT
    return FileLanguage.UNKNOWN


def detect_role(file_path: str, section_text: str = "") -> FileRole:
    """Detect file role from path patterns and architecture section content.

    Path patterns are checked first (strongest signal). Falls back to
    section content analysis for ambiguous filenames.
    """
    norm = file_path.replace("\\", "/")
    basename = os.path.basename(norm)

    for pattern, role in _ROLE_PATTERNS:
        if re.search(pattern, basename, re.IGNORECASE):
            return role

    # Content-based fallback
    if section_text:
        lower = section_text.lower()
        if any(kw in lower for kw in ["sqlalchemy", "column(", "tablename", "relationship("]):
            return FileRole.MODEL
        if any(kw in lower for kw in ["apirouter", "router.post", "router.get", "@router"]):
            return FileRole.ROUTER
        if any(kw in lower for kw in ["usestate", "useeffect", "jsx", "component"]):
            return FileRole.COMPONENT

    return FileRole.UNKNOWN


# =============================================================================
# IMPORT EXTRACTION
# =============================================================================

# Matches: "from fastapi import APIRouter, Depends" or "import logging"
_IMPORT_LINE_RE = re.compile(
    r"^(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w.]+))",
    re.MULTILINE,
)

# Matches imports inside ```python ... ``` blocks
_CODE_BLOCK_RE = re.compile(r"```(?:python|typescript|tsx?)?\s*\n(.*?)```", re.DOTALL)

# Category classification for Python imports
_STDLIB_MODULES = frozenset({
    "abc", "ast", "asyncio", "base64", "collections", "contextlib",
    "copy", "csv", "datetime", "decimal", "enum", "functools",
    "hashlib", "hmac", "http", "importlib", "inspect", "io",
    "itertools", "json", "logging", "math", "os", "pathlib",
    "pickle", "platform", "random", "re", "secrets", "shutil",
    "signal", "socket", "sqlite3", "string", "struct", "subprocess",
    "sys", "tempfile", "textwrap", "threading", "time", "traceback",
    "typing", "unittest", "urllib", "uuid", "warnings",
    "__future__",
})

_THIRD_PARTY_MODULES = frozenset({
    "fastapi", "pydantic", "sqlalchemy", "starlette", "uvicorn",
    "httpx", "aiohttp", "requests", "celery", "redis",
    "alembic", "jwt", "passlib", "bcrypt", "cryptography",
    "PIL", "pillow", "numpy", "pandas",
    "react", "lucide",
})


def _classify_import_category(module: str) -> str:
    """Classify an import's top-level module into stdlib/third_party/local."""
    top = module.split(".")[0].lstrip(".")
    if top in _STDLIB_MODULES:
        return "stdlib"
    if top in _THIRD_PARTY_MODULES:
        return "third_party"
    return "local"


def extract_imports(section_text: str, language: FileLanguage) -> List[ParsedImport]:
    """Extract import statements from an architecture section.

    Searches both inline import references and code blocks.
    """
    imports: List[ParsedImport] = []
    seen: set = set()

    # Gather text from code blocks (higher priority) + raw section
    sources = []
    for m in _CODE_BLOCK_RE.finditer(section_text):
        sources.append(m.group(1))
    sources.append(section_text)

    for source in sources:
        # Join multi-line imports before parsing.
        # Python: "from X import (\n    A,\n    B,\n)" → single line
        # TS: multi-line imports are rare in architecture docs
        joined_lines = _join_multiline_imports(source.splitlines())

        for stripped in joined_lines:
            if not stripped:
                continue

            if language == FileLanguage.PYTHON:
                parsed = _parse_python_import_line(stripped)
            elif language == FileLanguage.TYPESCRIPT:
                parsed = _parse_ts_import_line(stripped)
            else:
                continue

            if parsed and parsed.statement not in seen:
                seen.add(parsed.statement)
                imports.append(parsed)

    return imports


def _join_multiline_imports(lines: List[str]) -> List[str]:
    """Join multi-line import statements into single lines.

    Detects open-paren at end of an import line and collects
    continuation lines until the closing paren.
    """
    result: List[str] = []
    accumulator = ""
    in_multiline = False

    for line in lines:
        stripped = line.strip()

        if in_multiline:
            # Accumulate continuation, strip trailing comma
            cleaned = stripped.rstrip(",").strip()
            if cleaned == ")" or cleaned.endswith(")"):
                # Closing paren — possibly with a last symbol
                final = cleaned.rstrip(")").rstrip(",").strip()
                if final:
                    accumulator += f", {final}"
                result.append(accumulator.strip())
                accumulator = ""
                in_multiline = False
            elif cleaned:
                if accumulator.endswith("("):
                    accumulator = accumulator[:-1].strip() + " " + cleaned
                else:
                    accumulator += f", {cleaned}"
        elif (stripped.startswith("from ") or stripped.startswith("import ")) and stripped.endswith("("):
            # Start of multi-line import
            in_multiline = True
            accumulator = stripped
        else:
            result.append(stripped)

    # If we were still accumulating (malformed), flush it
    if accumulator:
        result.append(accumulator.strip())

    return result


def _parse_python_import_line(line: str) -> Optional[ParsedImport]:
    """Parse a single Python import line.

    Handles both single-line and the first/continuation lines of
    multi-line imports.  Multi-line blocks like:
        from .models import (
            X,
            Y,
        )
    are normalised to a single statement by extract_imports which
    joins continuation lines before calling this function.
    """
    # from X import Y, Z  (or multi-line already joined)
    m = re.match(r"^from\s+([\w.]+)\s+import\s+\(?([\w\s,*]+)\)?", line)
    if m:
        module = m.group(1)
        symbols_raw = m.group(2)
        symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        # Rebuild a clean single-line statement
        clean_stmt = f"from {module} import {', '.join(symbols)}"
        return ParsedImport(
            statement=clean_stmt,
            module=module,
            symbols=symbols,
            is_relative=module.startswith("."),
            category=_classify_import_category(module),
        )

    # import X
    m = re.match(r"^import\s+([\w.]+)", line)
    if m:
        module = m.group(1)
        return ParsedImport(
            statement=line.rstrip(),
            module=module,
            symbols=[],
            is_relative=False,
            category=_classify_import_category(module),
        )

    return None


def _parse_ts_import_line(line: str) -> Optional[ParsedImport]:
    """Parse a single TypeScript import line."""
    # import { X, Y } from 'module';
    m = re.match(r"""^import\s+\{([^}]+)\}\s+from\s+['"]([^'"]+)['"]""", line)
    if m:
        symbols = [s.strip() for s in m.group(1).split(",") if s.strip()]
        module = m.group(2)
        return ParsedImport(
            statement=line.rstrip(";").rstrip(),
            module=module,
            symbols=symbols,
            is_relative=module.startswith("."),
            category="local" if module.startswith(".") else "third_party",
        )

    # import X from 'module';
    m = re.match(r"""^import\s+(\w+)\s+from\s+['"]([^'"]+)['"]""", line)
    if m:
        return ParsedImport(
            statement=line.rstrip(";").rstrip(),
            module=m.group(2),
            symbols=[m.group(1)],
            is_relative=m.group(2).startswith("."),
            category="local" if m.group(2).startswith(".") else "third_party",
        )

    # import * as X from 'module';
    m = re.match(r"""^import\s+\*\s+as\s+(\w+)\s+from\s+['"]([^'"]+)['"]""", line)
    if m:
        return ParsedImport(
            statement=line.rstrip(";").rstrip(),
            module=m.group(2),
            symbols=[m.group(1)],
            is_relative=m.group(2).startswith("."),
            category="local" if m.group(2).startswith(".") else "third_party",
        )

    return None


# =============================================================================
# CLASS/MODEL EXTRACTION
# =============================================================================

# Matches class definitions in architecture text
_CLASS_HEADER_RE = re.compile(
    r"(?:^|\n)\s*class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:",
    re.MULTILINE,
)

# Matches field definitions like: title: str = Field(...)
_FIELD_RE = re.compile(
    r"^\s+(\w+)\s*:\s*(\S+(?:\[.*?\])?)\s*(?:=\s*(.+))?$",
    re.MULTILINE,
)

# Detects SQLAlchemy column patterns
_COLUMN_RE = re.compile(
    r"^\s+(\w+)\s*=\s*Column\((.+)\)$",
    re.MULTILINE,
)

# Detects relationship patterns
_RELATIONSHIP_RE = re.compile(
    r"^\s+(\w+)\s*=\s*relationship\((.+)\)$",
    re.MULTILINE,
)


def extract_classes(
    section_text: str,
    language: FileLanguage,
) -> List[ParsedClass]:
    """Extract class definitions from an architecture section."""
    classes: List[ParsedClass] = []

    # Extract from code blocks first
    code_blocks = _CODE_BLOCK_RE.findall(section_text)
    search_text = "\n".join(code_blocks) if code_blocks else section_text

    for m in _CLASS_HEADER_RE.finditer(search_text):
        name = m.group(1)
        parent = (m.group(2) or "").strip()

        # Find the class body (indented lines after the class header)
        class_start = m.end()
        body_lines = _extract_indented_block(search_text, class_start)
        body_text = "\n".join(body_lines)

        cls = ParsedClass(name=name, parent=parent)

        # Detect class kind
        if parent and ("BaseModel" in parent or "Schema" in parent):
            cls.is_pydantic = True
        elif parent and ("Base" in parent or "Model" in parent):
            cls.is_sqlalchemy = True

        # Extract fields (Pydantic style)
        for fm in _FIELD_RE.finditer(body_text):
            f = ParsedField(
                name=fm.group(1),
                type_str=fm.group(2),
                default=fm.group(3) or "",
                is_optional="Optional" in fm.group(2),
            )
            if "Field(" in f.default:
                pass  # Keep as-is
            cls.fields.append(f)

        # Extract SQLAlchemy columns
        for cm in _COLUMN_RE.finditer(body_text):
            f = ParsedField(
                name=cm.group(1),
                type_str=cm.group(2).split(",")[0].strip(),
                column_kwargs=cm.group(2),
            )
            if "primary_key" in f.column_kwargs.lower():
                f.is_primary_key = True
            if "ForeignKey" in f.column_kwargs:
                f.is_foreign_key = True
                fk_m = re.search(r"""ForeignKey\(['"]([^'"]+)['"]\)""", f.column_kwargs)
                if fk_m:
                    f.foreign_key_target = fk_m.group(1)
            cls.fields.append(f)

        # Extract relationships
        for rm in _RELATIONSHIP_RE.finditer(body_text):
            f = ParsedField(
                name=rm.group(1),
                type_str="relationship",
                column_kwargs=rm.group(2),
            )
            rel_target_m = re.match(r"""['"](\w+)['"]""", rm.group(2).strip())
            if rel_target_m:
                f.relationship_target = rel_target_m.group(1)
            cls.fields.append(f)

        # Extract __tablename__
        tn_m = re.search(r"""__tablename__\s*=\s*['"](\w+)['"]""", body_text)
        if tn_m:
            cls.tablename = tn_m.group(1)

        # Extract Config class
        config_m = re.search(r"class\s+Config\s*:", body_text)
        if config_m:
            config_body = _extract_indented_block(body_text, config_m.end())
            for cl in config_body:
                kv = cl.strip().split("=", 1)
                if len(kv) == 2:
                    cls.config_attrs[kv[0].strip()] = kv[1].strip()

        classes.append(cls)

    return classes


# =============================================================================
# FUNCTION/ENDPOINT EXTRACTION
# =============================================================================

# Matches decorator lines
_DECORATOR_RE = re.compile(r"^(\s*)@([\w.]+\(.*\)|[\w.]+)", re.MULTILINE)

# Matches function definitions — captures async keyword in group 1
_FUNCTION_RE = re.compile(
    r"^(\s*)(async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?\s*:",
    re.MULTILINE,
)

# Matches TypeScript function declarations
_TS_FUNCTION_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*(.+?))?(?:\s*\{)?$",
    re.MULTILINE,
)

# Matches TypeScript arrow functions
_TS_ARROW_RE = re.compile(
    r"^(?:export\s+)?const\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*(.+?))?\s*=>",
    re.MULTILINE,
)


def extract_functions(
    section_text: str,
    language: FileLanguage,
) -> List[ParsedFunction]:
    """Extract function/endpoint definitions from an architecture section."""
    if language == FileLanguage.PYTHON:
        return _extract_python_functions(section_text)
    elif language == FileLanguage.TYPESCRIPT:
        return _extract_ts_functions(section_text)
    return []


def _extract_python_functions(section_text: str) -> List[ParsedFunction]:
    """Extract Python function definitions with their decorators."""
    functions: List[ParsedFunction] = []
    code_blocks = _CODE_BLOCK_RE.findall(section_text)
    search_text = "\n".join(code_blocks) if code_blocks else section_text

    # Collect all decorator positions
    decorator_positions: Dict[int, List[str]] = {}
    for dm in _DECORATOR_RE.finditer(search_text):
        # Map decorator end position to decorator text
        line_num = search_text[:dm.start()].count("\n")
        decorator_positions.setdefault(line_num, []).append(f"@{dm.group(2)}")

    for m in _FUNCTION_RE.finditer(search_text):
        func_line = search_text[:m.start()].count("\n")
        name = m.group(3)
        params_raw = m.group(4).strip()
        return_type = (m.group(5) or "").strip()
        is_async = bool(m.group(2))  # group(2) is 'async ' or None

        params = _split_params(params_raw)

        # Gather decorators from preceding lines
        decorators = []
        for line_offset in range(1, 6):  # Check up to 5 lines above
            check_line = func_line - line_offset
            if check_line in decorator_positions:
                decorators = decorator_positions[check_line] + decorators

        # Extract body hint from docstring or comments after the function
        body_hint = _extract_body_hint(search_text, m.end())

        func = ParsedFunction(
            name=name,
            params=params,
            return_type=return_type,
            decorators=decorators,
            is_async=is_async,
            docstring=body_hint[:200] if body_hint else "",
            body_hint=body_hint,
            max_lines=_estimate_max_lines(body_hint, params),
        )
        functions.append(func)

    return functions


def _extract_ts_functions(section_text: str) -> List[ParsedFunction]:
    """Extract TypeScript function definitions."""
    functions: List[ParsedFunction] = []
    code_blocks = _CODE_BLOCK_RE.findall(section_text)
    search_text = "\n".join(code_blocks) if code_blocks else section_text

    # Named functions
    for m in _TS_FUNCTION_RE.finditer(search_text):
        name = m.group(1)
        params = _split_params(m.group(2).strip())
        return_type = (m.group(3) or "").strip()
        is_async = "async" in search_text[max(0, m.start() - 10):m.start() + 10]

        body_hint = _extract_body_hint(search_text, m.end())
        functions.append(ParsedFunction(
            name=name,
            params=params,
            return_type=return_type,
            is_async=is_async,
            docstring=body_hint[:200] if body_hint else "",
            body_hint=body_hint,
            max_lines=_estimate_max_lines(body_hint, params),
        ))

    # Arrow functions (const handler = async (...) => ...)
    for m in _TS_ARROW_RE.finditer(search_text):
        name = m.group(1)
        params = _split_params(m.group(2).strip())
        return_type = (m.group(3) or "").strip()
        is_async = "async" in search_text[max(0, m.start() - 10):m.start() + 15]

        body_hint = _extract_body_hint(search_text, m.end())
        functions.append(ParsedFunction(
            name=name,
            params=params,
            return_type=return_type,
            is_async=is_async,
            docstring=body_hint[:200] if body_hint else "",
            body_hint=body_hint,
            max_lines=_estimate_max_lines(body_hint, params),
        ))

    return functions


# =============================================================================
# ENUM EXTRACTION
# =============================================================================

_ENUM_RE = re.compile(
    r"class\s+(\w+)\s*\(([^)]*Enum[^)]*)\)\s*:",
    re.MULTILINE,
)

_ENUM_MEMBER_RE = re.compile(
    r"""^\s+(\w+)\s*=\s*['"]([^'"]+)['"]""",
    re.MULTILINE,
)


def extract_enums(section_text: str) -> List[ParsedEnum]:
    """Extract enum definitions from an architecture section."""
    enums: List[ParsedEnum] = []
    code_blocks = _CODE_BLOCK_RE.findall(section_text)
    search_text = "\n".join(code_blocks) if code_blocks else section_text

    for m in _ENUM_RE.finditer(search_text):
        name = m.group(1)
        parent = m.group(2).strip()
        body_lines = _extract_indented_block(search_text, m.end())
        body_text = "\n".join(body_lines)

        members = []
        for mm in _ENUM_MEMBER_RE.finditer(body_text):
            members.append((mm.group(1), mm.group(2)))

        enums.append(ParsedEnum(name=name, parent=parent, members=members))

    return enums


# =============================================================================
# MAIN PARSER: ARCHITECTURE → ParsedFileSpec LIST
# =============================================================================


def parse_architecture(
    architecture_text: str,
) -> List[ParsedFileSpec]:
    """Parse a complete architecture document into per-file specifications.

    Uses the existing parsing.py module for file inventory extraction,
    then applies deeper extraction on each file's architecture section.

    Returns a list of ParsedFileSpec, one per file in the architecture.
    """
    try:
        from app.overwatcher.architecture_executor.parsing import (
            parse_file_inventory,
            extract_section_for_file,
        )
    except ImportError:
        logger.warning("[arch_parser] Cannot import parsing module — returning empty")
        return []

    # Step 1: Get file inventory from architecture
    inventory = parse_file_inventory(architecture_text)
    if not inventory:
        logger.warning("[arch_parser] No files found in architecture inventory")
        return []

    logger.info("[arch_parser] Found %d files in architecture inventory", len(inventory))

    specs: List[ParsedFileSpec] = []

    for file_info in inventory:
        rel_path = file_info.get("path", "")
        operation = file_info.get("operation", "CREATE").upper()
        description = file_info.get("description", "")

        if not rel_path:
            continue

        # Only scaffold CREATE operations (v1 scope)
        if operation != "CREATE":
            logger.debug("[arch_parser] Skipping %s (%s — not CREATE)", rel_path, operation)
            continue

        # Extract the detailed section for this file
        section = extract_section_for_file(architecture_text, rel_path)
        if not section:
            logger.debug("[arch_parser] No architecture section found for %s", rel_path)
            section = ""

        language = detect_language(rel_path)
        role = detect_role(rel_path, section)

        spec = ParsedFileSpec(
            file_path=rel_path,
            operation=operation,
            purpose=description,
            language=language,
            role=role,
            imports=extract_imports(section, language),
            classes=extract_classes(section, language),
            functions=extract_functions(section, language),
            enums=extract_enums(section),
        )

        # Extract design notes (D-001, D-002 style)
        for dm in re.finditer(r"D-\d{3,4}[:\s]+(.+)", section):
            spec.design_notes.append(dm.group(1).strip())

        logger.info(
            "[arch_parser] Parsed %s: lang=%s role=%s imports=%d classes=%d "
            "functions=%d enums=%d",
            rel_path, language.value, role.value,
            len(spec.imports), len(spec.classes),
            len(spec.functions), len(spec.enums),
        )

        specs.append(spec)

    return specs


# =============================================================================
# HELPERS
# =============================================================================


def _extract_indented_block(text: str, start_pos: int) -> List[str]:
    """Extract lines indented relative to start position.

    Returns lines that are more indented than the line at start_pos.
    Stops at the first line with equal or less indentation.
    """
    remaining = text[start_pos:]
    lines = remaining.split("\n")
    block: List[str] = []

    if not lines:
        return block

    # Skip the first line (it's the : line)
    base_indent = None
    for line in lines[1:]:
        if not line.strip():
            block.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent < base_indent:
            break
        block.append(line)

    return block


def _split_params(params_raw: str) -> List[str]:
    """Split a parameter string respecting nested brackets.

    "request: CourseCreateRequest, db: Session = Depends(get_db)"
    → ["request: CourseCreateRequest", "db: Session = Depends(get_db)"]
    """
    if not params_raw.strip():
        return []

    params: List[str] = []
    depth = 0
    current = ""

    for ch in params_raw:
        if ch in "([{":
            depth += 1
            current += ch
        elif ch in ")]}":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            stripped = current.strip()
            if stripped and stripped != "self":
                params.append(stripped)
            current = ""
        else:
            current += ch

    stripped = current.strip()
    if stripped and stripped != "self":
        params.append(stripped)

    return params


def _extract_body_hint(text: str, func_end_pos: int) -> str:
    """Extract the body description/hint after a function definition.

    Looks for docstrings, comments, or description text following
    the function signature.
    """
    remaining = text[func_end_pos:func_end_pos + 500]
    lines = remaining.split("\n")

    hints: List[str] = []
    for line in lines[1:6]:  # Check first 5 lines of body
        stripped = line.strip()
        if not stripped:
            continue
        # Stop at next function/class def
        if stripped.startswith("def ") or stripped.startswith("class "):
            break
        if stripped.startswith("async def "):
            break
        # Collect docstrings and comments
        if stripped.startswith('"""') or stripped.startswith("'''"):
            hints.append(stripped.strip('"\'').strip())
        elif stripped.startswith("#"):
            hints.append(stripped.lstrip("# "))
        elif stripped.startswith("//"):
            hints.append(stripped.lstrip("/ "))
        else:
            hints.append(stripped)

    return " ".join(hints).strip()


def _estimate_max_lines(body_hint: str, params: List[str]) -> int:
    """Estimate maximum fill lines based on description complexity."""
    if not body_hint:
        return 8  # Default

    words = len(body_hint.split())
    param_count = len(params)

    # Simple heuristic: more complex descriptions → more lines
    if words < 10 and param_count <= 2:
        return 5
    if words < 25:
        return 10
    if words < 50:
        return 15
    return 20

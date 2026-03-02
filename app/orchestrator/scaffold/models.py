# FILE: app/orchestrator/scaffold/models.py
"""
Data models for the Scaffold Engine.

Pure dataclasses — no external dependencies, no logic beyond
serialisation. These flow through every scaffold subsystem:
arch_parser → convention_extractor → generators → manifest_writer → validator.

v1.0 (2026-03-01): Initial implementation.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class FileLanguage(str, Enum):
    """Supported scaffold languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


class FileRole(str, Enum):
    """Determines which generator handles the file.

    MODEL files are 100% deterministic (zero fills).
    ROUTER/SERVICE/COMPONENT files have LLM_FILL markers.
    """
    MODEL = "model"              # SQLAlchemy models, Pydantic-only schemas
    ROUTER = "router"            # FastAPI routers / Express routes
    SERVICE = "service"          # Business logic services
    COMPONENT = "component"      # React/TS UI components
    TYPES = "types"              # TypeScript interface/type files (zero fills)
    CONFIG = "config"            # Configuration files (zero fills)
    UNKNOWN = "unknown"          # Fallback — skip scaffolding


class FillDifficulty(str, Enum):
    """Estimated difficulty of a fill for experience tracking."""
    TRIVIAL = "trivial"          # 1-3 lines, simple CRUD
    STANDARD = "standard"        # 4-10 lines, typical logic
    COMPLEX = "complex"          # 10+ lines, algorithms/transformations


# =============================================================================
# ARCHITECTURE PARSING OUTPUTS
# =============================================================================


@dataclass
class ParsedImport:
    """A single import extracted from the architecture document."""
    statement: str               # Full import line: "from fastapi import APIRouter"
    module: str                  # Module path: "fastapi"
    symbols: List[str] = field(default_factory=list)  # ["APIRouter"]
    is_relative: bool = False    # True for "from ..db import get_db"
    category: str = ""           # "stdlib", "third_party", "local"


@dataclass
class ParsedField:
    """A field in a class or interface definition."""
    name: str                    # Field name: "title"
    type_str: str                # Type as string: "str", "Optional[int]"
    default: str = ""            # Default value: "None", "Field(...)"
    is_optional: bool = False
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_target: str = "" # e.g. "education_courses.id"
    relationship_target: str = ""  # e.g. "EducationCourse"
    column_kwargs: str = ""      # Extra Column() kwargs: "unique=True, index=True"


@dataclass
class ParsedFunction:
    """A function/method extracted from the architecture document."""
    name: str                    # Function name
    params: List[str] = field(default_factory=list)  # ["request: CourseCreateRequest", "db: Session"]
    return_type: str = ""        # "dict", "List[CourseResponse]"
    decorators: List[str] = field(default_factory=list)  # ["@router.post('/courses')"]
    is_async: bool = False
    docstring: str = ""          # Brief description
    body_hint: str = ""          # Architecture description of what the body should do
    max_lines: int = 10          # Estimated max lines for the fill


@dataclass
class ParsedClass:
    """A class definition extracted from the architecture document."""
    name: str                    # "EducationCourse", "CourseCreateRequest"
    parent: str = ""             # "Base", "BaseModel"
    fields: List[ParsedField] = field(default_factory=list)
    methods: List[ParsedFunction] = field(default_factory=list)
    is_pydantic: bool = False    # True for BaseModel subclasses
    is_sqlalchemy: bool = False  # True for Base subclasses
    tablename: str = ""          # __tablename__ value
    config_attrs: Dict[str, str] = field(default_factory=dict)  # e.g. {"from_attributes": "True"}


@dataclass
class ParsedEnum:
    """An enum definition extracted from the architecture document."""
    name: str                    # "EducationContentType"
    parent: str = "str, Enum"    # Base class(es)
    members: List[tuple] = field(default_factory=list)  # [("ARTICLE", "article"), ...]


@dataclass
class ParsedTypeAlias:
    """A TypeScript type alias or constant."""
    name: str
    definition: str              # Full definition string


@dataclass
class ParsedFileSpec:
    """Complete structured specification for a single file.

    Language-agnostic — the generators consume this.
    Produced by arch_parser from the architecture document.
    """
    file_path: str               # "app/education/api.py"
    operation: str               # "CREATE" or "MODIFY"
    purpose: str                 # Brief description from architecture
    language: FileLanguage = FileLanguage.UNKNOWN
    role: FileRole = FileRole.UNKNOWN

    imports: List[ParsedImport] = field(default_factory=list)
    classes: List[ParsedClass] = field(default_factory=list)
    functions: List[ParsedFunction] = field(default_factory=list)
    enums: List[ParsedEnum] = field(default_factory=list)
    type_aliases: List[ParsedTypeAlias] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)  # Raw constant lines

    # Architecture design decisions relevant to this file
    design_notes: List[str] = field(default_factory=list)

    def is_scaffold_eligible(self) -> bool:
        """Only CREATE operations with known language/role get scaffolded."""
        if self.operation != "CREATE":
            return False
        if self.language == FileLanguage.UNKNOWN:
            return False
        if self.role == FileRole.UNKNOWN:
            return False
        return True

    def needs_fills(self) -> bool:
        """Model/type/config files have zero fills — entirely deterministic."""
        return self.role not in (FileRole.MODEL, FileRole.TYPES, FileRole.CONFIG)


# =============================================================================
# CONVENTION EXTRACTION OUTPUTS
# =============================================================================


@dataclass
class ConventionProfile:
    """Reusable conventions extracted from a codebase pattern reference.

    Extracted once per job, cached, shared across all files in the job.
    """
    # Identity
    source_file: str = ""        # Which file these conventions came from
    language: FileLanguage = FileLanguage.UNKNOWN

    # Python conventions
    import_order: List[str] = field(default_factory=list)
    # Grouped import blocks as literal strings: ["import logging\nfrom datetime import...", ...]
    logger_pattern: str = ""     # e.g. 'logger = logging.getLogger(__name__)'
    router_pattern: str = ""     # e.g. 'router = APIRouter(prefix=..., ...)'
    auth_pattern: str = ""       # e.g. 'dependencies=[Depends(require_auth)]'
    db_pattern: str = ""         # e.g. 'db: Session = Depends(get_db)'
    error_pattern: str = ""      # e.g. 'raise HTTPException(status_code=404, detail=str(e))'
    docstring_style: str = ""    # "google", "numpy", "simple"
    pydantic_config: str = ""    # e.g. 'class Config:\n    from_attributes = True'

    # TypeScript conventions
    ts_import_order: List[str] = field(default_factory=list)
    ts_state_pattern: str = ""   # useState declaration style
    ts_effect_pattern: str = ""  # useEffect structure
    ts_component_pattern: str = ""  # export function vs export default
    ts_error_pattern: str = ""   # Error boundary / try-catch style
    ts_fetch_pattern: str = ""   # API call style (fetch, axios, custom)
    ts_loading_pattern: str = "" # Loading state rendering pattern

    # Shared
    spec_header_style: str = ""  # SPEC_ID/SPEC_HASH comment format
    section_separator: str = ""  # e.g. "# === Section ===" or "// ---"

    def is_valid(self) -> bool:
        """A convention profile needs at least some patterns to be useful."""
        if self.language == FileLanguage.PYTHON:
            return bool(self.logger_pattern or self.router_pattern)
        if self.language == FileLanguage.TYPESCRIPT:
            return bool(self.ts_component_pattern or self.ts_import_order)
        return False


# =============================================================================
# SCAFFOLD OUTPUTS
# =============================================================================


@dataclass
class FillMarker:
    """A single LLM_FILL marker in a scaffold file."""
    id: str                      # "FILL_001"
    location: str                # "create_course:body"
    line_start: int              # Line number where the fill begins (1-indexed)
    line_end: int                # Line number where the fill ends (1-indexed)
    context: str                 # Description of what to implement
    max_lines: int = 10          # Maximum expected lines for this fill
    inputs_available: List[str] = field(default_factory=list)  # ["request: CourseCreateRequest"]
    return_type: str = ""        # "dict", "List[CourseResponse]"
    difficulty: FillDifficulty = FillDifficulty.STANDARD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "location": self.location,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "context": self.context,
            "max_lines": self.max_lines,
            "inputs_available": self.inputs_available,
            "return_type": self.return_type,
            "difficulty": self.difficulty.value,
        }


@dataclass
class LockedRegion:
    """A region of the scaffold that must not be modified during fill."""
    line_start: int              # 1-indexed
    line_end: int                # 1-indexed, inclusive
    region_type: str             # "imports", "schemas", "endpoint_decorator", etc.
    content_hash: str = ""       # SHA256 of the region content for tamper detection

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_start": self.line_start,
            "line_end": self.line_end,
            "type": self.region_type,
            "content_hash": self.content_hash,
        }


@dataclass
class FillManifest:
    """Sidecar manifest for a scaffold file.

    Describes every fill marker and every locked region.
    Written as .fills.json alongside the scaffold file.
    """
    file_path: str               # Original file path: "app/education/api.py"
    scaffold_hash: str = ""      # SHA256 of the complete scaffold content
    fills: List[FillMarker] = field(default_factory=list)
    locked_regions: List[LockedRegion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "scaffold_hash": self.scaffold_hash,
            "fills": [f.to_dict() for f in self.fills],
            "locked_regions": [r.to_dict() for r in self.locked_regions],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def fill_count(self) -> int:
        return len(self.fills)

    @property
    def has_fills(self) -> bool:
        return len(self.fills) > 0


@dataclass
class ScaffoldFile:
    """A single scaffolded file ready for the Implementer."""
    file_path: str               # "app/education/api.py"
    content: str                 # The actual scaffold source code
    manifest: FillManifest       # Fill markers and locked regions
    language: FileLanguage = FileLanguage.UNKNOWN
    role: FileRole = FileRole.UNKNOWN

    @property
    def fill_count(self) -> int:
        return self.manifest.fill_count

    @property
    def is_complete(self) -> bool:
        """True if file has zero fills (100% deterministic)."""
        return not self.manifest.has_fills

    @property
    def line_count(self) -> int:
        return self.content.count("\n") + 1 if self.content else 0

    def compute_scaffold_hash(self) -> str:
        """Compute and store SHA256 of the scaffold content."""
        h = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        self.manifest.scaffold_hash = h
        return h


@dataclass
class ScaffoldResult:
    """Complete output of the Scaffold Engine for one segment."""
    segment_id: str
    files: List[ScaffoldFile] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)  # Files that couldn't be scaffolded
    warnings: List[str] = field(default_factory=list)
    generation_time_ms: float = 0.0

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_fills(self) -> int:
        return sum(f.fill_count for f in self.files)

    @property
    def total_lines(self) -> int:
        return sum(f.line_count for f in self.files)

    @property
    def complete_files(self) -> int:
        """Files with zero fills (100% deterministic)."""
        return sum(1 for f in self.files if f.is_complete)

    def get_scaffold_for_path(self, file_path: str) -> Optional[ScaffoldFile]:
        """Look up a scaffold by its original file path."""
        norm = file_path.replace("\\", "/")
        for sf in self.files:
            if sf.file_path.replace("\\", "/") == norm:
                return sf
        return None

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "files_scaffolded": self.file_count,
            "files_complete": self.complete_files,
            "files_with_fills": self.file_count - self.complete_files,
            "total_fills": self.total_fills,
            "total_lines": self.total_lines,
            "skipped_files": self.skipped_files,
            "warnings": self.warnings,
            "generation_time_ms": round(self.generation_time_ms, 1),
        }


# =============================================================================
# VALIDATION OUTPUTS
# =============================================================================


class ValidationSeverity(str, Enum):
    """Severity level for scaffold validation issues."""
    ERROR = "error"        # Hard fail — fill is broken
    WARNING = "warning"    # Soft issue — logged but not blocking
    INFO = "info"          # Informational — for experience store


@dataclass
class ValidationIssue:
    """A single issue found during scaffold validation."""
    fill_id: str                 # "FILL_001" or "LOCKED_imports" for region issues
    severity: ValidationSeverity
    message: str
    line_start: int = 0
    line_end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "severity": self.severity.value,
            "message": self.message,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass
class ValidationResult:
    """Output of the Scaffold Validator for a single file."""
    file_path: str
    passed: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "passed": self.passed,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
        }

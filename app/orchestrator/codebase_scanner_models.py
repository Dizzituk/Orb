# FILE: app/orchestrator/codebase_scanner_models.py
# Purpose: Data models for the enhanced codebase scanner.
# Called-by: app.llm.local_tools.zobie.streams._codebase_report_structural, app.orchestrator._codebase_scanner_health, app.orchestrator._codebase_scanner_utils_3, app.orchestrator._deterministic_architecture_utils_2 (+4 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Data models for the enhanced codebase scanner.

Shared between the scanner, segmenter, and any downstream consumers.

BUILD_ID: 2026-02-20-v1.0-codebase-scanner-models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

CODEBASE_SCANNER_MODELS_BUILD_ID = "2026-02-23-v2.0-smart-health-checks"
print(f"[CODEBASE_SCANNER_MODELS_LOADED] BUILD_ID={CODEBASE_SCANNER_MODELS_BUILD_ID}")


# =============================================================================
# SYMBOL TYPES
# =============================================================================

class SymbolKind(Enum):
    """Classification of a symbol extracted from source code."""
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    CONSTANT = "constant"
    DATA_STRUCTURE = "data_structure"


@dataclass
class SymbolInfo:
    """
    Complete information about a single symbol in the source code.

    This is richer than the refactor_segmenter's Symbol — it includes
    the full call graph edges and source code.
    """
    name: str
    kind: SymbolKind
    source_code: str = ""
    signature: str = ""
    docstring: str = ""
    line_start: int = 0
    line_end: int = 0
    is_async: bool = False
    is_private: bool = False
    is_dunder: bool = False
    decorators: List[str] = field(default_factory=list)

    # Call graph edges (populated by reference analysis)
    calls: List[str] = field(default_factory=list)           # functions this symbol calls
    references: List[str] = field(default_factory=list)      # constants/classes this symbol uses
    called_by: List[str] = field(default_factory=list)       # functions that call this symbol
    referenced_by: List[str] = field(default_factory=list)   # functions that use this constant/class

    # For classes
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.source_code)

    @property
    def estimated_lines(self) -> int:
        if self.line_start and self.line_end:
            return self.line_end - self.line_start + 1
        if self.source_code:
            return self.source_code.count("\n") + 1
        return 5

    @property
    def is_dead(self) -> bool:
        """A symbol is dead if nothing calls or references it."""
        return not self.called_by and not self.referenced_by

    @property
    def all_outgoing(self) -> Set[str]:
        """All symbols this one depends on."""
        return set(self.calls) | set(self.references)

    @property
    def all_incoming(self) -> Set[str]:
        """All symbols that depend on this one."""
        return set(self.called_by) | set(self.referenced_by)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "signature": self.signature,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "estimated_lines": self.estimated_lines,
            "is_async": self.is_async,
            "is_private": self.is_private,
            "calls": self.calls,
            "references": self.references,
            "called_by": self.called_by,
            "referenced_by": self.referenced_by,
        }


# =============================================================================
# IMPORT INFO
# =============================================================================

@dataclass
class ImportInfo:
    """A single import statement with parsed details."""
    raw_statement: str
    module: str = ""            # e.g. "os.path", "app.orchestrator.utils"
    names: List[str] = field(default_factory=list)  # e.g. ["join", "exists"]
    is_relative: bool = False
    is_stdlib: bool = False
    is_third_party: bool = False
    is_internal: bool = False   # imports from within the project
    line_number: int = 0
    used_names: List[str] = field(default_factory=list)    # names actually used in code
    unused_names: List[str] = field(default_factory=list)  # names imported but never used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw_statement,
            "module": self.module,
            "names": self.names,
            "is_stdlib": self.is_stdlib,
            "is_internal": self.is_internal,
            "unused_names": self.unused_names,
        }


# =============================================================================
# HEALTH ISSUES
# =============================================================================

class HealthCategory(Enum):
    """Categories of codebase health issues."""
    DEAD_CODE = "dead_code"
    DEAD_IMPORT = "dead_import"
    DUPLICATE_CODE = "duplicate_code"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    ORPHANED_FILE = "orphaned_file"
    UNREACHABLE_CODE = "unreachable_code"
    SHADOWED_BUILTIN = "shadowed_builtin"
    # Legacy size-only checks (kept for backward compat, no longer emitted)
    OVERSIZED_FUNCTION = "oversized_function"
    OVERSIZED_FILE = "oversized_file"
    # v2.0: Smart refactorability checks — flag *structure*, not size
    MULTI_RESPONSIBILITY = "multi_responsibility"        # File mixes unrelated concerns
    EXTRACTABLE_BLOCK = "extractable_block"              # Self-contained block could be its own module
    MONOLITHIC_FUNCTION = "monolithic_function"           # Single huge function that does too many things
    GOD_CLASS = "god_class"                              # Class with too many methods spanning multiple concerns
    TANGLED_DEPENDENCIES = "tangled_dependencies"        # File has high fan-in AND fan-out


class HealthSeverity(Enum):
    """Severity levels for health issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class HealthIssue:
    """A single codebase health issue found by the scanner."""
    category: HealthCategory
    severity: HealthSeverity
    file_path: str
    symbol_name: str = ""
    line_number: int = 0
    description: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "line_number": self.line_number,
            "description": self.description,
            "suggestion": self.suggestion,
        }


# =============================================================================
# FILE SCAN RESULT
# =============================================================================

@dataclass
class FileScanResult:
    """Complete scan result for a single source file."""
    file_path: str
    line_count: int = 0
    char_count: int = 0
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)  # keyed by name
    imports: List[ImportInfo] = field(default_factory=list)
    module_level_code: List[str] = field(default_factory=list)
    health_issues: List[HealthIssue] = field(default_factory=list)
    parse_error: Optional[str] = None

    @property
    def function_count(self) -> int:
        return sum(1 for s in self.symbols.values()
                   if s.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION))

    @property
    def class_count(self) -> int:
        return sum(1 for s in self.symbols.values() if s.kind == SymbolKind.CLASS)

    @property
    def constant_count(self) -> int:
        return sum(1 for s in self.symbols.values()
                   if s.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE))

    @property
    def dead_symbol_count(self) -> int:
        return sum(1 for s in self.symbols.values() if s.is_dead)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_count": self.line_count,
            "char_count": self.char_count,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "constant_count": self.constant_count,
            "dead_symbol_count": self.dead_symbol_count,
            "parse_error": self.parse_error,
            "symbols": {n: s.to_dict() for n, s in self.symbols.items()},
            "imports": [i.to_dict() for i in self.imports],
            "health_issues": [h.to_dict() for h in self.health_issues],
        }


# =============================================================================
# CODEBASE GRAPH
# =============================================================================

@dataclass
class CodebaseGraph:
    """
    The complete codebase graph — the single source of truth.

    Contains every file, every symbol, every edge between symbols,
    and every health issue. This is what the deterministic segmenter,
    architecture generator, and compiler all read from.
    """
    files: Dict[str, FileScanResult] = field(default_factory=dict)  # keyed by file_path
    health_issues: List[HealthIssue] = field(default_factory=list)  # cross-file issues
    scan_errors: List[str] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_symbols(self) -> int:
        return sum(len(f.symbols) for f in self.files.values())

    @property
    def total_functions(self) -> int:
        return sum(f.function_count for f in self.files.values())

    @property
    def total_health_issues(self) -> int:
        file_issues = sum(len(f.health_issues) for f in self.files.values())
        return file_issues + len(self.health_issues)

    def get_symbol(self, name: str) -> Optional[SymbolInfo]:
        """Find a symbol by name across all files."""
        for f in self.files.values():
            if name in f.symbols:
                return f.symbols[name]
        return None

    def get_all_symbols(self) -> Dict[str, SymbolInfo]:
        """Flat dict of all symbols across all files."""
        result: Dict[str, SymbolInfo] = {}
        for f in self.files.values():
            result.update(f.symbols)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_symbols": self.total_symbols,
            "total_functions": self.total_functions,
            "total_health_issues": self.total_health_issues,
            "scan_errors": self.scan_errors,
            "files": {fp: f.to_dict() for fp, f in self.files.items()},
            "cross_file_health_issues": [h.to_dict() for h in self.health_issues],
        }

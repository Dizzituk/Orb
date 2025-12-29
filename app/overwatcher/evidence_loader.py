# FILE: app/overwatcher/evidence_loader.py
"""Evidence Loader: Load zobie_map evidence packs into Overwatcher pipeline.

zobie_map.py (v2.0) generates rich evidence packs containing:
- Repository tree structure
- Symbol index (functions, classes, routes)
- Import graph
- Enum/constant tracking
- Test coverage mapping
- Human review paths

This module loads and parses these artifacts for use in:
- Architecture generation (Stage 4)
- Chunk planning (Block 7)
- Self-modification safety checks
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Default evidence output directory from zobie_map.py
DEFAULT_EVIDENCE_DIR = os.getenv(
    "ORB_EVIDENCE_DIR",
    r"D:\tools\zobie_mapper\output"
)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Symbol:
    """A code symbol (function, class, variable)."""
    name: str
    kind: str  # function, class, variable, const
    line: int
    file_path: str
    signature: str = ""
    docstring: str = ""


@dataclass
class Route:
    """A FastAPI route definition."""
    path: str
    method: str
    handler: str
    handler_line: int
    file_path: str
    decorator_target: str = "app"


@dataclass
class EnumDef:
    """An enum definition with members."""
    name: str
    file_path: str
    line: int
    base: str  # Enum, IntEnum, StrEnum
    members: List[str] = field(default_factory=list)


@dataclass
class DictConstant:
    """A tracked dictionary constant."""
    name: str
    file_path: str
    line: int
    keys: List[str] = field(default_factory=list)
    is_tracked: bool = False


@dataclass
class TestMapping:
    """Source file to test file mapping."""
    source_path: str
    test_paths: List[str] = field(default_factory=list)


@dataclass
class CoChangeHint:
    """Files that typically change together."""
    file_a: str
    file_b: str
    reason: str = ""


@dataclass
class Invariant:
    """A condition that must hold."""
    name: str
    description: str
    check_paths: List[str] = field(default_factory=list)


@dataclass
class EvidencePack:
    """Complete evidence pack from zobie_map.py."""
    
    # Metadata
    generated_at: str = ""
    controller_url: str = ""
    repo_root: str = ""
    
    # File tree
    all_paths: List[str] = field(default_factory=list)
    top_level_counts: Dict[str, int] = field(default_factory=dict)
    
    # Symbols and routes
    symbols: List[Symbol] = field(default_factory=list)
    routes: List[Route] = field(default_factory=list)
    import_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    # Self-modification support (v2.0)
    enums: List[EnumDef] = field(default_factory=list)
    dict_constants: List[DictConstant] = field(default_factory=list)
    test_mappings: List[TestMapping] = field(default_factory=list)
    co_change_hints: List[CoChangeHint] = field(default_factory=list)
    invariants: List[Invariant] = field(default_factory=list)
    human_review_paths: List[str] = field(default_factory=list)
    
    # Raw scanned file data
    scanned_files: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_symbol_by_name(self, name: str) -> Optional[Symbol]:
        """Find symbol by name."""
        for sym in self.symbols:
            if sym.name == name:
                return sym
        return None
    
    def get_symbols_in_file(self, file_path: str) -> List[Symbol]:
        """Get all symbols defined in a file."""
        return [s for s in self.symbols if s.file_path == file_path]
    
    def get_routes_in_file(self, file_path: str) -> List[Route]:
        """Get all routes defined in a file."""
        return [r for r in self.routes if r.file_path == file_path]
    
    def get_importers(self, module_path: str) -> List[str]:
        """Get files that import the given module."""
        importers = []
        for file_path, imports in self.import_graph.items():
            if module_path in imports:
                importers.append(file_path)
        return importers
    
    def get_tests_for_file(self, source_path: str) -> List[str]:
        """Get test files for a source file."""
        for mapping in self.test_mappings:
            if mapping.source_path == source_path:
                return mapping.test_paths
        return []
    
    def is_human_review_required(self, file_path: str) -> bool:
        """Check if file requires human review for modifications."""
        # Normalize path for comparison
        normalized = file_path.replace("\\", "/")
        for hrp in self.human_review_paths:
            if hrp.replace("\\", "/") in normalized or normalized in hrp.replace("\\", "/"):
                return True
        return False
    
    def get_co_change_files(self, file_path: str) -> List[str]:
        """Get files that typically change with this file."""
        related = []
        for hint in self.co_change_hints:
            if hint.file_a == file_path:
                related.append(hint.file_b)
            elif hint.file_b == file_path:
                related.append(hint.file_a)
        return related
    
    def to_architecture_context(self) -> str:
        """Generate architecture context string for LLM consumption."""
        lines = []
        lines.append("# Repository Evidence Summary")
        lines.append("")
        lines.append(f"Generated: {self.generated_at}")
        lines.append(f"Repo root: {self.repo_root}")
        lines.append(f"Total files: {len(self.all_paths)}")
        lines.append("")
        
        # Top-level layout
        lines.append("## Top-level Layout")
        for dir_name, count in sorted(self.top_level_counts.items()):
            lines.append(f"- {dir_name}: {count} files")
        lines.append("")
        
        # Key symbols
        lines.append("## Key Symbols")
        classes = [s for s in self.symbols if s.kind == "class"][:20]
        if classes:
            lines.append("### Classes")
            for sym in classes:
                lines.append(f"- `{sym.name}` ({sym.file_path}:{sym.line})")
        
        functions = [s for s in self.symbols if s.kind == "function"][:30]
        if functions:
            lines.append("### Functions")
            for sym in functions:
                lines.append(f"- `{sym.name}` ({sym.file_path}:{sym.line})")
        lines.append("")
        
        # Routes
        if self.routes:
            lines.append("## API Routes")
            for route in self.routes[:50]:
                lines.append(f"- {route.method.upper()} `{route.path}` -> {route.handler} ({route.file_path})")
            if len(self.routes) > 50:
                lines.append(f"- ... ({len(self.routes)} total)")
        lines.append("")
        
        # Enums
        if self.enums:
            lines.append("## Enums")
            for enum in self.enums[:20]:
                members = ", ".join(enum.members[:5])
                if len(enum.members) > 5:
                    members += f", ... ({len(enum.members)} total)"
                lines.append(f"- `{enum.name}` ({enum.base}): [{members}]")
        lines.append("")
        
        # Human review paths
        if self.human_review_paths:
            lines.append("## Human Review Required")
            lines.append("These paths require human approval for modifications:")
            for path in self.human_review_paths[:20]:
                lines.append(f"- `{path}`")
        
        return "\n".join(lines)


# =============================================================================
# Loader Functions
# =============================================================================

def find_latest_evidence_dir(base_dir: str = DEFAULT_EVIDENCE_DIR) -> Optional[str]:
    """Find the most recent evidence directory.
    
    zobie_map.py creates directories like: output/evidence_YYYYMMDD_HHMMSS/
    
    Returns:
        Path to latest evidence directory, or None
    """
    base = Path(base_dir)
    if not base.exists():
        return None
    
    # Find all evidence directories
    evidence_dirs = []
    for item in base.iterdir():
        if item.is_dir() and item.name.startswith("evidence_"):
            evidence_dirs.append(item)
    
    if not evidence_dirs:
        return None
    
    # Sort by name (timestamp) and return latest
    evidence_dirs.sort(key=lambda x: x.name, reverse=True)
    return str(evidence_dirs[0])


def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[evidence_loader] Failed to load {path}: {e}")
        return None


def load_evidence_pack(
    evidence_dir: Optional[str] = None,
) -> Optional[EvidencePack]:
    """Load complete evidence pack from directory.
    
    Args:
        evidence_dir: Path to evidence directory (auto-discovers latest if None)
    
    Returns:
        EvidencePack or None if loading fails
    """
    if evidence_dir is None:
        evidence_dir = find_latest_evidence_dir()
    
    if evidence_dir is None:
        logger.warning("[evidence_loader] No evidence directory found")
        return None
    
    evidence_path = Path(evidence_dir)
    logger.info(f"[evidence_loader] Loading from {evidence_dir}")
    
    pack = EvidencePack()
    
    # Find index file
    index_files = list(evidence_path.glob("REPO_INDEX_*.json"))
    if index_files:
        index_data = load_json_file(str(index_files[0]))
        if index_data:
            pack.generated_at = index_data.get("generated", "")
            pack.controller_url = index_data.get("controller", "")
            pack.repo_root = index_data.get("repo_root", "")
            pack.top_level_counts = index_data.get("top_level_counts", {})
            pack.scanned_files = index_data.get("scanned_files", [])
    
    # Load tree
    tree_files = list(evidence_path.glob("REPO_TREE_*.json"))
    if tree_files:
        tree_data = load_json_file(str(tree_files[0]))
        if tree_data:
            pack.all_paths = [entry.get("path", "") for entry in tree_data]
    
    # Load symbol index
    symbol_files = list(evidence_path.glob("SYMBOL_INDEX_*.json"))
    if symbol_files:
        symbol_data = load_json_file(str(symbol_files[0]))
        if symbol_data:
            for file_path, file_info in symbol_data.items():
                symbols_list = file_info.get("symbols", [])
                for sym in symbols_list:
                    pack.symbols.append(Symbol(
                        name=sym.get("name", ""),
                        kind=sym.get("kind", ""),
                        line=sym.get("line", 0),
                        file_path=file_path,
                        signature=sym.get("signature", ""),
                        docstring=sym.get("docstring", ""),
                    ))
    
    # Load import graph
    import_files = list(evidence_path.glob("IMPORT_GRAPH_*.json"))
    if import_files:
        import_data = load_json_file(str(import_files[0]))
        if import_data:
            pack.import_graph = import_data
    
    # Load route map
    route_files = list(evidence_path.glob("ROUTE_MAP_*.json"))
    if route_files:
        route_data = load_json_file(str(route_files[0]))
        if route_data and isinstance(route_data, list):
            for route in route_data:
                pack.routes.append(Route(
                    path=route.get("path", ""),
                    method=route.get("method", ""),
                    handler=route.get("handler", ""),
                    handler_line=route.get("handler_line", 0),
                    file_path=route.get("file", ""),
                    decorator_target=route.get("decorator_target", "app"),
                ))
    
    # Load self-modification artifacts (v2.0)
    
    # Enums
    enum_files = list(evidence_path.glob("ENUM_INDEX_*.json"))
    if enum_files:
        enum_data = load_json_file(str(enum_files[0]))
        if enum_data:
            for file_path, enums in enum_data.items():
                for e in enums:
                    pack.enums.append(EnumDef(
                        name=e.get("name", ""),
                        file_path=file_path,
                        line=e.get("line", 0),
                        base=e.get("base", "Enum"),
                        members=e.get("members", []),
                    ))
    
    # Dict constants
    const_files = list(evidence_path.glob("CONST_INDEX_*.json"))
    if const_files:
        const_data = load_json_file(str(const_files[0]))
        if const_data:
            for file_path, consts in const_data.items():
                for c in consts:
                    pack.dict_constants.append(DictConstant(
                        name=c.get("name", ""),
                        file_path=file_path,
                        line=c.get("line", 0),
                        keys=c.get("keys", []),
                        is_tracked=c.get("is_tracked", False),
                    ))
    
    # Test coverage mapping
    test_files = list(evidence_path.glob("TEST_COVERAGE_*.json"))
    if test_files:
        test_data = load_json_file(str(test_files[0]))
        if test_data:
            for source, tests in test_data.items():
                pack.test_mappings.append(TestMapping(
                    source_path=source,
                    test_paths=tests if isinstance(tests, list) else [],
                ))
    
    # Co-change hints
    cochange_files = list(evidence_path.glob("CO_CHANGE_HINTS_*.json"))
    if cochange_files:
        cochange_data = load_json_file(str(cochange_files[0]))
        if cochange_data and isinstance(cochange_data, list):
            for hint in cochange_data:
                pack.co_change_hints.append(CoChangeHint(
                    file_a=hint.get("file_a", ""),
                    file_b=hint.get("file_b", ""),
                    reason=hint.get("reason", ""),
                ))
    
    # Invariants
    invariant_files = list(evidence_path.glob("INVARIANTS_*.json"))
    if invariant_files:
        inv_data = load_json_file(str(invariant_files[0]))
        if inv_data and isinstance(inv_data, list):
            for inv in inv_data:
                pack.invariants.append(Invariant(
                    name=inv.get("name", ""),
                    description=inv.get("description", ""),
                    check_paths=inv.get("check_paths", []),
                ))
    
    # Human review paths
    human_files = list(evidence_path.glob("HUMAN_REVIEW_PATHS_*.json"))
    if human_files:
        human_data = load_json_file(str(human_files[0]))
        if human_data and isinstance(human_data, list):
            pack.human_review_paths = human_data
    
    logger.info(
        f"[evidence_loader] Loaded: {len(pack.all_paths)} paths, "
        f"{len(pack.symbols)} symbols, {len(pack.routes)} routes, "
        f"{len(pack.enums)} enums, {len(pack.human_review_paths)} human review paths"
    )
    
    return pack


def load_scanned_file_content(
    evidence_dir: str,
    file_path: str,
) -> Optional[Dict[str, Any]]:
    """Load scanned file content from evidence pack.
    
    Args:
        evidence_dir: Path to evidence directory
        file_path: Relative file path to look up
    
    Returns:
        Dict with file content and metadata, or None
    """
    index_files = list(Path(evidence_dir).glob("REPO_INDEX_*.json"))
    if not index_files:
        return None
    
    index_data = load_json_file(str(index_files[0]))
    if not index_data:
        return None
    
    scanned = index_data.get("scanned_files", [])
    for sf in scanned:
        if sf.get("path") == file_path:
            return sf
    
    return None


# =============================================================================
# Safety Checks for Self-Modification
# =============================================================================

def check_modification_safety(
    pack: EvidencePack,
    files_to_modify: List[str],
) -> Dict[str, Any]:
    """Check if proposed modifications are safe.
    
    Args:
        pack: Loaded evidence pack
        files_to_modify: List of file paths to be modified
    
    Returns:
        Dict with:
        - requires_human_review: List of files needing approval
        - co_change_warnings: Files that usually change together
        - affected_tests: Tests that should be run
        - invariant_checks: Invariants that may be affected
    """
    result = {
        "requires_human_review": [],
        "co_change_warnings": [],
        "affected_tests": [],
        "invariant_checks": [],
    }
    
    for file_path in files_to_modify:
        # Check human review requirement
        if pack.is_human_review_required(file_path):
            result["requires_human_review"].append(file_path)
        
        # Check co-change hints
        related = pack.get_co_change_files(file_path)
        for r in related:
            if r not in files_to_modify:
                result["co_change_warnings"].append({
                    "file": file_path,
                    "should_also_change": r,
                })
        
        # Get affected tests
        tests = pack.get_tests_for_file(file_path)
        result["affected_tests"].extend(tests)
    
    # Deduplicate tests
    result["affected_tests"] = list(set(result["affected_tests"]))
    
    # Check invariants
    for inv in pack.invariants:
        for check_path in inv.check_paths:
            if check_path in files_to_modify:
                result["invariant_checks"].append({
                    "invariant": inv.name,
                    "description": inv.description,
                    "triggered_by": check_path,
                })
                break
    
    return result


__all__ = [
    # Models
    "Symbol",
    "Route",
    "EnumDef",
    "DictConstant",
    "TestMapping",
    "CoChangeHint",
    "Invariant",
    "EvidencePack",
    # Loaders
    "find_latest_evidence_dir",
    "load_evidence_pack",
    "load_scanned_file_content",
    # Safety
    "check_modification_safety",
]

# FILE: app/orchestrator/segment_enrichment.py
"""
Stage 4B: Segment Enrichment

Runs ONCE per job after segmentation produces the manifest, BEFORE any
Critical Pipeline calls.  For each segment it produces an enrichment
bundle containing:

  Layer 1 — Deterministic Source Extraction (AST parsing, no LLM)
    Parse the monolith with Python ast and extract the exact code that
    belongs to each segment's scope.

  Layer 2 — Deterministic Cross-Segment Symbol Map (no LLM)
    Build export/import/binding maps across all segments and flag
    unresolved references.

  Layer 3 — LLM Intelligence Pass (one call per job)
    A single LLM call producing implementation ordering, per-segment
    design guidance, risk flags, and cross-segment integration notes.
    Also resolves any symbols that couldn't be deterministically assigned
    in Layer 1.

The enrichment bundle flows into build_segment_context() and from there
into the Critical Pipeline prompt so that architectures are designed from
hard evidence rather than prose descriptions.

Failure handling:  Enrichment is NON-FATAL.  If any layer fails the
pipeline continues exactly as it does today.

v1.0 (2026-02-16): Initial implementation
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

BUILD_ID = "2026-02-17-v1.2-multi-stem-bidirectional-match"
print(f"[SEGMENT_ENRICHMENT_LOADED] BUILD_ID={BUILD_ID}")

# =============================================================================
# ENV Configuration
# =============================================================================

ENRICHMENT_PROVIDER = os.getenv("SEGMENT_ENRICHMENT_PROVIDER", "anthropic")
ENRICHMENT_MODEL = os.getenv("SEGMENT_ENRICHMENT_MODEL", "claude-sonnet-4-6")
ENRICHMENT_MAX_TOKENS = int(os.getenv("SEGMENT_ENRICHMENT_MAX_OUTPUT_TOKENS", "16000"))
ENRICHMENT_TIMEOUT = int(os.getenv("SEGMENT_ENRICHMENT_TIMEOUT_SECONDS", "180"))


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SegmentEnrichment:
    """Per-segment enrichment data produced by Stage 4B."""

    # Layer 1: Extracted source code belonging to this segment
    source_extract: Dict[str, str] = field(default_factory=dict)  # {symbol_name: full_code}
    constants: List[Dict[str, str]] = field(default_factory=list)  # [{name, value, line_number}]
    functions: List[Dict[str, str]] = field(default_factory=list)  # [{name, signature, docstring, body}]
    classes: List[Dict[str, str]] = field(default_factory=list)    # [{name, bases, methods, body}]
    imports: List[str] = field(default_factory=list)               # Import statements needed

    # Layer 2: Cross-segment contract
    exports: List[str] = field(default_factory=list)               # Symbols this segment defines
    consumes: Dict[str, List[str]] = field(default_factory=dict)   # {other_segment_id: [symbol_names]}
    consumed_by: Dict[str, List[str]] = field(default_factory=dict)  # {other_segment_id: [symbol_names]}
    unresolved: List[str] = field(default_factory=list)            # Symbols needed but not found anywhere

    # Layer 3: LLM intelligence
    implementation_order: int = 0                # Recommended position (1 = first)
    design_guidance: str = ""                    # Segment-specific advice
    risk_level: str = "low"                      # "low" | "medium" | "high"
    risk_notes: str = ""                         # Why this segment is risky

    # Metadata
    source_file: str = ""                        # Path to the monolith this was extracted from
    extraction_stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_extract": self.source_extract,
            "constants": self.constants,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "exports": self.exports,
            "consumes": self.consumes,
            "consumed_by": self.consumed_by,
            "unresolved": self.unresolved,
            "implementation_order": self.implementation_order,
            "design_guidance": self.design_guidance,
            "risk_level": self.risk_level,
            "risk_notes": self.risk_notes,
            "source_file": self.source_file,
            "extraction_stats": self.extraction_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentEnrichment":
        return cls(
            source_extract=data.get("source_extract", {}),
            constants=data.get("constants", []),
            functions=data.get("functions", []),
            classes=data.get("classes", []),
            imports=data.get("imports", []),
            exports=data.get("exports", []),
            consumes=data.get("consumes", {}),
            consumed_by=data.get("consumed_by", {}),
            unresolved=data.get("unresolved", []),
            implementation_order=data.get("implementation_order", 0),
            design_guidance=data.get("design_guidance", ""),
            risk_level=data.get("risk_level", "low"),
            risk_notes=data.get("risk_notes", ""),
            source_file=data.get("source_file", ""),
            extraction_stats=data.get("extraction_stats", {}),
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def enrich_segments(
    manifest: Any,
    source_evidence: Dict[str, str],
    job_dir_path: str,
    db: Any,
    project_id: int,
) -> Dict[str, Dict]:
    """
    Main entry point for Stage 4B.

    Runs all three layers sequentially and returns a dict of
    {segment_id: enrichment_dict} for every segment in the manifest.

    Args:
        manifest: SegmentManifest instance
        source_evidence: {relative_path: file_content} from _load_source_file_evidence
        job_dir_path: Path to the job directory on disk
        db: SQLAlchemy session for experience memory queries
        project_id: Project ID for experience memory queries

    Returns:
        {segment_id: enrichment_dict} — empty dict if enrichment fails entirely
    """
    if not source_evidence:
        logger.info("[SEGMENT_ENRICHMENT] No source evidence — skipping enrichment")
        return {}

    segments = manifest.segments
    if not segments or len(segments) < 2:
        logger.info("[SEGMENT_ENRICHMENT] < 2 segments — skipping enrichment")
        return {}

    # Identify the primary source file (the monolith being refactored).
    # In a typical refactor there's one large source file that all segments
    # extract from.  If multiple source files exist, process the largest.
    source_path, source_code = _pick_primary_source(source_evidence)
    if not source_code:
        logger.warning("[SEGMENT_ENRICHMENT] No parseable source code found")
        return {}

    # v1.1: Warn about segments with empty file_scope (segmentation quality issue).
    # These segments can't receive any symbol assignments because they own no files.
    _empty_scope_segs = [s.segment_id for s in segments if not s.file_scope]
    if _empty_scope_segs:
        logger.warning(
            "[SEGMENT_ENRICHMENT] v1.1 %d segment(s) have EMPTY file_scope "
            "(segmentation quality issue — cannot assign symbols): %s",
            len(_empty_scope_segs), _empty_scope_segs,
        )
        print(
            f"[SEGMENT_ENRICHMENT] ⚠️ {len(_empty_scope_segs)} segment(s) have "
            f"empty file_scope: {_empty_scope_segs}"
        )

    logger.info(
        "[SEGMENT_ENRICHMENT] Primary source: %s (%d chars)",
        source_path, len(source_code),
    )

    # =====================================================================
    # Layer 1: Deterministic AST extraction
    # =====================================================================
    try:
        all_symbols = _extract_all_symbols(source_code)
    except SyntaxError as e:
        logger.warning("[SEGMENT_ENRICHMENT] AST parse failed: %s", e)
        return {}

    logger.info(
        "[SEGMENT_ENRICHMENT] Layer 1: Extracted %d constants, %d functions, %d classes",
        len(all_symbols["constants"]),
        len(all_symbols["functions"]),
        len(all_symbols["classes"]),
    )

    # =====================================================================
    # Layer 1b: Deterministic assignment — match symbols to segments
    # =====================================================================
    # Step 1: Assign everything we can confidently match from the spec text
    assignments, unassigned = _assign_symbols_deterministic(
        segments, all_symbols, source_path,
    )

    logger.info(
        "[SEGMENT_ENRICHMENT] Layer 1b: %d symbols assigned deterministically, "
        "%d unassigned",
        sum(len(v) for v in assignments.values()),
        len(unassigned),
    )

    # =====================================================================
    # Layer 2: Cross-segment symbol map (from deterministic assignments)
    # =====================================================================
    per_segment_extractions = _build_per_segment_extractions(
        segments, assignments, all_symbols, source_code,
    )
    symbol_map = _build_symbol_map(segments, per_segment_extractions)

    logger.info(
        "[SEGMENT_ENRICHMENT] Layer 2: %d cross-segment bindings, %d unresolved",
        sum(
            len(syms)
            for seg_consumers in symbol_map["consumed_by"].values()
            for syms in seg_consumers.values()
        ),
        len(symbol_map["unresolved"]),
    )

    # =====================================================================
    # Layer 3: LLM intelligence pass (one call per job)
    # Resolves unassigned symbols AND generates ordering/guidance/risk
    # =====================================================================
    experience_patterns = _load_experience_patterns(db, manifest)

    llm_intelligence = await _generate_implementation_intelligence(
        manifest=manifest,
        symbol_map=symbol_map,
        extractions=per_segment_extractions,
        unassigned_symbols=unassigned,
        experience_patterns=experience_patterns,
        source_path=source_path,
    )

    # Apply LLM assignments for previously unassigned symbols
    if llm_intelligence and llm_intelligence.get("symbol_assignments"):
        _apply_llm_assignments(
            llm_intelligence["symbol_assignments"],
            assignments,
            per_segment_extractions,
            all_symbols,
            source_code,
            segments,
        )
        # Rebuild symbol map with newly assigned symbols
        symbol_map = _build_symbol_map(segments, per_segment_extractions)
        logger.info(
            "[SEGMENT_ENRICHMENT] Layer 3: LLM resolved %d additional symbol(s)",
            len(llm_intelligence.get("symbol_assignments", {})),
        )

    # =====================================================================
    # Assemble enrichment bundles per segment
    # =====================================================================
    enrichment_data: Dict[str, Dict] = {}

    for seg in segments:
        seg_id = seg.segment_id
        seg_extract = per_segment_extractions.get(seg_id, {})
        seg_intelligence = {}
        if llm_intelligence and llm_intelligence.get("segments"):
            seg_intelligence = llm_intelligence["segments"].get(seg_id, {})

        enrichment = SegmentEnrichment(
            # Layer 1
            source_extract={
                s["name"]: s.get("body", s.get("value", ""))
                for category in ("constants", "functions", "classes")
                for s in seg_extract.get(category, [])
            },
            constants=seg_extract.get("constants", []),
            functions=seg_extract.get("functions", []),
            classes=seg_extract.get("classes", []),
            imports=seg_extract.get("imports", []),
            # Layer 2
            exports=list(symbol_map["exports"].get(seg_id, set())),
            consumes=symbol_map["consumes"].get(seg_id, {}),
            consumed_by=symbol_map["consumed_by"].get(seg_id, {}),
            unresolved=[
                u for u in symbol_map["unresolved"]
                if u.startswith(f"{seg_id} ")
            ],
            # Layer 3
            implementation_order=seg_intelligence.get("implementation_order", 0),
            design_guidance=seg_intelligence.get("design_guidance", ""),
            risk_level=seg_intelligence.get("risk_level", "low"),
            risk_notes=seg_intelligence.get("risk_notes", ""),
            # Metadata
            source_file=source_path,
            extraction_stats={
                "constants": len(seg_extract.get("constants", [])),
                "functions": len(seg_extract.get("functions", [])),
                "classes": len(seg_extract.get("classes", [])),
                "imports": len(seg_extract.get("imports", [])),
            },
        )

        enrichment_data[seg_id] = enrichment.to_dict()

    # Persist to disk
    _save_enrichment(enrichment_data, job_dir_path)

    logger.info(
        "[SEGMENT_ENRICHMENT] Complete: %d segment(s) enriched",
        len(enrichment_data),
    )
    return enrichment_data


# =============================================================================
# LAYER 1: AST EXTRACTION
# =============================================================================

def _pick_primary_source(
    source_evidence: Dict[str, str],
) -> Tuple[str, str]:
    """
    From the source evidence dict, pick the largest Python file as the
    primary monolith.  Returns (path, content) or ("", "") if none found.
    """
    best_path = ""
    best_content = ""
    for path, content in source_evidence.items():
        if not path.endswith(".py"):
            continue
        if len(content) > len(best_content):
            best_path = path
            best_content = content
    return best_path, best_content


def _extract_all_symbols(source_code: str) -> Dict[str, list]:
    """
    Parse the monolith with Python AST and extract every top-level symbol.

    Returns a dict with keys: constants, functions, classes, imports,
    module_level.  Each value is a list of dicts with symbol metadata.
    """
    tree = ast.parse(source_code)

    result = {
        "constants": [],
        "functions": [],
        "classes": [],
        "imports": [],
        "module_level": [],
    }

    for node in ast.iter_child_nodes(tree):

        # --- Constants: ALL_CAPS assignments ---
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_constant_name(target.id):
                    source_segment = ast.get_source_segment(source_code, node)
                    result["constants"].append({
                        "name": target.id,
                        "value": source_segment or "",
                        "line_number": node.lineno,
                    })

        # --- Functions ---
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_source = ast.get_source_segment(source_code, node)
            result["functions"].append({
                "name": node.name,
                "signature": _get_function_signature(node, source_code),
                "docstring": ast.get_docstring(node) or "",
                "body": func_source or "",
                "line_range": f"{node.lineno}-{node.end_lineno}",
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [
                    ast.get_source_segment(source_code, d) or ""
                    for d in node.decorator_list
                ],
            })

        # --- Classes ---
        elif isinstance(node, ast.ClassDef):
            class_source = ast.get_source_segment(source_code, node)
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
            result["classes"].append({
                "name": node.name,
                "bases": [_get_name(b) for b in node.bases],
                "methods": methods,
                "body": class_source or "",
                "line_range": f"{node.lineno}-{node.end_lineno}",
            })

        # --- Import statements ---
        elif isinstance(node, ast.Import):
            source_segment = ast.get_source_segment(source_code, node)
            if source_segment:
                result["imports"].append(source_segment)

        elif isinstance(node, ast.ImportFrom):
            source_segment = ast.get_source_segment(source_code, node)
            if source_segment:
                result["imports"].append(source_segment)

        # --- Module-level expressions (print(), logger setup, if __name__) ---
        elif isinstance(node, ast.Expr):
            source_segment = ast.get_source_segment(source_code, node)
            if source_segment:
                result["module_level"].append({
                    "code": source_segment,
                    "line_number": node.lineno,
                })

        elif isinstance(node, ast.If):
            # Capture if __name__ == "__main__" blocks
            source_segment = ast.get_source_segment(source_code, node)
            if source_segment and "__name__" in source_segment:
                result["module_level"].append({
                    "code": source_segment,
                    "line_number": node.lineno,
                })

    return result


def _is_constant_name(name: str) -> bool:
    """
    Check if a name looks like a Python constant (ALL_CAPS or ALL_CAPS_WITH_UNDERSCORES).
    Excludes dunder names like __all__.
    """
    if name.startswith("__") and name.endswith("__"):
        return False
    # Must have at least one uppercase letter and contain only uppercase, digits, underscores
    return bool(re.match(r'^[A-Z][A-Z0-9_]*$', name))


def _get_function_signature(node: ast.FunctionDef, source_code: str) -> str:
    """
    Extract the function signature (def line with parameters and return annotation).
    """
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"

    # Build parameter list from AST
    args = node.args
    params = []

    # positional args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        param = arg.arg
        if arg.annotation:
            ann = ast.get_source_segment(source_code, arg.annotation)
            if ann:
                param += f": {ann}"
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            default = ast.get_source_segment(source_code, args.defaults[default_idx])
            if default:
                param += f" = {default}"
        params.append(param)

    # *args
    if args.vararg:
        va = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            ann = ast.get_source_segment(source_code, args.vararg.annotation)
            if ann:
                va += f": {ann}"
        params.append(va)
    elif args.kwonlyargs:
        params.append("*")

    # keyword-only args
    for i, kwarg in enumerate(args.kwonlyargs):
        param = kwarg.arg
        if kwarg.annotation:
            ann = ast.get_source_segment(source_code, kwarg.annotation)
            if ann:
                param += f": {ann}"
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            default = ast.get_source_segment(source_code, args.kw_defaults[i])
            if default:
                param += f" = {default}"
        params.append(param)

    # **kwargs
    if args.kwarg:
        kw = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            ann = ast.get_source_segment(source_code, args.kwarg.annotation)
            if ann:
                kw += f": {ann}"
        params.append(kw)

    sig = f"{prefix} {node.name}({', '.join(params)})"

    if node.returns:
        ret = ast.get_source_segment(source_code, node.returns)
        if ret:
            sig += f" -> {ret}"

    return sig + ":"


def _get_name(node: ast.expr) -> str:
    """Extract a readable name from an AST name/attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        return f"{_get_name(node.value)}[...]"
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    return "?"


# =============================================================================
# LAYER 1b: DETERMINISTIC SYMBOL ASSIGNMENT
# =============================================================================

def _assign_symbols_deterministic(
    segments: list,
    all_symbols: Dict[str, list],
    source_path: str,
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    """
    Assign extracted symbols to segments using deterministic heuristics.

    Strategy (Option C — deterministic first, LLM resolves ambiguity):

    1. CONSTANTS segment: If a segment's file_scope targets a file with
       "constant" in the name (or the segment title says "constants"),
       ALL constants go there.

    2. EXPLICIT NAME MATCH: Scan each segment's title, requirements text,
       and file_scope filename for explicit mentions of function/class names.
       If a symbol name appears in a segment's text, assign it there.

    3. FILENAME HEURISTIC: If a segment targets e.g. "sandbox_helpers.py"
       and a function is named "_sandbox_*" or "sandbox_*", assign it there.

    4. FACADE / __init__ segment: If a segment targets "__init__.py" or
       its title mentions "facade", it gets module-level code and the
       import-time BUILD_ID print statement.

    Anything not matched goes into the unassigned list for the LLM.

    Args:
        segments: List of SegmentSpec objects
        all_symbols: Output from _extract_all_symbols
        source_path: Path to the monolith (for logging)

    Returns:
        (assignments, unassigned) where:
        - assignments = {segment_id: [symbol_name, ...]}
        - unassigned = [{name, type, ...}, ...] for LLM resolution
    """
    assignments: Dict[str, List[str]] = {seg.segment_id: [] for seg in segments}
    assigned_names: Set[str] = set()

    # Build a lookup of what text is associated with each segment
    seg_text_map: Dict[str, str] = {}
    seg_filename_map: Dict[str, List[str]] = {}  # segment_id -> target filename stems
    constants_segment_id: Optional[str] = None
    facade_segment_id: Optional[str] = None

    for seg in segments:
        # Combine all text for matching
        text_parts = [seg.title.lower()]
        text_parts.extend(r.lower() for r in seg.requirements)
        text_parts.extend(ac.lower() for ac in seg.acceptance_criteria)
        seg_text_map[seg.segment_id] = " ".join(text_parts)

        # Extract the target filename stems (e.g. "constants", "sandbox_helpers")
        # v1.2: Collect ALL filename stems for multi-file segments
        seg_filename_map[seg.segment_id] = []
        for fpath in seg.file_scope:
            fname = os.path.basename(fpath).replace(".py", "").lower()
            seg_filename_map[seg.segment_id].append(fname)

            # Detect constants segment
            if "constant" in fname:
                constants_segment_id = seg.segment_id
            # Detect facade / __init__ segment
            if fname == "__init__" or "facade" in fname:
                facade_segment_id = seg.segment_id

        # Also detect from title
        title_lower = seg.title.lower()
        if "constant" in title_lower and not constants_segment_id:
            constants_segment_id = seg.segment_id
        if ("facade" in title_lower or "__init__" in title_lower) and not facade_segment_id:
            facade_segment_id = seg.segment_id

    # --- Pass 1: Constants → constants segment ---
    if constants_segment_id:
        for const in all_symbols["constants"]:
            name = const["name"]
            assignments[constants_segment_id].append(name)
            assigned_names.add(name)
        logger.info(
            "[SEGMENT_ENRICHMENT] Pass 1: %d constants → %s",
            len(all_symbols["constants"]), constants_segment_id,
        )

    # --- Pass 2: Explicit name matching in segment text ---
    for func in all_symbols["functions"]:
        if func["name"] in assigned_names:
            continue
        name_lower = func["name"].lower()
        for seg in segments:
            seg_text = seg_text_map[seg.segment_id]
            # Look for the function name explicitly mentioned in the segment's text
            if name_lower in seg_text or func["name"] in seg_text:
                assignments[seg.segment_id].append(func["name"])
                assigned_names.add(func["name"])
                break

    for cls in all_symbols["classes"]:
        if cls["name"] in assigned_names:
            continue
        name_lower = cls["name"].lower()
        for seg in segments:
            seg_text = seg_text_map[seg.segment_id]
            if name_lower in seg_text or cls["name"] in seg_text:
                assignments[seg.segment_id].append(cls["name"])
                assigned_names.add(cls["name"])
                break

    # --- Pass 3: Filename stem heuristic ---
    # If a function name contains the target filename stem, assign it there.
    # e.g. _resolve_path → path_resolution segment, _sandbox_write → sandbox_helpers
    # v1.2: Iterates ALL filename stems for multi-file segments, and adds
    # bidirectional matching (function words checked against stem too).
    for func in all_symbols["functions"]:
        if func["name"] in assigned_names:
            continue
        name_lower = func["name"].lower().lstrip("_")
        # v1.2: Split function name into words for bidirectional matching
        func_words = [w for w in func["name"].lower().lstrip("_").split("_") if len(w) > 2]
        _matched = False
        for seg_id, fname_stems in seg_filename_map.items():
            if _matched:
                break
            for fname_stem in fname_stems:
                if not fname_stem or fname_stem in ("__init__", "constants"):
                    continue
                # Check if the function name contains significant words from the
                # filename stem.  Split stem on underscores and check each word.
                stem_words = [w for w in fname_stem.split("_") if len(w) > 2]
                if not stem_words:
                    continue
                # Forward match: stem word appears in function name
                fwd_matches = sum(1 for w in stem_words if w in name_lower)
                # v1.2: Reverse match: function word appears in filename stem
                # (e.g. "update" in func matches "updates" in _state_updates)
                rev_matches = sum(
                    1 for fw in func_words
                    if any(fw in sw or sw in fw for sw in stem_words)
                )
                total_matches = max(fwd_matches, rev_matches)
                if total_matches >= 1 and len(stem_words) <= 3:
                    assignments[seg_id].append(func["name"])
                    assigned_names.add(func["name"])
                    _matched = True
                    break
                elif total_matches >= 2:
                    assignments[seg_id].append(func["name"])
                    assigned_names.add(func["name"])
                    _matched = True
                    break

    # --- Pass 4: Module-level code → facade/__init__ segment ---
    if facade_segment_id:
        for ml in all_symbols["module_level"]:
            # BUILD_ID print statements and if __name__ blocks go to facade
            code = ml.get("code", "")
            if "BUILD_ID" in code or "__name__" in code:
                assignments[facade_segment_id].append(f"__module_level_L{ml['line_number']}")
                assigned_names.add(f"__module_level_L{ml['line_number']}")

    # --- Collect unassigned symbols for LLM ---
    unassigned: List[Dict[str, Any]] = []

    for func in all_symbols["functions"]:
        if func["name"] not in assigned_names:
            unassigned.append({
                "name": func["name"],
                "type": "function",
                "signature": func["signature"],
                "docstring": func.get("docstring", "")[:200],
                "line_range": func.get("line_range", ""),
            })

    for cls in all_symbols["classes"]:
        if cls["name"] not in assigned_names:
            unassigned.append({
                "name": cls["name"],
                "type": "class",
                "bases": cls.get("bases", []),
                "methods": cls.get("methods", []),
                "line_range": cls.get("line_range", ""),
            })

    # Constants that weren't assigned (no constants segment found)
    for const in all_symbols["constants"]:
        if const["name"] not in assigned_names:
            unassigned.append({
                "name": const["name"],
                "type": "constant",
                "value_preview": const.get("value", "")[:100],
            })

    logger.info(
        "[SEGMENT_ENRICHMENT] Assignment summary: %s",
        {seg_id: len(names) for seg_id, names in assignments.items() if names},
    )

    return assignments, unassigned


# =============================================================================
# LAYER 1c: BUILD PER-SEGMENT EXTRACTIONS
# =============================================================================

def _build_per_segment_extractions(
    segments: list,
    assignments: Dict[str, List[str]],
    all_symbols: Dict[str, list],
    source_code: str,
) -> Dict[str, Dict]:
    """
    Given symbol assignments, build per-segment extraction dicts containing
    the actual code for each assigned symbol.
    """
    # Build lookup maps by symbol name
    const_by_name = {c["name"]: c for c in all_symbols["constants"]}
    func_by_name = {f["name"]: f for f in all_symbols["functions"]}
    class_by_name = {c["name"]: c for c in all_symbols["classes"]}
    module_level_by_key = {
        f"__module_level_L{ml['line_number']}": ml
        for ml in all_symbols["module_level"]
    }

    # Identify which imports each function/class uses by scanning its body
    # for names that match import statements
    all_import_lines = all_symbols.get("imports", [])

    extractions: Dict[str, Dict] = {}

    for seg in segments:
        seg_id = seg.segment_id
        assigned_names = assignments.get(seg_id, [])

        seg_extract = {
            "constants": [],
            "functions": [],
            "classes": [],
            "imports": [],
            "module_level": [],
        }

        # Collect the symbols
        used_names: Set[str] = set()

        for name in assigned_names:
            if name in const_by_name:
                seg_extract["constants"].append(const_by_name[name])
                used_names.add(name)
            elif name in func_by_name:
                seg_extract["functions"].append(func_by_name[name])
                used_names.add(name)
                # Also add names used within the function body
                body = func_by_name[name].get("body", "")
                for other_name in list(const_by_name.keys()) + list(func_by_name.keys()):
                    if other_name in body:
                        used_names.add(other_name)
            elif name in class_by_name:
                seg_extract["classes"].append(class_by_name[name])
                used_names.add(name)
            elif name in module_level_by_key:
                seg_extract["module_level"].append(module_level_by_key[name])

        # Resolve imports: include any import whose imported names overlap
        # with the names used by this segment's code
        for imp_line in all_import_lines:
            # Extract imported names from the import statement
            imported_names = _extract_names_from_import(imp_line)
            if imported_names & used_names:
                if imp_line not in seg_extract["imports"]:
                    seg_extract["imports"].append(imp_line)

        extractions[seg_id] = seg_extract

    return extractions


def _extract_names_from_import(import_line: str) -> Set[str]:
    """
    Extract the names imported by an import statement.

    Examples:
        "import os" → {"os"}
        "from typing import Dict, List, Optional" → {"Dict", "List", "Optional"}
        "from app.llm.streaming import call_llm_text" → {"call_llm_text"}
        "import json" → {"json"}
    """
    names: Set[str] = set()
    line = import_line.strip()

    if line.startswith("from "):
        # from X import a, b, c
        match = re.search(r'import\s+(.+)', line)
        if match:
            imports_part = match.group(1)
            # Handle multi-line imports (parenthesised)
            imports_part = imports_part.strip("()")
            for part in imports_part.split(","):
                part = part.strip()
                if " as " in part:
                    # "foo as bar" → use the alias "bar"
                    names.add(part.split(" as ")[-1].strip())
                elif part and part != "*":
                    names.add(part.strip())
    elif line.startswith("import "):
        # import X, Y
        imports_part = line[7:]
        for part in imports_part.split(","):
            part = part.strip()
            if " as " in part:
                names.add(part.split(" as ")[-1].strip())
            elif part:
                # "import os.path" → use "os"
                names.add(part.split(".")[0].strip())

    return names


# =============================================================================
# LAYER 2: CROSS-SEGMENT SYMBOL MAP
# =============================================================================

def _build_symbol_map(
    segments: list,
    extractions: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Build cross-segment export/import/binding maps.

    Returns:
        {
            "exports": {segment_id: set(symbol_names)},
            "consumes": {segment_id: {other_segment_id: [symbols]}},
            "consumed_by": {segment_id: {other_segment_id: [symbols]}},
            "unresolved": [description_strings],
        }
    """
    # Step 1: Build exports (what each segment defines)
    exports: Dict[str, Set[str]] = {}
    # Also build a reverse map: symbol_name → segment_id that defines it
    symbol_to_segment: Dict[str, str] = {}

    for seg in segments:
        seg_id = seg.segment_id
        seg_extract = extractions.get(seg_id, {})
        defined: Set[str] = set()

        for c in seg_extract.get("constants", []):
            defined.add(c["name"])
            symbol_to_segment[c["name"]] = seg_id
        for f in seg_extract.get("functions", []):
            defined.add(f["name"])
            symbol_to_segment[f["name"]] = seg_id
        for cl in seg_extract.get("classes", []):
            defined.add(cl["name"])
            symbol_to_segment[cl["name"]] = seg_id

        exports[seg_id] = defined

    # Step 2: Determine what each segment's code references from other segments.
    # Scan each segment's function/class bodies for names defined in other segments.
    consumes: Dict[str, Dict[str, List[str]]] = {
        seg.segment_id: {} for seg in segments
    }
    consumed_by: Dict[str, Dict[str, List[str]]] = {
        seg.segment_id: {} for seg in segments
    }

    for seg in segments:
        seg_id = seg.segment_id
        seg_extract = extractions.get(seg_id, {})
        seg_exports = exports.get(seg_id, set())

        # Collect all code in this segment to scan for cross-references
        all_bodies = []
        for f in seg_extract.get("functions", []):
            all_bodies.append(f.get("body", ""))
        for cl in seg_extract.get("classes", []):
            all_bodies.append(cl.get("body", ""))
        combined_body = "\n".join(all_bodies)

        # Check which symbols from OTHER segments appear in this segment's code
        for other_seg in segments:
            if other_seg.segment_id == seg_id:
                continue
            other_exports = exports.get(other_seg.segment_id, set())
            for sym in other_exports:
                # Only flag if the symbol actually appears in the code body
                # and is NOT also defined in this segment (local override)
                if sym in combined_body and sym not in seg_exports:
                    # This segment consumes sym from other_seg
                    if other_seg.segment_id not in consumes[seg_id]:
                        consumes[seg_id][other_seg.segment_id] = []
                    if sym not in consumes[seg_id][other_seg.segment_id]:
                        consumes[seg_id][other_seg.segment_id].append(sym)

                    # The other segment is consumed by this segment
                    if seg_id not in consumed_by[other_seg.segment_id]:
                        consumed_by[other_seg.segment_id][seg_id] = []
                    if sym not in consumed_by[other_seg.segment_id][seg_id]:
                        consumed_by[other_seg.segment_id][seg_id].append(sym)

    # Step 3: Find unresolved symbols
    # Symbols that appear in a segment's code but aren't defined in ANY segment
    unresolved: List[str] = []
    all_defined = set()
    for exp_set in exports.values():
        all_defined.update(exp_set)

    # Also build a set of all standard library / third-party names to exclude
    # (we don't flag os.path, json.loads, etc. as unresolved)
    _stdlib_names = {
        "os", "sys", "json", "logging", "re", "ast", "hashlib", "uuid",
        "datetime", "pathlib", "typing", "collections", "functools",
        "asyncio", "traceback", "io", "copy", "shutil", "time",
        "Dict", "List", "Optional", "Any", "Tuple", "Set", "Union",
        "Callable", "Sequence", "Mapping",
        "dataclass", "field", "Enum",
        "logger", "print", "len", "str", "int", "float", "bool",
        "True", "False", "None", "self", "cls",
        "Exception", "RuntimeError", "ValueError", "TypeError",
        "KeyError", "AttributeError", "ImportError", "OSError",
        "FileNotFoundError", "IndexError",
    }

    for seg in segments:
        seg_id = seg.segment_id
        seg_extract = extractions.get(seg_id, {})

        # Scan function/class bodies for name references
        for f in seg_extract.get("functions", []):
            body = f.get("body", "")
            # Look for identifiers that look like they could be cross-references
            # (ALL_CAPS names not defined in this segment or any other)
            for match in re.finditer(r'\b([A-Z][A-Z0-9_]{2,})\b', body):
                name = match.group(1)
                if (
                    name not in all_defined
                    and name not in _stdlib_names
                    and name not in exports.get(seg_id, set())
                ):
                    msg = f"{seg_id} needs '{name}' but it is not defined in any segment"
                    if msg not in unresolved:
                        unresolved.append(msg)

    return {
        "exports": exports,
        "consumes": consumes,
        "consumed_by": consumed_by,
        "unresolved": unresolved,
    }


# =============================================================================
# LAYER 3: LLM INTELLIGENCE PASS
# =============================================================================

ENRICHMENT_SYSTEM_PROMPT = """You are a software architecture analyst. You are given:
1. A list of segments that a monolith Python file is being refactored into
2. A symbol map showing what each segment exports and imports from other segments
3. A list of symbols that could not be automatically assigned to any segment
4. Experience patterns from previous refactoring attempts

Your job is to produce implementation intelligence for each segment AND assign
any unresolved symbols to their correct segments.

Respond ONLY in valid JSON — no markdown fences, no preamble, no commentary."""


def _build_enrichment_user_prompt(
    manifest: Any,
    symbol_map: Dict[str, Any],
    extractions: Dict[str, Dict],
    unassigned_symbols: List[Dict[str, Any]],
    experience_patterns: str,
    source_path: str,
) -> str:
    """Build the user prompt for the single LLM intelligence call."""
    parts = []

    # Segments overview
    parts.append("## Segments\n")
    for seg in manifest.segments:
        deps = ", ".join(seg.dependencies) if seg.dependencies else "(none)"
        files = ", ".join(seg.file_scope)
        parts.append(
            f"- **{seg.segment_id}**: {seg.title}\n"
            f"  - Target files: {files}\n"
            f"  - Dependencies: {deps}\n"
        )

    # Exports per segment
    parts.append("\n## Symbol Map\n### Exports per segment:\n")
    for seg in manifest.segments:
        seg_exports = symbol_map["exports"].get(seg.segment_id, set())
        if seg_exports:
            parts.append(f"- **{seg.segment_id}**: {', '.join(sorted(seg_exports))}")
        else:
            parts.append(f"- **{seg.segment_id}**: (no symbols assigned yet)")

    # Cross-segment dependencies
    parts.append("\n### Cross-segment dependencies:\n")
    has_deps = False
    for seg in manifest.segments:
        seg_consumes = symbol_map["consumes"].get(seg.segment_id, {})
        for other_id, symbols in seg_consumes.items():
            parts.append(f"- {seg.segment_id} imports from {other_id}: {', '.join(symbols)}")
            has_deps = True
    if not has_deps:
        parts.append("(none detected yet — will be clearer after symbol assignment)")

    # Unresolved symbols
    parts.append("\n### Unresolved symbols (CRITICAL — these will cause boot failure):\n")
    for u in symbol_map.get("unresolved", []):
        parts.append(f"- {u}")
    if not symbol_map.get("unresolved"):
        parts.append("(none)")

    # Unassigned symbols for LLM to resolve
    if unassigned_symbols:
        parts.append(
            f"\n## Unassigned Symbols ({len(unassigned_symbols)} symbols need assignment)\n"
            "These symbols were extracted from the monolith but could not be "
            "deterministically assigned to any segment.  For each one, decide "
            "which segment it belongs to based on the segment descriptions and "
            "target file names.\n"
        )
        for sym in unassigned_symbols:
            if sym["type"] == "function":
                parts.append(
                    f"- **{sym['name']}** (function): `{sym.get('signature', '')}`\n"
                    f"  Docstring: {sym.get('docstring', '(none)')[:150]}\n"
                    f"  Lines: {sym.get('line_range', '?')}"
                )
            elif sym["type"] == "class":
                parts.append(
                    f"- **{sym['name']}** (class): bases={sym.get('bases', [])}, "
                    f"methods={sym.get('methods', [])}"
                )
            elif sym["type"] == "constant":
                parts.append(
                    f"- **{sym['name']}** (constant): {sym.get('value_preview', '')}"
                )

    # Experience patterns
    if experience_patterns:
        parts.append(f"\n## Experience Patterns (lessons from past runs)\n{experience_patterns}")

    # Instructions
    parts.append("""
## Instructions

Respond in JSON with this exact structure:
{
  "segments": {
    "<segment_id>": {
      "implementation_order": <integer, 1 = implement first>,
      "design_guidance": "<2-3 sentences of specific advice>",
      "risk_level": "<low|medium|high>",
      "risk_notes": "<why this segment is risky, if medium/high>"
    }
  },
  "symbol_assignments": {
    "<symbol_name>": "<segment_id it belongs to>"
  },
  "global_notes": "<any cross-cutting concerns>"
}

Pay special attention to:
- Constants/config modules: EVERY constant must be included (this is the #1 failure mode)
- Facade/init modules: Must re-export exactly the right symbols
- Modules with many cross-segment consumers: High risk if they miss exports
- For symbol_assignments: assign each unassigned symbol to the segment whose TARGET FILE
  will DEFINE (contain) the function implementation — NOT the segment that calls/uses it.
  Example: if 'can_execute_segment' is a dependency-checking helper that will live in
  '_dependencies.py', assign it to the segment targeting '_dependencies.py' even though
  it is called by the main orchestration segment.
  Key principle: each function belongs to the segment responsible for DEFINING it.
  The consuming segment will import it — it does not need to own it.
""")

    return "\n".join(parts)


async def _generate_implementation_intelligence(
    manifest: Any,
    symbol_map: Dict[str, Any],
    extractions: Dict[str, Dict],
    unassigned_symbols: List[Dict[str, Any]],
    experience_patterns: str,
    source_path: str,
) -> Optional[Dict]:
    """
    Single LLM call to produce ordering, guidance, risk flags, and
    resolve unassigned symbols.

    Returns parsed JSON dict or None on failure.
    """
    user_prompt = _build_enrichment_user_prompt(
        manifest, symbol_map, extractions,
        unassigned_symbols, experience_patterns, source_path,
    )

    try:
        from app.llm.streaming import call_llm_text

        raw_response = await call_llm_text(
            provider=ENRICHMENT_PROVIDER,
            model=ENRICHMENT_MODEL,
            system_prompt=ENRICHMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=ENRICHMENT_MAX_TOKENS,
            timeout_seconds=ENRICHMENT_TIMEOUT,
            route="segment_enrichment",
        )

        if not raw_response:
            logger.warning("[SEGMENT_ENRICHMENT] LLM returned empty response")
            return None

        # Clean response: strip markdown fences if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        result = json.loads(cleaned)
        logger.info("[SEGMENT_ENRICHMENT] LLM intelligence parsed successfully")
        return result

    except ImportError:
        logger.warning("[SEGMENT_ENRICHMENT] call_llm_text not available — skipping LLM layer")
        return None
    except json.JSONDecodeError as e:
        logger.warning("[SEGMENT_ENRICHMENT] Failed to parse LLM JSON: %s", e)
        return None
    except Exception as e:
        logger.warning("[SEGMENT_ENRICHMENT] LLM call failed: %s", e)
        return None


def _apply_llm_assignments(
    llm_assignments: Dict[str, str],
    assignments: Dict[str, List[str]],
    per_segment_extractions: Dict[str, Dict],
    all_symbols: Dict[str, list],
    source_code: str,
    segments: list,
) -> None:
    """
    Apply the LLM's symbol assignments to the existing assignment and
    extraction structures.  Mutates in place.
    """
    func_by_name = {f["name"]: f for f in all_symbols["functions"]}
    class_by_name = {c["name"]: c for c in all_symbols["classes"]}
    const_by_name = {c["name"]: c for c in all_symbols["constants"]}
    valid_seg_ids = {seg.segment_id for seg in segments}

    for symbol_name, target_seg_id in llm_assignments.items():
        if target_seg_id not in valid_seg_ids:
            logger.warning(
                "[SEGMENT_ENRICHMENT] LLM assigned '%s' to unknown segment '%s' — skipping",
                symbol_name, target_seg_id,
            )
            continue

        # Add to assignments
        if symbol_name not in assignments.get(target_seg_id, []):
            assignments.setdefault(target_seg_id, []).append(symbol_name)

        # Add to per-segment extraction
        seg_extract = per_segment_extractions.setdefault(target_seg_id, {
            "constants": [], "functions": [], "classes": [], "imports": [], "module_level": [],
        })

        if symbol_name in func_by_name:
            if not any(f["name"] == symbol_name for f in seg_extract["functions"]):
                seg_extract["functions"].append(func_by_name[symbol_name])
        elif symbol_name in class_by_name:
            if not any(c["name"] == symbol_name for c in seg_extract["classes"]):
                seg_extract["classes"].append(class_by_name[symbol_name])
        elif symbol_name in const_by_name:
            if not any(c["name"] == symbol_name for c in seg_extract["constants"]):
                seg_extract["constants"].append(const_by_name[symbol_name])


# =============================================================================
# EXPERIENCE MEMORY INTEGRATION
# =============================================================================

def _load_experience_patterns(db: Any, manifest: Any) -> str:
    """
    Query the experience/learning system for patterns relevant to this job.

    Returns formatted injection text, or empty string if unavailable.
    """
    if db is None:
        return ""

    try:
        from app.experience.retrieval import retrieve_for_stage, format_injection

        spec_summary = ", ".join(
            seg.title for seg in manifest.segments[:5]
        )
        patterns = retrieve_for_stage(
            db,
            stage="segment_enrichment",
            context=f"Enriching segments for refactor: {spec_summary}",
            job_type="refactor",
            max_results=8,
        )
        if patterns:
            return format_injection(patterns, stage="segment_enrichment")
        return ""

    except ImportError:
        logger.debug("[SEGMENT_ENRICHMENT] Experience retrieval not available")
        return ""
    except Exception as e:
        logger.debug("[SEGMENT_ENRICHMENT] Experience query failed: %s", e)
        return ""


# =============================================================================
# PERSISTENCE
# =============================================================================

def _save_enrichment(
    enrichment: Dict[str, Dict],
    job_dir_path: str,
) -> None:
    """Write enrichment.json per segment and a combined enrichment_summary.json."""
    for seg_id, data in enrichment.items():
        seg_dir = os.path.join(job_dir_path, "segments", seg_id)
        os.makedirs(seg_dir, exist_ok=True)
        path = os.path.join(seg_dir, "enrichment.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            logger.info("[SEGMENT_ENRICHMENT] Saved: %s", path)
        except Exception as e:
            logger.warning("[SEGMENT_ENRICHMENT] Failed to save %s: %s", path, e)

    # Also save a combined summary at the job level
    summary_path = os.path.join(job_dir_path, "enrichment_summary.json")
    try:
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "build_id": BUILD_ID,
            "total_segments": len(enrichment),
            "segments": {},
        }
        for seg_id, data in enrichment.items():
            summary["segments"][seg_id] = {
                "constants": data.get("extraction_stats", {}).get("constants", 0),
                "functions": data.get("extraction_stats", {}).get("functions", 0),
                "classes": data.get("extraction_stats", {}).get("classes", 0),
                "exports": len(data.get("exports", [])),
                "risk_level": data.get("risk_level", "low"),
                "implementation_order": data.get("implementation_order", 0),
            }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        logger.info("[SEGMENT_ENRICHMENT] Summary saved: %s", summary_path)
    except Exception as e:
        logger.warning("[SEGMENT_ENRICHMENT] Failed to save summary: %s", e)


def load_enrichment(job_dir_path: str, segment_id: str) -> Optional[Dict]:
    """Load cached enrichment for a segment (for resume/retry)."""
    path = os.path.join(job_dir_path, "segments", segment_id, "enrichment.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[SEGMENT_ENRICHMENT] Failed to load %s: %s", path, e)
        return None

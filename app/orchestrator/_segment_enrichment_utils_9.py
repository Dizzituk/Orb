from __future__ import annotations
import logging
import os
from app.orchestrator._segment_enrichment_utils_8 import _extract_names_from_import
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


ENRICHMENT_MODEL = os.getenv("SEGMENT_ENRICHMENT_MODEL", "claude-sonnet-4-6")

ENRICHMENT_MAX_TOKENS = int(os.getenv("SEGMENT_ENRICHMENT_MAX_OUTPUT_TOKENS", "16000"))

ENRICHMENT_TIMEOUT = int(os.getenv("SEGMENT_ENRICHMENT_TIMEOUT_SECONDS", "180"))

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

ENRICHMENT_SYSTEM_PROMPT = """You are a software architecture analyst. You are given:
1. A list of segments that a monolith Python file is being refactored into
2. A symbol map showing what each segment exports and imports from other segments
3. A list of symbols that could not be automatically assigned to any segment
4. Experience patterns from previous refactoring attempts

Your job is to produce implementation intelligence for each segment AND assign
any unresolved symbols to their correct segments.

Respond ONLY in valid JSON — no markdown fences, no preamble, no commentary."""

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

    # v1.3: Build reverse lookup of existing deterministic assignments
    # so we can detect conflicts where the LLM overrides a good assignment.
    _det_assigned: Dict[str, str] = {}  # symbol -> segment that deterministic placed it
    for _da_seg, _da_syms in assignments.items():
        for _da_sym in _da_syms:
            _det_assigned[_da_sym] = _da_seg

    for symbol_name, target_seg_id in llm_assignments.items():
        if target_seg_id not in valid_seg_ids:
            logger.warning(
                "[SEGMENT_ENRICHMENT] LLM assigned '%s' to unknown segment '%s' — skipping",
                symbol_name, target_seg_id,
            )
            continue

        # v1.3: If deterministic already assigned this symbol, keep the
        # deterministic assignment — it's more reliable than LLM bulk guessing.
        if symbol_name in _det_assigned:
            _existing = _det_assigned[symbol_name]
            if _existing != target_seg_id:
                logger.info(
                    "[SEGMENT_ENRICHMENT] v1.3 LLM wants '%s' in %s but "
                    "deterministic already placed it in %s — keeping deterministic",
                    symbol_name, target_seg_id, _existing,
                )
                continue

        # v1.3: Validate target file affinity — check that at least one
        # word (>3 chars) from the function name appears in the target
        # segment's file stems or title. If zero overlap, log a warning
        # but still accept (the LLM may know something we don't).
        _target_seg = next((s for s in segments if s.segment_id == target_seg_id), None)
        if _target_seg and symbol_name in func_by_name:
            _sym_words = {w for w in symbol_name.lower().lstrip("_").split("_") if len(w) > 3}
            _seg_words = set()
            for _fp in _target_seg.file_scope:
                _stem = os.path.basename(_fp).replace(".py", "").lower().lstrip("_")
                _seg_words.update(w for w in _stem.split("_") if len(w) > 3)
            _seg_words.update(w for w in _target_seg.title.lower().split() if len(w) > 3)
            _overlap = _sym_words & _seg_words
            if not _overlap and _sym_words:
                logger.warning(
                    "[SEGMENT_ENRICHMENT] v1.3 LOW AFFINITY: '%s' → %s "
                    "(no word overlap between function and target files/title). "
                    "Words: func=%s, seg=%s",
                    symbol_name, target_seg_id, _sym_words, _seg_words,
                )

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

# FILE: app/orchestrator/enrichment_deterministic_layer3.py
"""
Deterministic Layer 3 Replacement for Segment Enrichment.

Replaces the LLM intelligence pass with five deterministic heuristics
for unassigned symbol resolution, plus rule-based implementation
ordering and risk assessment.

Heuristics for symbol assignment (in priority order):
1. Import graph proximity — symbol imported by files in segment X only
2. File path co-location — symbol's source file in same dir as segment targets
3. Call graph analysis — symbol called by functions already assigned to segment
4. Decorator/annotation matching — route decorators, component exports
5. Entrypoint gravity — truly ambiguous symbols go to entrypoint segment

v1.0 (2026-02-27): Initial implementation — Stage 4 of deterministic
verification migration.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DET_LAYER3_BUILD_ID = "2026-02-27-v1.0-det-enrichment-layer3"


# =========================================================================
# HEURISTIC 1: Import Graph Proximity
# =========================================================================

def _heuristic_import_proximity(
    symbol_name: str,
    symbol_info: Dict[str, Any],
    segments: List[Any],
    all_symbols: Dict[str, list],
    source_code: str,
) -> Optional[str]:
    """
    If a symbol is imported by functions/code in one segment only,
    assign to that segment.
    """
    # Find which other symbols reference this one
    consumers: Dict[str, int] = {}  # segment_id -> count
    sym_pattern = re.compile(r'\b' + re.escape(symbol_name) + r'\b')

    for seg in segments:
        scope_basenames = {
            os.path.basename(f).replace(".py", "").lower()
            for f in (seg.file_scope or [])
        }
        if not scope_basenames:
            continue
        # Count references from other functions assigned to this segment
        for func in all_symbols.get("functions", []):
            func_name = func.get("name", "")
            body = func.get("body", "")
            if not body or func_name == symbol_name:
                continue
            # Is this function in this segment's scope?
            if func.get("_assigned_segment") == seg.segment_id:
                if sym_pattern.search(body):
                    consumers.setdefault(seg.segment_id, 0)
                    consumers[seg.segment_id] += 1

    if len(consumers) == 1:
        seg_id = next(iter(consumers))
        return seg_id
    return None


# =========================================================================
# HEURISTIC 2: File Path Co-location
# =========================================================================

def _heuristic_file_colocation(
    symbol_name: str,
    symbol_info: Dict[str, Any],
    segments: List[Any],
    source_path: str,
) -> Optional[str]:
    """
    Symbol's source file is in the same directory as a segment's
    target files — assign to that segment.
    """
    source_dir = os.path.dirname(source_path).replace("\\", "/").lower()
    if not source_dir:
        return None

    matching_segs: List[str] = []
    for seg in segments:
        for fp in (seg.file_scope or []):
            target_dir = os.path.dirname(fp).replace("\\", "/").lower()
            if target_dir == source_dir:
                matching_segs.append(seg.segment_id)
                break

    if len(matching_segs) == 1:
        return matching_segs[0]
    return None


# =========================================================================
# HEURISTIC 3: Call Graph Analysis
# =========================================================================

def _heuristic_call_graph(
    symbol_name: str,
    symbol_info: Dict[str, Any],
    segments: List[Any],
    assignments: Dict[str, List[str]],
    all_symbols: Dict[str, list],
) -> Optional[str]:
    """
    If symbol calls functions already assigned to a specific segment,
    assign to the same segment.
    """
    body = symbol_info.get("body", "")
    if not body:
        return None

    # Find all function names called in this symbol's body
    called_names: Set[str] = set()
    for func in all_symbols.get("functions", []):
        fn = func.get("name", "")
        if fn and fn != symbol_name and fn in body:
            called_names.add(fn)

    if not called_names:
        return None

    # Count which segments own the called functions
    seg_scores: Dict[str, int] = {}
    for seg_id, assigned in assignments.items():
        overlap = called_names & set(assigned)
        if overlap:
            seg_scores[seg_id] = len(overlap)

    if not seg_scores:
        return None

    # If one segment dominates
    max_score = max(seg_scores.values())
    winners = [sid for sid, score in seg_scores.items() if score == max_score]
    if len(winners) == 1:
        return winners[0]
    return None


# =========================================================================
# HEURISTIC 4: Decorator/Annotation Matching
# =========================================================================

def _heuristic_decorator_match(
    symbol_name: str,
    symbol_info: Dict[str, Any],
    segments: List[Any],
) -> Optional[str]:
    """
    Route decorators (@app.get, @router.post) assign to entrypoint/router segment.
    React component exports assign to owning UI segment.
    """
    body = symbol_info.get("body", "")
    decorators = symbol_info.get("decorators", [])

    is_route = any(
        d in str(decorators) for d in ("app.", "router.", "@get", "@post", "@put", "@delete")
    )
    if is_route:
        # Find the segment that owns the router file
        for seg in segments:
            for fp in (seg.file_scope or []):
                basename = os.path.basename(fp).lower()
                if "router" in basename or "route" in basename:
                    return seg.segment_id

    # React/component pattern
    if body and ("export default" in body or "export function" in body):
        for seg in segments:
            for fp in (seg.file_scope or []):
                if fp.endswith((".tsx", ".jsx")):
                    return seg.segment_id

    return None


# =========================================================================
# HEURISTIC 5: Entrypoint Gravity
# =========================================================================

def _heuristic_entrypoint_gravity(
    symbol_name: str,
    symbol_info: Dict[str, Any],
    segments: List[Any],
) -> Optional[str]:
    """
    Truly ambiguous utility symbols go to the entrypoint/utils segment.
    Prefers segments with 'utils', 'helpers', 'common' in their scope,
    or the first segment as fallback.
    """
    # Prefer a utils/helpers segment
    for seg in segments:
        for fp in (seg.file_scope or []):
            basename = os.path.basename(fp).lower()
            if any(k in basename for k in ("utils", "helpers", "common", "shared")):
                return seg.segment_id

    # Fallback: first segment with file_scope
    for seg in segments:
        if seg.file_scope:
            return seg.segment_id

    return None


# =========================================================================
# DETERMINISTIC SYMBOL RESOLVER
# =========================================================================

def resolve_unassigned_symbols(
    unassigned: List[Dict[str, Any]],
    segments: List[Any],
    assignments: Dict[str, List[str]],
    all_symbols: Dict[str, list],
    source_code: str,
    source_path: str,
) -> Dict[str, str]:
    """
    Resolve unassigned symbols using five heuristics in priority order.

    Returns {symbol_name: segment_id} for resolved symbols.
    """
    resolved: Dict[str, str] = {}

    for sym in unassigned:
        name = sym.get("name", "")
        if not name:
            continue

        # Try each heuristic in priority order
        result = _heuristic_import_proximity(
            name, sym, segments, all_symbols, source_code,
        )
        if not result:
            result = _heuristic_file_colocation(
                name, sym, segments, source_path,
            )
        if not result:
            result = _heuristic_call_graph(
                name, sym, segments, assignments, all_symbols,
            )
        if not result:
            result = _heuristic_decorator_match(
                name, sym, segments,
            )
        if not result:
            result = _heuristic_entrypoint_gravity(
                name, sym, segments,
            )

        if result:
            resolved[name] = result

    if resolved:
        logger.info(
            "[enrichment_det_l3] Resolved %d/%d unassigned symbols deterministically",
            len(resolved), len(unassigned),
        )

    return resolved


# =========================================================================
# DETERMINISTIC IMPLEMENTATION INTELLIGENCE
# =========================================================================

def generate_deterministic_intelligence(
    manifest: Any,
    symbol_map: Dict[str, Any],
    extractions: Dict[str, Dict],
    source_path: str,
) -> Dict[str, Any]:
    """
    Generate implementation ordering, risk flags, and design guidance
    without any LLM calls.

    Uses topological sort for ordering and heuristic rules for risk.
    """
    segments = manifest.segments
    result: Dict[str, Any] = {"segments": {}}

    # Build dependency graph for topological ordering
    deps: Dict[str, Set[str]] = {}
    for seg in segments:
        sid = seg.segment_id
        seg_deps = set(getattr(seg, "dependencies", []) or [])
        # Also add implicit deps from consumes map
        for provider_seg in symbol_map.get("consumes", {}).get(sid, {}).values():
            if isinstance(provider_seg, str):
                seg_deps.add(provider_seg)
            elif isinstance(provider_seg, list):
                seg_deps.update(provider_seg)
        deps[sid] = seg_deps

    # Topological sort (Kahn's algorithm)
    in_degree: Dict[str, int] = {sid: 0 for sid in deps}
    for sid, dep_set in deps.items():
        for dep in dep_set:
            if dep in in_degree:
                in_degree[sid] = in_degree.get(sid, 0) + 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    order: List[str] = []
    while queue:
        queue.sort()  # Deterministic ordering
        node = queue.pop(0)
        order.append(node)
        for sid, dep_set in deps.items():
            if node in dep_set:
                in_degree[sid] -= 1
                if in_degree[sid] == 0:
                    queue.append(sid)

    # Add any remaining (cyclic) segments at the end
    for sid in deps:
        if sid not in order:
            order.append(sid)

    # Generate per-segment intelligence
    for idx, seg in enumerate(segments):
        sid = seg.segment_id
        seg_extract = extractions.get(sid, {})

        # Order from topological sort
        impl_order = order.index(sid) + 1 if sid in order else idx + 1

        # Risk assessment (heuristic)
        func_count = len(seg_extract.get("functions", []))
        class_count = len(seg_extract.get("classes", []))
        dep_count = len(deps.get(sid, set()))
        exports = len(symbol_map.get("exports", {}).get(sid, set()))
        consumed_by_count = sum(
            len(syms) for syms in
            symbol_map.get("consumed_by", {}).get(sid, {}).values()
        )

        # Risk scoring
        risk_score = 0
        risk_notes_parts: List[str] = []

        if func_count > 15:
            risk_score += 2
            risk_notes_parts.append(f"{func_count} functions (high complexity)")
        if class_count > 3:
            risk_score += 1
            risk_notes_parts.append(f"{class_count} classes")
        if dep_count > 3:
            risk_score += 1
            risk_notes_parts.append(f"{dep_count} dependencies")
        if consumed_by_count > 5:
            risk_score += 2
            risk_notes_parts.append(f"consumed by {consumed_by_count} symbols across segments")
        if exports > 10:
            risk_score += 1
            risk_notes_parts.append(f"{exports} exports (wide interface)")

        risk_level = "low"
        if risk_score >= 4:
            risk_level = "high"
        elif risk_score >= 2:
            risk_level = "medium"

        # Design guidance (rule-based)
        guidance_parts: List[str] = []
        is_facade = any(
            os.path.basename(f) == "__init__.py"
            for f in (seg.file_scope or [])
        )
        if is_facade:
            guidance_parts.append(
                "This is a facade/init segment — focus on re-exports and "
                "minimal logic. Import from sibling modules only."
            )
        if consumed_by_count > 3:
            guidance_parts.append(
                "High fan-out: many segments depend on this one. "
                "Keep the interface stable and minimal."
            )
        if dep_count == 0:
            guidance_parts.append(
                "No dependencies — implement first as a foundation segment."
            )

        result["segments"][sid] = {
            "implementation_order": impl_order,
            "design_guidance": " ".join(guidance_parts) if guidance_parts else "",
            "risk_level": risk_level,
            "risk_notes": "; ".join(risk_notes_parts) if risk_notes_parts else "",
        }

    return result


__all__ = [
    "resolve_unassigned_symbols",
    "generate_deterministic_intelligence",
    "DET_LAYER3_BUILD_ID",
]

# Purpose: evidence loop utils 2
# Called-by: app.llm.pipeline._evidence_loop_utils_3, app.llm.pipeline.evidence_loop
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


_BUILD_ID = "2026-02-09-v2.3-empty-path-guard"

_BLOCK_BOUNDARY_MARKERS = [
    "EVIDENCE_REQUEST:",
    "CRITICAL_CLAIMS:",
    "DECISION:",
    "HUMAN_REQUIRED:",
    "RESOLVED_REQUEST:",
    "FORCED_RESOLUTION:",
]

def _block_boundary_pattern() -> re.Pattern:
    """Build a lookahead pattern matching any known block boundary or end-of-string."""
    escaped = [re.escape(m) for m in _BLOCK_BOUNDARY_MARKERS]
    return re.compile(
        r'(?=\n(?:' + '|'.join(escaped) + r')|\Z)',
        re.DOTALL,
    )

def _restructure_flat_evidence_request(parsed: dict) -> dict:
    """Handle flat-indentation EVIDENCE_REQUEST blocks.

    When the LLM outputs EVIDENCE_REQUEST: with no indentation on child keys,
    YAML parses it as sibling top-level keys:
        {EVIDENCE_REQUEST: None, id: "ER-001", severity: "CRITICAL", ...}

    This restructures into the expected format:
        {id: "ER-001", severity: "CRITICAL", ...}
    """
    if "EVIDENCE_REQUEST" not in parsed:
        return None

    # If properly nested, return the nested value
    nested = parsed.get("EVIDENCE_REQUEST")
    if isinstance(nested, dict) and nested.get("id"):
        return nested

    # Flat structure: EVIDENCE_REQUEST is None, real data is in siblings
    if nested is None or (isinstance(nested, dict) and not nested.get("id")):
        flat_copy = dict(parsed)
        flat_copy.pop("EVIDENCE_REQUEST", None)

        if not flat_copy.get("id"):
            return None  # No id field -> not a valid request

        # Reconstruct scope if roots/max_files are floating as siblings
        if "roots" in flat_copy and "scope" not in flat_copy:
            flat_copy["scope"] = {
                "roots": flat_copy.pop("roots"),
                "max_files": flat_copy.pop("max_files", 500),
            }
        elif flat_copy.get("scope") is None and "roots" in flat_copy:
            flat_copy["scope"] = {
                "roots": flat_copy.pop("roots"),
                "max_files": flat_copy.pop("max_files", 500),
            }

        return flat_copy

    return None

def _request_block_pattern(req_id: str) -> re.Pattern:
    """Match an EVIDENCE_REQUEST block by ID using block boundaries."""
    escaped_markers = [re.escape(m) for m in _BLOCK_BOUNDARY_MARKERS]
    return re.compile(
        rf'EVIDENCE_REQUEST:\s*\n\s*id:\s*"{re.escape(req_id)}".*?'
        rf'(?=\n(?:' + '|'.join(escaped_markers) + r')|\Z)',
        re.DOTALL,
    )

@dataclass
class StageResult:
    """Output from a single pipeline stage invocation."""
    output: str = ""
    success: bool = True
    error: Optional[str] = None
    unresolved_human_required: List[Dict] = field(default_factory=list)

@dataclass
class JobContext:
    """Shared context passed through pipeline stages.

    Accumulates evidence across the evidence-request fulfillment loop.
    """
    evidence_bundle: Optional[object] = None  # EvidenceBundle from evidence_collector
    fulfilled_evidence: Dict[str, Dict] = field(default_factory=dict)
    fulfilled_evidence_ids: Set[str] = field(default_factory=set)
    evidence_results: Dict[str, List] = field(default_factory=dict)
    force_resolve_only: bool = False
    force_resolve: Dict[str, Dict] = field(default_factory=dict)

def _extract_path_from_rag_hit(hit) -> Optional[str]:
    """Pull a file path from a RAG/embeddings search result."""
    if isinstance(hit, dict):
        for key in ("file_path", "path", "source", "document"):
            val = hit.get(key)
            if val and isinstance(val, str):
                return val
    return None

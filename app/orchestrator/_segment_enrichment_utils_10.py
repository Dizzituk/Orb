from __future__ import annotations
import json
import logging
import os
import re
from app.orchestrator._segment_enrichment_utils_8 import ENRICHMENT_PROVIDER, _build_enrichment_user_prompt
from app.orchestrator._segment_enrichment_utils_9 import ENRICHMENT_MAX_TOKENS, ENRICHMENT_MODEL, ENRICHMENT_SYSTEM_PROMPT, ENRICHMENT_TIMEOUT
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
BUILD_ID = "2026-02-18-v1.3-llm-assignment-conflict-guard"


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

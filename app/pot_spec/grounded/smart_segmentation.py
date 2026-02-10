# FILE: app/pot_spec/grounded/smart_segmentation.py
"""
Smart Segmentation — Concept-Aware Segment Generation.

Phase 3B of Pipeline Evolution.

Replaces the layer-based (file path heuristic) segmentation with concept-aware
grouping. Uses a lightweight LLM call to group files by related concepts,
targeting 2-3 needles per segment.

The key insight: architectural layers (backend/frontend/config) are an
implementation detail. What matters for coherent architecture generation is
that each segment contains a self-contained set of CONCEPTS. A "voice
transcription" segment should contain the service, its API route, AND its
frontend component — even though those span three layers.

Integration with Phase 3A:
    - NeedleEstimate tells us HOW MANY segments (recommended_segment_count)
    - Smart Segmentation tells us WHICH FILES go in which segment
    - The two work together: needle count drives the target, this module
      does the actual grouping

Fallback: if the LLM call fails, falls back to the existing layer-based
grouping in segmentation.py.

v1.0 (2026-02-10): Initial implementation — Phase 3B.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SMART_SEGMENTATION_BUILD_ID = "2026-02-10-v1.0-concept-aware-segmentation"
print(f"[SMART_SEGMENTATION_LOADED] BUILD_ID={SMART_SEGMENTATION_BUILD_ID}")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

GROUPING_SYSTEM_PROMPT = """\
You are a software specification decomposer. Your job is to group a list of \
files into coherent segments for independent architecture generation.

GOAL: Each segment should be a self-contained unit that an AI architect can \
design WITHOUT needing to think about the other segments. This means:

1. Related concepts stay together. A service + its API route + its UI component \
= ONE segment, even though they span backend/frontend/config.

2. Each segment should have 2-3 "needles" (independent concepts to track). \
Never put more than 4 needles in one segment.

3. Dependencies between segments should be minimal — ideally just interface \
contracts (function signatures, data types, API endpoints).

4. Files that import from each other or share data types should be in the SAME \
segment.

5. Config/schema/migration files go with the feature they serve, not in their \
own segment.

RULES:
- Every file in the input MUST appear in exactly one segment.
- Segment count should match the target_segments parameter (±1 is acceptable).
- Each segment needs a short descriptive title (e.g., "Voice Transcription", \
"Route Optimisation", "Core Data Models").
- For each segment, list which other segments it depends on (by index).
- Dependencies should form a DAG (no cycles).

OUTPUT FORMAT:
Return ONLY a JSON object, no markdown:
{
  "segments": [
    {
      "title": "Short descriptive title",
      "files": ["path/to/file1.py", "path/to/file2.py"],
      "concepts": ["concept1", "concept2"],
      "depends_on": []
    },
    {
      "title": "Another segment",
      "files": ["path/to/file3.py"],
      "concepts": ["concept3"],
      "depends_on": [0]
    }
  ],
  "reasoning": "Brief explanation of grouping logic"
}

depends_on uses zero-indexed segment indices.
"""


# =============================================================================
# LLM CALL — Concept-aware grouping
# =============================================================================

def _build_grouping_prompt(
    spec_markdown: str,
    file_scope: List[str],
    target_segments: int,
    requirements: List[str],
) -> str:
    """Build user prompt for concept-aware grouping."""
    # Trim spec for context window efficiency
    _spec = spec_markdown
    if len(_spec) > 10000:
        _half = 4500
        _spec = (
            _spec[:_half]
            + f"\n\n... ({len(spec_markdown) - 2*_half} chars trimmed) ...\n\n"
            + _spec[-_half:]
        )

    req_text = ""
    if requirements:
        req_text = "\nRequirements:\n" + "\n".join(f"- {r}" for r in requirements[:20])
        if len(requirements) > 20:
            req_text += f"\n... (+{len(requirements)-20} more)"

    return f"""\
Group these files into {target_segments} segments (±1 is acceptable).

Files ({len(file_scope)} total):
{chr(10).join(f'- {f}' for f in file_scope)}
{req_text}

Specification context:
{_spec}

Return ONLY the JSON object.
"""


def _parse_grouping_response(
    llm_output: str,
    file_scope: List[str],
) -> Optional[List[Dict[str, Any]]]:
    """
    Parse the LLM's grouping response.

    Returns list of segment dicts: [{title, files, concepts, depends_on}]
    Returns None if parsing fails or validation fails.
    """
    if not llm_output or not llm_output.strip():
        return None

    text = llm_output.strip()

    # Strip markdown fences
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    text = re.sub(r',\s*\]', ']', text)
    text = re.sub(r',\s*\}', '}', text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("[smart_seg] JSON parse failed: %s", e)
        return None

    if not isinstance(data, dict) or "segments" not in data:
        logger.error("[smart_seg] Expected {segments: [...]}")
        return None

    segments = data["segments"]
    if not isinstance(segments, list) or len(segments) < 2:
        logger.error("[smart_seg] Need at least 2 segments, got %d", len(segments) if isinstance(segments, list) else 0)
        return None

    # Validate: every file appears exactly once
    all_grouped_files = []
    for seg in segments:
        files = seg.get("files", [])
        if not isinstance(files, list):
            return None
        all_grouped_files.extend(files)

    scope_set = set(f.replace("\\", "/").lower() for f in file_scope)
    grouped_set = set(f.replace("\\", "/").lower() for f in all_grouped_files)

    # Check for files assigned but not in scope (LLM hallucination)
    hallucinated = grouped_set - scope_set
    if hallucinated:
        logger.warning("[smart_seg] LLM hallucinated %d file(s): %s",
                       len(hallucinated), list(hallucinated)[:3])

    # Check for files in scope but not assigned
    missing = scope_set - grouped_set
    if missing:
        logger.warning("[smart_seg] %d file(s) not assigned to any segment", len(missing))
        # Assign missing files to the last segment (best-effort)
        if segments:
            missing_original = [f for f in file_scope
                                if f.replace("\\", "/").lower() in missing]
            segments[-1].setdefault("files", []).extend(missing_original)

    # Validate depends_on indices
    n = len(segments)
    for seg in segments:
        deps = seg.get("depends_on", [])
        seg["depends_on"] = [d for d in deps if isinstance(d, int) and 0 <= d < n]

    return segments


async def generate_concept_segments(
    spec_markdown: str,
    file_scope: List[str],
    target_segments: int,
    requirements: List[str] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate concept-aware segment groupings via LLM.

    Args:
        spec_markdown: Full PoT spec
        file_scope: All files in scope
        target_segments: How many segments to target (from needle_estimate)
        requirements: Job requirements for context
        provider_id/model_id: Override model selection

    Returns:
        List of segment dicts, or None if LLM call fails (caller falls back).
    """
    requirements = requirements or []

    if target_segments < 2:
        return None  # No segmentation needed

    if len(file_scope) < 2:
        return None  # Can't segment 1 file

    # Resolve model — use the same lightweight model as needle classifier
    _provider = provider_id
    _model = model_id

    if not _provider or not _model:
        try:
            from app.llm.stage_models import get_stage_config
            config = get_stage_config("SMART_SEGMENTATION")
            _provider = _provider or config.provider
            _model = _model or config.model
        except (ImportError, Exception) as _cfg_err:
            logger.warning("[smart_seg] stage_models unavailable: %s", _cfg_err)

    if not _provider or not _model:
        logger.error("[smart_seg] Model not configured — cannot generate concept segments")
        return None

    logger.info(
        "[smart_seg] Generating %d concept segments for %d files — provider=%s model=%s",
        target_segments, len(file_scope), _provider, _model,
    )

    user_prompt = _build_grouping_prompt(spec_markdown, file_scope, target_segments, requirements)

    try:
        from app.providers.registry import llm_call

        result = await llm_call(
            provider_id=_provider,
            model_id=_model,
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=GROUPING_SYSTEM_PROMPT,
            max_tokens=2000,
            timeout_seconds=45,
        )

        if not result.is_success():
            logger.warning("[smart_seg] LLM call failed: %s", result.error_message)
            return None

        raw = (result.content or "").strip()
        segments = _parse_grouping_response(raw, file_scope)

        if segments is None:
            logger.warning("[smart_seg] Failed to parse grouping response")
            return None

        logger.info(
            "[smart_seg] Generated %d concept segments: %s",
            len(segments),
            [s.get("title", "?") for s in segments],
        )
        return segments

    except ImportError:
        logger.warning("[smart_seg] Provider registry unavailable")
        return None
    except Exception as e:
        logger.exception("[smart_seg] Unexpected error: %s", e)
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "generate_concept_segments",
    "SMART_SEGMENTATION_BUILD_ID",
]

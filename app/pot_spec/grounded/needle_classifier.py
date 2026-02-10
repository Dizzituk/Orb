# FILE: app/pot_spec/grounded/needle_classifier.py
"""
Needle Classifier — Cognitive Load Estimator for Specifications.

Phase 3A of Pipeline Evolution.

Analyses a PoT spec to estimate how many "needles" (independent facts the
architecture LLM must track simultaneously) it contains. This replaces the
crude file-count threshold with a concept-aware measurement.

Three needle dimensions:
    blast_radius_count  — Number of distinct files/modules that must change
                          in a coordinated way. A change that touches 5 files
                          that all need consistent interfaces = 5 blast-radius
                          needles.
    concept_count       — Number of distinct technical concepts the LLM must
                          hold in context. e.g. "OBD2 parsing", "OCR receipt
                          scanning", "route optimisation" = 3 concept needles.
    interface_count     — Number of cross-boundary interfaces that must be
                          designed. Each API endpoint, shared class, or data
                          contract between components = 1 interface needle.

The final needle estimate = max(blast_radius, concepts, interfaces).
This is the number the segmentation logic uses to decide whether/how to
segment the job.

Needle-count theory (from architecture notes):
    - 1-2 needles: trivial, any model handles it
    - 3 needles:   moderate, Sonnet-class handles well
    - 4 needles:   hard, needs frontier-class model
    - 5+ needles:  must segment — no single LLM call is reliable

v1.0 (2026-02-10): Initial implementation — Phase 3A.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

NEEDLE_CLASSIFIER_BUILD_ID = "2026-02-10-v1.0-needle-classifier"
print(f"[NEEDLE_CLASSIFIER_LOADED] BUILD_ID={NEEDLE_CLASSIFIER_BUILD_ID}")


# =============================================================================
# RESULT SCHEMA
# =============================================================================

@dataclass
class NeedleEstimate:
    """
    Result of needle classification for a PoT spec.
    """
    blast_radius_count: int = 1     # Files/modules that must change together
    concept_count: int = 1          # Distinct technical concepts
    interface_count: int = 1        # Cross-boundary interfaces
    needle_estimate: int = 1        # max(blast_radius, concepts, interfaces)
    reasoning: str = ""             # LLM's reasoning (for debugging)
    model_used: str = ""            # Which model produced this estimate
    from_cache: bool = False        # Whether this was a cached result

    def __post_init__(self):
        # Ensure needle_estimate is always the max
        self.needle_estimate = max(
            self.blast_radius_count,
            self.concept_count,
            self.interface_count,
        )

    @property
    def needs_segmentation(self) -> bool:
        """5+ needles must be segmented."""
        return self.needle_estimate >= 5

    @property
    def recommended_segment_count(self) -> int:
        """How many segments to create (targeting 2-3 needles each)."""
        if self.needle_estimate <= 3:
            return 1
        # Target 2-3 needles per segment, round up
        return max(2, -(-self.needle_estimate // 3))  # ceil division

    @property
    def difficulty_tier(self) -> str:
        """Human-readable difficulty classification."""
        n = self.needle_estimate
        if n <= 2:
            return "trivial"
        if n <= 3:
            return "moderate"
        if n <= 4:
            return "hard"
        return "must_segment"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blast_radius_count": self.blast_radius_count,
            "concept_count": self.concept_count,
            "interface_count": self.interface_count,
            "needle_estimate": self.needle_estimate,
            "reasoning": self.reasoning,
            "model_used": self.model_used,
            "from_cache": self.from_cache,
            "difficulty_tier": self.difficulty_tier,
            "needs_segmentation": self.needs_segmentation,
            "recommended_segment_count": self.recommended_segment_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeedleEstimate":
        return cls(
            blast_radius_count=data.get("blast_radius_count", 1),
            concept_count=data.get("concept_count", 1),
            interface_count=data.get("interface_count", 1),
            needle_estimate=data.get("needle_estimate", 1),
            reasoning=data.get("reasoning", ""),
            model_used=data.get("model_used", ""),
            from_cache=data.get("from_cache", False),
        )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

NEEDLE_CLASSIFIER_SYSTEM_PROMPT = """\
You are a cognitive load estimator for software specifications. Your job is to \
count how many independent "needles" (facts the architecture designer must track \
simultaneously) a specification contains.

Count three dimensions:

1. blast_radius_count: How many distinct files or modules must change in a \
coordinated way? Count each file that needs modifications that depend on changes \
in other files. A simple single-file change = 1. A change that requires 5 files \
to all agree on a new interface = 5.

2. concept_count: How many distinct technical concepts must the designer hold \
in working memory simultaneously? Each independent technology, algorithm, \
protocol, or domain concept = 1. Examples: "OBD2 Bluetooth parsing" = 1 concept, \
"OCR receipt scanning" = 1 concept, "SQLite schema migration" = 1 concept. \
Closely related sub-concepts that always appear together count as 1.

3. interface_count: How many cross-boundary interfaces must be designed? Each \
API endpoint, shared data type, event bus channel, or integration point between \
distinct components = 1. Internal helper functions within a single module = 0.

RULES:
- Be precise. Don't inflate counts.
- Closely related items that always travel together = 1 needle, not N.
- A "todo app" with one model, one API, one UI = blast_radius:3, concepts:1, interfaces:2.
- A "multi-service platform" with 4 services, 3 databases, 8 APIs = much higher.
- Count what the ARCHITECT must track, not what the end user sees.

OUTPUT FORMAT:
Return ONLY a JSON object, no markdown, no commentary:
{
  "blast_radius_count": <int>,
  "concept_count": <int>,
  "interface_count": <int>,
  "reasoning": "<brief explanation of what you counted>"
}
"""


# =============================================================================
# DETERMINISTIC PRE-FILTER
# =============================================================================

def _deterministic_estimate(spec_markdown: str, file_scope: list) -> Optional[NeedleEstimate]:
    """
    Fast deterministic check for obvious cases that don't need an LLM call.

    Returns a NeedleEstimate if the case is obvious, None if LLM is needed.
    """
    file_count = len(file_scope) if file_scope else 0

    # Single file, short spec → trivial
    if file_count <= 1 and len(spec_markdown) < 2000:
        return NeedleEstimate(
            blast_radius_count=1,
            concept_count=1,
            interface_count=0,
            reasoning="Deterministic: single file, short spec",
            model_used="deterministic",
        )

    # 2-3 files, short spec → likely moderate
    if file_count <= 3 and len(spec_markdown) < 3000:
        return NeedleEstimate(
            blast_radius_count=file_count,
            concept_count=1,
            interface_count=max(0, file_count - 1),
            reasoning=f"Deterministic: {file_count} files, short spec",
            model_used="deterministic",
        )

    # Very large file scope → obviously needs segmentation
    if file_count >= 20:
        return NeedleEstimate(
            blast_radius_count=file_count,
            concept_count=max(3, file_count // 4),
            interface_count=max(3, file_count // 5),
            reasoning=f"Deterministic: {file_count} files — clearly needs segmentation",
            model_used="deterministic",
        )

    return None  # Need LLM


# =============================================================================
# LLM CALL
# =============================================================================

def _parse_needle_response(llm_output: str) -> Optional[NeedleEstimate]:
    """Parse the LLM's JSON response into a NeedleEstimate."""
    if not llm_output or not llm_output.strip():
        return None

    text = llm_output.strip()

    # Strip markdown fences
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    # Lenient JSON
    text = re.sub(r',\s*\}', '}', text)
    text = re.sub(r',\s*\]', ']', text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("[needle_classifier] JSON parse failed: %s", e)
        return None

    if not isinstance(data, dict):
        return None

    try:
        return NeedleEstimate(
            blast_radius_count=max(1, int(data.get("blast_radius_count", 1))),
            concept_count=max(1, int(data.get("concept_count", 1))),
            interface_count=max(0, int(data.get("interface_count", 0))),
            reasoning=str(data.get("reasoning", "")),
        )
    except (ValueError, TypeError) as e:
        logger.error("[needle_classifier] Failed to extract counts: %s", e)
        return None


async def classify_needles(
    spec_markdown: str,
    file_scope: list = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> NeedleEstimate:
    """
    Main entry point: classify the needle count for a PoT spec.

    Tries deterministic pre-filter first. Falls back to lightweight
    LLM call (model from NEEDLE_CLASSIFIER_MODEL env / stage_models).

    Args:
        spec_markdown: The full PoT spec markdown
        file_scope: List of file paths in scope (for deterministic check)
        provider_id: Override provider
        model_id: Override model

    Returns:
        NeedleEstimate with counts and reasoning
    """
    file_scope = file_scope or []

    # Try deterministic first
    det = _deterministic_estimate(spec_markdown, file_scope)
    if det is not None:
        logger.info(
            "[needle_classifier] Deterministic: needle=%d (%s)",
            det.needle_estimate, det.reasoning,
        )
        return det

    # Resolve provider/model
    _provider = provider_id
    _model = model_id

    if not _provider or not _model:
        try:
            from app.llm.stage_models import get_stage_config
            config = get_stage_config("NEEDLE_CLASSIFIER")
            _provider = _provider or config.provider
            _model = _model or config.model
        except (ImportError, Exception) as _cfg_err:
            logger.warning("[needle_classifier] stage_models unavailable: %s", _cfg_err)

    if not _provider or not _model:
        raise RuntimeError(
            "Needle Classifier model not configured. "
            "Set NEEDLE_CLASSIFIER_PROVIDER and NEEDLE_CLASSIFIER_MODEL env vars, "
            "or ensure app.llm.stage_models is importable."
        )

    logger.info(
        "[needle_classifier] LLM call: provider=%s model=%s spec_len=%d files=%d",
        _provider, _model, len(spec_markdown), len(file_scope),
    )

    # Build user prompt — send the spec (trimmed if huge)
    _spec_for_prompt = spec_markdown
    if len(_spec_for_prompt) > 12000:
        # Trim middle, keep start and end which have most structural info
        _half = 5500
        _spec_for_prompt = (
            _spec_for_prompt[:_half]
            + f"\n\n... ({len(spec_markdown) - 2*_half} chars trimmed) ...\n\n"
            + _spec_for_prompt[-_half:]
        )

    user_prompt = f"""\
Analyse this specification and count the needles.

File scope ({len(file_scope)} files):
{chr(10).join(f'- {f}' for f in file_scope[:30])}
{'... and ' + str(len(file_scope) - 30) + ' more' if len(file_scope) > 30 else ''}

Specification:
{_spec_for_prompt}

Return ONLY the JSON object with blast_radius_count, concept_count, interface_count, reasoning.
"""

    try:
        from app.providers.registry import llm_call

        result = await llm_call(
            provider_id=_provider,
            model_id=_model,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            system_prompt=NEEDLE_CLASSIFIER_SYSTEM_PROMPT,
            max_tokens=500,     # Response is tiny JSON
            timeout_seconds=30,  # Fast call
        )

        if not result.is_success():
            logger.warning(
                "[needle_classifier] LLM call failed: %s — falling back to heuristic",
                result.error_message,
            )
            return _heuristic_fallback(spec_markdown, file_scope)

        raw = (result.content or "").strip()
        estimate = _parse_needle_response(raw)
        if estimate is None:
            logger.warning("[needle_classifier] Failed to parse response — falling back")
            return _heuristic_fallback(spec_markdown, file_scope)

        estimate.model_used = f"{_provider}/{_model}"
        logger.info(
            "[needle_classifier] LLM result: blast=%d concept=%d interface=%d → needle=%d (%s)",
            estimate.blast_radius_count, estimate.concept_count,
            estimate.interface_count, estimate.needle_estimate,
            estimate.difficulty_tier,
        )
        return estimate

    except ImportError:
        logger.warning("[needle_classifier] Provider registry unavailable — heuristic fallback")
        return _heuristic_fallback(spec_markdown, file_scope)
    except Exception as e:
        logger.exception("[needle_classifier] Unexpected error: %s", e)
        return _heuristic_fallback(spec_markdown, file_scope)


# =============================================================================
# HEURISTIC FALLBACK — when LLM is unavailable
# =============================================================================

def _heuristic_fallback(spec_markdown: str, file_scope: list) -> NeedleEstimate:
    """
    Rough heuristic when the LLM call fails or is unavailable.

    Uses file count + spec length + keyword density as proxy.
    """
    file_count = len(file_scope) if file_scope else 0
    spec_len = len(spec_markdown)

    # Blast radius ~ file count
    blast = max(1, file_count)

    # Concept count ~ unique technical keywords density
    # Simple proxy: count section headers in the spec
    headers = re.findall(r'^#{1,3}\s+.+', spec_markdown, re.MULTILINE)
    concepts = max(1, len(headers) // 3)  # Rough: 3 headers per concept

    # Interface count ~ "endpoint", "API", "import", "class" mentions
    interface_keywords = len(re.findall(
        r'\b(endpoint|API|route|import|interface|schema|contract|class)\b',
        spec_markdown, re.IGNORECASE,
    ))
    interfaces = max(0, interface_keywords // 3)

    estimate = NeedleEstimate(
        blast_radius_count=blast,
        concept_count=concepts,
        interface_count=interfaces,
        reasoning=f"Heuristic fallback: {file_count} files, {spec_len} chars, {len(headers)} headers",
        model_used="heuristic",
    )
    logger.info(
        "[needle_classifier] Heuristic: blast=%d concept=%d interface=%d → needle=%d",
        blast, concepts, interfaces, estimate.needle_estimate,
    )
    return estimate


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NeedleEstimate",
    "classify_needles",
    "NEEDLE_CLASSIFIER_BUILD_ID",
]

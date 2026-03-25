# FILE: app/memory/document_knowledge_promoter.py
"""
Document Knowledge Promoter — extract durable facts from uploaded
documents and promote them into ASTRA's permanent memory.

Two-stage gate:
  1. Classify the document: personal vs reference vs mixed.
  2. Extract facts appropriate to the classification:
     - Personal docs (CV, ID, contracts, plans): full extraction
     - Mixed docs (notes with personal decisions): decisions only
     - Reference docs (articles, guides, reports): no promotion

Uses the same ASTRA preference system as knowledge_extractor.py,
with appropriate strength tiers:
  - Immutable facts (name, DOB, education)     → HARD_RULE
  - Current-state facts (job, address, goals)   → DEFAULT  (can be superseded)
  - Soft preferences / decisions                → SOFT     (decay over time)

Hooks into chat_attachments.py after embeddings indexing.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session as DbSession

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

def _get_model() -> tuple[str, str]:
    """Return (provider, model) for classification + extraction."""
    provider = os.getenv("SUMMARY_PROVIDER", "google")
    model = os.getenv("SUMMARY_MODEL", "gemini-2.5-flash-lite")
    return provider, model


# =========================================================================
# Stage 1: Document classification
# =========================================================================

_CLASSIFY_PROMPT = """\
You are a document classifier. Given the filename and a preview of a \
document's content, classify it into one of three categories.

CATEGORIES:
- personal: A document that is ABOUT the user or belongs to them personally. \
  Examples: CV/resume, passport scan, driving licence, tenancy agreement, \
  bank statement, personal plan, goals document, medical record, personal letter.
- mixed: A document that is mostly reference material but contains personal \
  decisions, preferences, or plans from the user. Examples: project notes \
  with personal decisions, meeting minutes where the user committed to actions, \
  research with personal annotations or conclusions.
- reference: External information not about the user. Examples: articles, \
  guides, regulations, reports, someone else's work, downloaded PDFs, \
  technical documentation, news clippings.

RULES:
- Output ONLY one word: personal, mixed, or reference.
- If uncertain, default to reference (safer — avoids polluting memory).
- The document belonging to a user's project does NOT make it personal. \
  A downloaded HMRC guide is reference even if the user needs it for their tax return.
"""


async def _classify_document(
    filename: str,
    content_preview: str,
) -> str:
    """
    Classify a document as personal / mixed / reference.

    Returns one of: 'personal', 'mixed', 'reference'.
    """
    provider, model = _get_model()

    user_prompt = (
        f"Filename: {filename}\n\n"
        f"Content preview (first 2000 chars):\n{content_preview[:2000]}"
    )

    try:
        from app.llm._streaming_utils_3 import call_llm_text

        raw = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=_CLASSIFY_PROMPT,
            user_prompt=user_prompt,
            max_tokens=10,
            timeout_seconds=15,
        )
        result = raw.strip().lower().split()[0] if raw else "reference"
        if result not in ("personal", "mixed", "reference"):
            result = "reference"
        return result
    except Exception as e:
        logger.warning("[doc_promoter] Classification failed: %s", e)
        return "reference"


# =========================================================================
# Stage 2: Knowledge extraction
# =========================================================================

_EXTRACT_PERSONAL_PROMPT = """\
You are a knowledge extractor. Given a PERSONAL document, extract facts \
about the user that should be stored in long-term memory.

RULES:
- Output ONLY valid JSON array — no markdown fences, no preamble.
- Each item: {"category", "key", "value", "permanence"}
- Categories: biographical, preference, philosophy, project, learning
- Permanence levels:
  - "permanent": Facts that never change (name, date of birth, nationality, \
    educational qualifications, family members)
  - "current_state": Facts true now but expected to change over time \
    (current job, current address, current salary, active goals, skills being learned)
  - "soft": Preferences, decisions, and plans that may evolve \
    (career goals, project choices, lifestyle targets)
- Only extract CLEAR facts — not speculation.
- Maximum 15 extractions per document.
- Key should be a short snake_case identifier.
- Value should be a concise factual statement.

OUTPUT FORMAT:
[
  {"category": "biographical", "key": "full_name", "value": "Taz Smith", "permanence": "permanent"},
  {"category": "biographical", "key": "current_occupation", "value": "Delivery driver at Yodel", "permanence": "current_state"}
]
"""

_EXTRACT_DECISIONS_PROMPT = """\
You are a knowledge extractor. This document is mainly reference material, \
but may contain PERSONAL DECISIONS or PREFERENCES from the user.

RULES:
- Output ONLY valid JSON array — no markdown fences, no preamble.
- ONLY extract facts where THE USER is the subject — their decisions, \
  choices, commitments, preferences, or plans.
- Do NOT extract facts about external entities, regulations, or the world.
- Each item: {"category", "key", "value", "permanence"}
- Categories: preference, philosophy, project, learning
- Permanence: "current_state" or "soft" only (reference material decisions are never permanent)
- If no personal decisions are found, return an empty array: []
- Maximum 5 extractions.

OUTPUT FORMAT:
[
  {"category": "preference", "key": "visa_route_choice", "value": "Chose D7 visa route for Portugal", "permanence": "soft"}
]
"""


def _parse_extractions(raw: str) -> List[Dict[str, str]]:
    """Parse LLM extraction response into fact dicts."""
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("[doc_promoter] JSON parse failed: %s", e)
        return []

    if not isinstance(data, list):
        return []

    valid_categories = {"biographical", "preference", "philosophy", "project", "learning"}
    valid_permanence = {"permanent", "current_state", "soft"}

    valid = []
    for item in data:
        if (
            isinstance(item, dict)
            and item.get("category") in valid_categories
            and item.get("key")
            and item.get("value")
            and item.get("permanence", "soft") in valid_permanence
        ):
            valid.append(item)

    return valid[:15]


# =========================================================================
# Stage 3: Write to ASTRA memory
# =========================================================================

# Map permanence → PreferenceStrength
_PERMANENCE_TO_STRENGTH = {
    "permanent": "HARD_RULE",
    "current_state": "DEFAULT",
    "soft": "SOFT",
}

# Map category → namespace
_CATEGORY_TO_NAMESPACE = {
    "biographical": "user_personal",
    "preference": "user_personal",
    "philosophy": "dev_principles",
    "project": "project_context",
    "learning": "user_personal",
}


def _write_facts_to_memory(
    db: DbSession,
    extractions: List[Dict[str, str]],
    source_filename: str,
) -> int:
    """
    Write extracted facts to ASTRA preference system.

    Handles superseding: if a key already exists with a different value,
    the old record gets superseded and a new one is created.

    Returns count of facts written.
    """
    written = 0

    # Safety: ensure session is clean before writing
    try:
        db.rollback()
    except Exception:
        pass

    try:
        from app.astra_memory.preference_service import create_preference
        from app.astra_memory.preference_models import (
            PreferenceRecord,
            PreferenceStrength,
            RecordStatus,
            SignalType,
        )
        from app.astra_memory.confidence_scoring import (
            append_preference_evidence,
        )
    except ImportError:
        logger.debug("[doc_promoter] ASTRA preference service not available")
        return 0

    for item in extractions:
        category = item["category"]
        key = item["key"]
        value = item["value"]
        permanence = item.get("permanence", "soft")

        pref_key = f"doc_extract:{category}:{key}"
        strength_str = _PERMANENCE_TO_STRENGTH.get(permanence, "SOFT")
        strength = PreferenceStrength(strength_str.lower())
        namespace = _CATEGORY_TO_NAMESPACE.get(category, "user_personal")
        context_ptr = f"document:{source_filename}"

        try:
            existing = (
                db.query(PreferenceRecord)
                .filter(PreferenceRecord.preference_key == pref_key)
                .first()
            )

            if existing:
                if existing.preference_value == value:
                    # Same value — reinforce confidence
                    append_preference_evidence(
                        db=db,
                        preference_key=pref_key,
                        signal_type=SignalType.IMPLICIT,
                        context_pointer=context_ptr,
                        details={
                            "action": "reinforce_from_document",
                            "source": source_filename,
                        },
                    )
                    logger.debug("[doc_promoter] Reinforced: %s", pref_key)
                else:
                    # Different value — supersede the old record
                    existing.status = RecordStatus.SUPERSEDED
                    db.commit()

                    create_preference(
                        db=db,
                        preference_key=pref_key,
                        preference_value=value,
                        strength=strength,
                        source="document_extraction",
                        namespace=namespace,
                        context_pointer=context_ptr,
                    )
                    logger.info(
                        "[doc_promoter] Superseded: %s (old=%s, new=%s)",
                        pref_key,
                        str(existing.preference_value)[:50],
                        value[:50],
                    )
            else:
                # New fact — create
                create_preference(
                    db=db,
                    preference_key=pref_key,
                    preference_value=value,
                    strength=strength,
                    source="document_extraction",
                    namespace=namespace,
                    context_pointer=context_ptr,
                )
                logger.debug("[doc_promoter] Created: %s = %s", pref_key, value[:60])

            written += 1

        except Exception as e:
            logger.warning("[doc_promoter] Failed to write %s: %s", pref_key, e)

    return written


# =========================================================================
# Main entry point
# =========================================================================

async def promote_document_knowledge(
    db: DbSession,
    filename: str,
    raw_text: str,
    structured_data: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify a document and promote durable knowledge to ASTRA memory.

    Called after upload text extraction and embeddings indexing.

    Args:
        db: Database session.
        filename: Original filename.
        raw_text: Extracted text content.
        structured_data: Optional JSON string of structured parse
                         (e.g. from CV parser).

    Returns:
        Dict with classification, extraction count, and write count.
    """
    result = {
        "filename": filename,
        "classification": "reference",
        "extractions": 0,
        "written": 0,
        "skipped_reason": None,
    }

    if not raw_text or len(raw_text.strip()) < 50:
        result["skipped_reason"] = "Content too short for extraction"
        return result

    # ── Stage 1: Classify ────────────────────────────────────────
    classification = await _classify_document(filename, raw_text)
    result["classification"] = classification

    if classification == "reference":
        result["skipped_reason"] = "Reference document — no knowledge promotion"
        logger.info(
            "[doc_promoter] %s classified as reference — skipping promotion",
            filename,
        )
        return result

    # ── Stage 2: Extract ─────────────────────────────────────────
    provider, model = _get_model()

    # Build content for extraction
    content_for_extraction = ""
    if structured_data:
        content_for_extraction += f"Structured data:\n{structured_data}\n\n"
    content_for_extraction += f"Raw text:\n{raw_text[:6000]}"

    prompt = (
        _EXTRACT_PERSONAL_PROMPT
        if classification == "personal"
        else _EXTRACT_DECISIONS_PROMPT
    )

    try:
        from app.llm._streaming_utils_3 import call_llm_text

        raw_response = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=prompt,
            user_prompt=(
                f"Document: {filename}\n\n"
                f"{content_for_extraction}"
            ),
            max_tokens=2000,
            timeout_seconds=30,
        )
    except Exception as e:
        logger.error("[doc_promoter] Extraction LLM call failed: %s", e)
        result["skipped_reason"] = f"LLM call failed: {e}"
        return result

    extractions = _parse_extractions(raw_response)
    result["extractions"] = len(extractions)

    if not extractions:
        result["skipped_reason"] = "No extractable facts found"
        logger.info("[doc_promoter] %s: no facts extracted", filename)
        return result

    # ── Stage 3: Write to memory ─────────────────────────────────
    written = _write_facts_to_memory(db, extractions, filename)
    result["written"] = written

    logger.info(
        "[doc_promoter] %s: classified=%s, extracted=%d, written=%d",
        filename, classification, len(extractions), written,
    )

    return result


def promote_document_knowledge_sync(
    db: DbSession,
    filename: str,
    raw_text: str,
    structured_data: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for promote_document_knowledge.

    Handles the async-from-sync dance needed inside FastAPI
    synchronous endpoints.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            promote_document_knowledge(db, filename, raw_text, structured_data),
            loop,
        )
        return future.result(timeout=60)
    except RuntimeError:
        return asyncio.run(
            promote_document_knowledge(db, filename, raw_text, structured_data)
        )

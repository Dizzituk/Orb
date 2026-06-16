# FILE: app/memory/knowledge_extractor.py
# Purpose: Layer 3: Extract durable knowledge from conversation summaries
# Called-by: app.memory.session_lifecycle, tests.test_conversation_memory
# Depends-on: app.astra_memory.confidence_scoring, app.astra_memory.preference_models, app.astra_memory.preference_service, app.llm._streaming_utils_3 (+3 more)
# Last-renovated: 2026-06-11
"""
Layer 3: Extract durable knowledge from conversation summaries
and promote it to ASTRA's permanent memory system.

Called when a session transitions to 'archived' (Phase 5 lifecycle).
Uses a cheap LLM to classify summary content into extraction categories:
  - biographical:  permanent (user facts, background)
  - preference:    evolving  (style, tool, workflow preferences)
  - philosophy:    permanent (development principles, design rules)
  - project:       ephemeral (current task context, WIP decisions)
  - learning:      evolving  (knowledge gaps, skill building)

Each extraction is written to ASTRA memory via the preference service
with appropriate confidence tiers. New extractions start at low confidence
and get promoted after 2+ reinforcements (existing ASTRA behaviour).

Part of CONV-MEMORY-001: Phase 6 (RAG Memory Extraction).
"""

import json
import logging
import os
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session as DbSession

from app.memory.conversation_models import ConversationSession
from app.memory.conversation_service import get_latest_summary

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

def _get_extraction_model() -> tuple[str, str]:
    """
    Return (provider, model) for knowledge extraction.

    Separate from summary model -- extraction needs comprehension,
    not just compression. Summary can use a cheap model; extraction
    must understand commitment level, consequence weight, and
    durability to classify facts correctly.

    Config: EXTRACTION_PROVIDER / EXTRACTION_MODEL env vars.
    Defaults to OpenAI gpt-5.4-mini (strong linguistics, low cost).
    """
    provider = os.getenv("EXTRACTION_PROVIDER", "openai")
    model = os.getenv("EXTRACTION_MODEL", "gpt-5.4-mini")
    return provider, model

# =========================================================================
# Extraction prompt
#
# ROLE CHANGE (2026-04-17): The conversation LLM now writes memory facts
# directly via tool calls (save_to_memory / save_residence / update_memory)
# DURING the turn. This extractor still runs on session archive, but its
# role has narrowed to a SAFETY NET: catch durable facts that the live
# tool path missed, not re-extract everything.
#
# Duplicates are handled by the fuzzy-match reinforcement logic in
# _write_to_astra_memory — if the key already exists, it's reinforced
# (which is the correct behaviour either way: the same fact being
# captured by two paths should strengthen confidence, not create a
# duplicate row).
# =========================================================================

_EXTRACTION_SYSTEM_PROMPT = """\
You are a safety-net knowledge extractor for a personal AI memory system. \
The conversation AI already writes durable facts to memory DURING the \
conversation via explicit tool calls. Your job is to catch anything it \
missed, by re-reading the session summary after the session has gone idle.

CRITICAL PRINCIPLES:
1. QUALITY OVER QUANTITY. Extract 0-4 facts max. Empty output is strongly \
   preferred over noise — the live path has likely caught the obvious facts \
   already. Only extract what clearly slipped through.
2. DURABILITY TEST: Would this fact still matter in 2 weeks? If not, skip it.
3. COMMITMENT vs ASPIRATION: Distinguish active commitments from idle wishes.
   - "I want to move to Portugal" = aspiration (low weight, preference category)
   - "I'm pursuing a D7 visa" = active commitment (high weight, biographical/project)
   - "I'm working with Leigh Day solicitors" = concrete action (high weight, project)
   Key signals: legal/financial actions taken, deadlines set, money spent,
   professionals engaged, repeated across sessions. Wishes use "I want to",
   "I'd like to", "maybe". Commitments use "I'm doing", "I've started",
   "I'm working on", "the plan is".
4. CONSEQUENCE WEIGHT: Does this fact affect other decisions? If changing it
   would cascade into other areas (finance, timeline, architecture), it's
   high-value. If it's isolated, it's lower-value.
5. CROSS-DOMAIN RECURRENCE: If a theme appears across multiple conversation
   topics (e.g. Portugal appears in legal, financial, lifestyle contexts),
   that's structural importance, not just frequency.
6. SKIP RESIDENCE HISTORY: Historical places the user has lived are handled \
   by a dedicated live tool (save_residence). Do NOT extract them here — \
   they'll be duplicated or miscategorised. Only extract CURRENT location \
   if it's clearly stated and missing.

OUTPUT RULES:
- Output ONLY valid JSON array — no markdown fences, no preamble.
- Each item: {"category", "key", "value", "weight"}
- "weight" is 1-5: 1=weak signal, 3=moderate, 5=load-bearing fact.
- Use STABLE keys: "current_location" not "location_mentioned_today".
  Same concept must always get the same key so reinforcement works.
- Maximum 4 extractions per summary.
- Deduplicate aggressively. The live tool path has already written the
  obvious facts — your job is to catch the gaps.

CATEGORIES (strict):
- biographical: Personal hard facts — name, DOB, CURRENT location, occupation,
                nationality, family. ONLY from direct user statements.
                NEVER infer. Weight 4-5. Do NOT extract past residences.
- preference:   DURABLE style/tool/workflow preferences that generalise
                beyond one session. Weight 2-4.
- philosophy:   Hard rules and design principles stated as absolutes
                ("always", "never", "the rule is"). Weight 5.
- project:      STABLE project context — project names, architectures,
                long-running goals, active legal/business proceedings.
                NOT session tasks or WIP. Weight 3-5.
- learning:     DURABLE skill-building patterns or knowledge interests
                that persist across sessions. NOT one-off questions. Weight 2-3.

STRICT EXCLUSIONS (do NOT extract):
- Past residences (handled by save_residence tool).
- Current session tasks, WIP decisions, active debugging context.
- One-off questions ("user asked about X today").
- Technical details specific to a single session.
- Test commands, TTS checks, build artefacts.
- Transient UI states or tool parameters.
- Anything that fails the 2-week durability test.
- API keys, secrets, tokens, credentials of any kind.
- If in doubt, DO NOT extract. Empty output beats noise.

KEY NAMING RULES (critical for reinforcement):
- Use snake_case, short, generic keys.
- Same concept = same key always. Examples:
  "hands_free_operation" (not "hands_free_essential" or "driving_hands_free")
  "tts_preference" (not "tts_naturalness" or "tts_speed_preference")
  "legal_challenge" (not "inpost_legal_case" or "leigh_day_action")
- Prefer broad keys that can absorb related facts over narrow ones.

OUTPUT FORMAT:
[
  {"category": "biographical", "key": "date_of_birth", "value": "1980-06-15", "weight": 5},
  {"category": "preference", "key": "code_style", "value": "Small modular files under 20KB", "weight": 4},
  {"category": "project", "key": "legal_challenge", "value": "Pursuing worker status claim against InPost via Leigh Day", "weight": 5}
]
"""

# =========================================================================
# Response parsing
# =========================================================================

def _parse_extractions(raw: str) -> List[Dict[str, str]]:
    """Parse LLM extraction response into a list of fact dicts."""
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
        logger.warning(
            "[knowledge_ext] Failed to parse extraction JSON: %s", e,
        )
        return []

    if not isinstance(data, list):
        logger.warning("[knowledge_ext] Extraction response is not a list")
        return []

    valid = []
    for item in data:
        if (
            isinstance(item, dict)
            and "category" in item
            and "key" in item
            and "value" in item
            and item["category"] in (
                "biographical", "preference", "philosophy",
                "project", "learning",
            )
        ):
            # Normalise key to snake_case, strip whitespace
            item["key"] = (
                item["key"].strip().lower()
                .replace(" ", "_").replace("-", "_")
                .replace("__", "_").rstrip("_")
            )
            # Ensure weight is an int 1-5, default 3
            try:
                item["weight"] = max(1, min(5, int(item.get("weight", 3))))
            except (ValueError, TypeError):
                item["weight"] = 3
            valid.append(item)

    return valid[:10]  # Hard cap


# =========================================================================
# ASTRA memory writing
# =========================================================================

# Map extraction categories to ASTRA preference namespaces and strengths
_CATEGORY_CONFIG = {
    "biographical": {
        "namespace": "user_personal",
        "source": "conversation_extraction",
        "is_permanent": True,
    },
    "preference": {
        "namespace": "user_personal",
        "source": "conversation_extraction",
        "is_permanent": False,
    },
    "philosophy": {
        "namespace": "dev_principles",
        "source": "conversation_extraction",
        "is_permanent": True,
    },
    "project": {
        "namespace": "project_context",
        "source": "conversation_extraction",
        "is_permanent": False,
    },
    "learning": {
        "namespace": "user_personal",
        "source": "conversation_extraction",
        "is_permanent": False,
    },
}


def _write_to_astra_memory(
    db: DbSession,
    extractions: List[Dict[str, str]],
    session_id: int,
) -> List[Dict[str, Any]]:
    """
    Write extracted facts to ASTRA memory via preference service.

    Uses create_preference for new facts and append_preference_evidence
    for existing ones (reinforcement). On an EXACT-key conflict the new value
    supersedes the old via update_preference_value (hard-rule and
    established-confidence protection is enforced inside that helper).

    Returns the finalised items actually touched, each a dict
    {pref_id, key, value, category, changed}; the membrane encodes these for
    retrieval. `changed` is True when the stored value was created or
    overwritten (so it warrants re-embedding).
    """
    touched: List[Dict[str, Any]] = []

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
        logger.debug("[knowledge_ext] ASTRA preference service not available")
        return 0

    for item in extractions:
        category = item["category"]
        key = item["key"]
        value = item["value"]
        weight = item.get("weight", 3)  # 1-5 commitment/consequence weight

        # Security: never store API keys, secrets, or tokens
        _key_lower = key.lower()
        _val_lower = str(value).lower() if value else ""
        if any(x in _key_lower for x in ("api_key", "secret", "token", "client_key", "client_id", "app_id")):
            continue
        if any(_val_lower.startswith(x) for x in ("sk-", "sk_", "aizasy", "gocspx")):
            continue
        config = _CATEGORY_CONFIG.get(category, _CATEGORY_CONFIG["project"])

        pref_key = f"conv_extract:{category}:{key}"
        context_ptr = f"session:{session_id}"

        try:
            # Check if this fact (or a semantically similar one) already exists.
            # Exact key match first, then fuzzy match within same category.
            existing = (
                db.query(PreferenceRecord)
                .filter(PreferenceRecord.preference_key == pref_key)
                .first()
            )

            # Fuzzy match: if no exact key match, look for a similar key
            # in the same category. This catches cases like
            # "hands_free_operation" and "hands_free_essential" being
            # the same preference extracted with slightly different keys.
            if not existing:
                _cat_prefix = f"conv_extract:{category}:"
                # Use the enum constant, not the string "EXPIRED". SQLAlchemy's
                # SQLEnum serialises enum NAMES by default, so the string form
                # happens to work today -- but any change to the enum config
                # (e.g. values_callable) would silently break this filter and
                # let expired rows match as fuzzy reinforcement targets.
                _candidates = (
                    db.query(PreferenceRecord)
                    .filter(
                        PreferenceRecord.preference_key.like(f"{_cat_prefix}%"),
                        PreferenceRecord.status != RecordStatus.EXPIRED,
                    )
                    .all()
                )
                # Simple token overlap: if >60% of words in the new value
                # appear in an existing value, treat as reinforcement.
                _new_tokens = set(value.lower().split())
                for cand in _candidates:
                    _cand_tokens = set(str(cand.preference_value).lower().split())
                    if not _new_tokens or not _cand_tokens:
                        continue
                    _overlap = len(_new_tokens & _cand_tokens)
                    _ratio = _overlap / min(len(_new_tokens), len(_cand_tokens))
                    if _ratio >= 0.6:
                        existing = cand
                        logger.debug(
                            "[knowledge_ext] Fuzzy match: %s ≈ %s (%.0f%% overlap)",
                            pref_key, cand.preference_key, _ratio * 100,
                        )
            if existing:
                # Reinforce existing fact with new evidence
                # Use extraction weight as evidence weight — high-commitment
                # facts reinforce more strongly than casual mentions.
                append_preference_evidence(
                    db=db,
                    preference_key=existing.preference_key,
                    signal_type=SignalType.IMPLICIT,
                    context_pointer=context_ptr,
                    weight_override=float(weight),  # 1.0-5.0 from extraction
                    details={
                        "action": "reinforce_from_conversation",
                        "new_value": value,
                        "extraction_weight": weight,
                    },
                )
                # Conflict resolution: on an EXACT-key match whose stored value
                # changed, let the new value supersede the old. Fuzzy matches
                # are left as pure reinforcement (the match is only a guess at
                # sameness, so we never overwrite the value on it).
                prev_value = existing.preference_value
                if existing.preference_key == pref_key and str(prev_value) != str(value):
                    try:
                        from app.astra_memory.preference_service import (
                            update_preference_value,
                        )
                        update_preference_value(
                            db,
                            preference_key=existing.preference_key,
                            new_value=value,
                            is_explicit=False,
                            context_pointer=context_ptr,
                        )
                        db.refresh(existing)
                    except Exception as _ue:
                        logger.debug(
                            "[knowledge_ext] value update skipped for %s: %s",
                            existing.preference_key, _ue,
                        )
                final_value = existing.preference_value
                touched.append({
                    "pref_id": existing.id,
                    "key": key,
                    "value": final_value,
                    "category": category,
                    "changed": str(final_value) != str(prev_value),
                })
                logger.debug(
                    "[knowledge_ext] Reinforced: %s (weight=%d)", existing.preference_key, weight,
                )
            else:
                # Create new preference
                # Use extraction weight to determine strength:
                # weight 5 + permanent category = HARD_RULE
                # weight 4-5 = DEFAULT (starts with moderate confidence)
                # weight 1-3 = SOFT (needs reinforcement to gain confidence)
                if config["is_permanent"] and weight >= 4:
                    strength = PreferenceStrength.HARD_RULE
                elif weight >= 4:
                    strength = PreferenceStrength.DEFAULT
                else:
                    strength = PreferenceStrength.SOFT
                _rec = create_preference(
                    db=db,
                    preference_key=pref_key,
                    preference_value=value,
                    strength=strength,
                    source=config["source"],
                    namespace=config["namespace"],
                    context_pointer=context_ptr,
                )
                if _rec is not None:
                    touched.append({
                        "pref_id": _rec.id,
                        "key": key,
                        "value": value,
                        "category": category,
                        "changed": True,
                    })
                logger.debug(
                    "[knowledge_ext] Created: %s = %s (weight=%d, strength=%s)",
                    pref_key, value[:60], weight, strength.value,
                )

            # Notify Self-Model of the new/reinforced fact
            try:
                from app.self_model.hooks import on_knowledge_extracted
                on_knowledge_extracted(category=category, key=key, value=value, source=context_ptr)
            except Exception:
                pass

        except Exception as e:
            logger.warning(
                "[knowledge_ext] Failed to write %s: %s", pref_key, e,
            )

    return touched


# =========================================================================
# Consolidation helpers (Job 3): transcript tail, encode, graph
# =========================================================================

def _load_unsummarised_tail(
    db: DbSession,
    session: ConversationSession,
    latest_summary,
    max_msgs: int = 6,
    max_chars: int = 1500,
) -> str:
    """Return the few session turns added after the last rolling summary.

    The summary fires every 5 messages, so a handful of trailing messages are
    typically unsummarised at archive. Feeding them keeps the freshest turns in
    scope for extraction and link detection. Best-effort — returns "".
    """
    try:
        from app.memory.models import Message
        through = getattr(latest_summary, "summarised_through_message_id", None) or 0
        q = db.query(Message).filter(Message.session_id == session.id)
        if through:
            q = q.filter(Message.id > through)
        msgs = q.order_by(Message.id.asc()).limit(max_msgs).all()
        lines = []
        for m in msgs:
            content = (m.content or "").strip()
            if content:
                lines.append(f"{m.role}: {content}")
        return "\n".join(lines)[:max_chars]
    except Exception as exc:
        logger.debug("[knowledge_ext] tail load failed: %s", exc)
        return ""


def _encode_touched(
    db: DbSession,
    session: ConversationSession,
    touched: List[Dict[str, Any]],
) -> None:
    """Encode finalised items for retrieval (tag + embed). Best-effort."""
    if not touched:
        return
    try:
        from app.memory.consolidation_encode import encode_memory_item
    except Exception as exc:
        logger.debug("[knowledge_ext] encode module unavailable: %s", exc)
        return
    project_id = getattr(session, "project_id", None)
    for item in touched:
        try:
            encode_memory_item(
                db,
                project_id=project_id,
                pref_id=item["pref_id"],
                key=item["key"],
                value=item["value"],
                reembed=bool(item.get("changed", True)),
            )
        except Exception as exc:
            logger.debug(
                "[knowledge_ext] encode failed for %s: %s", item.get("key"), exc,
            )


async def _consolidate_graph(consolidation_input: str, session_id: int) -> None:
    """Capture cross-domain links into the knowledge graph. Best-effort."""
    try:
        from app.memory.consolidation_graph import detect_and_write_links
        await detect_and_write_links(consolidation_input, session_id)
    except Exception as exc:
        logger.debug("[knowledge_ext] graph consolidation failed: %s", exc)


# =========================================================================
# Main entry point
# =========================================================================

async def extract_and_promote_async(
    db: DbSession,
    session: ConversationSession,
) -> bool:
    """
    Async version of extract_and_promote.
    Called from the async lifecycle scheduler.
    """
    latest = get_latest_summary(db, session.id)
    if not latest or not latest.summary_json:
        logger.debug(
            "[knowledge_ext] No summary for session %d — skipping",
            session.id,
        )
        return False

    summary_json = latest.summary_json
    summary_text = json.dumps(summary_json, indent=2)
    # Job 3: also feed the handful of turns the rolling summary has not yet
    # folded in (it fires every 5 messages, so a few trailing messages are
    # unsummarised at archive). Used for BOTH fact extraction and link detection.
    tail = _load_unsummarised_tail(db, session, latest)
    consolidation_input = summary_text + (
        f"\n\nRecent messages (not yet summarised):\n{tail}" if tail else ""
    )

    provider, model = _get_extraction_model()

    try:
        from app.llm._streaming_utils_3 import call_llm_text
        raw = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=(
                f"Extract durable knowledge from this conversation summary:\n\n"
                f"{consolidation_input}"
            ),
            max_tokens=1500,
            timeout_seconds=30,
        )
    except Exception as e:
        logger.error(
            "[knowledge_ext] LLM extraction failed for session %d: %s",
            session.id, e,
        )
        raw = ""  # fact extraction failed, but graph consolidation is independent

    extractions = _parse_extractions(raw)
    touched = _write_to_astra_memory(db, extractions, session.id) if extractions else []

    # Job 3 task 4: encode finalised items so Job 4 retrieval can find them.
    _encode_touched(db, session, touched)
    # Job 3 task 3: capture the cross-domain links the user drew into the graph.
    await _consolidate_graph(consolidation_input, session.id)

    if not extractions:
        logger.info(
            "[knowledge_ext] No extractable facts from session %d "
            "(safety net had nothing to add — live tool path likely handled them)",
            session.id,
        )
    else:
        logger.info(
            "[knowledge_ext] Session %d: safety-net extracted %d facts, finalised %d",
            session.id, len(extractions), len(touched),
        )
    return len(touched) > 0


def extract_and_promote(
    db: DbSession,
    session: ConversationSession,
) -> bool:
    """
    Synchronous wrapper around extract_and_promote_async.

    The async form is the single implementation (it also runs the Job 3
    consolidation steps — encode + cross-domain graph). This wrapper exists for
    sync callers/tooling; the live lifecycle scheduler calls the async form
    directly. Returns True if any durable item was finalised.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside a running loop — hand off to it and wait on a worker thread.
        future = asyncio.run_coroutine_threadsafe(
            extract_and_promote_async(db, session), loop,
        )
        return future.result(timeout=90)
    return asyncio.run(extract_and_promote_async(db, session))

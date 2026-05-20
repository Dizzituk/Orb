# FILE: app/debug/executors/memory_tools.py
"""
Memory tool executors for the chat LLM.

These tools let the conversation model WRITE to ASTRA's tiered memory
directly during the turn, rather than relying on the knowledge_extractor
background job that only runs on session archive. This eliminates
confabulation: the model can only claim to have saved something if it
actually called the tool and got a successful result back.

Tools exposed:
    save_to_memory        — Write a new fact (biographical / preference /
                            philosophy / project / learning).
    update_memory         — Change an existing fact's value.
    forget_memory         — Mark a fact for expiry.
    search_memory         — Retrieve facts by query or category.
    save_residence        — Append a structured entry to the identity
                            store's residence_history list.

All writes go through the same confidence/tier machinery that the
knowledge_extractor uses, so tiering, decay, and promotion are
unchanged. The LLM chooses permanence/weight; the machinery maps those
to PreferenceStrength and initial confidence.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# PERMANENCE -> STRENGTH MAPPING
# =============================================================================

def _resolve_strength(permanence: str, weight: int):
    """
    Map LLM-chosen (permanence, weight) to PreferenceStrength.

    permanence:
        "permanent"  - biographical hard facts, philosophy rules. Tier 1.
        "evolving"   - preferences, ongoing projects. Tier 2.
        "ephemeral"  - session context, WIP state. Tier 3.

    weight: 1-5. Load-bearing facts (5) in permanent slots become HARD_RULE
    so they bypass decay. Moderate weight (3-4) becomes DEFAULT.
    Low weight (1-2) stays SOFT and needs reinforcement to gain confidence.
    """
    from app.astra_memory.preference_models import PreferenceStrength

    weight = max(1, min(5, int(weight)))
    permanence = (permanence or "evolving").lower()

    if permanence == "permanent" and weight >= 4:
        return PreferenceStrength.HARD_RULE
    if weight >= 4:
        return PreferenceStrength.DEFAULT
    return PreferenceStrength.SOFT


# Map LLM category to (namespace, is_identity_candidate)
_CATEGORY_CONFIG = {
    "biographical": {"namespace": "user_personal",   "mirror_to_identity": True},
    "preference":   {"namespace": "user_personal",   "mirror_to_identity": False},
    "philosophy":   {"namespace": "dev_principles",  "mirror_to_identity": False},
    "project":      {"namespace": "project_context", "mirror_to_identity": False},
    "learning":     {"namespace": "user_personal",   "mirror_to_identity": False},
}

# Biographical keys that should ALSO mirror to the always-injected
# Tier 1 identity store (same rule the knowledge_extractor uses).
_IDENTITY_ROUTED_KEYS = {
    "name", "preferred_name", "date_of_birth", "birthplace",
    "current_location", "address", "nationality", "occupation",
    "email", "phone",
}


# =============================================================================
# save_to_memory
# =============================================================================

async def execute_save_to_memory(params: Dict[str, Any]) -> str:
    """
    Write a fact to the tiered memory system.

    params:
        category:   biographical | preference | philosophy | project | learning
        key:        stable snake_case identifier (e.g. "residence_exeter",
                    "code_style", "favourite_band"). Same concept = same key.
        value:      the fact itself (string)
        weight:     1-5 commitment/consequence weight (default 3)
        permanence: permanent | evolving | ephemeral (default evolving)
        reason:     optional short note explaining why this is being saved

    Returns a structured status line the LLM can echo back to the user
    truthfully.
    """
    from app.db import get_db_session
    from app.astra_memory.preference_service import create_preference
    from app.astra_memory.preference_models import (
        PreferenceRecord, SignalType,
    )
    from app.astra_memory.confidence_scoring import append_preference_evidence

    category = str(params.get("category", "")).strip().lower()
    key = str(params.get("key", "")).strip().lower().replace(" ", "_").replace("-", "_")
    value = str(params.get("value", "")).strip()
    weight = params.get("weight", 3)
    permanence = str(params.get("permanence", "evolving")).strip().lower()
    reason = str(params.get("reason", "")).strip()

    if not category or category not in _CATEGORY_CONFIG:
        return (
            "save_to_memory failed: category must be one of "
            f"{sorted(_CATEGORY_CONFIG.keys())}"
        )
    if not key:
        return "save_to_memory failed: key is required"
    if not value:
        return "save_to_memory failed: value is required"

    # Security: never store secrets by accident
    _key_lower = key.lower()
    _val_lower = value.lower()
    if any(x in _key_lower for x in ("api_key", "secret", "token", "client_key", "client_id")):
        return "save_to_memory refused: key looks like a credential"
    if any(_val_lower.startswith(x) for x in ("sk-", "sk_", "aizasy", "gocspx")):
        return "save_to_memory refused: value looks like a credential"

    config = _CATEGORY_CONFIG[category]
    strength = _resolve_strength(permanence, weight)
    pref_key = f"conv_extract:{category}:{key}"

    db = get_db_session()
    try:
        # Check for existing fact under the same key
        existing = (
            db.query(PreferenceRecord)
            .filter(PreferenceRecord.preference_key == pref_key)
            .first()
        )

        if existing:
            # Reinforce with new evidence - weight flows through as
            # evidence weight_override so high-commitment mentions
            # accelerate confidence.
            append_preference_evidence(
                db=db,
                preference_key=existing.preference_key,
                signal_type=SignalType.EXPLICIT,
                context_pointer="tool:save_to_memory",
                weight_override=float(weight),
                details={
                    "action": "reinforce_via_tool",
                    "new_value": value,
                    "reason": reason,
                    "weight": weight,
                    "permanence": permanence,
                },
            )

            # If the incoming value differs from what's stored, update it.
            # The LLM reaches save_to_memory (not update_memory) when it
            # considers this a fresh statement. If the value is literally
            # the same string we treat it as pure reinforcement; if it's
            # different it's a correction-by-restatement and the stored
            # value should reflect the user's latest wording.
            value_changed = str(existing.preference_value).strip() != value.strip()
            if value_changed:
                from datetime import datetime as _dt, timezone as _tz
                existing.preference_value = value
                existing.updated_at = _dt.now(_tz.utc)
                existing.last_reinforced_at = _dt.now(_tz.utc)
                db.commit()
                logger.info(
                    "[memory_tool] reinforced + value-updated %s: %r (weight=%d)",
                    pref_key, value[:60], weight,
                )
                action = "updated"
            else:
                logger.info("[memory_tool] reinforced %s (weight=%d)", pref_key, weight)
                action = "reinforced"
            confidence = round(existing.confidence or 0.0, 3)
        else:
            create_preference(
                db=db,
                preference_key=pref_key,
                preference_value=value,
                strength=strength,
                source="tool:save_to_memory",
                namespace=config["namespace"],
                context_pointer="tool:save_to_memory",
            )
            logger.info(
                "[memory_tool] created %s = %r (weight=%d, strength=%s, permanence=%s)",
                pref_key, value[:60], weight, strength.value, permanence,
            )
            action = "saved"
            # Re-query for the freshly-computed confidence
            fresh = (
                db.query(PreferenceRecord)
                .filter(PreferenceRecord.preference_key == pref_key)
                .first()
            )
            confidence = round(fresh.confidence if fresh else 0.0, 3)

        # Mirror biographical hard facts into the Tier 1 identity store
        # so they end up in the always-injected identity block.
        # Mirror on BOTH permanent and evolving permanence (because
        # current_location is legitimately evolving but still belongs
        # in the always-injected block).
        #
        # Phase 1 audit (2026-05-02): the LLM tool path was the second
        # writer that corrupted identity.json (alongside knowledge_extractor).
        # Default behaviour is now LOG-ONLY: the PreferenceRecord still
        # gets written so the fact isn't lost, but the Tier 1 mirror is
        # suppressed until Phase 3's arbiter ships. Set
        # ASTRA_MEMORY_TOOLS_IDENTITY_WRITE=true to restore the old
        # behaviour. The result includes identity_mirror="blocked_phase1"
        # so the conversation LLM knows to be careful in its reply
        # ("I've noted that" is fine; "I've saved that as your address"
        # is not).
        identity_mirror = None
        if (config["mirror_to_identity"]
                and key in _IDENTITY_ROUTED_KEYS
                and permanence in ("permanent", "evolving")):
            # Phase 3 audit (2026-05-02): observe via arbiter regardless of
            # mirror flag.
            try:
                from app.self_model.write_arbiter import propose as _arbiter_propose
                _arbiter_propose(
                    field_name=key,
                    proposed_value=value,
                    source="tool:save_to_memory",
                    evidence={
                        "category": category,
                        "weight": weight,
                        "permanence": permanence,
                        "reason": reason,
                    },
                )
            except Exception as _arb_err:
                logger.debug("[memory_tool] arbiter observe failed: %s", _arb_err)

            _mirror_enabled = os.getenv(
                "ASTRA_MEMORY_TOOLS_IDENTITY_WRITE", "false",
            ).lower() in ("1", "true", "yes")
            if not _mirror_enabled:
                logger.info(
                    "[memory_tool] identity mirror SUPPRESSED (Phase 1): "
                    "would have written %s = %r (source=tool:save_to_memory)",
                    key, value,
                )
                identity_mirror = "blocked_phase1"
            else:
                try:
                    from app.self_model.identity import get_identity_store
                    r = get_identity_store().set(
                        key, value, source="tool:save_to_memory",
                    )
                    identity_mirror = r.get("action")
                    logger.info(
                        "[memory_tool] mirrored %s to identity store: %s",
                        key, identity_mirror,
                    )
                except Exception as id_err:
                    logger.debug("[memory_tool] identity mirror failed: %s", id_err)

        result = {
            "saved": True,
            "action": action,
            "key": pref_key,
            "category": category,
            "value": value,
            "weight": weight,
            "permanence": permanence,
            "strength": strength.value,
            "confidence": confidence,
        }
        if identity_mirror:
            result["identity_mirror"] = identity_mirror
        return _format_result(result)

    except Exception as e:
        logger.error("[memory_tool] save_to_memory failed: %s", e)
        return f"save_to_memory failed: {e}"
    finally:
        db.close()


# =============================================================================
# update_memory
# =============================================================================

async def execute_update_memory(params: Dict[str, Any]) -> str:
    """
    Update the value of an existing memory fact.

    params:
        key:       the preference key (as returned by save_to_memory or
                   search_memory). May be the bare key or the full
                   "conv_extract:category:key" form.
        new_value: the replacement value
        reason:    why the update (optional, for audit trail)
    """
    from app.db import get_db_session
    from app.astra_memory.preference_service import update_preference_value

    key = str(params.get("key", "")).strip()
    new_value = str(params.get("new_value", "")).strip()
    reason = str(params.get("reason", "")).strip()

    if not key:
        return "update_memory failed: key is required"
    if not new_value:
        return "update_memory failed: new_value is required"

    db = get_db_session()
    try:
        # Resolve bare key to full preference_key if needed
        pref_key = _resolve_preference_key(db, key)
        if not pref_key:
            return f"update_memory failed: no preference found for key={key!r}"

        result = update_preference_value(
            db=db,
            preference_key=pref_key,
            new_value=new_value,
            is_explicit=True,
            context_pointer=f"tool:update_memory:{reason[:60]}",
        )
        if not result:
            return f"update_memory failed: update_preference_value returned None for {pref_key}"

        return _format_result({
            "saved": True,
            "action": "updated",
            "key": pref_key,
            "new_value": new_value,
            "confidence": round(result.confidence or 0.0, 3),
            "reason": reason or "not specified",
        })
    except Exception as e:
        logger.error("[memory_tool] update_memory failed: %s", e)
        return f"update_memory failed: {e}"
    finally:
        db.close()


# =============================================================================
# forget_memory
# =============================================================================

async def execute_forget_memory(params: Dict[str, Any]) -> str:
    """
    Mark a memory fact for expiry (soft delete via status).

    The evidence ledger is preserved for audit. Only the preference's
    status changes to EXPIRED; retrieval queries will ignore it.

    params:
        key:    the preference key to forget
        reason: why (optional, for audit trail)
    """
    from app.db import get_db_session
    from app.astra_memory.preference_service import expire_preference

    key = str(params.get("key", "")).strip()
    reason = str(params.get("reason", "")).strip() or "user requested"

    if not key:
        return "forget_memory failed: key is required"

    db = get_db_session()
    try:
        pref_key = _resolve_preference_key(db, key)
        if not pref_key:
            return f"forget_memory: no preference found for key={key!r}"

        result = expire_preference(db, pref_key, reason=reason)
        if not result:
            return f"forget_memory failed: could not expire {pref_key}"

        return _format_result({
            "saved": True,
            "action": "forgotten",
            "key": pref_key,
            "reason": reason,
        })
    except Exception as e:
        logger.error("[memory_tool] forget_memory failed: %s", e)
        return f"forget_memory failed: {e}"
    finally:
        db.close()


# =============================================================================
# search_memory
# =============================================================================

async def execute_search_memory(params: Dict[str, Any]) -> str:
    """
    Search stored memory by keyword, category, or both.

    Returns a compact list of matching facts (key, value, confidence,
    category) suitable for the LLM to use when answering "what do you
    know about X" style questions.

    params:
        query:    free-text search (matches against value)
        category: optional filter (biographical / preference / etc.)
        limit:    max results to return (default 20, cap 50)
    """
    from app.db import get_db_session
    from app.astra_memory.preference_models import (
        PreferenceRecord, RecordStatus,
    )

    query = str(params.get("query", "")).strip()
    category = str(params.get("category", "")).strip().lower()
    limit = min(50, max(1, int(params.get("limit", 20))))

    db = get_db_session()
    try:
        q = db.query(PreferenceRecord).filter(
            PreferenceRecord.status == RecordStatus.ACTIVE,
        )
        if category and category in _CATEGORY_CONFIG:
            q = q.filter(
                PreferenceRecord.preference_key.like(f"conv_extract:{category}:%"),
            )
        rows = q.order_by(PreferenceRecord.confidence.desc()).limit(200).all()

        # Post-filter on value match (SQLAlchemy's LIKE on JSON columns
        # is provider-specific; keep this simple and in Python).
        results: List[Dict[str, Any]] = []
        q_lower = query.lower() if query else None
        for r in rows:
            val_str = str(r.preference_value or "")
            if q_lower and q_lower not in val_str.lower() and q_lower not in r.preference_key.lower():
                continue
            results.append({
                "key": r.preference_key,
                "value": val_str,
                "confidence": round(r.confidence or 0.0, 3),
                "strength": r.strength.value if r.strength else "soft",
                "namespace": r.namespace,
            })
            if len(results) >= limit:
                break

        # Also include identity store for biographical queries (always-injected
        # facts that aren't in the preference table)
        identity_results: List[Dict[str, Any]] = []
        if not category or category == "biographical":
            try:
                from app.self_model.identity import get_identity_store
                store = get_identity_store()
                for field_name, f in store.all().items():
                    val_str = str(f.value)
                    if q_lower and q_lower not in val_str.lower() and q_lower not in field_name.lower():
                        continue
                    identity_results.append({
                        "field": field_name,
                        "value": val_str,
                        "source": f.source,
                    })
            except Exception:
                pass

        return _format_result({
            "saved": False,  # search is read-only
            "action": "searched",
            "query": query or "(all)",
            "category": category or "(all)",
            "preference_results": results,
            "identity_results": identity_results,
            "total_preferences": len(results),
            "total_identity": len(identity_results),
        })
    except Exception as e:
        logger.error("[memory_tool] search_memory failed: %s", e)
        return f"search_memory failed: {e}"
    finally:
        db.close()


# =============================================================================
# save_residence
# =============================================================================

async def execute_save_residence(params: Dict[str, Any]) -> str:
    """
    Append (or update) an entry in the user's residence_history.

    Structured residence timeline belongs in the identity store - it's
    always-injected, same tier as biographical facts, deduped by place.

    params:
        place:          location name (e.g. "Exeter", "Algarve, Portugal")
        from_year:      year moved in (int or string; optional)
        to_year:        year left (optional, omit for current)
        duration_years: explicit duration (optional, when years aren't known)
        notes:          short free-text note (e.g. "teenage years")
    """
    place = str(params.get("place", "")).strip()
    if not place:
        return "save_residence failed: place is required"

    entry: Dict[str, Any] = {"place": place}
    for src, dst in (
        ("from_year", "from"),
        ("to_year", "to"),
        ("duration_years", "duration_years"),
        ("notes", "notes"),
    ):
        v = params.get(src)
        if v not in (None, "", []):
            entry[dst] = v

    try:
        from app.self_model.identity import get_identity_store
        store = get_identity_store()
        result = store.append_to_list(
            field="residence_history",
            entry=entry,
            dedupe_key="place",
            source="tool:save_residence",
        )
    except Exception as e:
        logger.error("[memory_tool] save_residence failed: %s", e)
        return f"save_residence failed: {e}"

    return _format_result({
        "saved": result.get("action") in ("appended", "updated"),
        "action": result.get("action"),
        "field": "residence_history",
        "entry": entry,
        "list_size": result.get("list_size"),
    })


# =============================================================================
# helpers
# =============================================================================

def _resolve_preference_key(db, key: str) -> Optional[str]:
    """
    Accept either a bare key ("favourite_band") or a full preference_key
    ("conv_extract:preference:favourite_band"). Returns the full key if
    a matching preference exists, None otherwise.
    """
    from app.astra_memory.preference_models import PreferenceRecord

    # Exact match first
    exact = (
        db.query(PreferenceRecord)
        .filter(PreferenceRecord.preference_key == key)
        .first()
    )
    if exact:
        return exact.preference_key

    # Bare key - search for any preference ending in :key
    candidates = (
        db.query(PreferenceRecord)
        .filter(PreferenceRecord.preference_key.like(f"%:{key}"))
        .all()
    )
    if len(candidates) == 1:
        return candidates[0].preference_key
    if len(candidates) > 1:
        # Ambiguous - prefer the highest-confidence match
        candidates.sort(key=lambda p: p.confidence or 0.0, reverse=True)
        return candidates[0].preference_key
    return None


def _format_result(d: Dict[str, Any]) -> str:
    """Format a result dict as a readable multi-line string for the LLM."""
    import json
    try:
        return json.dumps(d, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(d)

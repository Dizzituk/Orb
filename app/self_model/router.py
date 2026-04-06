# FILE: app/self_model/router.py
"""
Self-Model API Router

FastAPI endpoints for the Self-Model & Evolution Layer.
Exposes capability map, user model, pattern observations,
suggestions, and evolution journal to the frontend.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.self_model.capability_map import get_capability_map, refresh_capabilities
from app.self_model.journal import get_journal
from app.self_model.observer import get_observer
from app.self_model.suggestions import get_suggestion_engine
from app.self_model.user_model import get_user_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/self-model", tags=["self-model"])


# ── Request schemas ───────────────────────────────────────

class CanDoRequest(BaseModel):
    query: str


class SuggestionActionRequest(BaseModel):
    suggestion_id: str


class FactCorrectionRequest(BaseModel):
    category: str
    key: str
    new_value: str


class AddFactRequest(BaseModel):
    category: str
    key: str
    value: str
    source: str = "user_provided"

class DeclineFactRequest(BaseModel):
    category: str
    key: str


# ── Capabilities ──────────────────────────────────────────

@router.get("/capabilities")
async def list_capabilities():
    cap_map = get_capability_map()
    return {
        "capabilities": [c.to_dict() for c in cap_map.get_all()],
        "summary": cap_map.summary(),
    }


@router.get("/capabilities/active")
async def list_active_capabilities():
    return {"capabilities": [c.to_dict() for c in get_capability_map().get_active()]}


@router.get("/capabilities/in-development")
async def list_dev_capabilities():
    return {"capabilities": [c.to_dict() for c in get_capability_map().get_in_development()]}


@router.post("/capabilities/can-i")
async def can_i_do(request: CanDoRequest):
    return get_capability_map().can_do(request.query)


@router.post("/capabilities/refresh")
async def trigger_refresh():
    return refresh_capabilities()


@router.get("/capabilities/plain")
async def capabilities_plain():
    return {"text": get_capability_map().to_plain_language()}


# ── User Model ────────────────────────────────────────────

@router.get("/user")
async def get_user_facts():
    model = get_user_model()
    return {
        "facts": [f.to_dict() for f in model.get_all()],
        "summary": model.summary(),
    }


@router.get("/user/{category}")
async def get_user_facts_by_category(category: str):
    return {"facts": [f.to_dict() for f in get_user_model().get_by_category(category)]}


@router.post("/user/correct")
async def correct_user_fact(request: FactCorrectionRequest):
    fact = get_user_model().correct_fact(request.category, request.key, request.new_value)
    if not fact:
        return {"status": "not_found"}
    journal = get_journal()
    journal.record_learning(
        f"User corrected {request.category}:{request.key}",
        f"New value: {request.new_value}",
    )
    return {"status": "corrected", "fact": fact.to_dict()}

@router.post("/user/decline")
async def decline_user_fact(request: DeclineFactRequest):
    """Remove a fact the user has declined/rejected from the self-model."""
    removed = get_user_model().remove_fact(request.category, request.key)
    if not removed:
        return {"status": "not_found"}
    journal = get_journal()
    journal.record_learning(
        f"User declined {request.category}:{request.key}",
        "Fact removed from self-model by user request",
    )
    return {"status": "declined", "category": request.category, "key": request.key}

@router.post("/user/confirm")
async def confirm_user_fact(request: DeclineFactRequest):
    """Confirm a fact is correct - reinforces it (increases confidence/count)."""
    model = get_user_model()
    existing = model.get_fact(request.category, request.key)
    if not existing:
        return {"status": "not_found"}
    from app.self_model.models import UserFact
    # Re-add with same value — add_fact handles reinforcement
    reinforced = model.add_fact(UserFact(
        category=existing.category,
        key=existing.key,
        value=existing.value,
        source="user_confirmed",
    ))
    journal = get_journal()
    journal.record_learning(
        f"User confirmed {request.category}:{request.key}",
        f"Reinforced to count={reinforced.reinforcement_count}, confidence={reinforced.confidence}",
    )
    return {"status": "confirmed", "fact": reinforced.to_dict()}



@router.post("/user/add")
async def add_user_fact(request: AddFactRequest):
    from app.self_model.models import UserFact
    fact = UserFact(
        category=request.category,
        key=request.key,
        value=request.value,
        source=request.source,
    )
    get_user_model().add_fact(fact)
    journal = get_journal()
    journal.record_learning(f"New fact added: {request.category}:{request.key}", request.value)
    return {"status": "added", "fact": fact.to_dict()}


@router.get("/user/plain")
async def user_plain():
    return {"text": get_user_model().to_plain_language()}


# ── Patterns ──────────────────────────────────────────────

@router.get("/patterns")
async def get_patterns():
    observer = get_observer()
    return {
        "patterns": [p.to_dict() for p in observer.get_all_patterns()],
        "summary": observer.summary(),
    }


@router.get("/patterns/actionable")
async def get_actionable_patterns():
    return {"patterns": [p.to_dict() for p in get_observer().get_actionable_patterns()]}


@router.get("/patterns/plain")
async def patterns_plain():
    return {"text": get_observer().to_plain_language()}


# ── Suggestions ───────────────────────────────────────────

@router.get("/suggestions")
async def get_suggestions():
    engine = get_suggestion_engine()
    return {
        "suggestions": [s.to_dict() for s in engine.get_all()],
        "summary": engine.summary(),
    }


@router.get("/suggestions/pending")
async def get_pending_suggestions():
    return {"suggestions": [s.to_dict() for s in get_suggestion_engine().get_pending()]}


@router.post("/suggestions/approve")
async def approve_suggestion(request: SuggestionActionRequest):
    s = get_suggestion_engine().approve(request.suggestion_id)
    if not s:
        return {"status": "not_found"}
    journal = get_journal()
    journal.record_approval(s.suggestion_id, s.title)
    return {"status": "approved", "suggestion": s.to_dict()}


@router.post("/suggestions/reject")
async def reject_suggestion(request: SuggestionActionRequest):
    s = get_suggestion_engine().reject(request.suggestion_id)
    if not s:
        return {"status": "not_found"}
    journal = get_journal()
    journal.record_rejection(s.suggestion_id, s.title)
    return {"status": "rejected", "suggestion": s.to_dict()}


@router.post("/suggestions/defer")
async def defer_suggestion(request: SuggestionActionRequest):
    s = get_suggestion_engine().defer(request.suggestion_id)
    if not s:
        return {"status": "not_found"}
    journal = get_journal()
    journal.record_deferral(s.suggestion_id, s.title)
    return {"status": "deferred", "suggestion": s.to_dict()}


@router.get("/suggestions/plain")
async def suggestions_plain():
    return {"text": get_suggestion_engine().to_plain_language()}


# ── Journal ───────────────────────────────────────────────

@router.get("/journal")
async def get_journal_entries(n: int = 20):
    journal = get_journal()
    return {
        "entries": [e.to_dict() for e in journal.get_recent(n)],
        "summary": journal.summary(),
    }


@router.get("/journal/plain")
async def journal_plain(n: int = 10):
    return {"text": get_journal().to_plain_language(n)}


# ── Overview ──────────────────────────────────────────────

@router.get("/overview")
async def self_model_overview():
    """Complete overview of the Self-Model state."""
    return {
        "capabilities": get_capability_map().summary(),
        "user_model": get_user_model().summary(),
        "patterns": get_observer().summary(),
        "suggestions": get_suggestion_engine().summary(),
        "journal": get_journal().summary(),
    }


@router.get("/overview/plain")
async def self_model_overview_plain():
    """Plain-language overview of everything the Self-Model knows."""
    parts = [
        get_capability_map().to_plain_language(),
        "\n---\n",
        get_user_model().to_plain_language(),
        "\n---\n",
        get_observer().to_plain_language(),
        "\n---\n",
        get_suggestion_engine().to_plain_language(),
        "\n---\n",
        get_journal().to_plain_language(),
    ]
    return {"text": "\n".join(parts)}

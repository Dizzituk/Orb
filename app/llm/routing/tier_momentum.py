# FILE: app/llm/routing/tier_momentum.py
# Purpose: Conversation-level tier momentum — decaying complexity signal so model
#          tiering moves BOTH directions within a session.
# Called-by: app.llm.routing.chat_model_selection
# Depends-on: stdlib only
# Last-renovated: 2026-06-12
"""
Tier momentum (memory architecture Job 6, 2026-06-12).

Before this module, tier selection was per-message keywords plus a sticky
latch: any complexity-driven escalation called set_sticky_model(), and sticky
is consulted before complexity on every later turn — so ONE architectural
question pinned the whole session at the top tier, and de-escalation never
happened. The reverse problem would appear the moment the latch was removed:
one casual aside ("ha, fair enough") mid-deep-session would drop straight
back to the cheap model.

This module adds the light, deterministic middle ground:

  - Every routed turn records a complexity score (from the classifier's own
    signals — no new NLP).
  - Momentum = exponentially decayed sum over the last few turns.
  - A casual turn during high momentum HOLDS the elevated tier (bounded lag);
    momentum decays each casual turn, so after DECAY-driven drop below the
    release threshold the session settles back to the cheap tier.
  - Architectural turns raise momentum; sustained casual chat empties it.

Deterministic and inspectable: every decision is logged with its reason and
kept in a per-process ring buffer (get_recent_decisions()).

Explicit user choices (frontend dropdown, "use opus", cognitive cues, agentic
tabs) still use the permanent sticky latch — momentum only governs
complexity-driven escalation.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tuning (env-overridable, deterministic defaults) ─────────────────
# Per-turn decay: each earlier turn contributes score * DECAY^age.
DECAY = float(os.getenv("TIER_MOMENTUM_DECAY", "0.55"))
WINDOW = int(os.getenv("TIER_MOMENTUM_WINDOW", "6"))
# Hold the elevated tier while momentum >= HOLD_THRESHOLD.
HOLD_THRESHOLD = float(os.getenv("TIER_MOMENTUM_HOLD", "1.0"))

# Per-turn scores by what the turn actually was
SCORE_ARCHITECT = 2.0
SCORE_REASONING = 1.2
SCORE_EXPLORE = 0.5
SCORE_CASUAL = 0.0

_DECISION_RING_SIZE = 200

# Per-project turn-score history and held models
_turn_scores: Dict[int, Deque[float]] = {}
_held_model: Dict[int, Tuple[str, str]] = {}
_decisions: Deque[dict] = deque(maxlen=_DECISION_RING_SIZE)


def _scores(project_id: int) -> Deque[float]:
    if project_id not in _turn_scores:
        _turn_scores[project_id] = deque(maxlen=WINDOW)
    return _turn_scores[project_id]


def score_for_turn(tier_label: str) -> float:
    """Map a routed tier label onto a momentum score."""
    return {
        "architect": SCORE_ARCHITECT,
        "reasoning": SCORE_REASONING,
        "explore": SCORE_EXPLORE,
    }.get(tier_label, SCORE_CASUAL)


def current_momentum(project_id: int) -> float:
    """Decayed sum of recent turn scores. Newest turn weights most."""
    scores = _scores(project_id)
    momentum = 0.0
    for age, score in enumerate(reversed(scores)):
        momentum += score * (DECAY ** age)
    return round(momentum, 4)


def record_turn(
    project_id: int,
    tier_label: str,
    provider: str,
    model: str,
    reason: str,
) -> float:
    """Record the routed turn, log the decision, return new momentum.

    Call AFTER the tier decision for the turn. Elevated tiers also update
    the held (provider, model) so a momentum hold knows what to keep.
    """
    score = score_for_turn(tier_label)
    _scores(project_id).append(score)
    momentum = current_momentum(project_id)

    if tier_label in ("architect", "reasoning"):
        _held_model[project_id] = (provider, model)

    decision = {
        "ts": time.time(),
        "project_id": project_id,
        "tier": tier_label,
        "provider": provider,
        "model": model,
        "score": score,
        "momentum": momentum,
        "reason": reason,
    }
    _decisions.append(decision)
    logger.info(
        "[tier_momentum] project=%s tier=%s model=%s/%s score=%.2f momentum=%.2f reason=%s",
        project_id, tier_label, provider, model, score, momentum, reason,
    )
    print(
        f"[TIER_DECISION] tier={tier_label} model={provider}/{model} "
        f"momentum={momentum:.2f} reason={reason}"
    )
    return momentum


def momentum_hold(project_id: int) -> Optional[Tuple[str, str, float]]:
    """When the conversation still has elevated momentum, return the held
    (provider, model, momentum) to keep using; None when released.

    The hold only ever returns a model that an elevated turn actually used —
    it never invents an escalation on its own.
    """
    momentum = current_momentum(project_id)
    held = _held_model.get(project_id)
    if held and momentum >= HOLD_THRESHOLD:
        return held[0], held[1], momentum
    return None


def get_recent_decisions(limit: int = 50) -> List[dict]:
    """Inspectable decision log (newest last)."""
    return list(_decisions)[-limit:]


def reset(project_id: Optional[int] = None) -> None:
    """Testing/diagnostics hook."""
    if project_id is None:
        _turn_scores.clear()
        _held_model.clear()
        _decisions.clear()
    else:
        _turn_scores.pop(project_id, None)
        _held_model.pop(project_id, None)


__all__ = [
    "record_turn",
    "current_momentum",
    "momentum_hold",
    "score_for_turn",
    "get_recent_decisions",
    "reset",
    "HOLD_THRESHOLD",
    "DECAY",
    "WINDOW",
]

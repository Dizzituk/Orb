# FILE: app/llm/routing/tier_momentum.py
# Purpose: Conversation-level tier momentum — decaying complexity signal so model
#          tiering moves BOTH directions within a session.
# Called-by: app.llm.routing.chat_model_selection
# Depends-on: app.llm.frontier_models (step-down resolution only)
# Last-renovated: 2026-07-01
"""
Tier momentum (memory architecture Job 6, 2026-06-12; retuned 2026-07-01).

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
  - A casual turn during high momentum HOLDS an elevated tier (bounded lag);
    momentum decays each casual turn, so the session settles back to the
    cheap tier.
  - Architectural turns raise momentum; sustained casual chat empties it.

Lane A retune (2026-07-01), from the 2026-07-01 chat log where one
architect turn held Opus for ~2 more turns (PSU question + web search both
billed at the top tier):

  1. DECAY default 0.55 -> 0.45, so a single architect spike
     (score 2.0) drops below HOLD_THRESHOLD=1.0 after ONE casual turn.
  2. A held casual turn STEPS DOWN: when the held tier is "architect",
     the hold returns the REASONING/FRONTIER-tier model (env-resolved)
     instead of keeping the top model. Top-tier pricing never
     outlives the architect turn itself. Kill-switch:
     TIER_MOMENTUM_STEPDOWN=0 restores hold-what-was-used.
  3. All knobs env-tunable: TIER_MOMENTUM_DECAY / _WINDOW / _HOLD /
     _STEPDOWN / _SCORE_ARCHITECT / _SCORE_REASONING / _SCORE_EXPLORE /
     _SCORE_CASUAL (read at import; restart to apply, like the rest of .env).

Deterministic and inspectable: every decision is logged with its reason and
kept in a per-process ring buffer (get_recent_decisions()).

Explicit user choices (frontend dropdown, "use opus", agentic tabs) still
use the permanent sticky latch — momentum only governs complexity-driven
escalation.
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
# 0.45 (2026-07-01): one architect spike (2.0) decays to 0.9 < 1.0 after a
# single casual turn — the hold lasts at most ~1 light turn per spike.

def _env_float(name: str, default: float) -> float:
    """Env float that survives present-but-empty/garbage values — a bad
    TIER_MOMENTUM_* edit must not crash the first chat turn at import."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("[tier_momentum] ignoring invalid %s=%r", name, raw)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("[tier_momentum] ignoring invalid %s=%r", name, raw)
        return default


DECAY = _env_float("TIER_MOMENTUM_DECAY", 0.45)
WINDOW = _env_int("TIER_MOMENTUM_WINDOW", 6)
# Hold an elevated tier while momentum >= HOLD_THRESHOLD.
HOLD_THRESHOLD = _env_float("TIER_MOMENTUM_HOLD", 1.0)
# Step a held ARCHITECT turn down to the reasoning/frontier tier (1=on).
STEPDOWN_ENABLED = os.getenv("TIER_MOMENTUM_STEPDOWN", "1").strip().lower() not in (
    "0", "false", "no", "off",
)

# Per-turn scores by what the turn actually was (env-tunable, 2026-07-01)
SCORE_ARCHITECT = _env_float("TIER_MOMENTUM_SCORE_ARCHITECT", 2.0)
SCORE_REASONING = _env_float("TIER_MOMENTUM_SCORE_REASONING", 1.2)
SCORE_EXPLORE = _env_float("TIER_MOMENTUM_SCORE_EXPLORE", 0.5)
SCORE_CASUAL = _env_float("TIER_MOMENTUM_SCORE_CASUAL", 0.0)

_DECISION_RING_SIZE = 200

# Per-project turn-score history and held models.
# _held_model values are (provider, model, tier_label) — the tier label
# lets momentum_hold() step an architect hold down to the mid tier.
_turn_scores: Dict[int, Deque[float]] = {}
_held_model: Dict[int, Tuple[str, str, str]] = {}
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
    the held (provider, model, tier) so a momentum hold knows what to keep.
    """
    score = score_for_turn(tier_label)
    _scores(project_id).append(score)
    momentum = current_momentum(project_id)

    if tier_label in ("architect", "reasoning"):
        _held_model[project_id] = (provider, model, tier_label)

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


def _stepped_down(provider: str, model: str) -> Tuple[str, str]:
    """Resolve the mid tier for an architect hold (env-only, guarded).

    REASONING_* -> FRONTIER_* -> DEFAULT_* via the one resolver. If nothing
    resolves (bare env), keep the held model rather than fail the turn.
    """
    try:
        from app.llm.frontier_models import get_role_model
        return get_role_model("REASONING", "FRONTIER")
    except Exception as e:
        logger.warning(
            "[tier_momentum] step-down unresolved (%s) — keeping held model", e,
        )
        return provider, model


def momentum_hold(project_id: int) -> Optional[Tuple[str, str, float]]:
    """When the conversation still has elevated momentum, return the
    (provider, model, momentum) to keep using; None when released.

    The hold only ever returns a model an elevated turn actually earned —
    it never invents an escalation on its own. Retune 2026-07-01: a held
    ARCHITECT tier steps DOWN to the reasoning/frontier tier, so casual
    turns never ride the top model (TIER_MOMENTUM_STEPDOWN=0 to disable).
    """
    momentum = current_momentum(project_id)
    held = _held_model.get(project_id)
    if held and momentum >= HOLD_THRESHOLD:
        provider, model, tier_label = held
        if tier_label == "architect" and STEPDOWN_ENABLED:
            provider, model = _stepped_down(provider, model)
        return provider, model, momentum
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
    "STEPDOWN_ENABLED",
    "SCORE_ARCHITECT",
    "SCORE_REASONING",
    "SCORE_EXPLORE",
    "SCORE_CASUAL",
]

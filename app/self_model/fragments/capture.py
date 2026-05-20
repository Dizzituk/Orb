# FILE: app/self_model/fragments/capture.py
"""
Capture pipeline — user message → fragment.

Called from the chat entry-points (Bridge + web). Runs:
  1. classify() → (signal_type, claim_type)
  2. generate_embedding(text) → embedding vector (Gemini Embedding 2)
  3. append fragment to store
  4. assign_fragment_to_theme() to incrementally cluster

Fire-and-forget. Never raises. Logs outcome.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.self_model.fragments.classifier import classify
from app.self_model.fragments.cluster import assign_fragment_to_theme
from app.self_model.fragments.models import (
    Fragment, SignalType, ClaimType, SIGNAL_BASE_WEIGHT,
)
from app.self_model.fragments.store import get_fragment_store

logger = logging.getLogger(__name__)

# Minimum message length worth capturing as a fragment
_MIN_LEN = 8

# Voice-to-text from the keyword spotter / ambient listener produces a lot
# of short PASSIVE/UNKNOWN fragments ("We're done", "okay", "um", "got
# messages to start with"). These have no signal the classifier can pin
# and pollute the theme store. Skip short classifier-negative fragments
# to keep the store signal-dense.
_NOISE_PASSIVE_MAX_LEN = 40
_VOICE_SOURCES = {"bridge", "voice_ambient", "chat-and-speak", "bridge-tts"}


def _looks_like_voice_noise(
    message: str,
    signal_type: str,
    claim_type: str,
    source: str,
) -> bool:
    """True if the fragment is almost certainly voice-to-text noise.

    Criteria (all must hold):
      - source looks voice-ish
      - classifier matched nothing (PASSIVE signal + UNKNOWN claim)
      - message is short (< _NOISE_PASSIVE_MAX_LEN chars)
    A typed short message with intent still passes because the
    classifier will have flagged DECLARATION / CORRECTION / etc.
    """
    if signal_type != "passive" or claim_type != "unknown":
        return False
    if len(message) >= _NOISE_PASSIVE_MAX_LEN:
        return False
    src_lower = (source or "").lower()
    return any(v in src_lower for v in _VOICE_SOURCES)


def capture_fragment(
    message: str,
    source: str = "chat",
    session_id: Optional[int] = None,
    project_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the full capture pipeline for a user message.
    Returns: {"fragment_id": id, "theme_id": id, "signal": ..., "claim": ...}
    or {"skipped": reason}.
    """
    if not message or len(message.strip()) < _MIN_LEN:
        return {"skipped": "too_short"}

    # 1. Classify
    signal, claim = classify(message)

    # 1a. Voice-to-text noise filter: skip short classifier-negative
    # fragments coming from voice sources. These are almost always
    # transcription artefacts ("we're done", "okay", "a bit of") that
    # add no signal but pollute theme clustering.
    if _looks_like_voice_noise(message.strip(), signal.value, claim.value, source):
        logger.debug(
            "[fragment_capture] skipped voice noise: src=%s msg=%r",
            source, message.strip()[:50],
        )
        return {"skipped": "voice_noise"}

    # 2. Embed
    embedding = None
    embedding_model = ""
    try:
        from app.embeddings.gemini_provider import generate_embedding, GEMINI_EMBEDDING_MODEL
        embedding = generate_embedding(message)
        embedding_model = GEMINI_EMBEDDING_MODEL if embedding else ""
    except Exception as exc:
        logger.debug("[fragment_capture] embedding failed: %s", exc)

    # 3. Build fragment
    fragment = Fragment(
        text=message.strip(),
        signal_type=signal.value,
        claim_type=claim.value,
        base_weight=SIGNAL_BASE_WEIGHT.get(signal.value, 1.0),
        session_id=session_id,
        project_id=project_id,
        source=source,
        embedding=embedding,
        embedding_model=embedding_model,
    )

    # 4. Append
    store = get_fragment_store()
    store.append(fragment)

    # 5. Cluster (only if we have an embedding)
    theme_id = None
    if embedding:
        try:
            theme_id = assign_fragment_to_theme(fragment)
        except Exception as exc:
            logger.debug("[fragment_capture] clustering failed: %s", exc)

    return {
        "fragment_id": fragment.fragment_id,
        "theme_id": theme_id,
        "signal": signal.value,
        "claim": claim.value,
        "embedded": embedding is not None,
    }
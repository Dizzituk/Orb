# FILE: app/self_model/fragments/models.py
"""
Fragment store data model.

A "fragment" is a single recorded mention — one piece of evidence
captured from one user utterance. Fragments are the atomic unit
of Astra's long-term associative memory.

Fragments are APPEND-ONLY. They are never deleted. Weight decays
over time, but the record stays forever so cold-layer semantic
retrieval can always find them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SignalType(str, Enum):
    """Strength of signal that created this fragment."""
    PASSIVE      = "passive"       # weight +1: incidental mention in a message
    DECLARATION  = "declaration"   # weight +3: explicit "I want / I like / I'm into X"
    CORRECTION   = "correction"    # weight +10: "I told you to X, you're not"
    REINFORCE    = "reinforce"     # weight +5: affirmative response to a surfaced memory


class ClaimType(str, Enum):
    """What kind of claim this fragment represents."""
    BEHAVIOUR   = "behaviour"     # how Astra should act / communicate
    INTEREST    = "interest"      # topic the user cares about
    PREFERENCE  = "preference"    # stated preference (tool, style, format)
    STANCE      = "stance"        # user's position on something
    UNKNOWN     = "unknown"       # classifier unsure — still stored


# Signal → base weight (per-fragment weight at time of creation)
SIGNAL_BASE_WEIGHT: Dict[str, float] = {
    SignalType.PASSIVE.value:     1.0,
    SignalType.DECLARATION.value: 3.0,
    SignalType.CORRECTION.value:  10.0,
    SignalType.REINFORCE.value:   5.0,
}


@dataclass
class Fragment:
    """
    One recorded mention.

    Invariant: `base_weight` and `created_at` are set once and never
    change. Current effective weight is *computed* from these at read
    time via the decay function (see weights.py).
    """
    fragment_id: str = field(default_factory=lambda: str(uuid4())[:12])
    text: str = ""
    signal_type: str = SignalType.PASSIVE.value
    claim_type: str = ClaimType.UNKNOWN.value
    base_weight: float = 1.0
    created_at: str = field(default_factory=_utc_now)
    # Context pointers so cold surfacing can show "you said this in session X"
    session_id: Optional[int] = None
    project_id: Optional[int] = None
    source: str = ""  # free-text source label (e.g. "bridge:42", "chat_capture")
    # Embedding is a list of floats — stored inline in JSON for simplicity.
    # Populated by the capture pipeline after embedding call succeeds.
    embedding: Optional[List[float]] = None
    embedding_model: str = ""
    # Theme membership (filled in by clustering pass, may be None if orphan)
    theme_id: Optional[str] = None
    # Surface history: when this fragment (or its theme) was last shown to user
    last_surfaced_at: Optional[str] = None
    # Dismissal flag: set when user dismisses a surfaced memory — accelerates decay
    dismissed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fragment":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
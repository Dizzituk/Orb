# Purpose: trust gate
# Called-by: app.optimize.loop_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    SUPERVISED = "supervised"
    SEMI_AUTO = "semi_auto"
    AUTONOMOUS = "autonomous"


_current_trust_level = TrustLevel.SEMI_AUTO


def get_trust_level() -> TrustLevel:
    return _current_trust_level


def set_trust_level(trust_level: TrustLevel | str) -> TrustLevel:
    global _current_trust_level
    _current_trust_level = trust_level if isinstance(trust_level, TrustLevel) else TrustLevel(trust_level)
    return _current_trust_level


def can_auto_continue(trust_level: TrustLevel | str, evidence_strength: float) -> bool:
    level = trust_level if isinstance(trust_level, TrustLevel) else TrustLevel(trust_level)
    if level == TrustLevel.SUPERVISED:
        return False
    if level == TrustLevel.SEMI_AUTO:
        return evidence_strength >= 0.65
    return evidence_strength >= 0.35

# FILE: app/scene_director/research/gate.py
# Purpose: Decide whether an intent needs FACTS before composing — triggers on specific/real/
#          factual/historical subjects the model shouldn't invent; conservative by default so
#          imaginative scenes skip research and pay no latency.
# Called-by: app.scene_director.research (maybe_research)
# Depends-on: stdlib only (heuristic); optional LLM classifier behind a flag
# Last-renovated: 2026-06-13
"""should_research(intent, era) -> (bool, reason).

Explainable heuristic first (no latency): research when the brief references a real,
documented, or historical subject — a year, an era/history cue, or a named real place/
landmark/event. Default NO. An optional LLM classifier (SCENE_RESEARCH_GATE_LLM=on) can
override the heuristic for ambiguous briefs.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

# 4-digit years 1000-2099 (a strong "real/documented" signal).
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

_HISTORY_CUES = (
    "history", "historical", "ancient", "medieval", "mediaeval", "tudor", "victorian",
    "georgian", "jacobean", "roman", "viking", "renaissance", "war", "battle", "siege",
    "great fire", "plague", "revolution", "empire", "dynasty", "century", "bc", "ad",
)
# Words that usually denote a real, named place/landmark/institution.
_PLACE_CUES = (
    "bridge", "cathedral", "palace", "tower", "square", "museum", "castle", "abbey",
    "monument", "harbour", "harbor", "station", "stadium", "temple", "pyramid", "wall",
    "river thames", "thames", "colosseum", "parthenon", "acropolis", "landmark",
)
_REAL_CUES = ("the real ", "real-life", "real life", "actual ", "accurate", "based on")


def _enabled_gate_llm() -> bool:
    return os.getenv("SCENE_RESEARCH_GATE_LLM", "false").strip().lower() in ("true", "1", "yes")


def should_research(intent: str, era: Optional[str] = None) -> Tuple[bool, str]:
    """Conservative heuristic. Returns (needs_research, human reason)."""
    if not intent:
        return False, "empty intent"
    low = intent.lower()

    if era and era.strip().lower() not in ("", "modern", "any", "none"):
        return True, f"explicit historical era '{era}'"
    if _YEAR.search(intent):
        return True, "references a specific year"
    for cue in _HISTORY_CUES:
        if cue in low:
            return True, f"historical cue '{cue}'"
    for cue in _PLACE_CUES:
        if cue in low:
            return True, f"named real place/landmark cue '{cue}'"
    for cue in _REAL_CUES:
        if cue in low:
            return True, f"asks for real/accurate detail ('{cue.strip()}')"
    # A capitalised multi-word proper name mid-sentence (e.g. "Tower of London") is a
    # weaker signal — only trip when it is NOT just the leading word of the sentence.
    proper = re.findall(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-z]+(?:\s+(?:of|the|de|el)\s+)?[A-Z][a-z]+)\b", intent)
    if proper:
        return True, f"named proper noun '{proper[0]}'"
    return False, "imaginative/generic — no lookup needed"

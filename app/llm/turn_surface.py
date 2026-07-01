# FILE: app/llm/turn_surface.py
# Purpose: Per-turn surface tag (desktop|bridge) + pending artifact registry, via contextvars.
# Called-by: app.bridge.capability_layer (begin/drain), app.reports.tools_registration (read/add)
# Depends-on: stdlib contextvars only
# Last-renovated: 2026-07-01
"""
The chat tool loop passes no execution context down to registry tools, so a
surface-aware tool (ship a document to the phone vs open a desktop window)
needs a turn-scoped side channel. Contextvars ride the request's async task:
the bridge capability layer calls begin_turn("bridge") at turn entry; desktop
paths never set it and read the default.

Artifact flow mirrors the image-marker precedent in capability_layer: a tool
registers its artifact here, and the capability layer appends the
[ASTRA_ARTIFACT:kind:filename] markers to the reply AFTER the LLM stream
finishes — the model never has to echo a marker verbatim.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import List, Tuple

SURFACE_DESKTOP = "desktop"
SURFACE_BRIDGE = "bridge"

_surface: ContextVar[str] = ContextVar("astra_turn_surface", default=SURFACE_DESKTOP)
# Immutable tuple so ContextVar semantics stay copy-on-write.
_artifacts: ContextVar[Tuple[Tuple[str, str], ...]] = ContextVar("astra_turn_artifacts", default=())


def begin_turn(surface: str) -> None:
    """Tag the current async context as a turn on `surface` with no artifacts."""
    _surface.set(surface or SURFACE_DESKTOP)
    _artifacts.set(())


def get_turn_surface() -> str:
    return _surface.get()


def add_turn_artifact(kind: str, filename: str) -> None:
    _artifacts.set(_artifacts.get() + ((kind, filename),))


def drain_turn_artifacts() -> List[Tuple[str, str]]:
    pending = list(_artifacts.get())
    _artifacts.set(())
    return pending


def format_artifact_markers(artifacts: List[Tuple[str, str]]) -> str:
    """Markers ready to append to a reply ('' when none)."""
    return " ".join(f"[ASTRA_ARTIFACT:{kind}:{filename}]" for kind, filename in artifacts)

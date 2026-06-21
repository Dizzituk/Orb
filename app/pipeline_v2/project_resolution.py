# FILE: app/pipeline_v2/project_resolution.py
# Purpose: Resolve which build target(s) a natural-language message refers to.
# Called-by: app.pipeline_v2.target_registry (re-export shim; importers resolve through it)
# Depends-on: app.pipeline_v2.target_profiles
# Last-renovated: 2026-06-21
"""
Project resolution — split out of target_registry.py (SPLIT BATCH 9, 2026-06-21).

Message -> target scoring: detect_all_projects_from_message (union of every hit)
and resolve_project_from_message (scored single winner), with the weighted signal
banks (_SIGNAL_WEIGHTS / _PROJECT_SIGNALS / _SELF_WORK_SIGNALS) and the helpers
_check_explicit_build_target_directive / _score_projects_from_text. Reads the
shared _REGISTRY + the 4 profile literals from target_profiles. Moved VERBATIM;
target_registry.py re-exports these names so importers resolve unchanged.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from app.pipeline_v2.target_profiles import (
    _REGISTRY,
    ASTRA_BACKEND,
    ASTRA_BRIDGE,
    ASTRA_FRONTEND,
    DRIVER_COPILOT,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Fix 2 (2026-04-18): Scored target resolution
# ======================================================================
#
# Previous implementation was first-match-wins ordered Bridge → CoPilot
# → backend → frontend. Bridge 'driving domain' signals (on-road,
# hands-free, wake word, etc.) appear legitimately in non-Bridge specs
# — a Driver CoPilot spec that mentions 'on-road use' would silently
# resolve to Astra-Bridge. This caused catastrophic misrouting where
# SpecGate loaded the wrong project's codebase and produced a 0-file
# manifest.
#
# The scored resolver fixes this by:
#   1. Checking for an explicit 'Build target: X' directive — if present
#      and X is a registered project_id, that wins unconditionally.
#   2. Checking for explicit repo path or project_id mentions — these
#      are hard signals that cannot be fuzzy.
#   3. Scoring each project by signal hits (weighted by signal strength).
#   4. Returning the single highest-scoring project if there's a clear
#      winner, else falling back to history, else None.

# Weighted signal sets. Higher weight = stronger evidence for that target.
_SIGNAL_WEIGHTS: Dict[str, int] = {
    "product_name": 10,   # 'driver copilot', 'astra bridge'
    "path": 8,            # repo path literals
    "domain_unique": 4,   # 'wake word', 'yodel app' — unique to a project
    "domain_shared": 2,   # 'hands-free', 'on-road' — could apply to either Android project
    "stack": 1,           # 'kotlin', 'compose' — could be either Android project
}

# Per-project scored signal lists. Each tuple is (signal_str, weight_key).
_PROJECT_SIGNALS: Dict[str, list] = {
    ASTRA_BRIDGE.project_id: [
        ("astra bridge", "product_name"),
        ("astrabridge", "product_name"),
        ("astra-bridge", "product_name"),
        ("android bridge", "product_name"),
        ("phone bridge", "product_name"),
        ("mobile bridge", "product_name"),
        ("bridge app", "product_name"),
        ("companion app", "product_name"),
        # Domain — unique to the Bridge app, not Driver CoPilot
        ("wake word", "domain_unique"),
        ("wakeword", "domain_unique"),
        ("astra pause", "domain_unique"),
        ("astra continue", "domain_unique"),
        ("astra wake", "domain_unique"),
        ("astra repeat", "domain_unique"),
        ("astra playback", "domain_unique"),
        ("tts playback", "domain_unique"),
        ("tts auto-play", "domain_unique"),
        ("auto-play back", "domain_unique"),
        ("playback should resume", "domain_unique"),
        ("buffered cutoff", "domain_unique"),
        ("exoplayer", "domain_unique"),
        ("chatterbox", "domain_unique"),
        # Domain — shared across Android projects (weak signal on its own)
        ("in the van", "domain_shared"),
        ("in-van", "domain_shared"),
        ("on-road", "domain_shared"),
        ("on the road", "domain_shared"),
        ("while driving", "domain_shared"),
        ("driving use", "domain_shared"),
        ("hands-free", "domain_shared"),
        ("voice-command-friendly", "domain_shared"),
        ("voice command", "domain_shared"),
        # 'the bridge' is kept as a weak signal — it's generic enough
        # that scoring against stronger signals should overrule it
        ("the bridge", "domain_shared"),
    ],
    DRIVER_COPILOT.project_id: [
        ("driver copilot", "product_name"),
        ("drivercopilot", "product_name"),
        ("androiddrivercopilot", "product_name"),
        ("driver co-pilot", "product_name"),
        ("copilot app", "product_name"),
        # Domain — unique to Driver CoPilot
        ("yodel app", "domain_unique"),
        ("round planner", "domain_unique"),
        ("manifest stop", "domain_unique"),
        ("delivery driver", "domain_unique"),
        ("parcel delivery", "domain_unique"),
        ("delivery route", "domain_unique"),
        ("trouble stop", "domain_unique"),
        ("problem address", "domain_unique"),
        ("address finder", "domain_unique"),
        ("address resolution", "domain_unique"),
        ("daily log", "domain_unique"),
        ("pay settings", "domain_unique"),
        ("mellanoweth", "domain_unique"),  # Cornwall addresses we've seen
        # Shared Android stack signals (weak alone)
        ("kotlin", "stack"),
        ("jetpack", "stack"),
        ("compose", "stack"),
        ("composable", "stack"),
        ("jetpack compose", "stack"),
        ("gradle", "stack"),
        ("room database", "stack"),
        ("room entity", "stack"),
        ("viewmodel", "stack"),
        ("androidmanifest", "stack"),
        ("accessibility service", "stack"),
    ],
    ASTRA_BACKEND.project_id: [
        ("astra itself", "product_name"),
        ("astra backend", "product_name"),
        ("orb backend", "product_name"),
        ("the orb", "product_name"),
        ("pipeline_v2", "domain_unique"),
        ("specgate", "domain_unique"),
        ("spec gate", "domain_unique"),
        ("overwatcher", "domain_unique"),
        ("fastapi", "domain_unique"),
        ("the pipeline", "domain_unique"),
        ("python code", "stack"),
        ("astra code", "stack"),
        ("app/bridge", "stack"),
        ("app\\bridge", "stack"),
        ("chat_and_speak", "stack"),
        ("tts_proxy", "stack"),
        # Debug tab is a backend concept
        ("debug tab", "domain_unique"),
        ("build project", "domain_unique"),
    ],
    ASTRA_FRONTEND.project_id: [
        ("astra desktop", "product_name"),
        ("orb-desktop", "product_name"),
        ("desktop app", "product_name"),
        ("electron", "product_name"),
        ("react component", "domain_unique"),
        ("frontend", "domain_unique"),
        ("the ui", "domain_unique"),
        ("the tabs", "domain_unique"),
        ("tab ui", "domain_unique"),
        ("dashboard component", "domain_unique"),
        ("sidebar", "domain_unique"),
        ("react", "stack"),
        ("typescript", "stack"),
        ("tsx", "stack"),
        ("jsx", "stack"),
    ],
}

# 'Work on yourself' signals point to the backend (Astra itself)
_SELF_WORK_SIGNALS: list = [
    "work on yourself", "work on your", "improve yourself",
    "self-improvement", "self-optimize", "optimize yourself",
    "your own code", "your own architecture",
    "astra work on", "let's work on astra",
]


def _check_explicit_build_target_directive(text: str) -> "BuildTargetProfile | None":
    """If the text contains 'build target: X' where X is a registered
    project_id, return that profile. This is an authoritative override:
    the spec is LITERALLY saying which project it's for.
    """
    import re as _re
    for pid, profile in _REGISTRY.items():
        # Accept 'build target: driver-copilot', 'Build target:driver-copilot',
        # 'target: driver-copilot', 'target = driver-copilot', etc.
        pattern = (
            r"(?:build\s+target|target)\s*[:=]\s*"
            + _re.escape(pid.lower())
            + r"(?![-\w])"
        )
        if _re.search(pattern, text):
            return profile
    return None


def _score_projects_from_text(text: str) -> Dict[str, int]:
    """Compute a weighted score per project from the given text.

    Returns a dict of project_id → accumulated score. Only projects with
    score > 0 are included. Callers can compare scores to pick a winner.
    """
    scores: Dict[str, int] = {}

    # Hard signals first: explicit project_id and repo path literals
    for pid, profile in _REGISTRY.items():
        hard = 0
        if pid.lower() in text:
            hard += _SIGNAL_WEIGHTS["product_name"] * 2  # explicit id is strongest
        root_lower = profile.project_root.replace("\\", "/").lower()
        if root_lower and root_lower in text:
            hard += _SIGNAL_WEIGHTS["path"]
        if hard:
            scores[pid] = scores.get(pid, 0) + hard

    # Weighted signal lists
    for pid, signals in _PROJECT_SIGNALS.items():
        for sig, weight_key in signals:
            if sig in text:
                scores[pid] = scores.get(pid, 0) + _SIGNAL_WEIGHTS[weight_key]

    # Self-work signals → backend
    for sig in _SELF_WORK_SIGNALS:
        if sig in text:
            scores[ASTRA_BACKEND.project_id] = (
                scores.get(ASTRA_BACKEND.project_id, 0)
                + _SIGNAL_WEIGHTS["domain_unique"]
            )
            break

    return scores


def detect_all_projects_from_message(message: str) -> set:
    """Detect EVERY known project the message touches.

    Unlike resolve_project_from_message (scored winner, single result),
    this scans every signal list and returns the union of project IDs that
    matched. Used by pipeline_bridge to detect multi-target jobs that the
    single-target resolver would silently collapse.

    v1.0 (2026-04-11): Added to fix multi-target collapse bug — see
        pipeline_bridge.save_weaver_extraction.
    v1.1 (2026-04-18): Added hard path/explicit-target signals so that
        Weaver job descriptions mentioning 'driver-copilot' or the repo
        path literally register even if softer 'bridge signals' also hit.
    """
    text = message.lower()
    hits = set()

    # Fix 2a (2026-04-18): Hard signals — explicit project_id mentions and
    # repo path references always count. These cannot be ambiguous.
    for pid, profile in _REGISTRY.items():
        if pid.lower() in text:
            hits.add(pid)
        root_lower = profile.project_root.replace("\\", "/").lower()
        if root_lower and root_lower in text:
            hits.add(pid)

    bridge_signals = [
        "astra bridge", "astrabridge", "astra-bridge", "android bridge",
        "phone bridge", "bridge app", "mobile bridge", "the bridge",
        "companion app", "in the van", "in-van", "on-road", "on the road",
        "while driving", "driving use", "hands-free", "wake word", "wakeword",
        "astra pause", "astra continue", "astra wake", "astra repeat",
        "astra playback", "voice-command-friendly", "voice command",
        "tts playback", "tts auto-play", "auto-play back",
        "playback should resume", "buffered cutoff", "exoplayer", "chatterbox",
    ]
    if any(sig in text for sig in bridge_signals):
        hits.add(ASTRA_BRIDGE.project_id)
    android_signals = [
        "driver copilot", "copilot app",
        "drivercopilot", "driver co-pilot", "yodel app",
        "round planner", "manifest stop",
        "androiddrivercopilot",
    ]
    if any(sig in text for sig in android_signals):
        hits.add(DRIVER_COPILOT.project_id)
    backend_signals = [
        "astra itself", "the pipeline", "fastapi", "the orb", "orb backend",
        "pipeline_v2", "specgate", "spec gate", "overwatcher",
        "astra backend", "python code", "astra code",
        "d:/orb", "d:\\orb", "app/bridge", "app\\bridge",
        "chat_and_speak", "router.py", "tts_proxy",
    ]
    if any(sig in text for sig in backend_signals):
        hits.add(ASTRA_BACKEND.project_id)
    frontend_signals = [
        "desktop app", "electron", "react component", "orb-desktop",
        "astra desktop", "d:/orb-desktop", "d:\\orb-desktop",
    ]
    if any(sig in text for sig in frontend_signals):
        hits.add(ASTRA_FRONTEND.project_id)
    return hits


def resolve_project_from_message(
    message: str,
    conversation_history: Optional[list] = None,
) -> Optional[BuildTargetProfile]:
    """Detect which project the user is talking about from message context.

    Returns None if genuinely ambiguous (caller should ask or default).

    Fix 2 (2026-04-18): Rewrote from first-match-wins to scored selection.
    The old implementation had ordered signal lists checked in sequence;
    any hit in the first list returned immediately, which meant a Driver
    CoPilot spec mentioning 'hands-free' or 'on-road' (legitimate driver-
    context terms) silently resolved to Astra-Bridge because bridge was
    checked first. The scored resolver compares weighted signal hits across
    all projects and returns the clear winner, or None if tied.

    Resolution order:
      1. Explicit 'build target: X' directive — authoritative.
      2. Score the current message. If one project scores >= 1.5x the
         runner-up, return it.
      3. Combine current-message score (weighted 3x) with recent-history
         score (weighted 1x). Return clear winner.
      4. Return None — genuinely ambiguous.
    """
    text = (message or "").lower()

    # Step 1: Explicit override.
    explicit = _check_explicit_build_target_directive(text)
    if explicit is not None:
        logger.info(
            "[target_registry] Resolved via explicit 'build target' directive: %s",
            explicit.project_id,
        )
        return explicit

    # Step 2: Score current message.
    current_scores = _score_projects_from_text(text)

    def _clear_winner(scores: Dict[str, int], min_margin: float = 1.5) -> Optional[str]:
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) == 1:
            # Single project scored anything at all — it's the winner
            # unless the score is trivial (< 2, i.e. a single stack hit).
            return ranked[0][0] if ranked[0][1] >= 2 else None
        top_pid, top_score = ranked[0]
        runner_score = ranked[1][1] if ranked[1][1] > 0 else 1  # avoid /0
        if top_score >= runner_score * min_margin:
            return top_pid
        return None

    winner = _clear_winner(current_scores)
    if winner:
        profile = _REGISTRY.get(winner)
        if profile is not None:
            logger.info(
                "[target_registry] Resolved via scored current-message: %s (scores=%s)",
                winner,
                {k: v for k, v in sorted(
                    current_scores.items(), key=lambda kv: kv[1], reverse=True
                )},
            )
            return profile

    # Step 3: Combine with history.
    if conversation_history:
        history_scores: Dict[str, int] = {}
        for msg in conversation_history[-10:]:
            content = (msg.get("content") or "").lower()
            h_scores = _score_projects_from_text(content)
            for pid, s in h_scores.items():
                history_scores[pid] = history_scores.get(pid, 0) + s

        combined: Dict[str, int] = {}
        for pid, s in current_scores.items():
            combined[pid] = combined.get(pid, 0) + s * 3
        for pid, s in history_scores.items():
            combined[pid] = combined.get(pid, 0) + s

        winner = _clear_winner(combined, min_margin=1.5)
        if winner:
            profile = _REGISTRY.get(winner)
            if profile is not None:
                logger.info(
                    "[target_registry] Resolved via scored combined (current+history): "
                    "%s (combined=%s, current=%s, history=%s)",
                    winner,
                    combined, current_scores, history_scores,
                )
                return profile

    # Step 4: Genuinely ambiguous.
    logger.info(
        "[target_registry] Could not resolve target — scores=%s",
        current_scores,
    )
    return None

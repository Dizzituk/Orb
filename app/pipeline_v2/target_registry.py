# FILE: app/pipeline_v2/target_registry.py
# Purpose: Build Target Registry.
# Called-by: app.builds.pipeline_bridge, app.cloud.build_and_deploy, app.debug.debug_chat, app.debug.orchestrator.decomposer (+15 more)
# Depends-on: app.pipeline_v2.build_targets
# Last-renovated: 2026-06-11
"""
Build Target Registry.

Central registry of all known build target profiles. The pipeline
looks up the active profile by ID to configure every stage.

v1.0 (2026-03-10): Initial — 3 profiles.
v1.1 (2026-03-10): Expanded signal keywords, JAVA_HOME for Gradle.
v1.2 (2026-03-11): Added AstraBridge profile, "work on yourself" keywords,
    Astra Bridge signal keywords.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)

# Android Studio bundled JDK — required for Gradle builds
_JAVA_HOME = r'C:\Program Files\Android\Android Studio\jbr'
_SET_JAVA = f'$env:JAVA_HOME = "{_JAVA_HOME}" ; '


# ═══════════════════════════════════════════════════════════════════
# Predefined profiles
# ═══════════════════════════════════════════════════════════════════

ASTRA_BACKEND = BuildTargetProfile(
    project_id="astra-backend",
    project_name="ASTRA Backend",
    project_root="D:/Orb",
    language="python",
    build_system="pip",
    framework="fastapi",
    source_root="app",
    package_name="app",
    architecture_pattern="module-router",
    key_directories={
        "models": "*/models.py",
        "routers": "*/router.py",
        "services": "*/service.py",
        "schemas": "*/schemas.py",
    },
    syntax_check_cmd='& "D:\\Orb\\.venv\\Scripts\\python.exe" -m py_compile "{file}"',
    build_cmd=(
        'cd "D:\\Orb" ; '
        '& "D:\\Orb\\.venv\\Scripts\\python.exe" -c '
        '"from main import app; print(\'BOOT_OK\')"'
    ),
    boot_cmd=(
        'cd "D:\\Orb" ; '
        '& "D:\\Orb\\.venv\\Scripts\\python.exe" -c '
        '"import sys; sys.path.insert(0, r\'D:\\Orb\'); '
        'from app.db import init_db; init_db(); '
        'from main import app; print(\'BOOT_CHECK_PASS\')"'
    ),
    clean_cmd="",
    verification_mode="browser",
    screenshot_method="screenshot-tool",
    file_extension=".py",
    test_extension="_test.py",
    dependency_file="requirements.txt",
    dependency_add_pattern="pip install {package}",
    path_signals=[
        "app/bridge/", "app/routers/", "app/services/", "app/pot_spec/",
        "app/pipeline_v2/", "app/orchestrator/", "app/builds/", "app/api/",
        "app/db/", "app/memory/", "app/models/",
    ],
)

ASTRA_FRONTEND = BuildTargetProfile(
    project_id="astra-frontend",
    project_name="ASTRA Desktop",
    project_root="D:/orb-desktop",
    language="typescript",
    build_system="npm",
    framework="react",
    source_root="src",
    package_name="orb-desktop",
    architecture_pattern="component-page",
    key_directories={
        "components": "src/components/",
        "pages": "src/pages/",
    },
    syntax_check_cmd='cd "D:\\orb-desktop" ; npx tsc --noEmit 2>&1',
    build_cmd='cd "D:\\orb-desktop" ; npx tsc --noEmit 2>&1',
    boot_cmd=None,
    clean_cmd='cd "D:\\orb-desktop" ; Remove-Item -Recurse -Force node_modules\\.cache -ErrorAction SilentlyContinue',
    verification_mode="browser",
    screenshot_method="screenshot-tool",
    file_extension=".tsx",
    test_extension=".test.tsx",
    dependency_file="package.json",
    dependency_add_pattern="npm install {package}",
    path_signals=[
        "orb-desktop/", "src/components/", "src/pages/", "src/hooks/",
    ],
)

DRIVER_COPILOT = BuildTargetProfile(
    project_id="driver-copilot",
    project_name="Driver CoPilot",
    project_root="D:/Astra Android Folder/AndroidDriverCopilot",
    language="kotlin",
    build_system="gradle",
    framework="jetpack-compose",
    source_root="app/src/main/java/com/example/drivercopilot",
    package_name="com.example.drivercopilot",
    architecture_pattern="mvvm",
    key_directories={
        "data": "data/",
        "views": "ui_screens/",
        "viewmodels": "viewmodel/",
        "navigation": "navigation/",
        "security": "security/",
        "theme": "ui/theme/",
    },
    syntax_check_cmd=(
        _SET_JAVA +
        'cd "D:\\Astra Android Folder\\AndroidDriverCopilot" ; '
        '.\\gradlew.bat compileDebugKotlin 2>&1'
    ),
    build_cmd=(
        _SET_JAVA +
        'cd "D:\\Astra Android Folder\\AndroidDriverCopilot" ; '
        '.\\gradlew.bat assembleDebug 2>&1'
    ),
    boot_cmd=None,
    clean_cmd=(
        _SET_JAVA +
        'cd "D:\\Astra Android Folder\\AndroidDriverCopilot" ; '
        '.\\gradlew.bat clean 2>&1'
    ),
    verification_mode="emulator",
    emulator_config={
        "avd_name": "ASTRA_Test_Device",
        "api_level": 35,
        "system_image": "system-images;android-35;google_apis;x86_64",
        "device_profile": "pixel_6",
        "ram_mb": 4096,
    },
    screenshot_method="adb-screencap",
    file_extension=".kt",
    test_extension="Test.kt",
    manifest_file="app/src/main/AndroidManifest.xml",
    dependency_file="app/build.gradle.kts",
    dependency_add_pattern='implementation("{package}")',
    path_signals=[
        "drivercopilot/", "ui_screens/", "com/example/drivercopilot/",
        "AndroidDriverCopilot/",
    ],
)

ASTRA_BRIDGE = BuildTargetProfile(
    project_id="astra-bridge",
    project_name="Astra Bridge",
    project_root="D:/Astra Android Folder/Astra-Bridge",
    language="kotlin",
    build_system="gradle",
    framework="jetpack-compose",
    source_root="app/src/main/java/com/astra/astrabridge",
    package_name="com.astra.astrabridge",
    architecture_pattern="mvvm",
    key_directories={
        "data": "data/",
        "views": "ui/",
        "viewmodels": "viewmodel/",
        "navigation": "navigation/",
        "bridge": "bridge/",
    },
    syntax_check_cmd=(
        _SET_JAVA +
        'cd "D:\\Astra Android Folder\\Astra-Bridge" ; '
        '.\\gradlew.bat compileDebugKotlin 2>&1'
    ),
    build_cmd=(
        _SET_JAVA +
        'cd "D:\\Astra Android Folder\\Astra-Bridge" ; '
        '.\\gradlew.bat assembleDebug 2>&1'
    ),
    boot_cmd=None,
    clean_cmd=(
        _SET_JAVA +
        'cd "D:\\Astra Android Folder\\Astra-Bridge" ; '
        '.\\gradlew.bat clean 2>&1'
    ),
    verification_mode="emulator",
    emulator_config={
        "avd_name": "ASTRA_Test_Device",
        "api_level": 35,
        "system_image": "system-images;android-35;google_apis;x86_64",
        "device_profile": "pixel_6",
        "ram_mb": 4096,
    },
    screenshot_method="adb-screencap",
    file_extension=".kt",
    test_extension="Test.kt",
    manifest_file="app/src/main/AndroidManifest.xml",
    dependency_file="app/build.gradle.kts",
    dependency_add_pattern='implementation("{package}")',
    path_signals=[
        "astrabridge/", "com/astra/astrabridge/", "Astra-Bridge/",
        "voice/", "viewmodel/", "navigation/",
    ],
)


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

_REGISTRY: Dict[str, BuildTargetProfile] = {
    ASTRA_BACKEND.project_id: ASTRA_BACKEND,
    ASTRA_FRONTEND.project_id: ASTRA_FRONTEND,
    DRIVER_COPILOT.project_id: DRIVER_COPILOT,
    ASTRA_BRIDGE.project_id: ASTRA_BRIDGE,
}

# Public alias. Some modules (e.g. _spec_runner_utils_13._discover_project_roots)
# import ALL_PROFILES to enumerate known projects for root discovery. The bare
# _REGISTRY name is conventionally private, so we expose a stable public name.
# Fix 1 (2026-04-18): without this alias, `from ...target_registry import
# ALL_PROFILES` fails silently under a broad `except ImportError: pass`, which
# caused external Android project roots to never be merged into spec_runner's
# discovery index. That in turn made file-scope extraction fail for any
# Android build because the discovery only knew about D:\Orb and D:\orb-desktop.
ALL_PROFILES: Dict[str, BuildTargetProfile] = _REGISTRY


# ═══════════════════════════════════════════════════════════════════
# Job-level target hint (ambient context for resolve_target_for_files)
# ═══════════════════════════════════════════════════════════════════
# When the pipeline_bridge layer persists target_ids on the BuildProject
# (Phase 0 Job 3), the spec_runner should read those back and call
# set_job_target_hint() before segmentation begins. resolve_target_for_files
# falls back to this hint when callers do not pass job_target_ids
# explicitly — removes the need to thread context through every layer.
# v1.0 (2026-04-12): Phase 1 Job 14.

_job_target_hint: "set | None" = None


def set_job_target_hint(target_ids) -> None:
    """Set the ambient job-level target hint.

    target_ids: iterable of project_id strings, or None to clear.
    """
    global _job_target_hint
    if target_ids is None:
        _job_target_hint = None
        return
    try:
        _job_target_hint = set(target_ids)
        logger.info("[target_registry] Job target hint set: %s", sorted(_job_target_hint))
    except TypeError:
        _job_target_hint = None
        logger.warning("[target_registry] set_job_target_hint: invalid target_ids=%r", target_ids)


def get_job_target_hint() -> "set | None":
    """Return the current ambient job target hint (or None)."""
    return _job_target_hint


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


def get_profile(project_id: str) -> Optional[BuildTargetProfile]:
    """Look up a build target profile by ID."""
    return _REGISTRY.get(project_id)


def get_default_profile() -> BuildTargetProfile:
    """Return the default profile (ASTRA backend)."""
    target = os.getenv("ASTRA_V2_BUILD_TARGET", "astra-backend")
    return _REGISTRY.get(target, ASTRA_BACKEND)


def list_profiles() -> List[BuildTargetProfile]:
    """Return all registered profiles."""
    return list(_REGISTRY.values())


def register_profile(profile: BuildTargetProfile) -> None:
    """Register a new build target profile."""
    _REGISTRY[profile.project_id] = profile
    logger.info("[registry] Registered profile: %s (%s)", profile.project_id, profile.project_name)


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

# ═══════════════════════════════════════════════════════════════════
# Target Groups (v1.2, 2026-04-11)
# ═══════════════════════════════════════════════════════════════════
# A TargetGroup names a bundle of targets that frequently get worked on
# together as a coordinated system. When detect_all_projects_from_message
# returns multiple hits matching a known group, the build project inherits
# the group as its scope, signalling to the segmenter that this is a
# first-class multi-target job (not an accident).

from dataclasses import dataclass as _dc, field as _field


@_dc
class TargetGroup:
    group_id: str
    group_name: str
    target_ids: list           # member target IDs
    description: str = ""


_TARGET_GROUPS = {
    "astra-system": TargetGroup(
        group_id="astra-system",
        group_name="ASTRA System (backend + bridge + frontend)",
        target_ids=["astra-backend", "astra-bridge", "astra-frontend"],
        description="Coordinated upgrades spanning the ASTRA backend, the "
                    "Android bridge app, and the Electron desktop frontend.",
    ),
    "astra-mobile": TargetGroup(
        group_id="astra-mobile",
        group_name="ASTRA Backend + Bridge",
        target_ids=["astra-backend", "astra-bridge"],
        description="Backend + Android bridge coordinated work — the most "
                    "common multi-target pattern (e.g. hands-free upgrades, "
                    "offline sync, missed-reply recovery).",
    ),
    "delivery-system": TargetGroup(
        group_id="delivery-system",
        group_name="Backend + Driver CoPilot",
        target_ids=["astra-backend", "driver-copilot"],
        description="Backend + delivery driver Android app.",
    ),
}


def get_target_group(group_id: str):
    return _TARGET_GROUPS.get(group_id)


def list_target_groups() -> list:
    return list(_TARGET_GROUPS.values())


def find_group_for_targets(target_ids: set) -> "TargetGroup | None":
    """Find the smallest registered group that contains all given targets.

    Returns the most specific match. If multiple groups contain the targets,
    prefers the group with the fewest extra members (closest fit).
    """
    matches = []
    for g in _TARGET_GROUPS.values():
        if target_ids.issubset(set(g.target_ids)):
            matches.append((len(g.target_ids), g))
    if not matches:
        return None
    matches.sort(key=lambda m: m[0])
    return matches[0][1]


def _resolve_single_file(norm: str, candidate_ids=None) -> "str | None":
    """Resolve a single (normalised, lowercased, forward-slash) path to one
    target_id, or None if no profile matches.

    Tier 1: absolute path starts with a profile's project_root.
    Tier 2: project folder name appears as a path segment (e.g. /astra-bridge/).
    Tier 3: path_signals (distinctive relative directories) — scored, clear winner required.
    Tier 4: file_extension alone (only when exactly one profile's ext matches).

    If candidate_ids is provided, only profiles whose project_id is in that
    set are considered (used to scope resolution to the current job's targets).
    """
    pool = [p for p in _REGISTRY.values()
            if candidate_ids is None or p.project_id in candidate_ids]
    # Tier 1: absolute root match
    for profile in pool:
        root = profile.project_root.replace("\\", "/").rstrip("/").lower()
        if norm.startswith(root + "/") or norm == root:
            return profile.project_id
    # Tier 2: project-folder-name in path
    for profile in pool:
        root = profile.project_root.replace("\\", "/").rstrip("/").lower()
        seg = root.split("/")[-1]
        if seg and ("/" + seg + "/") in ("/" + norm + "/"):
            return profile.project_id
    # Tier 3: path_signals scoring (relative-path LLM output)
    sig_scores = {}
    for profile in pool:
        signals = getattr(profile, "path_signals", None) or []
        score = sum(3 for sig in signals if sig.lower() in norm)
        if score > 0:
            sig_scores[profile.project_id] = score
    if sig_scores:
        best_score = max(sig_scores.values())
        winners = [pid for pid, s in sig_scores.items() if s == best_score]
        if len(winners) == 1:
            return winners[0]
    # Tier 4: file_extension as last resort
    ext_winners = [p.project_id for p in pool
                   if p.file_extension and norm.endswith(p.file_extension.lower())]
    if len(ext_winners) == 1:
        return ext_winners[0]
    return None


def resolve_target_for_files(file_paths, job_target_ids=None) -> "tuple[str | None, set[str]]":
    """Resolve which registered target owns a list of file paths.

    Returns (target_id, all_hits) where:
      - target_id is the single target_id if all files belong to one target
        AND all files resolved cleanly.
      - target_id is None if files span multiple targets OR any file failed
        to resolve (genuinely ambiguous segment).
      - all_hits is the full set of target_ids touched (for diagnostics).

    Used by smart_segmentation (Phase 1 Job 5) to tag each segment with
    its owning target. Mixed segments (target_id=None) should be split
    along target lines before being written to the manifest.

    job_target_ids: optional set of project_ids to restrict resolution to.
        When the pipeline_bridge layer has already detected the set of
        targets for this job (Phase 0), passing it here avoids false
        matches against unrelated profiles.

    v1.0 (2026-04-11): Phase 1 Job 5 — initial absolute-path resolver.
    v1.1 (2026-04-12): Phase 1 Job 14 — added path_signals + extension
        fallback tiers for LLM-generated relative paths, plus job-scoped
        candidate restriction. Any unresolved file forces target_id=None.
    """
    # Fall back to ambient job hint when caller did not pass explicit scope.
    if job_target_ids is None and _job_target_hint is not None:
        job_target_ids = _job_target_hint
    hits = set()
    unresolved = 0
    for raw in file_paths:
        if not raw:
            continue
        norm = str(raw).replace("\\", "/").lower()
        tid = _resolve_single_file(norm, candidate_ids=job_target_ids)
        if tid is not None:
            hits.add(tid)
        else:
            unresolved += 1
    # If any file failed to resolve, flag the whole segment as mixed/ambiguous
    # rather than silently claiming a partial target.
    if unresolved > 0:
        return (None, hits)
    if len(hits) == 1:
        return (next(iter(hits)), hits)
    return (None, hits)
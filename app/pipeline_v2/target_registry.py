# FILE: app/pipeline_v2/target_registry.py
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

    Unlike resolve_project_from_message (first-match-wins, single result),
    this scans every signal list and returns the union of project IDs that
    matched. Used by pipeline_bridge to detect multi-target jobs that the
    single-target resolver would silently collapse.

    v1.0 (2026-04-11): Added to fix multi-target collapse bug — see
        pipeline_bridge.save_weaver_extraction.
    """
    text = message.lower()
    hits = set()
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
    ]
    if any(sig in text for sig in android_signals):
        hits.add(DRIVER_COPILOT.project_id)
    backend_signals = [
        "astra itself", "the pipeline", "fastapi", "the orb", "orb backend",
        "pipeline_v2", "specgate", "spec gate", "weaver", "overwatcher",
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


def resolve_project_from_message(
    message: str,
    conversation_history: Optional[list] = None,
) -> Optional[BuildTargetProfile]:
    """Detect which project the user is talking about from message context.

    Returns None if ambiguous (caller should ask).

    Order matters: Most specific signals checked first. Astra Bridge before
    Driver CoPilot (both Android). Backend last as it's the broadest match.
    """
    text = message.lower()

    # ── Astra Bridge signals (most specific Android match) ──
    # v2.1 (2026-04-08): Added in-van/driving domain signals. Jobs about
    # TTS playback, wake words, hands-free voice, connection resilience
    # while driving are AstraBridge work — they never refer to the ASTRA
    # backend or desktop app explicitly, but the domain is unambiguous.
    bridge_signals = [
        # Explicit product name
        "astra bridge", "astrabridge", "astra-bridge",
        "android bridge", "phone bridge",
        "bridge app", "mobile bridge", "the bridge",
        "companion app",
        # In-van / driving domain (unique to AstraBridge)
        "in the van", "in-van", "on-road", "on the road",
        "while driving", "driving use", "hands-free",
        "wake word", "wakeword", "astra pause", "astra continue",
        "astra wake", "astra repeat", "astra playback",
        "voice-command-friendly", "voice command",
        "tts playback", "tts auto-play", "auto-play back",
        "playback should resume", "buffered cutoff",
        "exoplayer", "chatterbox",
    ]
    if any(sig in text for sig in bridge_signals):
        return ASTRA_BRIDGE

    # ── Explicit Android / Driver CoPilot signals ──
    android_signals = [
        "driver copilot", "copilot app", 
        "kotlin", "the app", "drivercopilot", "driver co-pilot",
        "mobile app", "jetpack", "compose", "accessibility service",
        "yodel app", "gradle", "room database", "room entity",
        "viewmodel", "composable", "manifest stop", "round planner",
    ]
    if any(sig in text for sig in android_signals):
        return DRIVER_COPILOT

    # ── "Work on yourself" signals → ASTRA backend ──
    self_work_signals = [
        "work on yourself", "work on your", "improve yourself",
        "self-improvement", "self-optimize", "optimize yourself",
        "your own code", "your own architecture",
        "astra work on", "let's work on astra",
    ]
    if any(sig in text for sig in self_work_signals):
        return ASTRA_BACKEND

    # ── Explicit ASTRA backend signals ──
    backend_signals = [
        "astra itself", "the pipeline", "backend", "fastapi",
        "the orb", "orb backend", "pipeline_v2", "sandbox",
        "specgate", "spec gate", "weaver", "overwatcher",
        "astra backend", "python code", "astra code",
        "pipeline", "debug tab", "build project",
        "d:/orb", "d:\\orb",
    ]
    if any(sig in text for sig in backend_signals):
        return ASTRA_BACKEND

    # ── Explicit frontend signals ──
    frontend_signals = [
        "desktop app", "electron", "frontend", "the ui",
        "react component", "react", "typescript", "orb-desktop",
        "astra desktop", "the tabs", "tab ui",
        "tsx", "jsx", "dashboard component", "sidebar",
        "d:/orb-desktop", "d:\\orb-desktop",
    ]
    if any(sig in text for sig in frontend_signals):
        return ASTRA_FRONTEND

    # ── Check conversation history for recent context ──
    if conversation_history:
        all_signal_sets = [
            (bridge_signals, ASTRA_BRIDGE),
            (android_signals, DRIVER_COPILOT),
            (self_work_signals, ASTRA_BACKEND),
            (backend_signals, ASTRA_BACKEND),
            (frontend_signals, ASTRA_FRONTEND),
        ]
        for msg in reversed(conversation_history[-10:]):
            content = (msg.get("content") or "").lower()
            for signals, profile in all_signal_sets:
                if any(sig in content for sig in signals):
                    return profile

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
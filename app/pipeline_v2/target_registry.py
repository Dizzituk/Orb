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
)

ASTRA_BRIDGE = BuildTargetProfile(
    project_id="astra-bridge",
    project_name="Astra Bridge",
    project_root="D:/Astra Android Folder/AstraBridge",
    language="kotlin",
    build_system="gradle",
    framework="jetpack-compose",
    source_root="app/src/main/java/com/astra/bridge",
    package_name="com.astra.bridge",
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
        'cd "D:\\Astra Android Folder\\AstraBridge" ; '
        '.\\gradlew.bat compileDebugKotlin 2>&1'
    ),
    build_cmd=(
        _SET_JAVA +
        'cd "D:\\Astra Android Folder\\AstraBridge" ; '
        '.\\gradlew.bat assembleDebug 2>&1'
    ),
    boot_cmd=None,
    clean_cmd=(
        _SET_JAVA +
        'cd "D:\\Astra Android Folder\\AstraBridge" ; '
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
    bridge_signals = [
        "astra bridge", "android bridge", "phone bridge",
        "bridge app", "mobile bridge", "the bridge",
    ]
    if any(sig in text for sig in bridge_signals):
        return ASTRA_BRIDGE

    # ── Explicit Android / Driver CoPilot signals ──
    android_signals = [
        "driver copilot", "copilot app", "phone app", "android app",
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

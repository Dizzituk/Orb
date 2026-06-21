# FILE: app/pipeline_v2/target_profiles.py
# Purpose: Build-target profile DATA — the 4 BuildTargetProfile literals + registry.
# Called-by: app.pipeline_v2.target_registry (shim), app.pipeline_v2.project_resolution, app.pipeline_v2.file_target_resolution
# Depends-on: app.pipeline_v2.build_targets
# Last-renovated: 2026-06-21
"""
Build-target profile data — split out of target_registry.py (SPLIT BATCH 9, 2026-06-21).

The 4 known BuildTargetProfile literals (ASTRA_BACKEND / ASTRA_FRONTEND /
DRIVER_COPILOT / ASTRA_BRIDGE), the _REGISTRY dict indexing them by project_id,
and the public ALL_PROFILES alias. Pure config DATA moved VERBATIM;
target_registry.py re-exports these names and the resolver leaves import them
directly, so all importers resolve unchanged.
"""
from __future__ import annotations

from typing import Dict

from app.pipeline_v2.build_targets import BuildTargetProfile

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

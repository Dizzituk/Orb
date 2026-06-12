# FILE: app/pot_spec/grounded/_greenfield_spec_builder.py
# Purpose: Greenfield Spec Builder — generates a full project architecture spec
# Called-by: app.pot_spec.grounded.spec_runner
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Greenfield Spec Builder — generates a full project architecture spec
for new external projects (Android apps, web apps, etc.).

When the pipeline detects a greenfield build (empty project root, external
to ASTRA's own codebase), this builder creates a structured spec that tells
the agentic builder exactly what files to CREATE and how to structure them.

v1.0 (2026-03-11): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def build_greenfield_spec(
    goal: str,
    what_to_do: str,
    build_profile: Dict,
) -> str:
    """Build a comprehensive SPoT spec for a greenfield project.

    Uses the build profile metadata to generate a proper project structure
    with file creation targets, architecture decisions, and acceptance criteria.

    Returns:
        spec_markdown string. File paths are embedded in the spec and will be
        extracted by _extract_file_scope_from_spec() during segmentation.
    """
    language = build_profile.get("language", "unknown")
    framework = build_profile.get("framework", "unknown")
    build_system = build_profile.get("build_system", "unknown")
    project_root = build_profile.get("project_root", "")
    project_name = build_profile.get("project_name", "Untitled")
    package_name = build_profile.get("package_name", "")
    source_root = build_profile.get("source_root", "")
    arch_pattern = build_profile.get("architecture_pattern", "modular")
    project_id = build_profile.get("project_id", "untitled")

    # Select architecture template based on language/framework
    if language == "kotlin" and framework == "jetpack-compose":
        spec = _build_android_spec(
            goal, what_to_do, project_root, project_name,
            package_name, source_root, arch_pattern, project_id,
        )
    elif language == "typescript" and framework == "react":
        spec = _build_react_spec(
            goal, what_to_do, project_root, project_name,
            package_name, source_root,
        )
    elif language == "python" and framework == "fastapi":
        spec = _build_python_spec(
            goal, what_to_do, project_root, project_name,
            package_name, source_root,
        )
    else:
        spec = _build_generic_spec(
            goal, what_to_do, project_root, project_name,
            language, framework, build_system,
        )

    logger.info("[greenfield_spec] Generated %s spec: %d chars", language, len(spec))
    return spec


def _build_android_spec(
    goal: str,
    what_to_do: str,
    project_root: str,
    project_name: str,
    package_name: str,
    source_root: str,
    arch_pattern: str,
    project_id: str,
) -> str:
    """Generate a Kotlin/Jetpack Compose Android project spec."""
    # Clean package path
    pkg_path = package_name.replace(".", "/") if package_name else f"com/astra/{project_id.replace('-', '')}"
    pkg_name = package_name or f"com.astra.{project_id.replace('-', '')}"

    lines = [
        "# SPoT Spec — Greenfield Android Project",
        "",
        "## Goal",
        "",
        goal.split('\\n')[0].strip() if goal else f"Create {project_name} Android application",
        "",
        "## Project Target",
        "",
        f"- **Type:** GREENFIELD (all files must be CREATED, nothing exists yet)",
        f"- **Root:** `{project_root}`",
        f"- **Language:** Kotlin",
        f"- **Framework:** Jetpack Compose",
        f"- **Build System:** Gradle (Kotlin DSL)",
        f"- **Architecture:** {arch_pattern.upper()}",
        f"- **Package:** `{pkg_name}`",
        "",
        "## What to build",
        "",
    ]

    # Include the Weaver's what_to_do
    if what_to_do:
        for line in what_to_do.split('\n')[:20]:
            if line.strip():
                lines.append(line.strip())
    lines.append("")

    # Android project structure
    lines.extend([
        "## Files to CREATE",
        "",
        "### Build Configuration",
        f"- `{project_root}/build.gradle.kts` — Root build file (AGP + Kotlin plugin)",
        f"- `{project_root}/settings.gradle.kts` — Project settings (module includes)",
        f"- `{project_root}/gradle.properties` — Gradle JVM args, AndroidX, compose flags",
        f"- `{project_root}/app/build.gradle.kts` — App module (dependencies, compose config, SDK versions)",
        "",
        "### Android Manifest",
        f"- `{project_root}/app/src/main/AndroidManifest.xml` — Permissions (INTERNET, RECORD_AUDIO, CAMERA, READ/WRITE_EXTERNAL_STORAGE), application entry",
        "",
        "### Application Entry",
        f"- `{project_root}/app/src/main/java/{pkg_path}/AstraApp.kt` — Application class (DI container init)",
        f"- `{project_root}/app/src/main/java/{pkg_path}/MainActivity.kt` — Single-activity host (Compose setContent, navigation host)",
        "",
        "### Navigation",
        f"- `{project_root}/app/src/main/java/{pkg_path}/navigation/AppNavGraph.kt` — NavHost with route definitions",
        f"- `{project_root}/app/src/main/java/{pkg_path}/navigation/Routes.kt` — Sealed class/object route definitions",
        "",
        "### UI Layer (Compose Screens)",
        f"- `{project_root}/app/src/main/java/{pkg_path}/ui/home/HomeScreen.kt` — Main dashboard/landing screen",
        f"- `{project_root}/app/src/main/java/{pkg_path}/ui/chat/ChatScreen.kt` — Conversational interface with message list",
        f"- `{project_root}/app/src/main/java/{pkg_path}/ui/theme/Theme.kt` — Material3 theme (colors, typography)",
        f"- `{project_root}/app/src/main/java/{pkg_path}/ui/theme/Color.kt` — Color palette",
        f"- `{project_root}/app/src/main/java/{pkg_path}/ui/components/VoiceButton.kt` — Mic/voice toggle composable",
        "",
        "### ViewModel Layer",
        f"- `{project_root}/app/src/main/java/{pkg_path}/viewmodel/ChatViewModel.kt` — Chat state, message send/receive, TTS/STT coordination",
        f"- `{project_root}/app/src/main/java/{pkg_path}/viewmodel/ConnectionViewModel.kt` — Backend connection state, health checks",
        "",
        "### Network / Bridge Layer",
        f"- `{project_root}/app/src/main/java/{pkg_path}/network/AstraApiClient.kt` — Retrofit/OkHttp client targeting PC backend",
        f"- `{project_root}/app/src/main/java/{pkg_path}/network/AstraApiService.kt` — API interface (chat, upload, commands)",
        f"- `{project_root}/app/src/main/java/{pkg_path}/network/ConnectionManager.kt` — Backend discovery, connection lifecycle, reconnect",
        f"- `{project_root}/app/src/main/java/{pkg_path}/network/models/ChatMessage.kt` — Request/response DTOs",
        "",
        "### Voice / Audio Layer",
        f"- `{project_root}/app/src/main/java/{pkg_path}/voice/SpeechRecognitionService.kt` — Android SpeechRecognizer wrapper (STT)",
        f"- `{project_root}/app/src/main/java/{pkg_path}/voice/TextToSpeechService.kt` — Android TTS wrapper with interruption support",
        f"- `{project_root}/app/src/main/java/{pkg_path}/voice/WakeWordDetector.kt` — Wake word detection (\"Hey Astra\")",
        f"- `{project_root}/app/src/main/java/{pkg_path}/voice/VoiceSessionManager.kt` — Coordinates STT→send→receive→TTS flow",
        "",
        "### Media Layer",
        f"- `{project_root}/app/src/main/java/{pkg_path}/media/MediaUploader.kt` — Image/video upload via multipart to backend",
        f"- `{project_root}/app/src/main/java/{pkg_path}/media/CameraCapture.kt` — CameraX integration for photo/video capture",
        f"- `{project_root}/app/src/main/java/{pkg_path}/media/MediaPicker.kt` — Gallery picker (photo/video selection)",
        "",
        "### DI / Config",
        f"- `{project_root}/app/src/main/java/{pkg_path}/di/AppModule.kt` — Hilt/manual DI module (network, voice, media providers)",
        "",
        "## Architecture Decisions",
        "",
        "- Single-activity Compose architecture with NavHost",
        "- MVVM: Screen → ViewModel → Repository/Service",
        "- Network layer uses OkHttp + Retrofit targeting the PC backend at a configurable IP:port",
        "- Voice pipeline: WakeWord → STT → API call → TTS response (with interruption)",
        "- Media uploads via multipart/form-data POST to backend `/upload` endpoint",
        "- Backend connection is user-configured (IP address of PC on local network)",
        "",
        "## Acceptance Criteria",
        "",
        "- [ ] Project compiles with `./gradlew assembleDebug`",
        "- [ ] App launches and shows home screen",
        "- [ ] Navigation between screens works",
        "- [ ] Backend connection can be configured (IP/port entry)",
        "- [ ] Chat messages can be sent to and received from the backend",
        "- [ ] Voice input (STT) captures speech and sends to backend",
        "- [ ] Voice output (TTS) reads backend responses aloud",
        "- [ ] Image/video upload sends media to backend",
        "- [ ] Wake word detection activates voice mode",
        "",
    ])

    spec = "\n".join(lines)
    logger.info("[greenfield_spec] Generated Android spec: %d files, %d chars", 
                spec.count("- `"), len(spec))
    return spec


def _build_react_spec(
    goal: str,
    what_to_do: str,
    project_root: str,
    project_name: str,
    package_name: str,
    source_root: str,
) -> str:
    """Generate a React/TypeScript web project spec."""
    lines = [
        "# SPoT Spec — Greenfield React Web Project",
        "",
        "## Goal",
        "",
        goal.split('\\n')[0].strip() if goal else f"Create {project_name} web application",
        "",
        "## Project Target",
        "",
        f"- **Type:** GREENFIELD (all files must be CREATED)",
        f"- **Root:** `{project_root}`",
        f"- **Language:** TypeScript",
        f"- **Framework:** React + Vite",
        f"- **Package:** `{package_name}`",
        "",
        "## What to build",
        "",
    ]
    if what_to_do:
        for line in what_to_do.split('\n')[:20]:
            if line.strip():
                lines.append(line.strip())
    lines.extend([
        "",
        "## Files to CREATE",
        "",
        f"- `{project_root}/package.json`",
        f"- `{project_root}/tsconfig.json`",
        f"- `{project_root}/vite.config.ts`",
        f"- `{project_root}/index.html`",
        f"- `{project_root}/src/main.tsx`",
        f"- `{project_root}/src/App.tsx`",
        f"- `{project_root}/src/vite-env.d.ts`",
        "",
        "## Acceptance Criteria",
        "",
        "- [ ] `npm install` succeeds",
        "- [ ] `npm run dev` starts dev server",
        "- [ ] App renders in browser",
        "",
    ])
    return "\n".join(lines)


def _build_python_spec(
    goal: str,
    what_to_do: str,
    project_root: str,
    project_name: str,
    package_name: str,
    source_root: str,
) -> str:
    """Generate a Python/FastAPI project spec."""
    lines = [
        "# SPoT Spec — Greenfield Python/FastAPI Project",
        "",
        "## Goal",
        "",
        goal.split('\\n')[0].strip() if goal else f"Create {project_name} API service",
        "",
        "## Project Target",
        "",
        f"- **Type:** GREENFIELD (all files must be CREATED)",
        f"- **Root:** `{project_root}`",
        f"- **Language:** Python 3.11+",
        f"- **Framework:** FastAPI + Uvicorn",
        "",
        "## What to build",
        "",
    ]
    if what_to_do:
        for line in what_to_do.split('\n')[:20]:
            if line.strip():
                lines.append(line.strip())
    lines.extend([
        "",
        "## Files to CREATE",
        "",
        f"- `{project_root}/main.py` — FastAPI app entry",
        f"- `{project_root}/requirements.txt` — Dependencies",
        f"- `{project_root}/{source_root or 'app'}/__init__.py`",
        f"- `{project_root}/{source_root or 'app'}/routers/__init__.py`",
        f"- `{project_root}/{source_root or 'app'}/models/__init__.py`",
        "",
        "## Acceptance Criteria",
        "",
        "- [ ] `pip install -r requirements.txt` succeeds",
        "- [ ] `uvicorn main:app` starts server",
        "- [ ] `/docs` endpoint shows OpenAPI spec",
        "",
    ])
    return "\n".join(lines)


def _build_generic_spec(
    goal: str,
    what_to_do: str,
    project_root: str,
    project_name: str,
    language: str,
    framework: str,
    build_system: str,
) -> str:
    """Generate a generic greenfield project spec."""
    lines = [
        "# SPoT Spec — Greenfield Project",
        "",
        "## Goal",
        "",
        goal.split('\\n')[0].strip() if goal else f"Create {project_name}",
        "",
        "## Project Target",
        "",
        f"- **Type:** GREENFIELD (all files must be CREATED)",
        f"- **Root:** `{project_root}`",
        f"- **Language:** {language}",
        f"- **Framework:** {framework}",
        f"- **Build System:** {build_system}",
        "",
        "## What to build",
        "",
    ]
    if what_to_do:
        for line in what_to_do.split('\n')[:20]:
            if line.strip():
                lines.append(line.strip())
    lines.extend([
        "",
        "## Acceptance Criteria",
        "",
        "- [ ] Project compiles/runs successfully",
        "- [ ] Core functionality works as described",
        "",
    ])
    return "\n".join(lines)

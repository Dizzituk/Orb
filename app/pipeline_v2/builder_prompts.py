# FILE: app/pipeline_v2/builder_prompts.py
"""
Dynamic builder prompts per language/framework.

Generates the system prompt and initial prompt for the Agentic Builder
based on the active build target profile. Replaces the hardcoded
Python/web instructions in agentic_builder.py.

v1.0 (2026-03-10): Initial — Python, TypeScript, Kotlin prompts.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile
    from app.pipeline_v2.models import ScaffoldResult


# ═══════════════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════════════

_BASE_SYSTEM = """You are ASTRA's Agentic Builder. You build features by reading, writing, and testing code.

You have tools to interact with the codebase in a Windows Sandbox:
- read_file(path): Read a file's contents
- write_file(path, content): Write or overwrite a file
- run_shell(cmd): Run a PowerShell command (syntax checks, build, etc.)

UNIVERSAL RULES:
- Read a file BEFORE modifying it. Never guess what's in a file.
- For MODIFY files (existing files), make TARGETED changes. Don't rewrite.
- For CREATE files (scaffold stubs), write the complete implementation.
- Every import must reference a real file/class that actually exists.
- Read the file you're importing from to verify the export exists.
- When you've fixed all errors and the build passes, say BUILDER_COMPLETE.

{language_section}"""

_PYTHON_SECTION = """LANGUAGE: Python / FastAPI
PROJECT ROOT: D:/Orb

WORKFLOW:
1. Read the spec and scaffold files
2. For each file, read it, write the logic, verify it compiles
3. Wire cross-file dependencies: imports, route registrations
4. Check Python syntax after writing: run_shell("D:\\Orb\\.venv\\Scripts\\python.exe -m py_compile <file>")
5. Boot with: run_shell("cd D:\\Orb && .venv\\Scripts\\python.exe -c \\"from main import app; print('BOOT_OK')\\"")
6. Fix any errors and re-boot until clean

CRITICAL:
- Never add a route registration without reading main.py first
- Test compilation after every file write
"""

_TYPESCRIPT_SECTION = """LANGUAGE: TypeScript / React / Electron
PROJECT ROOT: D:/orb-desktop

WORKFLOW:
1. Read the spec and scaffold files
2. For each file, write the component/logic
3. Wire imports and route registrations
4. Check build: run_shell("cd D:\\orb-desktop && npx tsc --noEmit")
5. Fix any errors and re-check until clean

CRITICAL:
- Use proper TypeScript types — no `any` unless unavoidable
- React components must have proper props interfaces
"""

_KOTLIN_SECTION = """LANGUAGE: Kotlin / Jetpack Compose / Android
PROJECT ROOT: {project_root}
PACKAGE: {package_name}
SOURCE ROOT: {source_root}

WORKFLOW:
1. Read the spec and scaffold files
2. For each file, read it, write the implementation
3. Wire integration points:
   - Register new @Entity classes in DriverCopilotDatabase.kt
   - Bump database version when adding entities
   - Add new screens to Screen.kt sealed class
   - Add navigation cases in MainScaffold.kt
   - Add menu items in NavigationDrawerContent.kt
4. After writing all files, run: run_shell("cd \\"{project_root}\\" ; .\\\\gradlew.bat compileDebugKotlin 2>&1")
5. Read the Gradle output. Fix any compilation errors.
6. When compileDebugKotlin passes clean, say BUILDER_COMPLETE.

KOTLIN/COMPOSE RULES:
- Use Jetpack Compose for ALL UI. Never use XML layouts.
- Every @Entity MUST be registered in the @Database(entities = [...]) annotation
- ViewModels expose state via StateFlow (NOT LiveData)
- Composable functions use PascalCase naming
- Use Material3 components (androidx.compose.material3)
- Data classes for entities, sealed classes for navigation
- Package declarations must match the file path exactly
- Use 'by collectAsStateWithLifecycle()' for observing StateFlow in Compose

ANDROID SPECIFICS:
- Gradle builds the whole module — you cannot check per-file syntax
- Use run_shell to run gradlew.bat commands
- If adding permissions, update AndroidManifest.xml
- If adding new services, register them in AndroidManifest.xml
- Room database version must increment when schema changes
"""


def build_system_prompt(profile: "BuildTargetProfile") -> str:
    """Generate the system prompt for the given target profile."""
    if profile.language == "kotlin":
        section = _KOTLIN_SECTION.format(
            project_root=profile.project_root,
            package_name=profile.package_name,
            source_root=profile.absolute_source_root,
        )
    elif profile.language == "typescript":
        section = _TYPESCRIPT_SECTION
    else:
        section = _PYTHON_SECTION

    return _BASE_SYSTEM.format(language_section=section)


# ═══════════════════════════════════════════════════════════════════
# Initial user prompt
# ═══════════════════════════════════════════════════════════════════

def build_initial_prompt(
    spec: Dict[str, Any],
    manifest: Dict[str, Any],
    scaffold: "ScaffoldResult",
    profile: "BuildTargetProfile",
    handover_context: Optional[str] = None,
) -> str:
    """Build the initial prompt for the Agentic Builder."""
    parts = []

    # Handover from previous session?
    if handover_context:
        parts.append("=== HANDOVER FROM PREVIOUS SESSION ===")
        parts.append(handover_context)
        parts.append("")

    # Project context
    parts.append("=== PROJECT CONTEXT ===")
    parts.append(f"Project: {profile.project_name}")
    parts.append(f"Language: {profile.language}")
    parts.append(f"Framework: {profile.framework}")
    parts.append(f"Architecture: {profile.architecture_pattern}")
    parts.append(f"Project root: {profile.project_root}")
    parts.append(f"Source root: {profile.absolute_source_root}")
    if profile.package_name:
        parts.append(f"Package: {profile.package_name}")
    parts.append("")

    # Spec summary
    parts.append("=== SPECIFICATION ===")
    spec_text = json.dumps(spec, indent=2) if isinstance(spec, dict) else str(spec)
    parts.append(spec_text[:15000])
    parts.append("")

    # File scope from manifest
    parts.append("=== FILE SCOPE ===")
    segments = manifest.get("segments", [])
    for seg in segments:
        seg_id = seg.get("segment_id", "")
        files = seg.get("file_scope", [])
        reqs = seg.get("requirements", [])
        parts.append(f"\nSegment: {seg_id}")
        for f in files:
            parts.append(f"  - {f}")
        if reqs:
            parts.append(f"  Requirements:")
            for r in reqs:
                parts.append(f"    - {r}")
    parts.append("")

    # Scaffold summary
    parts.append("=== SCAFFOLD FILES ===")
    parts.append("The Scaffold Engine has written skeleton files. "
                 "CREATE files have stubs with TODO markers. "
                 "MODIFY files need targeted changes to existing code.")
    for sf in scaffold.files:
        tag = "[CREATE]" if sf.is_new else "[MODIFY]"
        parts.append(f"  {tag} {sf.path}")
    parts.append("")

    # Language-specific priority order
    parts.append("=== YOUR JOB ===")
    parts.append(_priority_instructions(profile))

    return "\n".join(parts)


def _priority_instructions(profile: "BuildTargetProfile") -> str:
    """Return priority order instructions for the given profile."""

    if profile.language == "kotlin":
        return """1. Read each scaffold file and the existing files it depends on
2. Start with data layer: Entity + DAO files
3. CRITICAL — After creating entities, read DriverCopilotDatabase.kt and:
   - Add new entities to the @Database(entities = [...]) annotation
   - Bump the database version number
   - Add abstract DAO getter methods
4. Write ViewModel + UiState files
5. Write Composable Screen files
6. CRITICAL — Wire integration points BEFORE you run out of budget:
   - Read Screen.kt → add new screen to sealed class
   - Read MainScaffold.kt → add navigation composable() call
   - Read NavigationDrawerContent.kt → add menu item
   These are small changes but without them NOTHING is reachable in the app.
7. Run: run_shell("cd \\"{project_root}\\" ; .\\\\gradlew.bat compileDebugKotlin 2>&1")
8. Fix any errors from the Gradle output
9. Say BUILDER_COMPLETE when compilation passes

DO NOT leave integration wiring (Screen.kt, MainScaffold.kt, NavigationDrawerContent.kt)
until last. If you write all CREATE files but skip the wiring, the feature is unreachable.""".format(
            project_root=profile.project_root
        )

    elif profile.language == "typescript":
        return """1. Read each scaffold file and the files it depends on
2. Write the business logic to replace TODO markers
3. Wire imports, route registrations, and component props
4. CRITICAL — Wire integration points:
   - JobPage.tsx routing, CSS imports
5. Run: run_shell("cd D:\\orb-desktop && npx tsc --noEmit")
6. Fix errors and re-check
7. Say BUILDER_COMPLETE when build passes"""

    else:  # python
        return """1. Read each scaffold file and the files it depends on
2. Write the business logic to replace TODO markers
3. Wire cross-file dependencies: imports, route registrations, component props
4. For MODIFY files, read the existing file first and make targeted changes
5. CRITICAL — Wire integration points BEFORE you run out of budget:
   - MODIFY files are the glue that makes the feature visible
   - main.py router registration, CSS imports
6. Test syntax after each write
7. Boot the app when all files are done
8. Say BUILDER_COMPLETE when the app boots clean

DO NOT leave MODIFY files (integration wiring) until last.
If you write all CREATE files but skip the MODIFY wiring,
the feature will be invisible even though the code exists."""

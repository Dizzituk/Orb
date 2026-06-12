# FILE: app/pipeline_v2/build_targets.py
# Purpose: ASTRA Build Target Profiles.
# Called-by: app.llm.project_scoping_stream, app.pipeline_v2.agentic_builder, app.pipeline_v2.android_sandbox, app.pipeline_v2.builder_prompts (+18 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
ASTRA Build Target Profiles.

A Build Target Profile describes everything the pipeline needs to know
about the project it's building into: language, build system, paths,
verification strategy, and architecture conventions.

Every pipeline stage reads from the active profile instead of
hardcoding assumptions about Python/TypeScript/Kotlin.

v1.0 (2026-03-10): Initial implementation — 3 profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BuildTargetProfile:
    """Everything the pipeline needs to know about a build target."""

    # ── Identity ──
    project_id: str
    project_name: str
    project_root: str

    # ── Language & Build ──
    language: str               # "kotlin", "python", "typescript"
    build_system: str           # "gradle", "pip", "npm"
    framework: str              # "jetpack-compose", "fastapi", "react"
    source_root: str            # Relative from project_root
    package_name: str           # e.g. "com.example.drivercopilot"

    # ── Architecture ──
    architecture_pattern: str   # "mvvm", "mvi", "module-router", "component-page"
    key_directories: Dict[str, str] = field(default_factory=dict)

    # ── Build Commands ──
    syntax_check_cmd: str = ""
    build_cmd: str = ""
    boot_cmd: Optional[str] = None
    clean_cmd: str = ""

    # ── Verification ──
    verification_mode: str = "compilation-only"  # "emulator", "browser", "api-only", "compilation-only"
    emulator_config: Optional[Dict] = None
    screenshot_method: str = "none"              # "adb-screencap", "screenshot-tool", "none"

    # ── File Patterns ──
    file_extension: str = ".py"
    test_extension: str = "_test.py"
    manifest_file: Optional[str] = None

    # ── Dependencies ──
    dependency_file: str = ""
    dependency_add_pattern: str = ""   # e.g. "implementation(...)" or "pip install"    # ── Multi-target awareness (v1.1, 2026-04-11) ──
    # Other target IDs this profile is commonly co-built with. Used by the
    # multi-target detector and segmenter to recognize coordinated upgrades.
    related_targets: List[str] = field(default_factory=list)
    # Files or directories that act as cross-target interface contracts
    # (e.g. FastAPI route definitions consumed by an Android client). The
    # contract verifier reads these to detect mismatches across repos.
    shared_contracts: List[str] = field(default_factory=list)
    # Distinctive relative-path markers used by resolve_target_for_files() to
    # classify LLM-generated relative paths that don't contain the project
    # root name. Examples: "app/bridge/", "voice/", "com/astra/astrabridge/".
    # v1.2 (2026-04-12): Phase 1 Job 14 — fix multi-target resolver gap.
    path_signals: List[str] = field(default_factory=list)

    @property
    def absolute_source_root(self) -> str:
        """Full path to the source root."""
        root = self.project_root.replace("\\", "/").rstrip("/")
        src = self.source_root.replace("\\", "/").lstrip("/")
        return f"{root}/{src}"

    def resolve_path(self, relative: str) -> str:
        """Resolve a relative path to an absolute path in this project.

        v1.2 (2026-04-12): Phase 1 Job 15 — Kotlin-aware expansion.
        When the input is a Kotlin file that lacks the full Android source
        root prefix (e.g. "voice/state/Foo.kt"), expand it relative to the
        profile's source_root (e.g. "app/src/main/java/com/astra/astrabridge")
        so writes land in the correct package directory. This is necessary
        because SpecGate/LLMs often emit short Kotlin paths without the full
        package prefix.

        v1.3 (2026-04-18): Don't expand .kts build scripts (build.gradle.kts,
        settings.gradle.kts). Those always live at the module/project root,
        never inside the package tree, so auto-expanding them produced
        bogus paths like app/src/main/java/com/example/drivercopilot/app/
        build.gradle.kts that Gradle ignores entirely. Only .kt source files
        get the package-prefix treatment now.
        """
        norm = relative.replace("\\", "/")
        # Already absolute
        if len(norm) > 1 and norm[1] == ":":
            return norm
        root = self.project_root.replace("\\", "/").rstrip("/")
        # Kotlin source-root expansion — .kt source files only, NOT .kts scripts
        if self.language == "kotlin" and norm.lower().endswith(".kt"):
            src = (self.source_root or "").replace("\\", "/").strip("/")
            if src and not norm.lower().startswith(src.lower() + "/"):
                # Also check that the path does not already contain the package
                # dir (to avoid double-expansion if caller already supplied
                # the full path).
                _pkg_marker = src.lower()
                if _pkg_marker not in norm.lower():
                    return f"{root}/{src}/{norm.lstrip('/')}"
        return f"{root}/{norm}"

    def relative_from_absolute(self, absolute: str) -> str:
        """Convert an absolute path back to project-relative."""
        norm = absolute.replace("\\", "/")
        root = self.project_root.replace("\\", "/").rstrip("/") + "/"
        if norm.lower().startswith(root.lower()):
            return norm[len(root):]
        return norm

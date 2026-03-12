# FILE: app/pipeline_v2/build_targets.py
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
    dependency_add_pattern: str = ""   # e.g. "implementation(...)" or "pip install"

    @property
    def absolute_source_root(self) -> str:
        """Full path to the source root."""
        root = self.project_root.replace("\\", "/").rstrip("/")
        src = self.source_root.replace("\\", "/").lstrip("/")
        return f"{root}/{src}"

    def resolve_path(self, relative: str) -> str:
        """Resolve a relative path to an absolute path in this project."""
        norm = relative.replace("\\", "/")
        # Already absolute
        if len(norm) > 1 and norm[1] == ":":
            return norm
        root = self.project_root.replace("\\", "/").rstrip("/")
        return f"{root}/{norm}"

    def relative_from_absolute(self, absolute: str) -> str:
        """Convert an absolute path back to project-relative."""
        norm = absolute.replace("\\", "/")
        root = self.project_root.replace("\\", "/").rstrip("/") + "/"
        if norm.lower().startswith(root.lower()):
            return norm[len(root):]
        return norm

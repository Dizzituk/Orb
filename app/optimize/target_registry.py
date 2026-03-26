from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class OptimizeTargetDefinition:
    target_id: str
    project_id: str
    project_label: str
    subsystem_id: str
    subsystem_label: str
    root_path: str
    include_globs: List[str] = field(default_factory=list)
    exclude_globs: List[str] = field(default_factory=list)
    user_outcome: str = ""
    verification: str = ""

    @property
    def display_label(self) -> str:
        return f"{self.project_label} → {self.subsystem_label}"

    def matches_relative_path(self, relative_path: str) -> bool:
        normalised = relative_path.replace('\\', '/')
        # Strip root path prefix so globs work on relative paths
        root_prefix = self.root_path.replace('\\', '/').rstrip('/') + '/'
        if normalised.startswith(root_prefix):
            normalised = normalised[len(root_prefix):]
        if self.include_globs:
            matched = False
            for pattern in self.include_globs:
                if fnmatch(normalised, pattern):
                    matched = True
                    break
                # fnmatch ** matches 1+ path segments, not 0
                if "**/" in pattern:
                    collapsed = pattern.replace("**/", "", 1)
                    if fnmatch(normalised, collapsed):
                        matched = True
                        break
            if not matched:
                return False
        if self.exclude_globs and any(fnmatch(normalised, pattern) for pattern in self.exclude_globs):
            return False
        return True

    def to_api_dict(self) -> Dict[str, object]:
        return {
            "target_id": self.target_id,
            "project_id": self.project_id,
            "project_label": self.project_label,
            "subsystem_id": self.subsystem_id,
            "subsystem_label": self.subsystem_label,
            "display_label": self.display_label,
            "root_path": self.root_path,
            "include_globs": list(self.include_globs),
            "exclude_globs": list(self.exclude_globs),
            "user_outcome": self.user_outcome,
            "verification": self.verification,
        }


def _backend_target(subsystem_id: str, subsystem_label: str, include_globs: List[str], user_outcome: str) -> OptimizeTargetDefinition:
    return OptimizeTargetDefinition(
        target_id=f"astra-backend:{subsystem_id}",
        project_id="astra-backend",
        project_label="ASTRA Backend",
        subsystem_id=subsystem_id,
        subsystem_label=subsystem_label,
        root_path="D:/Orb",
        include_globs=include_globs,
        exclude_globs=["**/__pycache__/**", "**/.pytest_cache/**", "**/node_modules/**", "**/.venv/**"],
        user_outcome=user_outcome,
        verification="browser",
    )


def _desktop_target(subsystem_id: str, subsystem_label: str, include_globs: List[str], user_outcome: str) -> OptimizeTargetDefinition:
    return OptimizeTargetDefinition(
        target_id=f"astra-desktop:{subsystem_id}",
        project_id="astra-desktop",
        project_label="ASTRA Desktop",
        subsystem_id=subsystem_id,
        subsystem_label=subsystem_label,
        root_path="D:/orb-desktop",
        include_globs=include_globs,
        exclude_globs=["**/node_modules/**", "**/dist/**"],
        user_outcome=user_outcome,
        verification="browser",
    )


def _bridge_target(subsystem_id: str, subsystem_label: str, include_globs: List[str], user_outcome: str) -> OptimizeTargetDefinition:
    return OptimizeTargetDefinition(
        target_id=f"astra-bridge:{subsystem_id}",
        project_id="astra-bridge",
        project_label="Astra Bridge",
        subsystem_id=subsystem_id,
        subsystem_label=subsystem_label,
        root_path="D:/Astra Android Folder/Astra-Bridge",
        include_globs=include_globs,
        exclude_globs=["**/.gradle/**", "**/build/**"],
        user_outcome=user_outcome,
        verification="emulator",
    )


def _copilot_target(subsystem_id: str, subsystem_label: str, include_globs: List[str], user_outcome: str) -> OptimizeTargetDefinition:
    return OptimizeTargetDefinition(
        target_id=f"driver-copilot:{subsystem_id}",
        project_id="driver-copilot",
        project_label="Driver CoPilot",
        subsystem_id=subsystem_id,
        subsystem_label=subsystem_label,
        root_path="D:/Astra Android Folder/AndroidDriverCopilot",
        include_globs=include_globs,
        exclude_globs=["**/.gradle/**", "**/build/**"],
        user_outcome=user_outcome,
        verification="emulator",
    )


TARGET_REGISTRY: Dict[str, OptimizeTargetDefinition] = {
    target.target_id: target
    for target in [
        _backend_target(
            "full-system",
            "Full system",
            ["app/**/*.py", "main.py"],
            "Improve the overall backend reliability and responsiveness across ASTRA services.",
        ),
        _backend_target(
            "optimize",
            "Optimizer engine",
            ["app/optimize/**/*.py"],
            "Improve how ASTRA detects, evaluates, and executes optimisations.",
        ),
        _backend_target(
            "memory",
            "Memory subsystem",
            ["app/astra_memory/**/*.py", "app/memory/**/*.py"],
            "Improve retrieval quality, confidence handling, and memory responsiveness.",
        ),
        _backend_target(
            "routing",
            "Routing and orchestration",
            ["app/agentic_pipeline/**/*.py", "app/routing/**/*.py", "app/llm/**/*.py"],
            "Improve task routing, orchestration, and decision consistency.",
        ),
        _backend_target(
            "debug",
            "Debug tooling",
            ["app/debug/**/*.py", "app/sandbox_*.py"],
            "Improve debugging speed, evidence quality, and tool reliability.",
        ),
        _desktop_target(
            "full-system",
            "Full app",
            ["src/**/*.ts", "src/**/*.tsx", "src/**/*.css"],
            "Improve the desktop experience across major ASTRA surfaces.",
        ),
        _desktop_target(
            "optimizer-tab",
            "Optimizer tab",
            ["src/components/optimize/**/*", "src/services/api.ts"],
            "Make the Optimizer tab clear, scoped, and aligned with real subsystem optimisation.",
        ),
        _desktop_target(
            "chat-shell",
            "Chat shell",
            ["src/components/**/*chat*.*", "src/pages/**/*chat*.*", "src/services/api.ts"],
            "Improve conversation flow and system feedback in the desktop shell.",
        ),
        _bridge_target(
            "full-system",
            "Full app",
            ["app/src/main/java/com/astra/astrabridge/**/*.kt"],
            "Improve Bridge responsiveness and cross-system continuity.",
        ),
        _bridge_target(
            "bridge-layer",
            "Bridge layer",
            ["app/src/main/java/com/astra/astrabridge/bridge/**/*.kt", "app/src/main/java/com/astra/astrabridge/navigation/**/*.kt"],
            "Improve desktop-to-bridge handoff and bridge execution reliability.",
        ),
        _copilot_target(
            "full-system",
            "Full app",
            ["app/src/main/java/com/example/drivercopilot/**/*.kt"],
            "Improve Driver CoPilot performance and usability across the app.",
        ),
    ]
}


def get_target_definition(target_id: str) -> OptimizeTargetDefinition:
    return TARGET_REGISTRY.get(target_id, TARGET_REGISTRY["astra-backend:optimize"])


def list_target_definitions() -> List[OptimizeTargetDefinition]:
    return sorted(TARGET_REGISTRY.values(), key=lambda item: (item.project_label, item.subsystem_label))


def path_matches_target(target: OptimizeTargetDefinition, relative_path: str) -> bool:
    return target.matches_relative_path(relative_path)


def target_root_path(target: OptimizeTargetDefinition) -> Path:
    return Path(target.root_path)

# FILE: app/pipeline_v2/scaffolds/kotlin_scaffolds.py
# Purpose: Kotlin Scaffold Templates for Android/Jetpack Compose projects (dispatcher/router).
# Called-by: app.pipeline_v2.scaffold_templates (via scaffold_engine)
# Depends-on: app.pipeline_v2.scaffolds.kotlin_templates_base, app.pipeline_v2.scaffolds.kotlin_templates_android, app.pipeline_v2.scaffolds.kotlin_template_helpers
# Last-renovated: 2026-06-21
"""
Kotlin Scaffold Templates for Android/Jetpack Compose projects.

Generates deterministic skeleton files based on file naming conventions.
Package declarations are derived from the file path relative to source root.

v1.0 (2026-03-10): Initial implementation for Driver CoPilot.
BATCH 4 split: the v1 base templates moved to kotlin_templates_base.py, the v2
Android templates to kotlin_templates_android.py, and the shared formatting
helpers to kotlin_template_helpers.py; all are re-exported below so the public
surface (generate_kotlin_skeleton) is unchanged.
"""
from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

from app.pipeline_v2.scaffolds.kotlin_template_helpers import (
    _req_block,
    _to_snake,
    _to_display_name,
)
from app.pipeline_v2.scaffolds.kotlin_templates_base import (
    _entity_skeleton,
    _dao_skeleton,
    _viewmodel_skeleton,
    _uistate_skeleton,
    _screen_skeleton,
    _service_skeleton,
    _repository_skeleton,
    _adapter_skeleton,
    _bridge_skeleton,
    _result_skeleton,
    _parser_skeleton,
)
from app.pipeline_v2.scaffolds.kotlin_templates_android import (
    _application_skeleton,
    _main_activity_skeleton,
    _activity_skeleton,
    _nav_graph_skeleton,
    _routes_skeleton,
    _api_client_skeleton,
    _api_service_skeleton,
    _manager_skeleton,
    _detector_skeleton,
    _uploader_skeleton,
    _picker_skeleton,
    _capture_skeleton,
    _di_module_skeleton,
    _theme_skeleton,
    _color_skeleton,
    _data_model_skeleton,
    _composable_component_skeleton,
    _generic_kotlin_skeleton,
)


def generate_kotlin_skeleton(
    file_path: str,
    requirements: List[str],
    profile: "BuildTargetProfile",
) -> str:
    """Generate a Kotlin skeleton based on file name pattern.

    v2.0: Added specific patterns for Application, Activity, Navigation,
    ApiClient, Manager, Detector, Uploader, Picker, Capture, Module,
    Theme, Color files. These produce richer scaffolds than the generic stub.
    """
    basename = os.path.basename(file_path).replace(".kt", "")
    package = _derive_package(file_path, profile)
    norm = file_path.replace("\\", "/")

    # Suffix-based matching (most specific first)
    if basename.endswith("Entity"):
        return _entity_skeleton(basename, package, requirements)
    elif basename.endswith("Dao"):
        return _dao_skeleton(basename, package, requirements)
    elif basename.endswith("ViewModel"):
        return _viewmodel_skeleton(basename, package, requirements)
    elif basename.endswith("UiState"):
        return _uistate_skeleton(basename, package, requirements)
    elif basename.endswith("Screen"):
        return _screen_skeleton(basename, package, requirements)
    elif basename.endswith("Service"):
        return _service_skeleton(basename, package, requirements)
    elif basename.endswith("Repository"):
        return _repository_skeleton(basename, package, requirements)
    elif basename.endswith("Adapter"):
        return _adapter_skeleton(basename, package, requirements)
    elif basename.endswith("Bridge"):
        return _bridge_skeleton(basename, package, requirements)
    elif basename.endswith("Result"):
        return _result_skeleton(basename, package, requirements)
    elif basename.endswith("Parser"):
        return _parser_skeleton(basename, package, requirements)
    # v2.0: Specific name patterns for common Android files
    elif basename == "MainActivity":
        return _main_activity_skeleton(basename, package, requirements, profile)
    elif basename.endswith("Activity"):
        return _activity_skeleton(basename, package, requirements)
    elif basename.endswith("App") or basename == "AstraApp":
        return _application_skeleton(basename, package, requirements)
    elif basename == "AppNavGraph" or basename.endswith("NavGraph"):
        return _nav_graph_skeleton(basename, package, requirements)
    elif basename == "Routes":
        return _routes_skeleton(basename, package, requirements)
    elif basename.endswith("ApiClient"):
        return _api_client_skeleton(basename, package, requirements)
    elif basename.endswith("ApiService"):
        return _api_service_skeleton(basename, package, requirements)
    elif basename.endswith("Manager"):
        return _manager_skeleton(basename, package, requirements)
    elif basename.endswith("Detector"):
        return _detector_skeleton(basename, package, requirements)
    elif basename.endswith("Uploader"):
        return _uploader_skeleton(basename, package, requirements)
    elif basename.endswith("Picker"):
        return _picker_skeleton(basename, package, requirements)
    elif basename.endswith("Capture"):
        return _capture_skeleton(basename, package, requirements)
    elif basename == "AppModule" or basename.endswith("Module"):
        return _di_module_skeleton(basename, package, requirements)
    elif basename == "Theme":
        return _theme_skeleton(basename, package, requirements)
    elif basename == "Color":
        return _color_skeleton(basename, package, requirements)
    elif "Message" in basename or basename.endswith("Dto"):
        return _data_model_skeleton(basename, package, requirements)
    elif basename.endswith("Button") or basename.endswith("Component"):
        return _composable_component_skeleton(basename, package, requirements)
    else:
        return _generic_kotlin_skeleton(basename, package, requirements)


def _derive_package(file_path: str, profile: "BuildTargetProfile") -> str:
    """Derive the Kotlin package name from file path and profile."""
    norm = file_path.replace("\\", "/")

    # Strip absolute prefix if present
    proot = profile.project_root.replace("\\", "/").rstrip("/") + "/"
    if norm.lower().startswith(proot.lower()):
        norm = norm[len(proot):]

    # Find the java/ or kotlin/ source root marker
    for marker in ("java/", "kotlin/"):
        idx = norm.find(marker)
        if idx >= 0:
            pkg_path = norm[idx + len(marker):]
            # Remove filename
            pkg_path = pkg_path.rsplit("/", 1)[0] if "/" in pkg_path else ""
            return pkg_path.replace("/", ".")

    # Fallback: use profile package + relative subdirectory
    source_root = profile.source_root.replace("\\", "/").rstrip("/") + "/"
    if norm.lower().startswith(source_root.lower()):
        rel = norm[len(source_root):]
    else:
        rel = norm

    # Remove filename, convert to package
    if "/" in rel:
        subpkg = rel.rsplit("/", 1)[0].replace("/", ".")
        return f"{profile.package_name}.{subpkg}"
    return profile.package_name

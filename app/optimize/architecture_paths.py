from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.optimize.target_registry import OptimizeTargetDefinition


DEFAULT_ARCH_PATHS: Dict[str, Path] = {
    "import_graph": Path("D:/Orb/.architecture/IMPORT_GRAPH.json"),
    "index": Path("D:/Orb/.architecture/INDEX.json"),
}


def get_architecture_paths(target: OptimizeTargetDefinition) -> Dict[str, Path]:
    root = Path(target.root_path)
    project_arch = root / ".architecture"
    import_graph = project_arch / "IMPORT_GRAPH.json"
    index_path = project_arch / "INDEX.json"
    return {
        "import_graph": import_graph if import_graph.exists() else DEFAULT_ARCH_PATHS["import_graph"],
        "index": index_path if index_path.exists() else DEFAULT_ARCH_PATHS["index"],
    }

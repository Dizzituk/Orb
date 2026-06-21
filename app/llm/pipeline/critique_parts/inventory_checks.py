# FILE: app/llm/pipeline/critique_parts/inventory_checks.py
# Purpose: Re-export shim — preserves the inventory_checks public surface after the 2026-06-20 split.
# Called-by: app.llm.pipeline.critique_parts.deterministic_verdict
# Depends-on: inventory_file_checks, inventory_create_classify, inventory_import_contracts
# Last-renovated: 2026-06-20
"""
Deterministic Critique — File Inventory, Classification & Import Checks (FACADE).

This module is now a thin re-export shim. On 2026-06-20 the three
independent deterministic check families were split into focused modules
via the conservative move-and-shim pattern (logic byte-identical):

    inventory_file_checks.py       — CHECK 1: file inventory compliance
                                     (+ shared helpers _extract_arch_file_inventory,
                                      _normalise_path — CHECK 1 is their sole caller)
    inventory_create_classify.py   — CHECK 2: CREATE vs MODIFY classification
                                     (+ _load_filesystem_index, _extract_new_file_paths,
                                      _check_sandbox_file_exists, FS-index cache)
    inventory_import_contracts.py  — CHECK 3: import contract validation
                                     (+ _extract_imports_from_arch)

Importers (currently only deterministic_verdict.py) keep resolving
`from ...inventory_checks import check_*` identically through this shim.

Zero LLM calls. Pure structural comparison.

v1.0 (2026-02-27): Initial implementation — Stage 1 of deterministic
verification migration.
v1.1 (2026-03-01): Added CHECK 2 — CREATE/MODIFY classification
accuracy via INDEX.json and filesystem cross-reference.
v1.2 (2026-06-20): Split into three focused modules; this file is now a
re-export shim. Behaviour unchanged.
"""

from __future__ import annotations

from app.llm.pipeline.critique_parts.inventory_file_checks import (
    check_file_inventory_compliance,
)
from app.llm.pipeline.critique_parts.inventory_create_classify import (
    check_create_modify_classification,
)
from app.llm.pipeline.critique_parts.inventory_import_contracts import (
    check_import_contracts,
)

INVENTORY_CHECKS_BUILD_ID = "2026-03-01-v1.1-inventory-import-and-classification-checks"

__all__ = [
    "check_file_inventory_compliance",
    "check_create_modify_classification",
    "check_import_contracts",
    "INVENTORY_CHECKS_BUILD_ID",
]

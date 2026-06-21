# Purpose: simple create utils 15 (re-export shim).
# Called-by: app.pot_spec.grounded._simple_create_evidence, app.pot_spec.grounded._simple_create_utils_16, app.pot_spec.grounded._simple_create_utils_17, app.pot_spec.grounded.simple_create
# Depends-on: app.pot_spec.grounded._simple_create_patterns_15, app.pot_spec.grounded._simple_create_evidence_read_15, app.pot_spec.grounded._simple_create_render_15
# Last-renovated: 2026-06-21
# Split 2026-06-21 (BATCH 6): discovery/patterns, evidence-read, and render moved to single-
# responsibility modules. All public names re-exported so importers resolve unchanged.
from __future__ import annotations
from app.pot_spec.grounded._simple_create_patterns_15 import _find_integration_points, _extract_patterns
from app.pot_spec.grounded._simple_create_evidence_read_15 import _read_text_any_encoding, _resolve_evidence_path, _host_read_file
from app.pot_spec.grounded._simple_create_render_15 import build_create_spec

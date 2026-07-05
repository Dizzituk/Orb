# FILE: app/pipeline_v2/segmented/__init__.py
# Purpose: Derek phase 5 — segmented builder package facade (segments become execution units).
# Called-by: app.pipeline_v2.orchestrator
# Depends-on: app.pipeline_v2.segmented.segmented_builder
# Last-renovated: 2026-07-04
"""Segmented builder: scheduler -> scoped HANDS workers -> HEAVY integrator.

Selected via ASTRA_BUILDER_MODE=segmented (the classic single-context
builder remains fully working as ASTRA_BUILDER_MODE=single)."""

from app.pipeline_v2.segmented.segmented_builder import run_segmented_builder

__all__ = ["run_segmented_builder"]

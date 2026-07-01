# FILE: app/reports/__init__.py
# Purpose: Render-on-demand report pipeline — styled HTML from ledger series, desktop window + Bridge document.
# Called-by: app.tools.registry (tool), app.bridge.artifacts (document base dir), main.py (static mount)
# Depends-on: app.reports.renderer, app.reports.cache, app.reports.surface
# Last-renovated: 2026-07-01
"""
Everything here is composition over existing pieces: html_builder (styled
single-file HTML), chart_renderer (deterministic Plotly PNG, embedded as
base64), display_client (reports window on a display alias), bridge
artifacts (document chip on the phone). No LLM anywhere in the render path.
"""

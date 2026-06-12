# FILE: app/transparency/__init__.py
# Purpose: Pipeline Transparency & User Feedback System.
# Called-by: app.db
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pipeline Transparency & User Feedback System.

Provides:
- ReasoningEvent emission from pipeline stages
- Collapsible reasoning traces in the UI
- User corrections pinned to specific decisions
- Correction matching for deterministic learning

Modules:
- schemas: Data structures for events and corrections
- collector: ReasoningCollector — persist + stream events
- corrections: CorrectionStore — CRUD for user corrections
- matcher: CorrectionMatcher — query relevant past corrections
- models: SQLAlchemy models for DB tables
- router: FastAPI endpoints
- io_events: IOEvent dataclass for file read/write tracking
- io_tracker: IOTracker — context-var-based IO operation recorder
"""

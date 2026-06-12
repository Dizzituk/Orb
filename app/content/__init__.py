# FILE: app/content/__init__.py
# Purpose: ASTRA Content Creation Pipeline
# Called-by: app.content.item_router, app.content.project_router, app.content.router, app.content.style_analyser (+2 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
ASTRA Content Creation Pipeline

Transforms natural voice conversations into multi-format social media content.
See: docs/ASTRA_Content_Creation_Pipeline_Spec_v1.docx

Modules:
- models: SQLAlchemy ORM models for content memory database
- schemas: Pydantic schemas for API request/response validation
- service: Core business logic for content operations
- router: FastAPI endpoints for content pipeline
"""

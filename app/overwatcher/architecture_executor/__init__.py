# Purpose: Architecture Executor - trimmed to active utilities only.
# Called-by: app.agentic_pipeline.pipeline, app.agentic_pipeline.segment_printer, app.pipeline_v2.stages.builder
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""Architecture Executor - trimmed to active utilities only.

Retained modules:
- arch_code_extractor: extract code blocks from LLM output
- parsing: architecture document parsing
- helpers: phase checkout helpers

Dead step files and orchestrator removed 2026-03-08.
"""

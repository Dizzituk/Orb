# FILE: app/llm/preprocessor.py
# Purpose: Re-export shim for the content preprocessor (split batch 3, 2026-06-21).
# Called-by: app.llm
# Depends-on: app.llm.preprocess_results, app.llm.preprocess_modalities, app.llm.preprocess_context
# Last-renovated: 2026-06-21
"""Content Preprocessor for Orb Routing Pipeline -- RE-EXPORT SHIM.

Split 2026-06-21 (batch 3) into single-responsibility modules; this module
preserves the original public surface so importers resolve unchanged:
  - preprocess_results.py    -- the 5 result dataclasses (leaf)
  - preprocess_modalities.py -- config + per-modality preprocessors + preprocess_task
  - preprocess_context.py    -- build_task_context / build_critical_context

Usage (API unchanged):
    from app.llm.preprocessor import preprocess_task, build_task_context
"""
from app.llm.preprocess_results import (
    VideoPreprocessResult,
    ImagePreprocessResult,
    CodePreprocessResult,
    TextPreprocessResult,
    TaskPreprocessResult,
)
from app.llm.preprocess_modalities import (
    VIDEO_TRANSCRIPT_MAX_TOKENS,
    VIDEO_SUMMARY_MAX_CHARS,
    IMAGE_SUMMARY_MAX_CHARS,
    CODE_SUMMARY_MAX_CHARS,
    TEXT_SUMMARY_MAX_CHARS,
    ROUTER_DEBUG,
    preprocess_video,
    preprocess_image,
    preprocess_code,
    preprocess_code_sync,
    preprocess_text,
    preprocess_text_sync,
    preprocess_task,
)
from app.llm.preprocess_context import (
    build_task_context,
    build_critical_context,
)


__all__ = [
    # Results
    "VideoPreprocessResult",
    "ImagePreprocessResult",
    "CodePreprocessResult",
    "TextPreprocessResult",
    "TaskPreprocessResult",
    
    # Preprocessing functions
    "preprocess_video",
    "preprocess_image",
    "preprocess_code",
    "preprocess_code_sync",
    "preprocess_text",
    "preprocess_text_sync",
    "preprocess_task",
    
    # Context builders
    "build_task_context",
    "build_critical_context",
    
    # Configuration
    "VIDEO_TRANSCRIPT_MAX_TOKENS",
    "VIDEO_SUMMARY_MAX_CHARS",
    "IMAGE_SUMMARY_MAX_CHARS",
    "CODE_SUMMARY_MAX_CHARS",
    "TEXT_SUMMARY_MAX_CHARS",
]
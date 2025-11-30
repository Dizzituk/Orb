# FILE: app/llm/__init__.py
"""
LLM module exports.
"""
from app.llm.router import call_llm, quick_chat, request_code, review_work
from app.llm.schemas import LLMTask, LLMResult, JobType, Provider
from app.llm.gemini_vision import analyze_image, is_image_mime_type
from app.llm.file_analyzer import (
    extract_text_content,
    detect_document_type,
    parse_cv_with_llm,
    generate_document_summary,
)

__all__ = [
    "call_llm",
    "quick_chat", 
    "request_code",
    "review_work",
    "LLMTask",
    "LLMResult",
    "JobType",
    "Provider",
    "analyze_image",
    "is_image_mime_type",
    "extract_text_content",
    "detect_document_type",
    "parse_cv_with_llm",
    "generate_document_summary",
]
# FILE: app/llm/preprocess_results.py
# Purpose: Preprocessing result dataclasses (leaf shared by modalities + context).
# Called-by: app.llm.preprocessor (shim), app.llm.preprocess_modalities, app.llm.preprocess_context
# Depends-on: stdlib only
# Last-renovated: 2026-06-21
"""Preprocessing result dataclasses for the Orb routing pipeline.

Split out of preprocessor.py (batch 3, 2026-06-21) as the shared leaf data
vocabulary. Re-exported verbatim via the app.llm.preprocessor shim.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class VideoPreprocessResult:
    """Result of video preprocessing."""
    file_id: str
    filename: str
    
    # Full transcript (token-bounded)
    transcript: str = ""
    transcript_tokens: int = 0
    
    # Condensed summary
    summary: str = ""
    summary_tokens: int = 0
    
    # Metadata
    model_used: str = ""
    duration_ms: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "transcript_tokens": self.transcript_tokens,
            "summary_tokens": self.summary_tokens,
            "model_used": self.model_used,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class ImagePreprocessResult:
    """Result of image preprocessing."""
    file_id: str
    filename: str
    
    # OCR text (if applicable)
    ocr_text: str = ""
    
    # Semantic description
    description: str = ""
    
    # Metadata
    model_used: str = ""
    duration_ms: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "has_ocr": bool(self.ocr_text),
            "description_chars": len(self.description),
            "model_used": self.model_used,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class CodePreprocessResult:
    """Result of code preprocessing."""
    file_id: str
    filename: str
    
    # Structural summary
    summary: str = ""
    
    # Key snippets (function bodies, error locations)
    key_snippets: List[str] = field(default_factory=list)
    key_snippets_chars: int = 0
    
    # Extracted elements
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    
    # Full content available
    full_content: str = ""
    full_content_chars: int = 0
    
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "summary_chars": len(self.summary),
            "key_snippets_count": len(self.key_snippets),
            "classes": self.classes,
            "functions": self.functions[:10],
            "full_content_chars": self.full_content_chars,
            "error": self.error,
        }


@dataclass
class TextPreprocessResult:
    """Result of text preprocessing."""
    file_id: str
    filename: str
    
    # High-level summary
    summary: str = ""
    
    # Key excerpts
    key_excerpts: List[str] = field(default_factory=list)
    
    # Full content available
    full_content: str = ""
    full_content_chars: int = 0
    
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "summary_chars": len(self.summary),
            "excerpts_count": len(self.key_excerpts),
            "full_content_chars": self.full_content_chars,
            "error": self.error,
        }


@dataclass
class TaskPreprocessResult:
    """Complete preprocessing result for a task."""
    task_id: str
    file_ids: List[str]
    
    # Per-modality results
    video_results: List[VideoPreprocessResult] = field(default_factory=list)
    image_results: List[ImagePreprocessResult] = field(default_factory=list)
    code_results: List[CodePreprocessResult] = field(default_factory=list)
    text_results: List[TextPreprocessResult] = field(default_factory=list)
    
    # Combined context (built from above)
    combined_context: str = ""
    combined_context_tokens: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_ms: int = 0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "file_ids": self.file_ids,
            "video_count": len(self.video_results),
            "image_count": len(self.image_results),
            "code_count": len(self.code_results),
            "text_count": len(self.text_results),
            "combined_context_tokens": self.combined_context_tokens,
            "total_duration_ms": self.total_duration_ms,
            "errors": self.errors,
        }

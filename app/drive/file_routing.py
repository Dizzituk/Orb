# FILE: app/drive/file_routing.py
# Purpose: File-type-aware model routing for ASTRA Drive actions.
# Called-by: app.drive.router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
File-type-aware model routing for ASTRA Drive actions.

Routes "Ask ASTRA to read/summarise" requests to the optimal LLM
based on file type. Each file class maps to a specific provider/model
and content preparation strategy.

v0.14.0: Added MIME entries for xlsx, xls, pptx, docx, doc.

Routing rules (defined by Taz):
  - Images        -> Gemini 2.5 Flash (multimodal vision, fast)
  - Videos        -> Gemini 3 Pro Preview (video understanding)
  - Audio/Music   -> Gemini 3 Pro Preview (audio understanding)
  - Code files    -> Claude (architecture + code analysis)
  - Data files    -> Claude (structured data reasoning)
  - Documents     -> GPT-5.4 (best linguistics)
  - PDFs          -> GPT-5.4 (text extraction + comprehension)
  - Spreadsheets  -> GPT-5.4 (tabular data + language)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FileRoute:
    """Routing decision for a file type."""
    provider: str
    model: str
    content_mode: str  # "text", "base64_image", "base64_binary", "file_uri"
    mime_prefix: Optional[str] = None  # For multimodal: "image/", "video/", "audio/"


# ─── Extension → Route mapping ──────────────────────────────────

# Images → Gemini 2.5 Flash (multimodal vision)
IMAGE_ROUTE = FileRoute(
    provider="google",
    model="gemini-2.5-flash",
    content_mode="base64_image",
    mime_prefix="image/",
)

# Videos → Gemini 3 Pro Preview (video understanding)
VIDEO_ROUTE = FileRoute(
    provider="google",
    model="gemini-3.1-pro-preview-customtools",
    content_mode="file_uri",
    mime_prefix="video/",
)

# Audio → Gemini 3 Pro Preview (audio understanding)
AUDIO_ROUTE = FileRoute(
    provider="google",
    model="gemini-3.1-pro-preview-customtools",
    content_mode="file_uri",
    mime_prefix="audio/",
)

# Code → Claude (architecture + analysis)
CODE_ROUTE = FileRoute(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    content_mode="text",
)

# Data files → Claude (structured reasoning)
DATA_ROUTE = FileRoute(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    content_mode="text",
)

# Documents → GPT-5.4 (best linguistics)
DOCUMENT_ROUTE = FileRoute(
    provider="openai",
    model="gpt-5.4",
    content_mode="text",
)

# PDFs → GPT-5.4 (text comprehension)
PDF_ROUTE = FileRoute(
    provider="openai",
    model="gpt-5.4",
    content_mode="base64_binary",
    mime_prefix="application/",
)

# Spreadsheets → GPT-5.4 (tabular + language)
SPREADSHEET_ROUTE = FileRoute(
    provider="openai",
    model="gpt-5.4",
    content_mode="text",
)

# Default fallback
DEFAULT_ROUTE = FileRoute(
    provider="google",
    model="gemini-2.5-flash",
    content_mode="text",
)


EXTENSION_ROUTES: dict[str, FileRoute] = {
    # Images
    "png": IMAGE_ROUTE,
    "jpg": IMAGE_ROUTE,
    "jpeg": IMAGE_ROUTE,
    "gif": IMAGE_ROUTE,
    "webp": IMAGE_ROUTE,
    "svg": IMAGE_ROUTE,
    "bmp": IMAGE_ROUTE,
    "ico": IMAGE_ROUTE,
    "tiff": IMAGE_ROUTE,

    # Videos
    "mp4": VIDEO_ROUTE,
    "mkv": VIDEO_ROUTE,
    "avi": VIDEO_ROUTE,
    "mov": VIDEO_ROUTE,
    "wmv": VIDEO_ROUTE,
    "webm": VIDEO_ROUTE,
    "flv": VIDEO_ROUTE,

    # Audio
    "mp3": AUDIO_ROUTE,
    "wav": AUDIO_ROUTE,
    "flac": AUDIO_ROUTE,
    "ogg": AUDIO_ROUTE,
    "aac": AUDIO_ROUTE,
    "m4a": AUDIO_ROUTE,
    "wma": AUDIO_ROUTE,

    # Code
    "py": CODE_ROUTE,
    "js": CODE_ROUTE,
    "ts": CODE_ROUTE,
    "tsx": CODE_ROUTE,
    "jsx": CODE_ROUTE,
    "css": CODE_ROUTE,
    "html": CODE_ROUTE,
    "htm": CODE_ROUTE,
    "sql": CODE_ROUTE,

    # Data
    "csv": DATA_ROUTE,
    "json": DATA_ROUTE,
    "xml": DATA_ROUTE,
    "yaml": DATA_ROUTE,
    "yml": DATA_ROUTE,
    "toml": DATA_ROUTE,

    # Documents
    "txt": DOCUMENT_ROUTE,
    "md": DOCUMENT_ROUTE,
    "docx": DOCUMENT_ROUTE,
    "doc": DOCUMENT_ROUTE,
    "rst": DOCUMENT_ROUTE,

    # PDFs
    "pdf": PDF_ROUTE,

    # Spreadsheets
    "xlsx": SPREADSHEET_ROUTE,
    "xls": SPREADSHEET_ROUTE,

    # Presentations (text extraction, same as documents)
    "pptx": DOCUMENT_ROUTE,
    "ppt": DOCUMENT_ROUTE,
}


def get_file_route(extension: str) -> FileRoute:
    """Get the routing decision for a file extension."""
    return EXTENSION_ROUTES.get(extension.lower(), DEFAULT_ROUTE)


# MIME type helpers
MIME_MAP: dict[str, str] = {
    # Images
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    # Video
    "mp4": "video/mp4",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "webm": "video/webm",
    # Audio
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    # Documents
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint",
    "txt": "text/plain",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "yaml": "application/x-yaml",
    "yml": "application/x-yaml",
}


def get_mime_type(extension: str) -> str:
    """Get MIME type for a file extension."""
    return MIME_MAP.get(extension.lower(), "application/octet-stream")

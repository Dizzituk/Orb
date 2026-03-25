# FILE: app/debug/file_upload_router.py
"""
File Upload Router — accept files for debug assistant analysis.

v0.14.0: Now triggers universal knowledge hook (text extraction,
         embeddings indexing, knowledge promotion) for ALL uploads.

Handles images (screenshots, photos) and text files (logs, configs).
Images are uploaded to the Gemini Files API for multimodal analysis.
Text files have their content returned for inline embedding.

Endpoint: POST /api/debug/upload
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.auth import require_auth
from app.auth.middleware import AuthResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["Debug File Upload"])

# Where uploaded files are saved temporarily
UPLOADS_DIR = Path(os.getenv("ASTRA_DATA_DIR", "D:/Orb/data")) / "debug_uploads"

# MIME types that should be uploaded to Gemini Files API (binary / multimodal)
# Gemini supports: images, video, audio, PDFs, and more
GEMINI_UPLOAD_MIMES = {
    # Images
    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp",
    "image/bmp", "image/tiff",
    # Video
    "video/mp4", "video/webm", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "video/x-matroska",
    # Audio
    "audio/mpeg", "audio/wav", "audio/ogg", "audio/webm",
    "audio/mp4", "audio/flac",
    # Documents (binary but extractable)
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint",
}

# Max text file size to inline (500KB)
MAX_TEXT_INLINE_BYTES = 512_000


class FileUploadResponse(BaseModel):
    """Response from a successful debug file upload."""
    upload_id: str
    file_name: str
    mime_type: str
    size_bytes: int
    upload_type: str  # "image" or "text"
    # For images: Gemini file URI for multimodal content
    file_uri: Optional[str] = None
    gemini_file_name: Optional[str] = None
    # For text: the file content (truncated if large)
    text_content: Optional[str] = None
    # Local path for cleanup
    local_path: str = ""
    # v0.14.0: Knowledge hook status
    knowledge_hook: str = "pending"


def _run_knowledge_hook(file_path: str, original_name: str, mime_type: str):
    """
    Background task: run universal knowledge hook on uploaded file.

    Extracts text, stores document content, indexes embeddings,
    and promotes durable facts to ASTRA memory. Non-blocking.
    """
    try:
        from app.db import SessionLocal
        from app.memory.upload_knowledge_hook import process_uploaded_file_sync

        db = SessionLocal()
        try:
            result = process_uploaded_file_sync(
                db=db,
                file_path=file_path,
                original_name=original_name,
                mime_type=mime_type,
            )
            extracted = result.get("extracted", False)
            indexed = result.get("indexed", False)
            promoted = result.get("promoted", False)
            errors = result.get("errors", [])

            logger.info(
                "[debug_upload] Knowledge hook complete for %s: "
                "extracted=%s, indexed=%s, promoted=%s, errors=%d",
                original_name, extracted, indexed, promoted, len(errors),
            )
            if errors:
                for err in errors:
                    logger.warning("[debug_upload] Hook error: %s", err)
        finally:
            db.close()
    except Exception as e:
        logger.error("[debug_upload] Knowledge hook crashed for %s: %s", original_name, e)


@router.post("/upload", response_model=FileUploadResponse)
async def upload_debug_file(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    auth: AuthResult = Depends(require_auth),
):
    """Upload a file for use in the debug assistant chat.

    v0.14.0: ALL uploads now also trigger the universal knowledge hook
    in a background task — text extraction, embeddings, and knowledge
    promotion happen without blocking the upload response.

    Images → uploaded to Gemini Files API, returns file_uri for
             multimodal content parts.
    Text   → content read and returned directly for inline embedding.

    Accepts:
        file: The uploaded file
        metadata: Optional JSON string with context info

    Returns:
        FileUploadResponse with either file_uri (images) or
        text_content (text files).
    """
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}

    mime = file.content_type or "application/octet-stream"
    original_name = file.filename or "unnamed"

    # Save file to disk
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    upload_id = f"dbg-{timestamp}-{uuid.uuid4().hex[:8]}"
    ext = Path(original_name).suffix or _ext_from_mime(mime)
    save_path = UPLOADS_DIR / f"{upload_id}{ext}"

    try:
        contents = await file.read()
        save_path.write_bytes(contents)
        size_bytes = len(contents)
        logger.info(
            "[debug_upload] Saved %s (%s, %d bytes) to %s",
            original_name, mime, size_bytes, save_path,
        )
    except Exception as e:
        logger.error("[debug_upload] Save failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save file")

    # v0.14.0: Schedule knowledge hook as background task for ALL files
    background_tasks.add_task(_run_knowledge_hook, str(save_path), original_name, mime)

    # Route by type:
    # 1. Known Gemini-uploadable binary types → upload to Gemini Files API
    # 2. Text-like files → read content and return inline
    # 3. Unknown → try text first, fall back to Gemini upload

    if mime in GEMINI_UPLOAD_MIMES or mime.startswith(("image/", "video/", "audio/")):
        return await _handle_gemini_upload(upload_id, original_name, mime, size_bytes, save_path)

    if mime.startswith("text/") or _is_text_extension(ext):
        return _handle_text(upload_id, original_name, mime, size_bytes, save_path, contents)

    # Unknown type — try reading as text, fall back to Gemini upload
    try:
        contents.decode("utf-8")
        return _handle_text(upload_id, original_name, mime, size_bytes, save_path, contents)
    except (UnicodeDecodeError, ValueError):
        # Binary file — upload to Gemini (PDFs, archives, etc.)
        return await _handle_gemini_upload(upload_id, original_name, mime, size_bytes, save_path)


async def _handle_gemini_upload(
    upload_id: str,
    original_name: str,
    mime: str,
    size_bytes: int,
    save_path: Path,
) -> FileUploadResponse:
    """Upload any binary file to Gemini Files API for multimodal analysis."""
    try:
        from app.debug.screen_capture import upload_video_to_gemini

        info = await upload_video_to_gemini(
            file_path=str(save_path),
            mime_type=mime,
        )
        logger.info("[debug_upload] File uploaded to Gemini: %s (%s)", info.uri, mime)

        return FileUploadResponse(
            upload_id=upload_id,
            file_name=original_name,
            mime_type=mime,
            size_bytes=size_bytes,
            upload_type="image",
            file_uri=info.uri,
            gemini_file_name=info.name,
            local_path=str(save_path),
            knowledge_hook="scheduled",
        )
    except Exception as e:
        logger.error("[debug_upload] Gemini upload failed for %s: %s", mime, e)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to upload file to Gemini: {e}",
        )


def _handle_text(
    upload_id: str,
    original_name: str,
    mime: str,
    size_bytes: int,
    save_path: Path,
    raw_bytes: bytes,
) -> FileUploadResponse:
    """Read text file content and return it for inline embedding."""
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = raw_bytes.decode("latin-1", errors="replace")

    if len(text) > MAX_TEXT_INLINE_BYTES:
        text = text[:MAX_TEXT_INLINE_BYTES] + f"\n\n... [truncated at {MAX_TEXT_INLINE_BYTES // 1024}KB]"

    return FileUploadResponse(
        upload_id=upload_id,
        file_name=original_name,
        mime_type=mime,
        size_bytes=size_bytes,
        upload_type="text",
        text_content=text,
        local_path=str(save_path),
        knowledge_hook="scheduled",
    )


def _ext_from_mime(mime: str) -> str:
    """Get a file extension from MIME type."""
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "application/json": ".json",
        "text/markdown": ".md",
        "text/html": ".html",
    }
    return mapping.get(mime, "")


def _is_text_extension(ext: str) -> bool:
    """Check if a file extension is likely text."""
    text_exts = {
        ".txt", ".log", ".csv", ".json", ".xml", ".yaml", ".yml",
        ".md", ".py", ".kt", ".java", ".ts", ".tsx", ".js", ".jsx",
        ".html", ".css", ".scss", ".toml", ".ini", ".cfg", ".conf",
        ".sh", ".bat", ".ps1", ".gradle", ".properties",
    }
    return ext.lower() in text_exts

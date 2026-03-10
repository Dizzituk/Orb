# FILE: app/content/production/file_router.py
"""
Content file serving — video/image preview for Electron.

Supports HTTP Range requests for video seeking.
No auth dependency — localhost-only, safe for desktop app.
"""
import os
import stat
import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content/files",
    tags=["Content Files"],
)

MEDIA_TYPES = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


@router.get("/file")
async def serve_content_file(
    request: Request,
    path: str = Query(..., description="Relative path within data/content/"),
):
    """Serve a content file with Range request support.

    Required for HTML5 video playback (seeking, streaming).
    Localhost-only — no auth needed for desktop app.
    """
    base_dir = os.path.abspath(os.path.join("data", "content"))
    full_path = os.path.abspath(os.path.join(base_dir, path))

    if not full_path.startswith(base_dir):
        raise HTTPException(403, "Access denied")

    if not os.path.exists(full_path):
        raise HTTPException(404, "File not found")

    ext = full_path.rsplit(".", 1)[-1].lower() if "." in full_path else ""
    media_type = MEDIA_TYPES.get(ext, "application/octet-stream")

    file_size = os.path.getsize(full_path)
    range_header = request.headers.get("range")

    if range_header:
        # Parse Range header: bytes=start-end
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1

        if start >= file_size:
            raise HTTPException(416, "Range not satisfiable")

        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_range():
            with open(full_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(65536, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(content_length),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
            },
        )
    else:
        # Full file response
        def iter_file():
            with open(full_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(
            iter_file(),
            media_type=media_type,
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
            },
        )

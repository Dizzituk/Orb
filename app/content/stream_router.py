# FILE: app/content/stream_router.py
"""
Content SSE stream router — real-time pipeline events.
Prefix: /content

Note: EventSource API cannot send Authorization headers,
so this endpoint accepts auth via query param as fallback.
"""

from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse

from app.auth.config import validate_session
from app.content.sse_manager import sse_manager

router = APIRouter(
    prefix="/content",
    tags=["Content Stream"],
)


@router.get("/projects/{project_id}/stream")
async def project_stream(
    project_id: str,
    token: Optional[str] = Query(None),
):
    """SSE stream for project pipeline events.

    EventSource cannot send Authorization headers, so we
    accept the session token as a query parameter.
    """
    if not token or not validate_session(token):
        raise HTTPException(status_code=401, detail="Token required")

    return StreamingResponse(
        sse_manager.event_generator(project_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

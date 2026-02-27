# FILE: app/content/stream_router.py
"""
Content SSE stream router — real-time pipeline events.
Prefix: /content
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth import require_auth
from app.content.sse_manager import sse_manager

router = APIRouter(
    prefix="/content",
    tags=["Content Stream"],
    dependencies=[Depends(require_auth)],
)


@router.get("/projects/{project_id}/stream")
async def project_stream(project_id: str):
    return StreamingResponse(
        sse_manager.event_generator(project_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

# FILE: app/debug/orchestrator/endpoint.py
"""
FastAPI endpoint for the Debug Orchestrator.

POST /api/debug/projects/{project_id}/orchestrate
  Streams SSE events for each phase/subagent/verification step.
  Returns a final `resolution` event with the full DebugResolution.

The endpoint reads the Debug Project description as the bug list.

v1.0 (2026-04-13): Initial implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth import require_auth
from app.auth.middleware import AuthResult

from app.debug.orchestrator.loop_controller import run_orchestration
from app.debug.orchestrator.schemas import (
    DebugResolution,
    OrchestrationEvent,
    OrchestrationRequest,
)
from app.debug.project_service import get_project

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["Debug Orchestrator"])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class OrchestrateOptions(BaseModel):
    """Client-supplied overrides. Defaults come from OrchestrationRequest."""
    target_project: Optional[str] = None
    max_iterations: int = 3
    max_subagents_parallel: int = 5
    enable_behaviour_verify: bool = True


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: OrchestrationEvent) -> str:
    payload = event.model_dump(mode="json")
    return f"data: {json.dumps(payload)}\n\n"


def _sse_raw(event_type: str, data: dict) -> str:
    payload = {"event_type": event_type, **data}
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/orchestrate")
async def orchestrate(
    project_id: str,
    options: OrchestrateOptions,
    auth: AuthResult = Depends(require_auth),
):
    """Run the orchestrator against a Debug Project, streaming SSE events."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, detail=f"Debug project {project_id} not found")

    bug_list = (project.get("description") or "").strip()
    if not bug_list and project.get("error_summary"):
        bug_list = project["error_summary"]
    if not bug_list:
        raise HTTPException(400, detail="Debug project has no description or error_summary to orchestrate against.")

    request_obj = OrchestrationRequest(
        project_id=project_id,
        bug_list=bug_list,
        target_project=options.target_project,
        max_iterations=options.max_iterations,
        max_subagents_parallel=options.max_subagents_parallel,
        enable_behaviour_verify=options.enable_behaviour_verify,
    )

    # Event bridge: synchronous callback from the loop -> async queue -> SSE
    queue: asyncio.Queue = asyncio.Queue()

    def _on_event(evt: OrchestrationEvent) -> None:
        try:
            queue.put_nowait(evt)
        except Exception as e:
            logger.warning("[orchestrate] queue.put_nowait failed: %s", e)

    async def _runner() -> DebugResolution:
        try:
            return await run_orchestration(request=request_obj, on_event=_on_event)
        finally:
            # Sentinel to tell the generator we are done
            await queue.put(None)

    async def event_stream() -> AsyncGenerator[str, None]:
        yield _sse_raw("start", {"project_id": project_id, "bug_list_chars": len(bug_list)})

        task = asyncio.create_task(_runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(item)
        except asyncio.CancelledError:
            logger.info("[orchestrate] client disconnected — cancelling run")
            task.cancel()
            raise
        finally:
            if not task.done():
                try:
                    task.cancel()
                except Exception:
                    pass

        # Task complete: emit the final resolution object
        try:
            resolution = await task
            yield _sse_raw("final", {"resolution": resolution.model_dump(mode="json")})
        except asyncio.CancelledError:
            yield _sse_raw("cancelled", {})
        except Exception as e:
            logger.exception("[orchestrate] runner raised: %s", e)
            yield _sse_raw("error", {"message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Non-streaming variant: run and return the final resolution only (for tests)
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/orchestrate/sync")
async def orchestrate_sync(
    project_id: str,
    options: OrchestrateOptions,
    auth: AuthResult = Depends(require_auth),
):
    """Synchronous variant — runs the orchestrator and returns the final DebugResolution.

    Useful for scripted tests or when SSE is inconvenient. Has no live progress.
    """
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, detail=f"Debug project {project_id} not found")

    bug_list = (project.get("description") or "").strip() or project.get("error_summary") or ""
    if not bug_list:
        raise HTTPException(400, detail="Debug project has no description to orchestrate against.")

    request_obj = OrchestrationRequest(
        project_id=project_id,
        bug_list=bug_list,
        target_project=options.target_project,
        max_iterations=options.max_iterations,
        max_subagents_parallel=options.max_subagents_parallel,
        enable_behaviour_verify=options.enable_behaviour_verify,
    )
    resolution = await run_orchestration(request=request_obj, on_event=None)
    return resolution.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Activity timeline (used by the Info tab)
# ---------------------------------------------------------------------------

@router.get("/projects/{project_id}/activity")
async def get_project_activity(
    project_id: str,
    limit: int = 200,
    auth: AuthResult = Depends(require_auth),
):
    """Return the activity timeline for a debug project: messages, orchestration
    runs, file changes, etc. Newest first."""
    from app.debug.orchestrator.activity_store import list_activity
    entries = list_activity(debug_project_id=project_id, limit=limit)
    return {"project_id": project_id, "count": len(entries), "entries": entries}


@router.delete("/projects/{project_id}/activity")
async def clear_project_activity(
    project_id: str,
    auth: AuthResult = Depends(require_auth),
):
    from app.debug.orchestrator.activity_store import clear_activity
    deleted = clear_activity(debug_project_id=project_id)
    return {"project_id": project_id, "deleted": deleted}

# FILE: app/bridge/schemas.py
"""
Pydantic request/response models and auth helper for the Bridge API.

Extracted from router.py during modularisation (2026-04-06).
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.auth import config as auth_config


security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BridgeLoginRequest(BaseModel):
    password: str


class BridgeLoginResponse(BaseModel):
    session_token: str
    message: str



class BridgeArtifactRef(BaseModel):
    # Downloadable artifact attached to an assistant message — generated
    # image, document, etc. The phone uses kind for the chip icon, filename
    # for display, size_bytes for the chip label, and url to fetch the
    # bytes via the auth-gated /bridge/artifacts endpoint.
    kind: str
    filename: str
    size_bytes: int = 0
    mime_type: str = "application/octet-stream"
    url: str

class BridgeChatRequest(BaseModel):
    message: str
    project_id: Optional[int] = None
    # Upload IDs returned by POST /bridge/upload. The chat-and-speak handler
    # resolves each via app.bridge.uploads_store.find_by_id, runs vision on
    # image attachments, and injects the descriptions into the prompt. Empty
    # or None means no attachments - the message is processed as plain text.
    attachment_ids: Optional[list[str]] = None


class BridgeChatResponse(BaseModel):
    reply: str
    project_id: int
    project_name: str
    domain: Optional[str] = None
    # DB id of the assistant message this reply was stored as. The phone
    # advances its missed-reply cursor and durable played-registry from this
    # so a text/muted reply it already received is never re-spoken by the
    # reconnect recovery (the voice path already gets the id via X-Message-Id).
    # Optional so an older phone build / non-message gate paths stay valid.
    message_id: Optional[int] = None
    # Downloadable artifacts (generated images, etc.). Empty for most
    # turns; populated when the assistant produced files this turn.
    attachments: list[BridgeArtifactRef] = []
    # Phone-action directives parsed from [[astra:...]] reply markers
    # (app/bridge/directives.py) — e.g. {"name": "sync_health", "arg": "48"}.
    directives: list[dict] = []
    # Model badge parity (2026-07-03): which brain answered, mirroring the
    # desktop's per-message chips (DB rows already store both). Optional and
    # additive — phone builds without the chip UI ignore unknown JSON fields.
    provider: Optional[str] = None
    model: Optional[str] = None


class BridgeProjectOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    created_at: str
    updated_at: str
    pinned: bool = False


class BridgePinRequest(BaseModel):
    # Idempotent set, not toggle. The caller knows the desired final state
    # (so concurrent pin/unpin from desktop + phone is race-free at the data
    # layer; last write wins, but neither client can accidentally flip the
    # state to a value it didn't intend).
    pinned: bool


class BridgeMessageOut(BaseModel):
    id: int
    role: str
    content: str
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: str
    # Parsed from content markers on history reload. Lets the phone
    # re-render historical generated-image messages with download chips,
    # not just the bare reply text.
    attachments: list[BridgeArtifactRef] = []


class BridgeTTSRequest(BaseModel):
    text: str
    voice_name: Optional[str] = None
    speed: Optional[float] = None


class BridgeDebugRequest(BaseModel):
    # Body for POST /bridge/debug-and-speak (live-session plan §2.1). A
    # separate model from BridgeChatRequest rather than an extension of it --
    # debug_project_id/reasoning_dial are debug-brain-specific and duck-typing
    # against _resolve_or_create_project (which only reads .message/.project_id)
    # means no shared base class is needed.
    message: str
    project_id: Optional[int] = None
    debug_project_id: Optional[str] = None
    reasoning_dial: Optional[str] = None


class BridgeDebugCancelRequest(BaseModel):
    # Either identifies the run directly, or names the project so the caller
    # doesn't need to have captured a run_id (e.g. a stale/lost metadata header).
    # At least one should be set; run_id wins if both are given. int, not str:
    # run_id doubles as the tts_cache/tts_audio cache key (see
    # debug_run_registry.new_run_id), which is int-typed throughout that stack.
    run_id: Optional[int] = None
    project_id: Optional[int] = None


class BridgeDebugCancelResponse(BaseModel):
    cancelled: bool
    run_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def require_bridge_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """Validate session token from the bridge app."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Use /bridge/login first.",
        )
    token = credentials.credentials
    if auth_config.validate_session(token):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session. Please log in again via /bridge/login.",
    )


# ---------------------------------------------------------------------------
# Health Connect ingest (AstraBridge -> Orb)  — added 2026-05-31
# ---------------------------------------------------------------------------

class HealthWeightIn(BaseModel):
    client_id: str                      # Health Connect record UID (dedup key)
    recorded_at: str                    # ISO 8601 timestamp of the measurement
    weight_kg: float
    source: str = "garmin"


class HealthWorkoutIn(BaseModel):
    client_id: str
    started_at: str                     # ISO 8601 start time
    duration_mins: Optional[int] = None
    activity_type: str = "workout"      # phone maps HC exercise type -> tag
    title: Optional[str] = None
    calories_burned: Optional[float] = None
    distance_m: Optional[float] = None
    avg_hr: Optional[int] = None
    source: str = "garmin"


class HealthDayIn(BaseModel):
    date: str                           # YYYY-MM-DD
    steps: Optional[int] = None
    floors: Optional[int] = None
    active_calories: Optional[float] = None
    resting_hr: Optional[int] = None
    sleep_minutes: Optional[int] = None
    source: str = "garmin"


class HealthIngestRequest(BaseModel):
    weights: list[HealthWeightIn] = []
    workouts: list[HealthWorkoutIn] = []
    days: list[HealthDayIn] = []


class HealthIngestResponse(BaseModel):
    ok: bool = True
    weights_added: int = 0
    weights_skipped: int = 0
    workouts_added: int = 0
    workouts_skipped: int = 0
    days_upserted: int = 0
    errors: list[str] = []

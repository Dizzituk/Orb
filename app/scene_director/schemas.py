# FILE: app/scene_director/schemas.py
# Purpose: The SceneDoc contract v1 — Pydantic models for everything the renderer
#          (Unity) consumes. Renderer-agnostic; the single source of truth.
# Called-by: app.scene_director.director/router/state, Unity SceneDocModels.cs (mirror)
# Depends-on: stdlib/pydantic only
# Last-renovated: 2026-06-12
"""SceneDoc contract v1.

NUMERIC CONVENTIONS (the renderer relies on these — do not change casually):
  * Coordinate system: Unity's — LEFT-handed, Y-up, +Z forward.
  * Units: METRES. The ground plane is y=0.
  * position / spawn / waypoints: [x, y, z] in metres.
  * rotation_y: degrees, clockwise when viewed from above (Unity yaw).
  * speed_mps: metres per second (1.2 ≈ casual human walk).
  * at_seconds: seconds from timeline start.

The director must only ever emit prefab_ids that exist in the asset catalogue
(data/scene_assets/catalog.json); unknown ids are stripped into warnings by
director.sanitise_scene. The C# mirror of these shapes lives in the Unity
project as SceneDocModels.cs — field names are wire-compatible (snake_case
JSON on both sides).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field

Vec3 = Tuple[float, float, float]


def _new_scene_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkyboxSpec(BaseModel):
    """Sky for the scene. v1 = 'preset' only; 'generated' is reserved for the
    future Blockade Labs hook and is coerced to a preset with a warning."""
    kind: Literal["preset", "generated"] = "preset"
    id: str = "day_clear"


class PropSpec(BaseModel):
    """One placed environment piece (street tile, building, vehicle...)."""
    prop_id: str
    prefab_id: str
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation_y: float = 0.0
    scale: float = 1.0


class ActorSpec(BaseModel):
    """One animated character that can walk a waypoint path."""
    actor_id: str
    prefab_id: str
    spawn: Vec3 = (0.0, 0.0, 0.0)
    waypoints: list[Vec3] = Field(default_factory=list)
    animation: Literal["walk", "idle", "run"] = "walk"
    speed_mps: float = 1.2
    loop: bool = False


class TimelineEvent(BaseModel):
    """One scheduled beat. action semantics:
      narrate     → speak `text` via Chatterbox (actor_id ignored)
      actor_start → start `actor_id` moving along its waypoints
      wait        → no-op marker (pacing breadcrumb for readability)"""
    at_seconds: float = Field(ge=0.0)
    action: Literal["narrate", "actor_start", "wait"]
    text: Optional[str] = None
    actor_id: Optional[str] = None


class SceneDoc(BaseModel):
    """The complete renderer-agnostic scene description."""
    scene_id: str = Field(default_factory=_new_scene_id)
    version: int = 1
    title: str
    intent: str
    skybox: SkyboxSpec = Field(default_factory=SkyboxSpec)
    environment: list[PropSpec] = Field(default_factory=list)
    actors: list[ActorSpec] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)


class ComposeRequest(BaseModel):
    intent: str = Field(min_length=1, max_length=2000)


class ComposeResponse(BaseModel):
    scene: SceneDoc
    warnings: list[str] = Field(default_factory=list)

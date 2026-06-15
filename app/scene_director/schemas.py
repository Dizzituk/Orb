# FILE: app/scene_director/schemas.py
# Purpose: The SceneDoc contract — Pydantic models for everything the renderer (Unity)
#          consumes. Renderer-agnostic; the single source of truth. v2 (2026-06-13) adds
#          terrain, tiling, scatter, vehicles, lighting, era/style, provenance, density budget.
# Called-by: app.scene_director.director/critic/router/state/research, Unity SceneDocModels.cs (mirror)
# Depends-on: stdlib/pydantic only
# Last-renovated: 2026-06-13
"""SceneDoc contract (v2 — backward-compatible with v1).

NUMERIC CONVENTIONS (the renderer relies on these — do not change casually):
  * Coordinate system: Unity's — LEFT-handed, Y-up, +Z forward.
  * Units: METRES. The ground plane is y=0.
  * position / spawn / waypoints / start / centre: [x, y, z] in metres.
  * rotation_y / sun_angle: degrees.
  * speed_mps: metres per second (1.2 ≈ casual human walk; ~8–14 ≈ city car).

BACKWARD COMPATIBILITY (sacred): every v2 field is optional or defaulted, so a v1
SceneDoc JSON (no schema_version, no terrain/tiles/scatter/lighting) validates as a v2
doc with empty enrichments and renders exactly as before. `schema_version` absent → v1.
The C# mirror (SceneDocModels.cs) stays wire-compatible (snake_case both sides); new
fields the renderer doesn't yet read are simply ignored.

The director must only ever emit prefab_ids that exist in the asset catalogue
(data/scene_assets/catalog.json); unknown ids are stripped into warnings by
director.sanitise_scene.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field

Vec3 = Tuple[float, float, float]

SCHEMA_VERSION = 2
# Soft cap on total spawned instances (props + tiles + scatter) so the 4080 stays smooth.
DEFAULT_DENSITY_BUDGET = 400


def _new_scene_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── v1 primitives (unchanged shape; ActorSpec gains `type`) ──────────────────

class SkyboxSpec(BaseModel):
    """Sky for the scene. 'preset' only; 'generated' is reserved for the future
    Blockade Labs hook and is coerced to a preset with a warning."""
    kind: Literal["preset", "generated"] = "preset"
    id: str = "day_clear"


class PropSpec(BaseModel):
    """One individually placed environment piece (building, single tree, bench...)."""
    prop_id: str
    prefab_id: str
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation_y: float = 0.0
    scale: float = 1.0


class ActorSpec(BaseModel):
    """One moving actor. type='person' (animated character, v1 behaviour) or
    type='vehicle' (driven along the path; `animation` ignored for vehicles)."""
    actor_id: str
    prefab_id: str
    type: Literal["person", "vehicle"] = "person"
    spawn: Vec3 = (0.0, 0.0, 0.0)
    waypoints: list[Vec3] = Field(default_factory=list)
    animation: Literal["walk", "idle", "run"] = "walk"
    speed_mps: float = 1.2
    loop: bool = False


class TimelineEvent(BaseModel):
    """One scheduled beat:
      narrate     → speak `text` via Chatterbox (actor_id ignored)
      actor_start → start `actor_id` (person OR vehicle) along its waypoints
      wait        → no-op pacing marker"""
    at_seconds: float = Field(ge=0.0)
    action: Literal["narrate", "actor_start", "wait"]
    text: Optional[str] = None
    actor_id: Optional[str] = None


# ── v2 enrichments ───────────────────────────────────────────────────────────

class ElevationFeature(BaseModel):
    """A bump in the ground so backdrops have rolling hills, not a flat plane."""
    shape: Literal["hill", "mound", "slope"] = "hill"
    centre: Vec3 = (0.0, 0.0, 0.0)
    radius: float = 10.0
    height: float = 3.0


class TerrainSpec(BaseModel):
    """Ground + optional elevation. `ground` is a catalogue/preset id
    (e.g. flat, urban, grass); `material` an optional override id."""
    ground: str = "flat"
    material: Optional[str] = None
    elevation: list[ElevationFeature] = Field(default_factory=list)


class TileRun(BaseModel):
    """A run of one prefab repeated end-to-end (road/fence/wall) so the director
    extends a street without enumerating every piece. The director computes
    spacing_m from the catalogue footprint along `direction`."""
    prefab_id: str
    start: Vec3 = (0.0, 0.0, 0.0)
    direction: Vec3 = (1.0, 0.0, 0.0)   # renderer normalises; tiles face along it
    count: int = Field(default=1, ge=1, le=200)
    spacing_m: float = 4.0
    jitter: float = 0.0                  # small random lateral/longitudinal wobble
    rotation_y: float = 0.0              # base yaw applied to every tile


class ScatterRegion(BaseModel):
    """Many instances of vegetation/rocks/props filling a box region cheaply —
    one spec → many instances. density_per_100m2 is the budget knob.
    (Box region for v2; polygon regions are a noted future upgrade.)"""
    prefab_ids: list[str] = Field(default_factory=list)
    centre: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3 = (20.0, 0.0, 20.0)      # x,z extent (y ignored for area)
    density_per_100m2: float = 8.0
    seed: int = 1
    align_to_ground: bool = True
    random_rotation: bool = True


class StructurePrimitive(BaseModel):
    """A raw geometric building block the renderer constructs DIRECTLY (no catalogue asset
    needed) — so the director can BUILD things from scratch when no prefab fits: interior
    walls, floors, ceilings, desks, a whiteboard, simple furniture. This is the 'blank canvas'
    substrate (think LEGO): the director reasons about a room and assembles it from blocks.
    kind: box (walls/desks/furniture), plane or quad (floor/ceiling/board), cylinder (posts/legs).
    size is [x,y,z] dimensions in metres; color is a hex string the renderer parses."""
    prop_id: str
    kind: Literal["box", "plane", "cylinder", "quad"] = "box"
    position: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3 = (1.0, 1.0, 1.0)
    rotation_y: float = 0.0
    color: str = "#bfc4cc"
    role: Optional[str] = None


class LightingSpec(BaseModel):
    """Time-of-day lighting so 'dusk' lights the scene, not just the skybox.
    The renderer prefers explicit sun_angle/sun_intensity/ambient when given,
    else derives sensible values from the `time_of_day` label."""
    time_of_day: str = "day"            # named (day/dusk/night/dawn) or a numeric string
    sun_angle: Optional[float] = None   # degrees of elevation above horizon
    sun_intensity: Optional[float] = None
    ambient: Optional[float] = None
    sky_tint: Optional[str] = None


class SourceRef(BaseModel):
    title: str = ""
    url: str = ""


class ProvenanceSpec(BaseModel):
    """What a researched scene was built from. Empty for imaginative scenes."""
    facts: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    notes: str = ""


class SceneDoc(BaseModel):
    """The complete renderer-agnostic scene description (v2)."""
    scene_id: str = Field(default_factory=_new_scene_id)
    version: int = 1                    # monotonic, stamped by scene_state (NOT the schema rev)
    schema_version: int = SCHEMA_VERSION
    title: str
    intent: str
    skybox: SkyboxSpec = Field(default_factory=SkyboxSpec)
    # v2: ground/terrain, single props, tiled runs, scattered fills
    terrain: Optional[TerrainSpec] = None
    environment: list[PropSpec] = Field(default_factory=list)
    tiles: list[TileRun] = Field(default_factory=list)
    scatter: list[ScatterRegion] = Field(default_factory=list)
    structures: list[StructurePrimitive] = Field(default_factory=list)
    actors: list[ActorSpec] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    lighting: Optional[LightingSpec] = None
    # v2 forward-compat: era/style filtering + research provenance + density budget
    era: str = "modern"
    style: str = ""
    provenance: Optional[ProvenanceSpec] = None
    density_budget: int = DEFAULT_DENSITY_BUDGET
    created_at: str = Field(default_factory=_now_iso)


class ComposeRequest(BaseModel):
    intent: str = Field(min_length=1, max_length=2000)
    era: Optional[str] = None           # optional explicit era override (forward-compat)


class ComposeResponse(BaseModel):
    scene: SceneDoc
    warnings: list[str] = Field(default_factory=list)


# ── backward-compat helpers ──────────────────────────────────────────────────

def detect_schema_version(data: dict) -> int:
    """A SceneDoc dict without `schema_version` is a v1 doc."""
    try:
        return int(data.get("schema_version", 1))
    except (TypeError, ValueError):
        return 1


def upconvert_to_v2(data: dict) -> dict:
    """Idempotently bring a v1 SceneDoc dict up to v2. Since every v2 field is
    optional/defaulted, this only stamps schema_version (and leaves the new
    structures empty) — Pydantic fills the rest on validation. Returns a new dict."""
    out = dict(data)
    out["schema_version"] = SCHEMA_VERSION
    return out

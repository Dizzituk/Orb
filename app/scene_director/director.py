# FILE: app/scene_director/director.py
# Purpose: The scene director brain (v3) — turns an intent (+ optional era/research facts)
#          into a RICH, validated SceneDoc via an LLM, with v2-aware sanitising + a richer
#          deterministic fallback. Composes the first draft; the critic (critic.py) revises it.
# Called-by: app.scene_director.router (compose), app.scene_director.critic (re-sanitise)
# Depends-on: app.scene_director.schemas/asset_catalog, app.llm.clients, app.llm.frontier_models
# Last-renovated: 2026-06-13
"""compose_scene(intent, era, fact_pack) → ComposeResponse (the draft).

v3: the v1 minimalism cap is gone. The director now composes fully dressed scenes —
terrain + backdrop, vegetation via scatter, street furniture, roads extended with
tile-runs, moving vehicle-actors, and time-of-day lighting — reasoning with catalogue
footprints so pieces tile end-to-end and don't overlap/float, within a density budget.

Pipeline unchanged in spirit: build prompt → LLM → strict parse → retry once with errors
→ deterministic fallback. Every prefab_id is validated against the catalogue; unknown ids
(and over-budget density) are stripped/trimmed into warnings. The pipeline NEVER returns
nothing. The scene critic (critic.py) runs after this draft to interrogate + revise it.

Model is env-switchable: ASTRA_SCENE_DIRECTOR_PROVIDER / ASTRA_SCENE_DIRECTOR_MODEL
(default anthropic / anthropic:frontier-sonnet → claude-sonnet-4-6).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.scene_director import asset_catalog
from app.scene_director.schemas import (
    DEFAULT_DENSITY_BUDGET,
    ActorSpec,
    ComposeResponse,
    LightingSpec,
    PropSpec,
    ProvenanceSpec,
    ScatterRegion,
    SceneDoc,
    SkyboxSpec,
    SourceRef,
    TerrainSpec,
    TileRun,
    TimelineEvent,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are ASTRA's scene director for "ASTRA Room", a narrated 3D learning space rendered in Unity. You turn the user's intent into ONE SceneDoc JSON object describing a FULLY DRESSED, believable scene. Scenes are LEARNING spaces with narration written for SPOKEN delivery — short sentences, UK English, no markdown, teaching tone. Timelines run 30-90 seconds with a clear focal action.

NUMERIC CONVENTIONS: Unity coordinates — left-handed, Y-up, +Z forward, METRES, ground plane y=0. rotation_y is degrees (yaw). speed_mps: ~1.2 = casual walk, ~8-14 = a city car. positions are [x,y,z].

COMPOSE RICHLY — where the brief allows, do ALL of these, not a bare minimum:
- TERRAIN & BACKDROP: set `terrain` (ground preset flat/urban/grass + optional elevation hills/mounds) so the background isn't an empty plane. Fill the distance with hills or a treeline.
- EXTEND ROADS/PATHS with `tiles` (TileRun) instead of enumerating each piece: one run = a prefab repeated `count` times from `start` along `direction` at `spacing_m`. Compute spacing_m from the tile footprint (use ~the footprint length along the run so tiles meet end-to-end, no gaps/overlaps).
- VEGETATION via `scatter` (ScatterRegion): one spec fills a box region with many trees/plants/rocks at `density_per_100m2`. This is the cheap way to dress ground — prefer it over hundreds of single props.
- STREET FURNITURE as single `environment` props: streetlights, benches, signage, fences, bins — place them sensibly along pavements.
- TRAFFIC: add moving vehicles as actors with "type":"vehicle" (a drivable catalogue id), a waypoint path along the road, and speed_mps ~8-14. People are "type":"person".
- LIGHTING: set `lighting.time_of_day` (day/dusk/dawn/night) so the scene is lit to match the mood, not just the sky.
- BUILD FROM PRIMITIVES when the catalogue has no asset for something — INTERIORS ESPECIALLY: the catalogue is an OUTDOOR city pack with NO indoor furniture (no desks, chairs, whiteboards, or room walls). Use `structures`: raw blocks the renderer builds directly from a size + colour, no asset needed. box = walls / desks / chair seats / a teacher's desk; plane = floor / ceiling; quad = a whiteboard or blackboard on a wall; cylinder = posts / legs. To build a ROOM: make four wall boxes (~0.2 m thick, ~3 m tall) enclosing a floor plane, add a ceiling plane, then place desks and seats as boxes inside and a board quad flat against the far wall. Reason about coordinates so walls meet at the corners and every object sits ON the floor (its y = half its height). Colour things sensibly (walls off-white, board dark green/black, desks wood-brown).

OUTPUT CONTRACT — reply with STRICT JSON only (no fences, no commentary), this shape (omit sections that don't apply):
{
  "title": "<short title>",
  "skybox": {"kind":"preset","id":"<skybox id>"},
  "terrain": {"ground":"flat|urban|grass","elevation":[{"shape":"hill","centre":[x,y,z],"radius":r,"height":h}]},
  "environment": [{"prop_id":"<unique>","prefab_id":"<catalogue id>","position":[x,y,z],"rotation_y":0.0,"scale":1.0}],
  "tiles": [{"prefab_id":"<road/fence id>","start":[x,y,z],"direction":[1,0,0],"count":12,"spacing_m":20.0,"rotation_y":0.0}],
  "scatter": [{"prefab_ids":["<veg id>","<veg id>"],"centre":[x,y,z],"size":[40,0,40],"density_per_100m2":8.0,"seed":1}],
  "structures": [{"prop_id":"<unique>","kind":"box|plane|quad|cylinder","position":[x,y,z],"size":[x,y,z],"rotation_y":0.0,"color":"#rrggbb"}],
  "actors": [{"actor_id":"<unique>","prefab_id":"<catalogue id>","type":"person|vehicle","spawn":[x,y,z],"waypoints":[[x,y,z]],"animation":"walk|idle|run","speed_mps":1.2,"loop":false}],
  "timeline": [{"at_seconds":0.0,"action":"narrate|actor_start|wait","text":"<spoken, narrate only>","actor_id":"<actor_start only>"}],
  "lighting": {"time_of_day":"dusk"}
}

HARD RULES:
- prefab_id values MUST come from the asset catalogue provided. Never invent ids. Use the categories shown (BUILDINGS, VEGETATION, VEHICLES, STREET FURNITURE...).
- Reason with footprints so tiles connect and props sit on the ground (y=0), not floating or overlapping.
- Every actor needs ≥1 waypoint and a matching actor_start timeline event. Vehicles need a road-aligned path.
- Respect DENSITY BUDGET (given below): single props + total tile counts + estimated scatter instances should stay under it. Prefer tiling + scatter over hundreds of single props.
- narrate events carry text; actor_start carry actor_id; wait carry neither.
- The ASTRA ORB is ALWAYS already present as a permanent floating object in the room — do NOT place an orb, sphere, or particle effect for it yourself.
- NEVER use particle/effect prefab_ids (e.g. particles_fire_01, particles_dust_01) as furniture, lights, the board, or the orb. Lighting comes from `lighting`; build fixtures from `structures` if you want visible lamps.
- `structures` need NO catalogue id (they are raw shapes). prefab_ids in environment/tiles/scatter/actors still MUST be real catalogue ids.

%RICH WORKED EXAMPLES%"""

_RICH_EXAMPLES = """RICH WORKED EXAMPLE 1 — intent "A busy city street corner at dusk, people walking, a car waiting":
{"title":"A busy corner at dusk","skybox":{"kind":"preset","id":"dusk"},"terrain":{"ground":"urban","elevation":[{"shape":"hill","centre":[0,0,70],"radius":40,"height":10}]},"environment":[{"prop_id":"shop_a","prefab_id":"cafe_01","position":[12,0,10],"rotation_y":180,"scale":1},{"prop_id":"shop_b","prefab_id":"house_03","position":[-14,0,10],"rotation_y":180,"scale":1},{"prop_id":"light_a","prefab_id":"light_01","position":[6,0,4],"rotation_y":0,"scale":1},{"prop_id":"light_b","prefab_id":"light_01","position":[26,0,4],"rotation_y":0,"scale":1},{"prop_id":"bench_a","prefab_id":"bench_01","position":[16,0,4.5],"rotation_y":0,"scale":1}],"tiles":[{"prefab_id":"road_01","start":[-20,0,0],"direction":[1,0,0],"count":6,"spacing_m":20,"rotation_y":0}],"scatter":[{"prefab_ids":["tree_01","tree_03"],"centre":[0,0,40],"size":[80,0,30],"density_per_100m2":6,"seed":7}],"actors":[{"actor_id":"walker1","prefab_id":"person_male_casual_01","type":"person","spawn":[-10,0,2],"waypoints":[[30,0,2]],"animation":"walk","speed_mps":1.2,"loop":false},{"actor_id":"walker2","prefab_id":"person_female_casual_01","type":"person","spawn":[30,0,3],"waypoints":[[-8,0,3]],"animation":"walk","speed_mps":1.1,"loop":false},{"actor_id":"car1","prefab_id":"car_02","type":"vehicle","spawn":[40,0,-1.5],"waypoints":[[8,0,-1.5]],"speed_mps":6,"loop":false}],"timeline":[{"at_seconds":0,"action":"narrate","text":"Dusk settles over a busy corner. Watch how the street comes to life."},{"at_seconds":3,"action":"actor_start","actor_id":"walker1"},{"at_seconds":4,"action":"actor_start","actor_id":"walker2"},{"at_seconds":6,"action":"actor_start","actor_id":"car1"},{"at_seconds":18,"action":"narrate","text":"People cross paths while a car slows for the corner. Everyday motion, all at once."}],"lighting":{"time_of_day":"dusk"}}

RICH WORKED EXAMPLE 2 — intent "A walk through rolling green hills":
{"title":"Across the green hills","skybox":{"kind":"preset","id":"day_clear"},"terrain":{"ground":"grass","elevation":[{"shape":"hill","centre":[20,0,30],"radius":25,"height":8},{"shape":"hill","centre":[-30,0,50],"radius":30,"height":12}]},"environment":[],"tiles":[],"scatter":[{"prefab_ids":["tree_01","tree_02","tree_05"],"centre":[0,0,40],"size":[120,0,120],"density_per_100m2":4,"seed":3},{"prefab_ids":["grass_01","flower_01"],"centre":[0,0,20],"size":[60,0,60],"density_per_100m2":20,"seed":9}],"actors":[{"actor_id":"rambler","prefab_id":"person_female_casual_01","type":"person","spawn":[-20,0,0],"waypoints":[[40,0,15]],"animation":"walk","speed_mps":1.2,"loop":false}],"timeline":[{"at_seconds":0,"action":"narrate","text":"Open hills roll into the distance, dotted with trees."},{"at_seconds":3,"action":"actor_start","actor_id":"rambler"},{"at_seconds":14,"action":"narrate","text":"A single walker crosses the slope, giving the landscape its sense of scale."}],"lighting":{"time_of_day":"day"}}"""

_FALLBACK_NARRATION_1 = (
    "Here is a quiet city street. One person is about to walk it from end to end."
)
_FALLBACK_NARRATION_2 = (
    "A steady walking pace is about one point two metres per second. "
    "Notice how far that carries them in just a few seconds."
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _provider_and_model() -> Tuple[str, str]:
    """Provider + concrete model for the director, env-overridable."""
    from app.llm.frontier_models import resolve_model_alias

    provider = os.getenv("ASTRA_SCENE_DIRECTOR_PROVIDER", "anthropic").strip().lower()
    model = resolve_model_alias(
        os.getenv("ASTRA_SCENE_DIRECTOR_MODEL", "anthropic:frontier-sonnet")
    )
    return provider, model


def _system_prompt() -> str:
    return _SYSTEM_PROMPT.replace("%RICH WORKED EXAMPLES%", _RICH_EXAMPLES)


async def _call_director_llm(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    """One call through the house plumbing. Returns (content, usage)."""
    from app.llm import clients

    provider, model = _provider_and_model()
    caller = {
        "anthropic": clients.async_call_anthropic,
        "google": clients.async_call_google,
        "openai": clients.async_call_openai,
    }.get(provider, clients.async_call_anthropic)
    return await caller(_system_prompt(), messages, model=model)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing + sanitising
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_scene(content: str, intent: str) -> Tuple[Optional[SceneDoc], str]:
    """Strict parse into SceneDoc. Server-owned fields stripped before validation."""
    data = _extract_json(content)
    if data is None:
        return None, "reply contained no parseable JSON object"
    data["intent"] = intent
    for owned in ("scene_id", "version", "created_at", "provenance"):
        data.pop(owned, None)
    try:
        return SceneDoc.model_validate(data), ""
    except ValidationError as exc:
        return None, str(exc)


def _estimate_instances(doc: SceneDoc) -> int:
    """Rough total spawn count for the density budget: props + tiles + scatter."""
    total = len(doc.environment) + len(doc.structures)
    total += sum(t.count for t in doc.tiles)
    for s in doc.scatter:
        area = max(1.0, abs(s.size[0]) * abs(s.size[2]))
        total += int(round(s.density_per_100m2 * area / 100.0))
    return total


def sanitise_scene(doc: SceneDoc, warnings: List[str]) -> SceneDoc:
    """Enforce the catalogue boundary + internal consistency + density budget.
    Strips/trims, never raises — the renderer must always get resolvable vocabulary."""
    env_ids = asset_catalog.known_ids("environment")
    actor_ids = asset_catalog.known_ids("actor")
    skybox_ids = asset_catalog.known_ids("skybox")

    # Skybox: presets only.
    if doc.skybox.kind == "generated":
        warnings.append(
            f"skybox 'generated' reserved for a future hook — using a preset instead of '{doc.skybox.id}'")
        doc.skybox = SkyboxSpec(kind="preset", id="day_clear")
    if skybox_ids and doc.skybox.id not in skybox_ids:
        warnings.append(f"unknown skybox preset '{doc.skybox.id}' — using 'day_clear'")
        doc.skybox = SkyboxSpec(kind="preset", id="day_clear")

    # Environment props.
    kept_props: List[PropSpec] = []
    seen_prop_ids: set[str] = set()
    for prop in doc.environment:
        if env_ids and prop.prefab_id not in env_ids:
            warnings.append(f"stripped prop '{prop.prop_id}': unknown prefab_id '{prop.prefab_id}'")
            continue
        if prop.prop_id in seen_prop_ids:
            warnings.append(f"stripped duplicate prop_id '{prop.prop_id}'")
            continue
        seen_prop_ids.add(prop.prop_id)
        kept_props.append(prop)
    doc.environment = kept_props

    # Tiles (runs): valid ids only.
    kept_tiles: List[TileRun] = []
    for t in doc.tiles:
        if env_ids and t.prefab_id not in env_ids:
            warnings.append(f"stripped tile run: unknown prefab_id '{t.prefab_id}'")
            continue
        kept_tiles.append(t)
    doc.tiles = kept_tiles

    # Scatter: keep only known veg/prop ids; drop empty regions.
    kept_scatter: List[ScatterRegion] = []
    for s in doc.scatter:
        valid = [pid for pid in s.prefab_ids if (not env_ids) or pid in env_ids]
        if not valid:
            warnings.append("stripped scatter region: no valid prefab_ids")
            continue
        if len(valid) != len(s.prefab_ids):
            warnings.append("scatter region: dropped some unknown prefab_ids")
        s.prefab_ids = valid
        kept_scatter.append(s)
    doc.scatter = kept_scatter

    # Actors: people validate against character ids; vehicles against env (vehicle) ids.
    kept_actors: List[ActorSpec] = []
    seen_actor_ids: set[str] = set()
    for actor in doc.actors:
        pool = env_ids if actor.type == "vehicle" else actor_ids
        if pool and actor.prefab_id not in pool:
            warnings.append(f"stripped {actor.type} '{actor.actor_id}': unknown prefab_id '{actor.prefab_id}'")
            continue
        if actor.actor_id in seen_actor_ids:
            warnings.append(f"stripped duplicate actor_id '{actor.actor_id}'")
            continue
        if not actor.waypoints:
            actor.waypoints = [actor.spawn]
            warnings.append(f"actor '{actor.actor_id}' had no waypoints — pinned to spawn")
        seen_actor_ids.add(actor.actor_id)
        kept_actors.append(actor)
    doc.actors = kept_actors

    # Timeline: events must reference survivors.
    kept_events: List[TimelineEvent] = []
    for event in doc.timeline:
        if event.action == "narrate" and not (event.text or "").strip():
            warnings.append("stripped narrate event with no text")
            continue
        if event.action == "actor_start" and event.actor_id not in seen_actor_ids:
            warnings.append(f"stripped actor_start for unknown actor '{event.actor_id}'")
            continue
        kept_events.append(event)
    doc.timeline = sorted(kept_events, key=lambda e: e.at_seconds)

    # Density budget: if over, thin scatter densities first, then cap tile counts.
    budget = doc.density_budget or DEFAULT_DENSITY_BUDGET
    est = _estimate_instances(doc)
    if est > budget and doc.scatter:
        factor = max(0.2, budget / float(est))
        for s in doc.scatter:
            s.density_per_100m2 = round(s.density_per_100m2 * factor, 2)
        warnings.append(f"density {est} over budget {budget} — scatter thinned by x{round(factor,2)}")
        est = _estimate_instances(doc)
    if est > budget and doc.tiles:
        for t in doc.tiles:
            t.count = max(1, int(t.count * budget / float(est)))
        warnings.append(f"density still over budget — tile counts capped")
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback (richer in v3)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_by_tag(kind: str, tag: str, fallback_first: bool = True) -> Optional[str]:
    """First catalogue id of `kind` carrying `tag`, else first of kind."""
    for e in asset_catalog.list_by_kind(kind):
        if tag.lower() in [str(t).lower() for t in (e.get("tags") or [])]:
            return e["prefab_id"]
    if fallback_first:
        entries = asset_catalog.list_by_kind(kind)
        return entries[0]["prefab_id"] if entries else None
    return None


def deterministic_fallback_scene(intent: str) -> SceneDoc:
    """A modest but DRESSED street scene so the pipeline never returns nothing —
    a tiled road, a person, dusk-aware lighting. Uses real catalogue ids."""
    road = next((e["prefab_id"] for e in asset_catalog.list_by_kind("environment")
                 if str(e["prefab_id"]).startswith("road_")), None) \
        or _pick_by_tag("environment", "road") or "road_01"
    person = (("person_male_casual_01" if "person_male_casual_01" in asset_catalog.known_ids("actor")
               else (next(iter(asset_catalog.known_ids("actor")), "person_male_casual_01"))))
    tree = _pick_by_tag("environment", "tree", fallback_first=False)
    lowered = intent.lower()
    is_dusk = any(w in lowered for w in ("dusk", "evening", "sunset", "night"))
    sky = "dusk" if is_dusk and "dusk" in asset_catalog.known_ids("skybox") else (
        "day_clear" if "day_clear" in asset_catalog.known_ids("skybox") else
        next(iter(asset_catalog.known_ids("skybox")), "day_clear"))

    scatter = []
    if tree:
        scatter = [ScatterRegion(prefab_ids=[tree], centre=(0.0, 0.0, 25.0),
                                 size=(60.0, 0.0, 20.0), density_per_100m2=5.0, seed=1)]
    return SceneDoc(
        title="A walk down a city street",
        intent=intent,
        skybox=SkyboxSpec(kind="preset", id=sky),
        terrain=TerrainSpec(ground="urban"),
        tiles=[TileRun(prefab_id=road, start=(-20.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                       count=4, spacing_m=20.0)],
        scatter=scatter,
        actors=[ActorSpec(actor_id="walker", prefab_id=person, type="person",
                          spawn=(-8.0, 0.0, 2.0), waypoints=[(28.0, 0.0, 2.0)],
                          animation="walk", speed_mps=1.2, loop=False)],
        timeline=[
            TimelineEvent(at_seconds=0.0, action="narrate", text=_FALLBACK_NARRATION_1),
            TimelineEvent(at_seconds=4.0, action="actor_start", actor_id="walker"),
            TimelineEvent(at_seconds=12.0, action="narrate", text=_FALLBACK_NARRATION_2),
        ],
        lighting=LightingSpec(time_of_day="dusk" if is_dusk else "day"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_msg(intent: str, era: Optional[str], fact_pack: Optional[dict]) -> str:
    catalog_block = asset_catalog.as_prompt_block(era=era)
    parts = [
        f"ASSET CATALOGUE (the only prefab_ids you may use, grouped by category):\n{catalog_block}",
        f"DENSITY BUDGET: {DEFAULT_DENSITY_BUDGET} total instances (props + tile counts + scatter).",
    ]
    if era and era.lower() != "modern":
        parts.append(f"ERA: {era}. Use only era-appropriate assets from the catalogue above.")
    if fact_pack:
        facts = "\n".join(f"- {f}" for f in (fact_pack.get("facts") or [])[:12])
        summary = (fact_pack.get("summary") or "").strip()
        parts.append(
            "RESEARCHED FACTS (compose the scene FROM these — do not invent specifics):\n"
            + (summary + "\n" if summary else "") + facts)
    parts.append(f"USER INTENT: {intent}\n\nReply with ONE rich SceneDoc JSON object only.")
    return "\n\n".join(parts)


def _apply_provenance(doc: SceneDoc, era: Optional[str], fact_pack: Optional[dict]) -> None:
    if era:
        doc.era = era
    if fact_pack:
        sources = [SourceRef(title=s.get("title", ""), url=s.get("url", ""))
                   for s in (fact_pack.get("sources") or [])]
        doc.provenance = ProvenanceSpec(
            facts=list(fact_pack.get("facts") or [])[:20],
            sources=sources,
            notes=(fact_pack.get("summary") or "")[:2000],
        )


async def compose_scene(intent: str, era: Optional[str] = None,
                        fact_pack: Optional[dict] = None) -> ComposeResponse:
    """Compose the DRAFT SceneDoc. LLM → strict parse → retry once → fallback.
    Always returns a scene. era filters the catalogue; fact_pack (research) is
    injected into the prompt and recorded as provenance. The critic revises after."""
    warnings: List[str] = []

    # Era scaffolding (v3 Layer 5): if a non-modern era is requested but NO era-matching
    # assets are catalogued, warn loudly + fall back to the modern catalogue as explicit
    # placeholders (rather than silently dressing the past in modern blocks). Historical
    # fidelity is asset-bound — see docs/astra_room/HISTORICAL_SCENES.md.
    effective_era = era
    if era and era.strip().lower() not in ("", "modern", "any", "none"):
        if not asset_catalog.list_by_era(era):
            warnings.append(
                f"no '{era}'-era assets catalogued — composing with MODERN assets as "
                f"placeholders; historical fidelity is asset-bound "
                f"(see docs/astra_room/HISTORICAL_SCENES.md)")
            effective_era = None  # use the full catalogue for the prompt

    messages: List[Dict[str, str]] = [
        {"role": "user", "content": _build_user_msg(intent, effective_era, fact_pack)}]

    for attempt in (1, 2):
        try:
            content, usage = await _call_director_llm(messages)
        except Exception as exc:
            logger.warning("[scene] director LLM call failed (%s) — fallback", exc)
            warnings.append(f"director model unavailable ({exc}) — deterministic fallback scene used")
            scene = deterministic_fallback_scene(intent)
            _apply_provenance(scene, era, fact_pack)
            return ComposeResponse(scene=scene, warnings=warnings)
        if usage.get("error"):
            logger.warning("[scene] director LLM error: %s — fallback", usage.get("error"))
            warnings.append("director model returned an error — deterministic fallback scene used")
            scene = deterministic_fallback_scene(intent)
            _apply_provenance(scene, era, fact_pack)
            return ComposeResponse(scene=scene, warnings=warnings)

        doc, error_text = _parse_scene(content, intent)
        if doc is not None:
            doc = sanitise_scene(doc, warnings)
            _apply_provenance(doc, era, fact_pack)
            logger.info("[scene] drafted '%s' on attempt %d (%d props, %d tiles, %d scatter, %d actors, %d warning(s))",
                        doc.title, attempt, len(doc.environment), len(doc.tiles),
                        len(doc.scatter), len(doc.actors), len(warnings))
            return ComposeResponse(scene=doc, warnings=warnings)

        logger.warning("[scene] attempt %d failed validation: %s", attempt, error_text[:300])
        if attempt == 1:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": ("That JSON failed validation with the errors below. Reply again with "
                            f"ONE corrected SceneDoc JSON object only.\n\nERRORS:\n{error_text[:2000]}"),
            })

    warnings.append("director output failed validation twice — deterministic fallback scene used")
    scene = deterministic_fallback_scene(intent)
    _apply_provenance(scene, era, fact_pack)
    return ComposeResponse(scene=scene, warnings=warnings)

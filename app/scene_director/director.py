# FILE: app/scene_director/director.py
# Purpose: The scene director brain — turns a user intent into a validated SceneDoc
#          via an LLM (strict JSON, one retry), with a deterministic fallback scene.
# Called-by: app.scene_director.router
# Depends-on: app.scene_director.schemas/asset_catalog, app.llm.clients, app.llm.frontier_models
# Last-renovated: 2026-06-12
"""compose_scene(intent) → ComposeResponse.

Pipeline: build prompt (contract + catalogue + conventions + worked example)
→ call a strong model through the house plumbing (Claude-family preferred for
structured JSON) → parse strictly into SceneDoc → on validation failure retry
ONCE with the errors appended → on second failure (or LLM transport error)
fall back to deterministic_fallback_scene(intent). The pipeline NEVER returns
nothing. Every prefab_id is validated against the asset catalogue; unknown
ids are stripped into warnings (the renderer must never receive vocabulary
it cannot resolve).
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
    ActorSpec,
    ComposeResponse,
    PropSpec,
    SceneDoc,
    SkyboxSpec,
    TimelineEvent,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are ASTRA's scene director for "ASTRA Room", a narrated 3D learning space rendered in Unity. You turn the user's intent into ONE SceneDoc JSON object. Scenes are LEARNING spaces: bias towards a clear focal action, one or two actors maximum, and narration written for SPOKEN delivery — short sentences, UK English, no markdown, narration that teaches rather than captions. Timelines run 30-90 seconds.

NUMERIC CONVENTIONS: Unity coordinates — left-handed, Y-up, +Z forward, units in METRES, ground plane y=0. rotation_y is degrees (yaw, clockwise from above). speed_mps: 1.2 is a casual walk. positions are [x, y, z].

OUTPUT CONTRACT — reply with STRICT JSON only (no markdown fences, no commentary), exactly this shape:
{
  "title": "<short scene title>",
  "skybox": {"kind": "preset", "id": "<one of the skybox preset ids>"},
  "environment": [{"prop_id": "<unique id>", "prefab_id": "<catalogue id>", "position": [x,y,z], "rotation_y": 0.0, "scale": 1.0}],
  "actors": [{"actor_id": "<unique id>", "prefab_id": "<catalogue id>", "spawn": [x,y,z], "waypoints": [[x,y,z], ...], "animation": "walk|idle|run", "speed_mps": 1.2, "loop": false}],
  "timeline": [{"at_seconds": 0.0, "action": "narrate|actor_start|wait", "text": "<spoken line, narrate only>", "actor_id": "<actor_start only>"}]
}

HARD RULES:
- prefab_id values MUST come from the asset catalogue you are given. Never invent ids.
- Lay environment props so actor waypoints travel across them (streets run along their long axis).
- Every actor needs at least one waypoint and a matching actor_start timeline event.
- narrate events carry text; actor_start events carry actor_id; wait events carry neither.
- Keep it simple: a handful of props, 1-2 actors, 3-8 timeline events.

WORKED EXAMPLE (intent: "show me a man walking down a city street"):
{"title":"A walk down a city street","skybox":{"kind":"preset","id":"day_clear"},"environment":[{"prop_id":"road_a","prefab_id":"street_straight_01","position":[0,0,0],"rotation_y":0,"scale":1.0},{"prop_id":"road_b","prefab_id":"street_straight_01","position":[20,0,0],"rotation_y":0,"scale":1.0},{"prop_id":"shop_a","prefab_id":"building_shop_01","position":[10,0,8],"rotation_y":180,"scale":1.0}],"actors":[{"actor_id":"walker","prefab_id":"person_male_casual_01","spawn":[-8,0,2],"waypoints":[[28,0,2]],"animation":"walk","speed_mps":1.2,"loop":false}],"timeline":[{"at_seconds":0,"action":"narrate","text":"Here is an ordinary city street. Watch how one person moves through it."},{"at_seconds":4,"action":"actor_start","actor_id":"walker"},{"at_seconds":12,"action":"narrate","text":"He keeps a steady pace of about one point two metres per second. That is the rhythm most of us walk at without thinking."}]}"""

_FALLBACK_NARRATION_1 = (
    "Here is a quiet city street. One person is about to walk it from end to end."
)
_FALLBACK_NARRATION_2 = (
    "A steady walking pace is about one point two metres per second. "
    "Notice how far that carries him in just a few seconds."
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _provider_and_model() -> Tuple[str, str]:
    """Provider + concrete model for the director, env-overridable.
    Claude-family default (best structured-JSON behaviour in this codebase)."""
    from app.llm.frontier_models import resolve_model_alias

    provider = os.getenv("ASTRA_SCENE_DIRECTOR_PROVIDER", "anthropic").strip().lower()
    model = resolve_model_alias(
        os.getenv("ASTRA_SCENE_DIRECTOR_MODEL", "anthropic:frontier-sonnet")
    )
    return provider, model


async def _call_director_llm(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    """One call through the house plumbing. Returns (content, usage);
    usage['error'] set on failure (same contract as app.llm.clients)."""
    from app.llm import clients

    provider, model = _provider_and_model()
    caller = {
        "anthropic": clients.async_call_anthropic,
        "google": clients.async_call_google,
        "openai": clients.async_call_openai,
    }.get(provider, clients.async_call_anthropic)
    return await caller(_SYSTEM_PROMPT, messages, model=model)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing + sanitising
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of an LLM reply (tolerates fences)."""
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
    """Strict parse into SceneDoc. Returns (doc, "") or (None, error_text)."""
    data = _extract_json(content)
    if data is None:
        return None, "reply contained no parseable JSON object"
    # Server-owned fields — the director model never controls these.
    data["intent"] = intent
    data.pop("scene_id", None)
    data.pop("version", None)
    data.pop("created_at", None)
    try:
        return SceneDoc.model_validate(data), ""
    except ValidationError as exc:
        return None, str(exc)


def sanitise_scene(doc: SceneDoc, warnings: List[str]) -> SceneDoc:
    """Enforce the catalogue boundary + internal consistency, appending to
    warnings. Strips, never raises — the renderer must always get something
    it can resolve."""
    env_ids = asset_catalog.known_ids("environment")
    actor_ids = asset_catalog.known_ids("actor")
    skybox_ids = asset_catalog.known_ids("skybox")

    # Skybox: v1 renders presets only.
    if doc.skybox.kind == "generated":
        warnings.append(
            f"skybox kind 'generated' is reserved for v2 — using preset 'day_clear' "
            f"instead of '{doc.skybox.id}'"
        )
        doc.skybox = SkyboxSpec(kind="preset", id="day_clear")
    if skybox_ids and doc.skybox.id not in skybox_ids:
        warnings.append(f"unknown skybox preset '{doc.skybox.id}' — using 'day_clear'")
        doc.skybox = SkyboxSpec(kind="preset", id="day_clear")

    # Environment props: catalogue ids only, unique prop_ids.
    kept_props: List[PropSpec] = []
    seen_prop_ids: set[str] = set()
    for prop in doc.environment:
        if env_ids and prop.prefab_id not in env_ids:
            warnings.append(f"stripped environment prop '{prop.prop_id}': unknown prefab_id '{prop.prefab_id}'")
            continue
        if prop.prop_id in seen_prop_ids:
            warnings.append(f"stripped duplicate prop_id '{prop.prop_id}'")
            continue
        seen_prop_ids.add(prop.prop_id)
        kept_props.append(prop)
    doc.environment = kept_props

    # Actors: catalogue ids only, unique actor_ids, at least one waypoint.
    kept_actors: List[ActorSpec] = []
    seen_actor_ids: set[str] = set()
    for actor in doc.actors:
        if actor_ids and actor.prefab_id not in actor_ids:
            warnings.append(f"stripped actor '{actor.actor_id}': unknown prefab_id '{actor.prefab_id}'")
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

    # Timeline: events must reference things that survived.
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
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _pick_prefab(kind: str, preferred: str) -> str:
    """preferred id if catalogued, else the first catalogue entry of that
    kind, else the preferred literal (renderer logs unknown ids gracefully)."""
    ids = asset_catalog.known_ids(kind)
    if preferred in ids:
        return preferred
    entries = asset_catalog.list_by_kind(kind)
    return entries[0]["prefab_id"] if entries else preferred


def deterministic_fallback_scene(intent: str) -> SceneDoc:
    """Hardcoded minimal street + walker scene so the pipeline never returns
    nothing. Uses catalogue entries where available."""
    street = _pick_prefab("environment", "street_straight_01")
    person = _pick_prefab("actor", "person_male_casual_01")
    lowered = intent.lower()
    sky = "dusk" if any(w in lowered for w in ("dusk", "evening", "sunset", "night")) else "day_clear"
    sky = sky if sky in asset_catalog.known_ids("skybox") else "day_clear"
    return SceneDoc(
        title="A walk down a city street",
        intent=intent,
        skybox=SkyboxSpec(kind="preset", id=sky),
        environment=[
            PropSpec(prop_id="road_a", prefab_id=street, position=(0.0, 0.0, 0.0)),
            PropSpec(prop_id="road_b", prefab_id=street, position=(20.0, 0.0, 0.0)),
        ],
        actors=[
            ActorSpec(
                actor_id="walker",
                prefab_id=person,
                spawn=(-8.0, 0.0, 2.0),
                waypoints=[(28.0, 0.0, 2.0)],
                animation="walk",
                speed_mps=1.2,
                loop=False,
            )
        ],
        timeline=[
            TimelineEvent(at_seconds=0.0, action="narrate", text=_FALLBACK_NARRATION_1),
            TimelineEvent(at_seconds=4.0, action="actor_start", actor_id="walker"),
            TimelineEvent(at_seconds=12.0, action="narrate", text=_FALLBACK_NARRATION_2),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def compose_scene(intent: str) -> ComposeResponse:
    """Compose a SceneDoc for the intent. LLM → strict parse → retry once with
    validation errors → deterministic fallback. Always returns a scene."""
    warnings: List[str] = []
    catalog_block = asset_catalog.as_prompt_block()
    user_msg = (
        f"ASSET CATALOGUE (the only prefab_ids you may use):\n{catalog_block}\n\n"
        f"USER INTENT: {intent}\n\nReply with the SceneDoc JSON only."
    )
    messages: List[Dict[str, str]] = [{"role": "user", "content": user_msg}]

    for attempt in (1, 2):
        try:
            content, usage = await _call_director_llm(messages)
        except Exception as exc:  # transport/SDK failure — no point retrying
            logger.warning("[scene] director LLM call failed (%s) — fallback", exc)
            warnings.append(f"director model unavailable ({exc}) — deterministic fallback scene used")
            return ComposeResponse(scene=deterministic_fallback_scene(intent), warnings=warnings)
        if usage.get("error"):
            logger.warning("[scene] director LLM error: %s — fallback", usage.get("error"))
            warnings.append("director model returned an error — deterministic fallback scene used")
            return ComposeResponse(scene=deterministic_fallback_scene(intent), warnings=warnings)

        doc, error_text = _parse_scene(content, intent)
        if doc is not None:
            doc = sanitise_scene(doc, warnings)
            logger.info("[scene] composed '%s' on attempt %d (%d warning(s))",
                        doc.title, attempt, len(warnings))
            return ComposeResponse(scene=doc, warnings=warnings)

        logger.warning("[scene] attempt %d failed validation: %s", attempt, error_text[:300])
        if attempt == 1:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    "That JSON failed validation with the errors below. Reply again with "
                    f"ONE corrected SceneDoc JSON object only.\n\nERRORS:\n{error_text[:2000]}"
                ),
            })

    warnings.append("director output failed validation twice — deterministic fallback scene used")
    return ComposeResponse(scene=deterministic_fallback_scene(intent), warnings=warnings)

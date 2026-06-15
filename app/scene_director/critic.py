# FILE: app/scene_director/critic.py
# Purpose: The scene critic (v3) — Taz's SpecGate/Verifier applied to scenes. After the
#          director drafts a SceneDoc, interrogate it against a checklist and emit a revised,
#          richer, valid scene; bounded re-critique passes. Auto-fixes "why no trees / road too
#          short" instead of Taz catching it by eye.
# Called-by: app.scene_director.router (compose path)
# Depends-on: app.scene_director.director (sanitise/parse) + asset_catalog + schemas, app.llm.clients
# Last-renovated: 2026-06-13
"""critique_and_revise(scene, intent, warnings) → revised SceneDoc.

Flow: critique → if revisions, apply + re-sanitise → re-critique up to N bounded passes
(SCENE_CRITIC_MAX_PASSES, default 2) → return with residual issues logged as breadcrumbs.
Never loops unbounded; never ships invalid prefab_ids (every revision is re-sanitised);
never hard-fails (any critic/transport error returns the current scene unchanged).

Env: SCENE_CRITIC_ENABLED (default on); SCENE_CRITIC_PROVIDER / SCENE_CRITIC_MODEL
(SAFE default anthropic / anthropic:frontier-sonnet → claude-sonnet-4-6. NOTE: claude-opus-4-8
errored on this account, so the default is Sonnet, not Opus; point SCENE_CRITIC_MODEL at a
heavier id when one is available — e.g. on this setup the director+critic run gemini-3.5-flash
via .env, with gemini-3.1-pro-preview a one-line upgrade for deeper critic reasoning).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.scene_director import asset_catalog, director
from app.scene_director.schemas import ProvenanceSpec, SceneDoc

logger = logging.getLogger(__name__)

_CRITIC_SYSTEM = """You are ASTRA's scene CRITIC for "ASTRA Room" — a strict reviewer that interrogates a composed 3D SceneDoc against the user's brief, then FIXES it. You receive: the original brief, the composed SceneDoc JSON, and the asset catalogue. Judge it against this checklist:
1. Is the ground/backdrop DRESSED (terrain, hills, treeline) or bare/flat-empty?
2. Is there vegetation/terrain where the scene implies it (parks, countryside, streets with trees)?
3. Do tiled roads/paths actually CONNECT and span a sensible length? Any gaps, overlaps, or floating props (y not on the ground)?
4. Do actor paths make spatial sense? Is there traffic (moving vehicles) where a street needs it?
5. Does the scene genuinely match the brief — is EVERY requested element present (counts, actions, time of day)?
6. Is every prefab_id valid in the catalogue? Is lighting set to match the mood? Is density within the budget?
7. INTERIORS: if the brief is a room/indoor scene, is it actually built (walls + floor + ceiling + the requested furniture) from `structures` (box/plane/quad blocks) — since the catalogue has NO indoor assets? Is the ASTRA orb NOT duplicated (it always exists), and are NO particle-effect prefab_ids used as furniture/lights/board?

If the scene already passes (rich, matches the brief, valid), reply EXACTLY: {"ok": true, "issues": []}
Otherwise reply with STRICT JSON only (no fences, no commentary):
{"ok": false, "issues": ["<concise issue>", ...], "revised_scene": { <a FULLY corrected SceneDoc JSON, same contract as the director: title, skybox, terrain, environment, tiles, scatter, structures (raw box/plane/quad/cylinder blocks for interiors/walls/furniture, with size+color, no catalogue id needed), actors (type person|vehicle), timeline, lighting> }}
The revised_scene MUST fix the issues you list: add the missing dressing/terrain/vegetation/traffic, extend short roads with tiles, remove overlaps/floats, ensure every brief element is present, use ONLY catalogue prefab_ids, and stay within the density budget. Keep narration spoken/UK-English/teaching. Output one JSON object only."""


def _enabled() -> bool:
    return os.getenv("SCENE_CRITIC_ENABLED", "true").strip().lower() not in ("false", "0", "no")


def _max_passes() -> int:
    try:
        return max(1, min(4, int(os.getenv("SCENE_CRITIC_MAX_PASSES", "2"))))
    except ValueError:
        return 2


def _provider_and_model() -> Tuple[str, str]:
    from app.llm.frontier_models import resolve_model_alias
    provider = os.getenv("SCENE_CRITIC_PROVIDER", "anthropic").strip().lower()
    model = resolve_model_alias(os.getenv("SCENE_CRITIC_MODEL", "anthropic:frontier-sonnet"))
    return provider, model


async def _call_critic_llm(payload: str) -> Tuple[str, Dict[str, Any]]:
    from app.llm import clients
    provider, model = _provider_and_model()
    caller = {
        "anthropic": clients.async_call_anthropic,
        "google": clients.async_call_google,
        "openai": clients.async_call_openai,
    }.get(provider, clients.async_call_anthropic)
    return await caller(_CRITIC_SYSTEM, [{"role": "user", "content": payload}], model=model)


def _payload(scene: SceneDoc, intent: str, era: Optional[str]) -> str:
    catalog_block = asset_catalog.as_prompt_block(era=era)
    scene_json = scene.model_dump_json(exclude={"provenance"})
    return (
        f"ASSET CATALOGUE (only valid prefab_ids, grouped):\n{catalog_block}\n\n"
        f"DENSITY BUDGET: {scene.density_budget} total instances.\n\n"
        f"ORIGINAL BRIEF: {intent}\n\n"
        f"COMPOSED SCENEDOC:\n{scene_json}\n\n"
        "Interrogate against the checklist and reply per the contract."
    )


def _parse_critic(content: str) -> Optional[Dict[str, Any]]:
    data = director._extract_json(content)
    if not isinstance(data, dict):
        return None
    return data


async def critique_and_revise(scene: SceneDoc, intent: str, warnings: List[str],
                              era: Optional[str] = None) -> SceneDoc:
    """Run the bounded critic loop. Returns the (possibly revised) scene. Never raises."""
    if not _enabled():
        return scene

    original_provenance: Optional[ProvenanceSpec] = scene.provenance
    original_era = scene.era

    for pass_no in range(1, _max_passes() + 1):
        try:
            content, usage = await _call_critic_llm(_payload(scene, intent, era))
        except Exception as exc:
            logger.warning("[critic] pass %d call failed (%s) — keeping current scene", pass_no, exc)
            warnings.append(f"critic unavailable on pass {pass_no} — scene shipped without further revision")
            break
        if usage.get("error"):
            logger.warning("[critic] pass %d LLM error: %s", pass_no, usage.get("error"))
            warnings.append(f"critic returned an error on pass {pass_no} — scene shipped as drafted")
            break

        verdict = _parse_critic(content)
        if verdict is None:
            warnings.append(f"critic reply unparseable on pass {pass_no} — scene shipped as drafted")
            break

        issues = [str(i) for i in (verdict.get("issues") or [])]
        if verdict.get("ok") is True or not verdict.get("revised_scene"):
            if issues:
                warnings.append(f"critic pass {pass_no}: noted but not blocking — {'; '.join(issues[:4])}")
            logger.info("[critic] pass %d: OK (%d note(s))", pass_no, len(issues))
            break

        # Apply the revision: parse → sanitise → keep server-owned fields.
        revised_json = json.dumps(verdict["revised_scene"])
        doc, err = director._parse_scene(revised_json, intent)
        if doc is None:
            warnings.append(f"critic pass {pass_no} revision failed validation ({err[:120]}) — kept prior scene")
            logger.warning("[critic] pass %d revision invalid: %s", pass_no, err[:200])
            break
        doc = director.sanitise_scene(doc, warnings)
        doc.era = original_era
        doc.provenance = original_provenance
        scene = doc
        warnings.append(f"critic pass {pass_no} revised the scene: {'; '.join(issues[:5]) or 'enrichment'}")
        logger.info("[critic] pass %d revised — %d issue(s): %s", pass_no, len(issues), "; ".join(issues[:5]))

    return scene

# FILE: app/settings/stage_roles_api.py
# Purpose: Derek stage-role introspection + save API — GET/PUT /settings/stage-models.
# Called-by: app.settings.router (included under /settings, require_auth inherited)
# Depends-on: app.llm.stage_roles, app.settings.env_model_store
# Last-renovated: 2026-07-04 (Derek phase 2)
"""
The single source the Settings UI reads for the per-stage cards. The page
must never be able to lie: every response is a LIVE resolution against the
environment, carrying the source env key and any deprecation note.

Routes (inherit /settings prefix + require_auth from app/settings/router.py):
    GET /settings/stage-models          — every role, resolved, with provenance
    GET /settings/stage-models/allowed  — curated model list per provider for the dropdowns
    PUT /settings/stage-models/{role}   — write canonical env keys, hot-apply
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

stage_roles_router = APIRouter(prefix="/stage-models", tags=["Stage Models"])

_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/\-]{1,200}$")
_PROVIDER_RE = re.compile(r"^[a-z0-9_\-]{1,32}$")

# Curated dropdown lists — env-driven so Derek day is a .env edit, with the
# live local discovery merged in (vLLM/ollama models appear automatically).
_ALLOWED_ENV = {
    "anthropic": "ASTRA_STAGE_ALLOWED_MODELS_ANTHROPIC",
    "openai": "ASTRA_STAGE_ALLOWED_MODELS_OPENAI",
    "google": "ASTRA_STAGE_ALLOWED_MODELS_GOOGLE",
    "vllm_local": "ASTRA_STAGE_ALLOWED_MODELS_LOCAL",
}


class PutStageRoleRequest(BaseModel):
    model: str
    provider: Optional[str] = None


@stage_roles_router.get("")
def get_stage_models() -> dict:
    """Every stage role, resolved live, with provenance and tier metadata."""
    from app.llm.stage_roles import resolve_all_stage_roles

    roles = [r.to_dict() for r in resolve_all_stage_roles()]
    return {
        "roles": roles,
        "tiers": ["APEX", "HEAVY", "HANDS", "SMALL"],
        "count": len(roles),
        "errors": sum(1 for r in roles if r["error"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@stage_roles_router.get("/allowed")
async def get_allowed_models() -> dict:
    """Curated allowed-models per provider for the stage cards' dropdowns.

    Env lists (ASTRA_STAGE_ALLOWED_MODELS_*) are the curation; live local
    discovery (vLLM :8003 / ollama) is merged into vllm_local so Derek's
    models appear as soon as they are served.
    """
    out: dict = {}
    for provider, env_key in _ALLOWED_ENV.items():
        raw = os.getenv(env_key, "")
        out[provider] = [m.strip() for m in raw.split(",") if m.strip()]

    try:
        from app.settings.model_discovery import _fetch_local
        local_live = await _fetch_local()
        for m in local_live or []:
            if m not in out["vllm_local"]:
                out["vllm_local"].append(m)
    except Exception as exc:
        logger.debug("[stage_roles_api] local discovery merge skipped: %s", exc)

    return {"allowed": out, "source_env": _ALLOWED_ENV}


@stage_roles_router.put("/{role}")
def put_stage_role(role: str, body: PutStageRoleRequest) -> dict:
    """Write the canonical ASTRA_STAGE_* keys to .env and hot-apply.

    Uses the same atomic env store as the Models tab (backup + validate +
    swap + load_dotenv override). Returns the LIVE post-save resolution so
    the card shows truth, not intent.
    """
    from app.llm.stage_roles import (
        STAGE_ROLES, StageRoleResolutionError, resolve_stage_role,
    )

    key = role.upper().replace("-", "_")
    if key not in STAGE_ROLES:
        raise HTTPException(404, f"Unknown stage role: {role}")

    model = body.model.strip()
    if not _MODEL_RE.match(model):
        raise HTTPException(400, "Invalid model id")

    provider = (body.provider or "").strip().lower()
    if not provider:
        # A model-only save must not leave a stale PROVIDER key pointing at
        # the wrong provider. Infer and write the pair together.
        from app.llm.stage_roles import _infer_provider
        provider = _infer_provider(model)
        if not provider:
            raise HTTPException(
                400,
                f"Cannot infer a provider for model {model!r} — send provider explicitly",
            )
    if not _PROVIDER_RE.match(provider):
        raise HTTPException(400, "Invalid provider")

    spec = STAGE_ROLES[key]

    if key == "DEBUG_MAIN":
        # Provider-switch consumer: set the column toggle + the column's own
        # chat model var (openai vs anthropic). The debug system reads these.
        column_var = "DEBUG_CHAT_MODEL_ANTHROPIC" if provider == "anthropic" else "DEBUG_CHAT_MODEL"
        pairs = {"DEBUG_PROVIDER": provider, column_var: model}
    elif spec.home_is_legacy:
        # The legacy var is the permanent home (CHAT_MODEL/CHAT_PROVIDER) —
        # write it directly, no dead canonical key.
        pairs = {}
        for lk in spec.legacy_model_keys:
            pairs[lk] = model
        for lk in spec.legacy_provider_keys:
            pairs[lk] = provider
    else:
        # 2026-07-04 live fix (THE BUG): the card wrote only the canonical
        # ASTRA_STAGE_* key, but the actual stages read their OWN legacy vars
        # (Weaver reads WEAVER_MODEL, SpecGate reads SPEC_GATE_MODEL, ...), so
        # a saved change never took effect. MIRROR the value to every legacy
        # consumer key too. Canonical stays authoritative for introspection;
        # legacy makes the running stage honour the change (hot-applies for
        # per-call readers, restart for the import-time ones — badged).
        pairs = {
            f"ASTRA_STAGE_{key}_MODEL": model,
            f"ASTRA_STAGE_{key}_PROVIDER": provider,
        }
        for lk in spec.legacy_model_keys:
            pairs[lk] = model
        for lk in spec.legacy_provider_keys:
            pairs[lk] = provider

    from app.settings.env_model_store import set_env_vars, reload_env
    written = set_env_vars(pairs)
    reload_env()

    try:
        live = resolve_stage_role(key).to_dict()
    except StageRoleResolutionError as exc:
        live = {"error": str(exc)}

    return {
        "role": key,
        "written": {k: {"old": o, "new": n} for k, (o, n) in written.items()},
        "reloaded": True,
        "live": live,
    }


__all__ = ["stage_roles_router"]

# FILE: app/settings/model_settings.py
# Purpose: Model-settings API — GET /settings/models (UI contract), POST
#          /settings/models/reload (hot-apply), PUT /settings/models/{role}
#          (atomic .env rewrite + reload). LANE D Tasks 2 + 4.
# Called-by: app.settings.router (include_router; inherits require_auth)
# Depends-on: app.llm.model_roles, app.settings.env_model_store
# Last-renovated: 2026-07-02 (enum-valued rows: PUT validates against spec['values'] — DEBUG_PROVIDER switch)
"""
The UI contract (Task 4): GET /settings/models returns every role the
resolver knows — group, live provider/model (resolved at request time via
get_role_model / get_stage_config), the env vars behind it, restart_gated
and verifier flags. PUT rewrites that role's .env lines atomically and
triggers the same hot reload as POST /settings/models/reload.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

models_router = APIRouter(tags=["Settings — Models"])

_PROVIDER_RE = re.compile(r"^[a-z0-9_\-]{0,32}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/\-]{1,200}$")


class PutRoleRequest(BaseModel):
    provider: Optional[str] = None
    model: str


@models_router.get("/models", response_model=dict)
def get_models():
    """Every known role with its LIVE resolution (Task 4 UI contract)."""
    from app.llm.model_roles import GROUP_ORDER, list_roles

    roles = list_roles()
    return {
        "roles": roles,
        "groups": list(GROUP_ORDER),
        "count": len(roles),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@models_router.post("/models/reload", response_model=dict)
def reload_models():
    """Hot-apply: re-read .env (override) and report the model-var diff.

    The resolver reads env on access, so changes apply on the next request.
    Roles flagged restart_gated in GET /settings/models still need a restart.
    """
    from app.settings.env_model_store import reload_env

    diff = reload_env()
    return {
        "reloaded": True,
        "changed": {k: {"old": o, "new": n} for k, (o, n) in diff.items()},
        "changed_count": len(diff),
    }


@models_router.put("/models/{role}", response_model=dict)
def put_role_model(role: str, body: PutRoleRequest):
    """Atomically rewrite {ROLE}_PROVIDER/{ROLE}_MODEL in .env, then reload.

    In-place rewrite when the vars exist; otherwise appended under the
    "## UI-MANAGED MODELS" block. Refuses when the rewritten file fails to
    parse back (write-temp-then-replace keeps .env intact on failure).
    """
    from app.llm.model_roles import get_role_spec
    from app.settings.env_model_store import reload_env, set_env_vars

    spec = get_role_spec(role)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown role: {role}")

    model = (body.model or "").strip()
    if not _MODEL_RE.match(model):
        raise HTTPException(status_code=400, detail="Invalid model id")

    # Enum-valued rows (e.g. the DEBUG_PROVIDER switch) accept only their
    # declared values — keeps garbage out of .env.
    allowed_values = spec.get("values")
    if allowed_values and model not in allowed_values:
        raise HTTPException(
            status_code=400,
            detail=f"{spec['role']} accepts only: {', '.join(allowed_values)}",
        )

    pairs = {spec["model_env"]: model}
    provider = (body.provider or "").strip().lower()
    if provider:
        if not _PROVIDER_RE.match(provider):
            raise HTTPException(status_code=400, detail="Invalid provider")
        if spec["provider_env"] is None:
            locked = spec.get("provider") or ""
            if provider and locked and provider != locked:
                raise HTTPException(
                    status_code=400,
                    detail=f"Role {spec['role']} is provider-locked to {locked!r} "
                           f"(single env var {spec['model_env']})",
                )
        else:
            pairs[spec["provider_env"]] = provider

    try:
        written = set_env_vars(pairs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    reload_diff = reload_env()
    resolved = get_role_spec(role) or spec
    logger.info("[model_settings] PUT %s -> %s", spec["role"], pairs)
    return {
        "role": spec["role"],
        "written": {k: {"old": o, "new": n} for k, (o, n) in written.items()},
        "reloaded": True,
        "reload_changed_count": len(reload_diff),
        "live": {"provider": resolved.get("provider"), "model": resolved.get("model")},
        "restart_gated": resolved.get("restart_gated", False),
        "restart_note": resolved.get("restart_note"),
        "verifier": resolved.get("verifier", False),
    }


__all__ = ["models_router"]

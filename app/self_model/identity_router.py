# FILE: app/self_model/identity_router.py
"""
Identity store FastAPI endpoints.

Routes:
  GET    /self-model/identity             — full store view
  GET    /self-model/identity/{field}     — single field
  POST   /self-model/identity             — set/update a field
  DELETE /self-model/identity/{field}     — remove a field
  GET    /self-model/identity/_/history   — change history
  GET    /self-model/identity/_/preview   — rendered injection block
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.self_model.identity import get_identity_store, KNOWN_FIELDS
from app.self_model.identity_format import build_identity_block

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/self-model/identity", tags=["self-model-identity"])


class IdentitySetRequest(BaseModel):
    field: str
    value: Any
    source: str = "user_manual"


@router.get("")
def get_all_identity() -> dict:
    store = get_identity_store()
    fields = store.all()
    return {
        "fields": {k: f.to_dict() for k, f in fields.items()},
        "known_fields": KNOWN_FIELDS,
        "summary": store.summary(),
    }


@router.get("/_/history")
def get_history(field: Optional[str] = None) -> dict:
    store = get_identity_store()
    return {"history": store.history(field)}


@router.get("/_/preview")
def preview_block() -> dict:
    block = build_identity_block()
    return {"block": block or "", "present": block is not None}


@router.get("/{field}")
def get_field(field: str) -> dict:
    store = get_identity_store()
    f = store.get(field)
    if f is None:
        raise HTTPException(status_code=404, detail=f"Field '{field}' not set")
    return f.to_dict()


@router.post("")
def set_field(req: IdentitySetRequest) -> dict:
    if not req.field or not req.field.strip():
        raise HTTPException(status_code=400, detail="field required")
    store = get_identity_store()
    result = store.set(req.field.strip(), req.value, source=req.source)
    return result


@router.delete("/{field}")
def delete_field(field: str) -> dict:
    store = get_identity_store()
    if not store.delete(field, source="user_delete"):
        raise HTTPException(status_code=404, detail=f"Field '{field}' not set")
    return {"action": "deleted", "field": field}
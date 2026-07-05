# FILE: app/documents/router.py
# Purpose: /documents API — open (file -> Univer snapshot), save (snapshot ->
#          file, atomic + .bak-once), and the editor action channel.
# Called-by: router_registry (include, auth-gated); orb-desktop documentsApi.ts + univerBridge.ts
# Depends-on: app.documents converters, editor_actions, storage; app.drive.file_utils (path guard)
# Last-renovated: 2026-07-02
"""
Documents router.

Auth-gated at include (router_registry) like /drive — the desktop renderer
sends its session token (documentsApi.ts authHeaders). /open and /save also
enforce the shared Drive allowed-roots policy BEFORE any converter touches
the path, so this router cannot read or overwrite files outside those roots
(security hardening 2026-07-02).
Supported types: .xlsx/.csv -> sheet snapshots, .docx/.md -> doc snapshots.
Conversion loss policies live in the converter headers.

Editor action channel (mirrors web_automation's loop, one editor session):
  POST /documents/editor/execute            tools -> enqueue + await
  GET  /documents/editor/pending-action     pane long-poll
  POST /documents/editor/action-result/{id} pane posts outcome
  POST /documents/editor/status             pane reports open/close
  GET  /documents/editor/state              anyone asks what's open
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.documents import csv_md_convert, docx_convert, editor_actions, storage, xlsx_convert
from app.drive import file_utils as _file_utils

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

SHEET_EXTS = {".xlsx", ".csv"}
DOC_EXTS = {".docx", ".md"}
SUPPORTED_EXTS = SHEET_EXTS | DOC_EXTS


def _require_allowed_path(raw_path: str) -> Path:
    """Single choke point protecting every converter sink downstream.

    Resolves the path and rejects anything outside the shared Drive
    allowed-roots policy (app/drive/file_utils.get_allowed_roots). Runs
    BEFORE existence checks so out-of-root probes learn nothing.
    """
    path = Path(raw_path)
    try:
        allowed = _file_utils.is_safe_path(path, _file_utils.get_allowed_roots())
    except (OSError, ValueError):
        allowed = False
    if not allowed:
        logger.warning("[documents] path outside allowed roots rejected: %s", raw_path)
        raise HTTPException(status_code=403,
                            detail="path outside allowed roots")
    return path


class OpenRequest(BaseModel):
    path: str


class SaveRequest(BaseModel):
    path: str
    kind: str                    # "sheet" | "doc"
    snapshot: dict


class ExecuteRequest(BaseModel):
    action_type: str
    payload: Optional[dict] = None
    timeout_seconds: float = 10.0


class ActionResult(BaseModel):
    ok: bool
    result: Optional[Any] = None
    error: Optional[str] = None


class EditorStatus(BaseModel):
    open: bool
    path: Optional[str] = None
    kind: Optional[str] = None
    name: Optional[str] = None


def _kind_for(ext: str) -> str:
    return "sheet" if ext in SHEET_EXTS else "doc"


@router.post("/open")
def open_document(req: OpenRequest):
    """File -> {kind, snapshot}. Complex content degrades, never crashes."""
    path = _require_allowed_path(req.path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"no file at {req.path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=415,
                            detail=f"unsupported type {ext} — supported: "
                                   f"{sorted(SUPPORTED_EXTS)}")
    try:
        if ext == ".xlsx":
            snapshot = xlsx_convert.xlsx_to_snapshot(str(path))
        elif ext == ".csv":
            snapshot = csv_md_convert.csv_to_snapshot(str(path))
        elif ext == ".docx":
            snapshot = docx_convert.docx_to_snapshot(str(path))
        else:
            snapshot = csv_md_convert.md_to_snapshot(str(path))
    except Exception as exc:
        logger.exception("[documents] open failed for %s", req.path)
        raise HTTPException(status_code=500, detail=f"conversion failed: {exc}")
    return {
        "ok": True,
        "kind": _kind_for(ext),
        "snapshot": snapshot,
        "path": str(path),
        "name": path.name,
        "has_backup": storage.backup_path(str(path)).exists(),
    }


@router.post("/save")
def save_document(req: SaveRequest):
    """Snapshot -> file. Atomic temp+replace; .bak of the original on first save."""
    path = _require_allowed_path(req.path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=415, detail=f"unsupported type {ext}")
    try:
        bak_created = storage.ensure_first_save_backup(str(path))
        if ext == ".xlsx":
            storage.atomic_write_via(
                str(path), lambda tmp: xlsx_convert.snapshot_to_xlsx(req.snapshot, tmp))
        elif ext == ".csv":
            storage.atomic_write_via(
                str(path), lambda tmp: csv_md_convert.snapshot_to_csv(req.snapshot, tmp))
        elif ext == ".docx":
            storage.atomic_write_via(
                str(path), lambda tmp: docx_convert.snapshot_to_docx(req.snapshot, tmp))
        else:
            storage.atomic_write_via(
                str(path), lambda tmp: csv_md_convert.snapshot_to_md(req.snapshot, tmp))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[documents] save failed for %s", req.path)
        raise HTTPException(status_code=500, detail=f"save failed: {exc}")
    return {"ok": True, "path": str(path), "bak_created": bak_created,
            "bak_path": str(storage.backup_path(str(path)))}


# ── editor action channel ──────────────────────────────────────────────────

@router.post("/editor/execute")
async def editor_execute(req: ExecuteRequest):
    """Backend tools (or anything local) drive the open editor pane."""
    return await editor_actions.execute(
        req.action_type, req.payload, timeout_seconds=req.timeout_seconds)


@router.get("/editor/pending-action")
async def editor_pending_action(wait: float = 25.0):
    """Long-poll for the pane. {} when nothing arrived in the window."""
    action = await editor_actions.next_action(wait_seconds=wait)
    return action or {}


@router.post("/editor/action-result/{action_id}")
def editor_action_result(action_id: str, body: ActionResult):
    accepted = editor_actions.post_result(action_id, body.model_dump())
    return {"ok": accepted}


@router.post("/editor/status")
def editor_status(body: EditorStatus):
    """The pane reports open/close so tools can answer without a round trip."""
    return {"ok": True, "state": editor_actions.set_editor_state(
        body.open, body.path, body.kind, body.name)}


@router.get("/editor/state")
def editor_state():
    return {"ok": True, "state": editor_actions.editor_state()}

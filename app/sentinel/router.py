# FILE: app/sentinel/router.py
# Purpose: FastAPI surface for Sentinel — status, live connections, alerts lifecycle,
#          ask-first block/unblock (confirm required), trust, localhost debug inject.
# Called-by: main (include_router), desktop SecurityView/SentinelAlertWatcher
# Depends-on: app.auth, app.db, app.sentinel.agent_client/alerts/baseline/collector/enrich
# Last-renovated: 2026-06-12
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import require_auth
from app.db import get_db
from app.sentinel import agent_client, alerts, baseline, collector
from app.sentinel.agent_client import AgentOfflineError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/sentinel",
    tags=["Sentinel"],
    dependencies=[Depends(require_auth)],
)

# Debug router: NO bearer auth, localhost-guarded instead — acceptance testing
# fires inject-alert from the machine itself (curl/PowerShell, no token juggling).
debug_router = APIRouter(prefix="/sentinel/debug", tags=["Sentinel"])


class BlockBody(BaseModel):
    ip: str
    confirm: bool = False


class TrustBody(BaseModel):
    process: str
    remote: Optional[str] = None
    port: Optional[int] = None


class InjectAlertBody(BaseModel):
    severity: str = "severe"
    title: Optional[str] = None
    explanation: Optional[str] = None


@router.get("/status")
async def get_status():
    """Agent health, learn-mode state + days remaining, baseline size, alert counts."""
    return await collector.status_payload()


@router.get("/connections")
async def get_connections():
    """Live snapshot from the agent, grouped by process for the desktop table."""
    try:
        conns = await agent_client.get_connections()
    except AgentOfflineError:
        return {"agent_online": False, "groups": [], "total": 0}
    groups = defaultdict(list)
    exe_by_proc = {}
    for c in conns:
        name = c.get("process_name") or "unknown"
        groups[name].append(c)
        if c.get("exe_path"):
            exe_by_proc.setdefault(name, c["exe_path"])
    shaped = [
        {
            "process_name": name,
            "exe_path": exe_by_proc.get(name, ""),
            "count": len(items),
            "connections": sorted(
                items, key=lambda c: (c.get("status") or "", c.get("raddr_ip") or "")
            ),
        }
        for name, items in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]
    return {"agent_online": True, "groups": shaped, "total": len(conns)}


@router.get("/alerts")
def get_alerts(
    unacked: bool = False,
    include_suppressed: bool = False,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    rows = alerts.list_alerts(
        db, unacked_only=unacked, include_suppressed=include_suppressed, limit=limit
    )
    return {"alerts": [alerts.alert_to_dict(a) for a in rows]}


@router.post("/alerts/{alert_id}/ack")
def ack_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = alerts.ack_alert(db, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return {"ok": True, "alert": alerts.alert_to_dict(alert)}


@router.post("/block")
async def block(body: BlockBody, db: Session = Depends(get_db)):
    """Ask-first is ABSOLUTE: refuses without confirm=true in the body."""
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="Blocking requires explicit confirmation: pass confirm=true.",
        )
    try:
        result = await agent_client.block_ip(body.ip)
    except AgentOfflineError:
        raise HTTPException(status_code=503, detail="sentinel agent is offline")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"agent block failed: {exc}")
    stamped = alerts.mark_action_for_remote(db, body.ip, f"blocked {body.ip}")
    logger.warning("[sentinel] BLOCK confirmed by Taz: %s (alerts stamped: %s)", body.ip, stamped)
    return {"ok": True, "result": result, "alerts_updated": stamped}


@router.post("/unblock")
async def unblock(body: BlockBody, db: Session = Depends(get_db)):
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="Unblocking requires explicit confirmation: pass confirm=true.",
        )
    try:
        result = await agent_client.unblock_ip(body.ip)
    except AgentOfflineError:
        raise HTTPException(status_code=503, detail="sentinel agent is offline")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"agent unblock failed: {exc}")
    alerts.mark_action_for_remote(db, body.ip, f"unblocked {body.ip}")
    return {"ok": True, "result": result}


@router.get("/firewall-rules")
async def firewall_rules():
    """Proxy of the agent's GET /rules — every ASTRA-Sentinel-* rule in the firewall."""
    try:
        return await agent_client.list_rules()
    except AgentOfflineError:
        raise HTTPException(status_code=503, detail="sentinel agent is offline")


@router.post("/trust")
def trust(body: TrustBody, db: Session = Depends(get_db)):
    """Trust a process outright, or one exact (process, remote, port) pair."""
    if not body.process.strip():
        raise HTTPException(status_code=400, detail="process is required")
    if body.remote:
        baseline.trust_pair(db, body.process.strip(), body.remote.strip(), int(body.port or 0))
        scope = f"pair {body.remote}:{body.port or 0}"
    else:
        baseline.trust_process(db, body.process.strip())
        scope = "process"
    return {"ok": True, "trusted": body.process.strip(), "scope": scope}


@debug_router.post("/inject-alert")
def inject_alert(body: InjectAlertBody, request: Request, db: Session = Depends(get_db)):
    """Test-only alert injection — REQUIRED for acceptance testing. Localhost only."""
    client_host = request.client.host if request.client else ""
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="debug endpoints are localhost-only")
    severity = body.severity if body.severity in ("severe", "medium", "low") else "severe"
    alert = alerts.create_alert(
        db,
        severity=severity,
        rule_key=f"debug_inject_{severity}",
        title=body.title or f"Test {severity} alert from Sentinel debug inject",
        explanation=body.explanation
        or (
            "This is a synthetic alert fired by POST /sentinel/debug/inject-alert to "
            "prove the alert plumbing end-to-end: store, desktop toast/sound/TTS, "
            "phone notification, chat injection."
        ),
        process="debug",
        # unique remote per call so repeated test fires are never deduped
        remote=f"debug:{severity}:{int(time.time())}",
        recommended_action="ignore",
    )
    if alert is None:  # pragma: no cover — unreachable with unique remote
        raise HTTPException(status_code=500, detail="alert store refused the injection")
    return {"ok": True, "alert": alerts.alert_to_dict(alert)}

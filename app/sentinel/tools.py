# FILE: app/sentinel/tools.py
# Purpose: Chat tools for Sentinel — ask ASTRA about network state, alerts, trust,
#          and PROPOSE blocks (never execute; ask-first is absolute). Schemas inline.
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.sentinel.collector/alerts/baseline/models, app.db (style ref: app/tools/finance_tools.py)
# Last-renovated: 2026-06-12
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

_OBJ = {"type": "object"}  # permissive output schema — handlers return plain dicts


@contextmanager
def _db_session():
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# Handlers — every one returns {"ok": bool, ...} and never raises
# =============================================================================

async def get_security_status_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.sentinel import collector

        status = await collector.status_payload()
        return {"ok": True, **status}
    except Exception as exc:
        logger.exception("[sentinel_tools] get_security_status failed")
        return {"ok": False, "error": str(exc)}


async def get_recent_connections_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.sentinel.models import SentinelConnection

        limit = max(1, min(int(input_data.get("limit") or 25), 100))
        process = str(input_data.get("process") or "").strip()
        with _db_session() as db:
            q = db.query(SentinelConnection).order_by(SentinelConnection.ts.desc())
            if process:
                q = q.filter(SentinelConnection.process_name == process)
            rows = q.limit(limit).all()
            events = [
                {
                    "ts": r.ts.isoformat() if r.ts else "",
                    "event": r.event,
                    "process": r.process_name,
                    "remote": f"{r.raddr_ip}:{r.raddr_port}" if r.raddr_ip else f"listen:{r.laddr_port}",
                    "rdns": r.rdns,
                    "country": r.country,
                    "proto": r.proto,
                }
                for r in rows
            ]
        return {"ok": True, "count": len(events), "events": events}
    except Exception as exc:
        logger.exception("[sentinel_tools] get_recent_connections failed")
        return {"ok": False, "error": str(exc)}


async def get_security_alerts_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.sentinel import alerts

        unacked_only = bool(input_data.get("unacked_only", True))
        limit = max(1, min(int(input_data.get("limit") or 10), 50))
        with _db_session() as db:
            rows = alerts.list_alerts(db, unacked_only=unacked_only, limit=limit)
            payload = [alerts.alert_to_dict(a) for a in rows]
        return {"ok": True, "count": len(payload), "alerts": payload}
    except Exception as exc:
        logger.exception("[sentinel_tools] get_security_alerts failed")
        return {"ok": False, "error": str(exc)}


async def explain_alert_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.sentinel import alerts

        alert_id = int(input_data.get("alert_id") or 0)
        with _db_session() as db:
            alert = alerts.get_alert(db, alert_id)
            if alert is None:
                return {"ok": False, "error": f"no alert with id {alert_id}"}
            return {"ok": True, "alert": alerts.alert_to_dict(alert)}
    except Exception as exc:
        logger.exception("[sentinel_tools] explain_alert failed")
        return {"ok": False, "error": str(exc)}


async def request_block_remote_address_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Creates a PENDING block proposal alert. NEVER touches the firewall —
    Taz confirms (or refuses) in the Security tab / desktop dialog."""
    try:
        import ipaddress

        from app.sentinel import alerts

        ip = str(input_data.get("ip") or "").strip()
        reason = str(input_data.get("reason") or "").strip()
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            return {"ok": False, "error": f"not a valid IP literal: {ip!r}"}
        with _db_session() as db:
            alert = alerts.create_alert(
                db,
                severity="medium",
                rule_key="block_requested",
                title=f"ASTRA proposes blocking {ip}",
                explanation=(
                    (reason or "No reason given.")
                    + " — PENDING: nothing is blocked yet. Confirm or dismiss in the "
                    "Security tab (ask-first is absolute)."
                ),
                process=str(input_data.get("process") or ""),
                remote=ip,
                recommended_action="propose_block",
            )
        if alert is None:
            return {"ok": True, "pending": True,
                    "note": "an identical block proposal is already pending"}
        return {"ok": True, "pending": True, "alert_id": alert.id,
                "note": "Block proposal recorded. Taz must confirm before anything is blocked."}
    except Exception as exc:
        logger.exception("[sentinel_tools] request_block failed")
        return {"ok": False, "error": str(exc)}


async def trust_process_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.sentinel import baseline

        process = str(input_data.get("process") or "").strip()
        if not process:
            return {"ok": False, "error": "process is required"}
        with _db_session() as db:
            baseline.trust_process(db, process)
        return {"ok": True, "trusted": process,
                "note": "future connections from this process won't raise alerts"}
    except Exception as exc:
        logger.exception("[sentinel_tools] trust_process failed")
        return {"ok": False, "error": str(exc)}


# =============================================================================
# Registration
# =============================================================================

def register_sentinel_tools() -> None:
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="get_security_status",
        version="v1",
        description=(
            "Sentinel network-monitor status: agent online/elevated, learn-mode days "
            "remaining, baseline size, alert counts, collector health."
        ),
        input_schema={"type": "object", "properties": {}},
        output_schema=_OBJ,
        handler=get_security_status_handler,
    ))
    register_tool(ToolDefinition(
        name="get_recent_connections",
        version="v1",
        description=(
            "Recent network connection events (new outbound pairs / new listeners), "
            "optionally filtered by process name."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "max events, default 25"},
                "process": {"type": "string", "description": "filter to one process name"},
            },
        },
        output_schema=_OBJ,
        handler=get_recent_connections_handler,
    ))
    register_tool(ToolDefinition(
        name="get_security_alerts",
        version="v1",
        description="Current Sentinel security alerts (default: unacknowledged only).",
        input_schema={
            "type": "object",
            "properties": {
                "unacked_only": {"type": "boolean", "description": "default true"},
                "limit": {"type": "integer", "description": "max alerts, default 10"},
            },
        },
        output_schema=_OBJ,
        handler=get_security_alerts_handler,
    ))
    register_tool(ToolDefinition(
        name="explain_alert",
        version="v1",
        description="Full detail for one Sentinel alert by id (explanation, recommendation, state).",
        input_schema={
            "type": "object",
            "properties": {"alert_id": {"type": "integer"}},
            "required": ["alert_id"],
        },
        output_schema=_OBJ,
        handler=explain_alert_handler,
    ))
    register_tool(ToolDefinition(
        name="request_block_remote_address",
        version="v1",
        description=(
            "PROPOSE blocking a remote IP via Windows Firewall. Creates a pending "
            "proposal that Taz must explicitly confirm — this tool NEVER blocks anything itself."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ip": {"type": "string", "description": "remote IP to propose blocking"},
                "reason": {"type": "string", "description": "why blocking is warranted"},
                "process": {"type": "string", "description": "related process name, if known"},
            },
            "required": ["ip", "reason"],
        },
        output_schema=_OBJ,
        handler=request_block_remote_address_handler,
    ))
    register_tool(ToolDefinition(
        name="trust_process",
        version="v1",
        description="Mark a process as trusted so Sentinel stops alerting on its connections.",
        input_schema={
            "type": "object",
            "properties": {"process": {"type": "string", "description": "exact process name, e.g. chrome.exe"}},
            "required": ["process"],
        },
        output_schema=_OBJ,
        handler=trust_process_handler,
    ))

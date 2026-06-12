# FILE: app/sentinel/collector.py
# Purpose: 30s collection cycle — pull agent snapshot, diff against the in-memory
#          active map, write connection events, feed baseline + rules, triage + alert.
# Called-by: app.sentinel.scheduler (collect_once), app.sentinel.router/tools (status_payload)
# Depends-on: app.sentinel.agent_client/enrich/baseline/rules/triage/alerts/models, app.db
# Last-renovated: 2026-06-12
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.sentinel import agent_client, alerts, baseline, enrich, rules, triage
from app.sentinel.agent_client import AgentOfflineError
from app.sentinel.models import SentinelConnection

logger = logging.getLogger(__name__)

_PAIR_TTL_SECONDS = 300.0  # forget a pair not seen for 5 min (baseline stops re-alerts)
_DAILY_COUNTS_KEY = "daily_external_max"
_DAILY_COUNTS_KEEP = 8

# (process_name, raddr_ip, raddr_port) -> last-seen monotonic
_active_pairs: Dict[Tuple[str, str, int], float] = {}
# (process_name, laddr_port) -> last-seen monotonic
_active_listens: Dict[Tuple[str, int], float] = {}
_agent_offline_alerted = False  # latched per backend boot — one offline alert max

_status: Dict[str, Any] = {
    "agent_online": None,
    "last_collect_at": None,
    "last_error": None,
    "connections_last": 0,
    "external_last": 0,
    "new_pairs_last": 0,
    "collect_runs": 0,
}


def _is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return True  # unparseable = not interesting


def _diff_snapshot(conns: List[Dict[str, Any]]) -> Tuple[List[dict], List[dict], int]:
    """Compare snapshot against the in-memory active maps.

    Returns (new_pairs, new_listens, external_established_count)."""
    now_mono = time.monotonic()
    new_pairs: List[dict] = []
    new_listens: List[dict] = []
    external_count = 0

    for conn in conns:
        pname = str(conn.get("process_name") or "unknown")
        status = str(conn.get("status") or "")
        raddr = str(conn.get("raddr_ip") or "")
        rport = int(conn.get("raddr_port") or 0)
        lport = int(conn.get("laddr_port") or 0)

        if status == "LISTEN":
            key = (pname, lport)
            if key not in _active_listens:
                new_listens.append(conn)
            _active_listens[key] = now_mono
            continue

        if not raddr or _is_loopback(raddr):
            continue
        if status == "ESTABLISHED" and enrich.is_public_ip(raddr):
            external_count += 1
        pair = (pname, raddr, rport)
        if pair not in _active_pairs:
            new_pairs.append(conn)
        _active_pairs[pair] = now_mono

    for store in (_active_pairs, _active_listens):
        expired = [k for k, seen in store.items() if now_mono - seen > _PAIR_TTL_SECONDS]
        for k in expired:
            del store[k]

    return new_pairs, new_listens, external_count


def _update_daily_counts(db, external_count: int) -> Tuple[Optional[float], int]:
    """Track today's max concurrent external count; return (avg of prior days, day count)."""
    today = datetime.now().astimezone().date().isoformat()
    try:
        counts = json.loads(baseline.get_state(db, _DAILY_COUNTS_KEY) or "{}")
    except json.JSONDecodeError:
        counts = {}
    if external_count > int(counts.get(today, 0)):
        counts[today] = external_count
        for stale in sorted(counts)[:-_DAILY_COUNTS_KEEP]:
            del counts[stale]
        baseline.set_state(db, _DAILY_COUNTS_KEY, json.dumps(counts))
    prior = [v for d, v in counts.items() if d != today]
    if not prior:
        return None, 0
    return sum(prior) / len(prior), len(prior)


def _process_db(
    new_pairs: List[dict],
    new_listens: List[dict],
    external_count: int,
    enrich_map: Dict[str, dict],
) -> List[Tuple[dict, dict]]:
    """Sync DB pass: events + baseline + rules. Stores suppressed/offline-grade
    alerts directly; returns [(finding, context)] still needing LLM triage."""
    from app.db import SessionLocal

    db = SessionLocal()
    to_triage: List[Tuple[dict, dict]] = []
    try:
        baseline.ensure_learn_mode_started(db)
        learn_active = baseline.learn_mode_active(db)
        # First collection EVER: this snapshot defines "normal so far" — seed the
        # baseline silently and fire NO rules. Otherwise boot #1 would scream
        # severe for every standard Windows listener (svchost :135, System :445…)
        # and burst LLM triage calls for nothing.
        first_seed = baseline.get_state(db, "first_seed_done") is None

        def _handle(finding: dict, context: dict) -> None:
            if first_seed:
                return  # seeding cycle records baseline + events, never alerts
            if learn_active and finding["severity"] != rules.SEVERE:
                alerts.create_alert(
                    db,
                    severity=finding["severity"],
                    rule_key=finding["rule_key"],
                    title=finding["title"],
                    explanation=finding["detail"] + " (recorded quietly during learn mode)",
                    process=finding["process"],
                    remote=finding["remote"],
                    suppressed=True,
                )
            else:
                to_triage.append((finding, context))

        for conn in new_pairs:
            pname = str(conn.get("process_name") or "unknown")
            raddr = str(conn.get("raddr_ip") or "")
            rport = int(conn.get("raddr_port") or 0)
            info = enrich_map.get(raddr) or enrich.cached_info(raddr) or {}
            is_public = enrich.is_public_ip(raddr)
            rdns = str(info.get("rdns") or "")
            country = str(info.get("country_code") or "")
            remote_key = rdns or raddr

            trusted = baseline.is_trusted(db, pname, remote_key, rport)
            known = baseline.process_known(db, pname)
            known_ports = baseline.known_ports_for_process(db, pname) if known else set()
            known_countries = baseline.known_countries_for_process(db, pname) if known else set()
            _, created = baseline.observe(
                db, pname, remote_key, rport, kind="outbound", country=country or None
            )
            db.add(SentinelConnection(
                event="new",
                pid=int(conn.get("pid") or 0),
                process_name=pname,
                exe_path=str(conn.get("exe_path") or ""),
                laddr_ip=str(conn.get("laddr_ip") or ""),
                laddr_port=conn.get("laddr_port"),
                raddr_ip=raddr,
                raddr_port=rport,
                status=str(conn.get("status") or ""),
                proto=str(conn.get("proto") or ""),
                rdns=rdns,
                country=country,
            ))
            db.commit()
            if not created:
                continue  # pair already in baseline — re-seen, not an anomaly
            ctx = {
                "process_name": pname,
                "exe_path": str(conn.get("exe_path") or ""),
                "raddr_ip": raddr,
                "raddr_port": rport,
                "rdns": rdns,
                "country": country,
                "is_public": is_public,
                "process_known": known,
                "known_ports": known_ports,
                "known_countries": known_countries,
                "trusted": trusted,
            }
            for finding in rules.evaluate_new_pair(ctx):
                _handle(finding, {
                    "rdns": rdns, "country": info.get("country") or country,
                    "org": info.get("org") or "", "exe_path": ctx["exe_path"],
                    "first_sighting": True,
                })

        for conn in new_listens:
            pname = str(conn.get("process_name") or "unknown")
            lport = int(conn.get("laddr_port") or 0)
            trusted = baseline.is_trusted(db, pname)
            _, created = baseline.observe(db, pname, "", lport, kind="listen")
            db.add(SentinelConnection(
                event="listen",
                pid=int(conn.get("pid") or 0),
                process_name=pname,
                exe_path=str(conn.get("exe_path") or ""),
                laddr_ip=str(conn.get("laddr_ip") or ""),
                laddr_port=lport,
                status="LISTEN",
                proto=str(conn.get("proto") or ""),
            ))
            db.commit()
            for finding in rules.evaluate_listen({
                "process_name": pname,
                "exe_path": str(conn.get("exe_path") or ""),
                "laddr_port": lport,
                "listen_known": not created,
                "trusted": trusted,
            }):
                _handle(finding, {"exe_path": str(conn.get("exe_path") or ""), "listening": True})

        avg, days = _update_daily_counts(db, external_count)
        spike = rules.evaluate_count_spike(external_count, avg, days)
        if spike:
            _handle(spike, {"current": external_count, "rolling_avg": avg, "days": days})

        if first_seed:
            baseline.set_state(db, "first_seed_done", datetime.now(timezone.utc).isoformat())
            logger.info(
                "[sentinel] first snapshot seeded the baseline quietly (%d pairs, %d listeners)",
                len(new_pairs), len(new_listens),
            )

        return to_triage
    finally:
        db.close()


def _store_triaged_alert(finding: dict, triaged: dict) -> None:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        alerts.create_alert(
            db,
            severity=finding["severity"],
            rule_key=finding["rule_key"],
            title=finding["title"],
            explanation=triaged["explanation"],
            process=finding["process"],
            remote=finding["remote"],
            recommended_action=triaged["recommended_action"],
            confidence=triaged.get("confidence"),
        )
    finally:
        db.close()


def _store_agent_offline_alert() -> None:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        alerts.create_alert(
            db,
            severity="medium",
            rule_key="agent_offline",
            title="Sentinel agent is offline",
            explanation=(
                "The elevated sentinel_agent on 127.0.0.1:8771 is not answering, so "
                "network monitoring is paused. If this wasn't you stopping it, check "
                "the AstraSentinelAgent scheduled task (re-run scripts\\install_sentinel_agent.ps1)."
            ),
            process="sentinel_agent",
            remote="127.0.0.1:8771",
        )
    finally:
        db.close()


async def collect_once() -> None:
    """One collection cycle. Never raises — the scheduler loop must survive anything."""
    global _agent_offline_alerted
    _status["collect_runs"] += 1
    _status["last_collect_at"] = datetime.now(timezone.utc).isoformat()
    try:
        conns = await agent_client.get_connections()
    except AgentOfflineError as exc:
        _status["agent_online"] = False
        _status["last_error"] = str(exc)
        if not _agent_offline_alerted:
            _agent_offline_alerted = True  # one medium alert per boot, max
            try:
                await asyncio.to_thread(_store_agent_offline_alert)
            except Exception:
                logger.exception("[sentinel] failed to store agent-offline alert")
        return
    except Exception as exc:
        _status["last_error"] = f"collect failed: {exc}"
        logger.exception("[sentinel] collect failed")
        return

    _status["agent_online"] = True
    _status["last_error"] = None
    _status["connections_last"] = len(conns)

    try:
        new_pairs, new_listens, external_count = _diff_snapshot(conns)
        _status["external_last"] = external_count
        _status["new_pairs_last"] = len(new_pairs)
        if not new_pairs and not new_listens:
            # still update daily counters cheaply when nothing is new
            if external_count:
                from app.db import SessionLocal
                db = SessionLocal()
                try:
                    _update_daily_counts(db, external_count)
                finally:
                    db.close()
            return
        public_ips = [
            str(c.get("raddr_ip") or "") for c in new_pairs
            if enrich.is_public_ip(str(c.get("raddr_ip") or ""))
        ]
        enrich_map = await enrich.enrich_ips(public_ips) if public_ips else {}
        to_triage = await asyncio.to_thread(
            _process_db, new_pairs, new_listens, external_count, enrich_map
        )
        for finding, context in to_triage:
            triaged = await triage.triage_finding(finding, context)
            await asyncio.to_thread(_store_triaged_alert, finding, triaged)
    except Exception as exc:
        _status["last_error"] = f"processing failed: {exc}"
        logger.exception("[sentinel] snapshot processing failed")


async def status_payload() -> Dict[str, Any]:
    """Shared status shape for /sentinel/status and the get_security_status tool."""
    agent_health = await agent_client.health()

    def _db_bits() -> Dict[str, Any]:
        from app.db import SessionLocal
        db = SessionLocal()
        try:
            started = baseline.learn_mode_started_at(db)
            return {
                "learn_mode": {
                    "active": baseline.learn_mode_active(db),
                    "started_at": started.isoformat() if started else None,
                    "days_remaining": baseline.learn_mode_days_remaining(db),
                    "total_days": baseline.LEARN_MODE_DAYS,
                },
                "baseline_size": baseline.baseline_size(db),
                "alerts": alerts.alert_counts(db),
            }
        finally:
            db.close()

    bits = await asyncio.to_thread(_db_bits)
    return {
        "agent": {
            "online": agent_health is not None,
            **({k: agent_health[k] for k in ("version", "elevated", "uptime_seconds")
                if agent_health and k in agent_health} if agent_health else {}),
        },
        **bits,
        "collector": dict(_status),
    }

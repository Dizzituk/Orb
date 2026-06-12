# FILE: app/sentinel/rules.py
# Purpose: Deterministic anomaly triage — pure functions, NO LLM, NO DB writes.
#          Thresholds are the constants below; tune them here and only here.
# Called-by: app.sentinel.collector (+ pytest units)
# Depends-on: stdlib only
# Last-renovated: 2026-06-12
from __future__ import annotations

from typing import Any, Dict, List, Optional

SEVERE = "severe"
MEDIUM = "medium"
LOW = "low"

# ── Tunable thresholds ───────────────────────────────────────────────────────
# Exe locations that legitimate long-lived software has no business running
# from. Deliberately NARROW (Temp/Downloads/Public) — most of AppData is
# normal for Electron-style apps (Discord, Slack), so plain AppData stays
# at the rule's base severity rather than escalating to severe.
USER_WRITABLE_MARKERS = (
    "\\appdata\\local\\temp\\",
    "\\windows\\temp\\",
    "\\temp\\",
    "\\downloads\\",
    "\\users\\public\\",
    "\\programdata\\temp\\",
)
# Remote ports common enough that a raw-IP or new-port connection alone is noise.
COMMON_REMOTE_PORTS = {
    80, 443, 53, 123, 853, 8080, 8443,
    993, 995, 465, 587,          # mail
    3478, 5223, 5228,            # push/STUN
    1900, 5353,                  # SSDP / mDNS
}
SPIKE_MULTIPLIER = 3.0   # current external count vs 7-day average…
SPIKE_MIN_EXCESS = 20    # …and at least this many above it
SPIKE_MIN_HISTORY_DAYS = 3
LOW_LISTEN_PORT = 1024   # listeners below this are more interesting


def exe_in_user_writable(exe_path: str) -> bool:
    path = (exe_path or "").lower()
    return any(marker in path for marker in USER_WRITABLE_MARKERS)


def _finding(rule_key: str, severity: str, title: str, detail: str,
             process: str, remote: str) -> Dict[str, Any]:
    return {
        "rule_key": rule_key,
        "severity": severity,
        "title": title,
        "detail": detail,
        "process": process,
        "remote": remote,
    }


def evaluate_new_pair(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rules for a newly-observed (process, remote, port) pair.

    ctx keys: process_name, exe_path, raddr_ip, raddr_port, rdns, country,
    is_public (bool), process_known (bool), known_ports (set), known_countries (set),
    trusted (bool).
    """
    if ctx.get("trusted"):
        return []
    findings: List[Dict[str, Any]] = []
    process = str(ctx.get("process_name") or "unknown")
    ip = str(ctx.get("raddr_ip") or "")
    port = int(ctx.get("raddr_port") or 0)
    remote = f"{ip}:{port}"
    rdns = str(ctx.get("rdns") or "")
    country = str(ctx.get("country") or "")
    is_public = bool(ctx.get("is_public"))
    known = bool(ctx.get("process_known"))

    # R1 — new process making external connections
    if is_public and not known:
        in_temp = exe_in_user_writable(str(ctx.get("exe_path") or ""))
        findings.append(_finding(
            "new_process",
            SEVERE if in_temp else MEDIUM,
            f"New process talking to the internet: {process}",
            (f"{process} ({ctx.get('exe_path') or 'path unknown'}) made its first external "
             f"connection, to {rdns or ip} port {port}"
             + (f" in {country}" if country else "")
             + (". The executable runs from a user-writable temp location — "
                "a common malware pattern." if in_temp else ".")),
            process, remote,
        ))

    # R2 — connection to a raw IP with no reverse DNS
    if is_public and not rdns:
        findings.append(_finding(
            "raw_ip",
            LOW if port in COMMON_REMOTE_PORTS else MEDIUM,
            f"{process} connected to a bare IP ({ip})",
            (f"{process} connected to {ip}:{port}"
             + (f" in {country}" if country else "")
             + " — the address has no reverse-DNS name. Plenty of CDNs do this, "
               "but so does malware command-and-control."),
            process, remote,
        ))

    # R3 — unusual remote port for a known process
    if (
        is_public and known and port
        and port not in COMMON_REMOTE_PORTS
        and port not in (ctx.get("known_ports") or set())
    ):
        findings.append(_finding(
            "unusual_port",
            LOW,
            f"{process} used an unusual port ({port})",
            (f"{process} normally talks on {sorted(ctx.get('known_ports') or set())[:8]} "
             f"but just connected to {rdns or ip} on port {port}."),
            process, remote,
        ))

    # R6 — known process suddenly connecting to a new country
    known_countries = ctx.get("known_countries") or set()
    if is_public and known and country and known_countries and country not in known_countries:
        findings.append(_finding(
            "new_country",
            MEDIUM,
            f"{process} connected to a new country: {country}",
            (f"{process} has only ever been seen talking to {sorted(known_countries)[:6]}; "
             f"it just connected to {rdns or ip}:{port} in {country}."),
            process, remote,
        ))

    return findings


def evaluate_listen(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """R5 — a new LISTENING port appeared.

    ctx keys: process_name, exe_path, laddr_port, listen_known (bool), trusted (bool).
    """
    if ctx.get("trusted") or ctx.get("listen_known"):
        return []
    process = str(ctx.get("process_name") or "unknown")
    port = int(ctx.get("laddr_port") or 0)
    in_temp = exe_in_user_writable(str(ctx.get("exe_path") or ""))
    severity = SEVERE if (in_temp or port < LOW_LISTEN_PORT) else MEDIUM
    return [_finding(
        "new_listen",
        severity,
        f"New listening port {port} opened by {process}",
        (f"{process} ({ctx.get('exe_path') or 'path unknown'}) started LISTENING on port {port}. "
         "A new open port means something on this PC is accepting inbound connections"
         + (" — and the executable runs from a temp location." if in_temp else ".")),
        process, f"listen:{port}",
    )]


def evaluate_count_spike(current_external: int, history_avg: Optional[float],
                         history_days: int) -> Optional[Dict[str, Any]]:
    """R4 — concurrent external-connection count spike vs the rolling baseline."""
    if history_avg is None or history_days < SPIKE_MIN_HISTORY_DAYS or history_avg <= 0:
        return None
    threshold = max(history_avg * SPIKE_MULTIPLIER, history_avg + SPIKE_MIN_EXCESS)
    if current_external <= threshold:
        return None
    return _finding(
        "conn_spike",
        MEDIUM,
        f"Connection spike: {current_external} external connections",
        (f"This PC normally holds about {history_avg:.0f} concurrent external connections "
         f"(last {history_days} days); right now it has {current_external}. "
         "Could be a big download or sync — or data exfiltration."),
        "(machine-wide)", "machine-wide",  # constant remote so dedupe holds across ticks
    )

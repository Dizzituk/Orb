# FILE: sentinel_agent/connections.py
# Purpose: psutil connection enumeration for the Sentinel agent — one snapshot() call
#          returns every inet connection (incl. LISTEN) with owning-process metadata.
# Called-by: sentinel_agent.main (/connections endpoint)
# Depends-on: psutil
# Last-renovated: 2026-06-12
from __future__ import annotations

import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

# pid -> (name, exe, cached_at). Process identity is stable enough over our TTL;
# a recycled pid self-corrects within a minute.
_PROC_CACHE: Dict[int, Tuple[str, str, float]] = {}
_PROC_CACHE_TTL_SECONDS = 60.0

_PROTO_BY_TYPE = {
    socket.SOCK_STREAM: "tcp",
    socket.SOCK_DGRAM: "udp",
}


def _process_info(pid: Optional[int]) -> Tuple[str, str]:
    """Resolve (process_name, exe_path) for a pid, tolerating exits and access denials."""
    if not pid:  # pid 0 / None = kernel or unowned socket
        return ("system", "")
    now = time.monotonic()
    cached = _PROC_CACHE.get(pid)
    if cached and (now - cached[2]) < _PROC_CACHE_TTL_SECONDS:
        return (cached[0], cached[1])
    name, exe = "unknown", ""
    try:
        proc = psutil.Process(pid)
        name = proc.name() or "unknown"
        try:
            exe = proc.exe() or ""
        except (psutil.AccessDenied, psutil.ZombieProcess, OSError):
            exe = ""
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        pass
    _PROC_CACHE[pid] = (name, exe, now)
    return (name, exe)


def _addr(addr: Any) -> Tuple[str, Optional[int]]:
    """Normalise a psutil addr (namedtuple or empty tuple) into (ip, port)."""
    try:
        if addr and getattr(addr, "ip", None) is not None:
            return (str(addr.ip), int(addr.port))
    except (ValueError, AttributeError):
        pass
    return ("", None)


def snapshot() -> List[Dict[str, Any]]:
    """Every inet connection on the machine, including listening sockets.

    Shape per item: {pid, process_name, exe_path, laddr_ip, laddr_port,
    raddr_ip, raddr_port, status, proto}. Requires elevation for full
    per-process visibility; unelevated it degrades to what's accessible.
    """
    out: List[Dict[str, Any]] = []
    for conn in psutil.net_connections(kind="inet"):
        name, exe = _process_info(conn.pid)
        laddr_ip, laddr_port = _addr(conn.laddr)
        raddr_ip, raddr_port = _addr(conn.raddr)
        out.append(
            {
                "pid": conn.pid or 0,
                "process_name": name,
                "exe_path": exe,
                "laddr_ip": laddr_ip,
                "laddr_port": laddr_port,
                "raddr_ip": raddr_ip,
                "raddr_port": raddr_port,
                "status": conn.status or "NONE",
                "proto": _PROTO_BY_TYPE.get(conn.type, str(conn.type)),
            }
        )
    return out

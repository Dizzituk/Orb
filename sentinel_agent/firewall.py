# FILE: sentinel_agent/firewall.py
# Purpose: Windows Firewall wrappers for the Sentinel agent — add/remove/list
#          ASTRA-Sentinel-Block-* rules via netsh. Block needs elevation.
# Called-by: sentinel_agent.main (/block, /unblock, /rules endpoints)
# Depends-on: netsh advfirewall (Windows built-in); stdlib only
# Last-renovated: 2026-06-12
from __future__ import annotations

import ipaddress
import logging
import subprocess
from typing import Any, Dict, List

logger = logging.getLogger("sentinel_agent.firewall")

RULE_PREFIX = "ASTRA-Sentinel-Block-"
_NETSH_TIMEOUT_SECONDS = 30


class InvalidIpError(ValueError):
    pass


def _validate_ip(ip: str) -> str:
    """Strict IP-literal validation — the only thing ever interpolated into netsh args."""
    try:
        parsed = ipaddress.ip_address(str(ip).strip())
    except ValueError as exc:
        raise InvalidIpError(f"not a valid IP literal: {ip!r}") from exc
    if parsed.is_loopback or parsed.is_unspecified:
        raise InvalidIpError(f"refusing to manage firewall rules for {parsed}")
    return str(parsed)


def _run_netsh(args: List[str]) -> subprocess.CompletedProcess:
    cmd = ["netsh", "advfirewall", "firewall"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_NETSH_TIMEOUT_SECONDS,
        shell=False,
    )


def block_ip(ip: str) -> Dict[str, Any]:
    """Create inbound + outbound block rules named ASTRA-Sentinel-Block-{ip}."""
    clean = _validate_ip(ip)
    name = f"{RULE_PREFIX}{clean}"
    created, errors = [], []
    for direction in ("in", "out"):
        cp = _run_netsh(
            [
                "add", "rule",
                f"name={name}",
                f"dir={direction}",
                "action=block",
                f"remoteip={clean}",
                "enable=yes",
                "profile=any",
            ]
        )
        if cp.returncode == 0:
            created.append(direction)
        else:
            errors.append(f"{direction}: {(cp.stderr or cp.stdout or '').strip()[:300]}")
    ok = len(created) == 2
    if not ok:
        logger.error("block_ip(%s) partial/failed: %s", clean, errors)
        if created:  # don't leave a half-applied block behind
            _run_netsh(["delete", "rule", f"name={name}"])
    return {"ok": ok, "ip": clean, "rule_name": name, "created": created, "errors": errors}


def unblock_ip(ip: str) -> Dict[str, Any]:
    """Delete every rule (both directions) named ASTRA-Sentinel-Block-{ip}."""
    clean = _validate_ip(ip)
    name = f"{RULE_PREFIX}{clean}"
    cp = _run_netsh(["delete", "rule", f"name={name}"])
    detail = (cp.stdout or cp.stderr or "").strip()
    # netsh exits non-zero when no rule matches — report that as ok=true, removed=0.
    no_match = "No rules match" in detail
    return {
        "ok": cp.returncode == 0 or no_match,
        "ip": clean,
        "rule_name": name,
        "removed": 0 if no_match else (2 if cp.returncode == 0 else 0),
        "detail": detail[:300],
    }


def list_rules() -> List[Dict[str, Any]]:
    """All ASTRA-Sentinel-* rules currently in the firewall (parsed from netsh output)."""
    cp = _run_netsh(["show", "rule", "name=all"])
    rules: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for raw in (cp.stdout or "").splitlines():
        line = raw.strip()
        if line.startswith("Rule Name:"):
            if current.get("name", "").startswith(RULE_PREFIX):
                rules.append(current)
            current = {"name": line.split(":", 1)[1].strip()}
        elif ":" in line and current:
            key, value = (part.strip() for part in line.split(":", 1))
            if key == "Enabled":
                current["enabled"] = value
            elif key == "Direction":
                current["direction"] = value
            elif key == "Action":
                current["action"] = value
            elif key == "RemoteIP":
                current["remote_ip"] = value
    if current.get("name", "").startswith(RULE_PREFIX):
        rules.append(current)
    return rules

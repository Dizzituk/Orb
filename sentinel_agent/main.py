# FILE: sentinel_agent/main.py
# Purpose: ASTRA Sentinel elevated agent — localhost-only FastAPI giving the Orb backend
#          connection snapshots and ask-first firewall blocking. Runs OUTSIDE the backend
#          so the backend itself never holds admin rights.
# Called-by: Scheduled Task "AstraSentinelAgent" (scripts\install_sentinel_agent.ps1); app.sentinel.agent_client over 127.0.0.1
# Depends-on: sentinel_agent.connections, sentinel_agent.firewall, psutil, fastapi, uvicorn
# Last-renovated: 2026-06-12
from __future__ import annotations

import ctypes
import hmac
import logging
import os
import secrets as _secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

# Standalone process: when launched as a script, our own folder is sys.path[0].
try:
    import connections as conn_mod
    import firewall as fw_mod
except ImportError:  # pragma: no cover - allows `python -m sentinel_agent.main` from D:\Orb
    from sentinel_agent import connections as conn_mod  # type: ignore
    from sentinel_agent import firewall as fw_mod  # type: ignore

VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ASTRA_SENTINEL_DATA_DIR", REPO_ROOT / "data" / "sentinel"))
SECRET_FILE = Path(os.environ.get("ASTRA_SENTINEL_SECRET_FILE", DATA_DIR / "agent_secret"))
PORT = int(os.environ.get("ASTRA_SENTINEL_AGENT_PORT", "8771"))

_STARTED_MONO = time.monotonic()
_STARTED_AT = datetime.now(timezone.utc).isoformat()

logger = logging.getLogger("sentinel_agent")


def _setup_logging() -> None:
    """Warnings+ to data\\sentinel\\agent.log and stderr. The secret is never logged."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in (logging.FileHandler(DATA_DIR / "agent.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stderr)):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def _load_or_create_secret() -> str:
    """Shared secret for the backend. First-to-run creates it; installer does the same."""
    if SECRET_FILE.exists():
        value = SECRET_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    value = _secrets.token_hex(32)
    tmp = SECRET_FILE.with_suffix(".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(SECRET_FILE)
    logger.warning("generated new agent secret at %s", SECRET_FILE)
    return value


def _is_elevated() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


SECRET = ""

app = FastAPI(title="ASTRA Sentinel Agent", version=VERSION, docs_url=None, redoc_url=None)


def require_secret(x_sentinel_secret: str = Header(default="")) -> None:
    if not SECRET or not hmac.compare_digest(x_sentinel_secret, SECRET):
        raise HTTPException(status_code=403, detail="bad or missing X-Sentinel-Secret")


class IpBody(BaseModel):
    ip: str


@app.get("/health")
def health() -> dict:
    """Open endpoint (localhost-only bind) so commissioning can be checked in a browser."""
    return {
        "service": "astra-sentinel-agent",
        "version": VERSION,
        "elevated": _is_elevated(),
        "started_at": _STARTED_AT,
        "uptime_seconds": round(time.monotonic() - _STARTED_MONO, 1),
        "port": PORT,
        "pid": os.getpid(),
    }


@app.get("/connections", dependencies=[Depends(require_secret)])
def get_connections() -> dict:
    try:
        items = conn_mod.snapshot()
    except Exception as exc:
        logger.exception("connection snapshot failed")
        raise HTTPException(status_code=500, detail=f"snapshot failed: {exc}")
    return {"ts": datetime.now(timezone.utc).isoformat(), "count": len(items), "connections": items}


@app.post("/block", dependencies=[Depends(require_secret)])
def post_block(body: IpBody) -> dict:
    if not _is_elevated():
        raise HTTPException(status_code=503, detail="agent is not elevated; cannot modify firewall")
    try:
        result = fw_mod.block_ip(body.ip)
    except fw_mod.InvalidIpError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result)
    logger.warning("BLOCKED %s (rule %s)", result["ip"], result["rule_name"])
    return result


@app.post("/unblock", dependencies=[Depends(require_secret)])
def post_unblock(body: IpBody) -> dict:
    if not _is_elevated():
        raise HTTPException(status_code=503, detail="agent is not elevated; cannot modify firewall")
    try:
        result = fw_mod.unblock_ip(body.ip)
    except fw_mod.InvalidIpError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    logger.warning("UNBLOCKED %s (ok=%s)", result["ip"], result["ok"])
    return result


@app.get("/rules", dependencies=[Depends(require_secret)])
def get_rules() -> dict:
    try:
        rules = fw_mod.list_rules()
    except Exception as exc:
        logger.exception("rule listing failed")
        raise HTTPException(status_code=500, detail=f"rule listing failed: {exc}")
    return {"count": len(rules), "rules": rules}


def main() -> None:
    global SECRET
    _setup_logging()
    SECRET = _load_or_create_secret()
    import uvicorn

    # 127.0.0.1 ONLY — never expose this process beyond the machine.
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()

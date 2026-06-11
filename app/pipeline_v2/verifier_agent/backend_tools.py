# FILE: app/pipeline_v2/verifier_agent/backend_tools.py
"""
PROVING RUN (2026-06-10) - Backend driving tools for the agentic verifier.

The verifier's Android hands are ADB (perception_tools). For SELF-BUILDS the
app under test is ASTRA's own backend running in the SANDBOX CLONE, so the
hands are different: boot the clone's uvicorn, fire HTTP requests at it from
INSIDE the VM (the clone's 127.0.0.1 is unreachable from the host), and read
its logs. Every tool here is a thin wrapper that generates a PowerShell
script and executes it through sandbox_tools.run_shell with the build
profile - which for self-build targets routes into the VM agent. No new
network plumbing.

CLONE KEY NOTE: the live backend refuses to start without ORB_MASTER_KEY
(app/crypto/encryption.py exits). The clone is started with a deliberately
WORTHLESS, publicly-known throwaway key (all-zero bytes) - it boots, its
encrypted store decrypts to placeholders, and key-dependent features degrade
gracefully. That is by design: the clone holds no live secrets, and a
feature that needs a real key reports as blocked-by-environment rather than
falsely failing. Never put the real master key anywhere near this module.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# All-zero 32-byte key, urlsafe base64. Valid shape, protects nothing,
# decrypts nothing. THE POINT is that it is not a secret.
THROWAWAY_CLONE_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

CLONE_OUT_LOG = "D:\\Orb\\_sandbox_cache\\clone_backend.out.log"
CLONE_ERR_LOG = "D:\\Orb\\_sandbox_cache\\clone_backend.err.log"

_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}


BACKEND_TOOLS: List[Dict] = [
    {"type": "function", "function": {
        "name": "backend_start",
        "description": (
            "Boot the CLONE's backend (uvicorn) inside the sandbox VM and wait "
            "for /ping to respond. Uses the clone's own venv and a throwaway "
            "master key (real secrets never enter the clone). Returns READY or "
            "the startup error log."
        ),
        "parameters": {"type": "object", "properties": {
            "port": {"type": "integer", "description": "Port for the clone backend (default 8000)"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "backend_stop",
        "description": "Stop the clone backend started by backend_start (never touches the sandbox agent itself).",
        "parameters": {"type": "object", "properties": {
            "port": {"type": "integer", "description": "Port the clone backend runs on (default 8000)"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "http_request",
        "description": (
            "Send an HTTP request to the CLONE backend from inside the VM. "
            "Returns status code and response body (truncated). This is your "
            "primary way to test backend features - like curl in human hands."
        ),
        "parameters": {"type": "object", "properties": {
            "method": {"type": "string", "description": "GET, POST, PUT, DELETE or PATCH"},
            "path": {"type": "string", "description": "Request path, e.g. /system/uptime"},
            "body": {"type": "string", "description": "Optional JSON body for POST/PUT/PATCH"},
            "port": {"type": "integer", "description": "Default 8000"},
        }, "required": ["method", "path"]},
    }},
    {"type": "function", "function": {
        "name": "backend_logs",
        "description": "Tail the clone backend's stdout/stderr logs (startup errors, tracebacks, request lines).",
        "parameters": {"type": "object", "properties": {
            "lines": {"type": "integer", "description": "How many lines from each log (default 60)"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "wait",
        "description": "Wait N seconds (max 15) for the backend to settle.",
        "parameters": {"type": "object", "properties": {
            "seconds": {"type": "number"},
        }, "required": ["seconds"]},
    }},
]


def _ps_quote(s: str) -> str:
    """Single-quote a string for PowerShell (doubling embedded quotes)."""
    return "'" + (s or "").replace("'", "''") + "'"


def _start_script(port: int) -> str:
    return (
        "$ErrorActionPreference='SilentlyContinue'; "
        "New-Item -ItemType Directory -Path 'D:\\Orb\\_sandbox_cache' -Force | Out-Null; "
        f"$env:ORB_MASTER_KEY='{THROWAWAY_CLONE_KEY}'; "
        "$py = if (Test-Path 'D:\\Orb\\.venv\\Scripts\\python.exe') "
        "{ 'D:\\Orb\\.venv\\Scripts\\python.exe' } else { 'python' }; "
        f"$existing = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*uvicorn*' -and $_.CommandLine -like '*{port}*' "
        f"-and $_.CommandLine -notlike '*8765*' }}; "
        "if ($existing) { Write-Output ('ALREADY_RUNNING pid=' + ($existing[0].ProcessId)) } else { "
        f"Start-Process -FilePath $py -ArgumentList '-m','uvicorn','main:app','--host','127.0.0.1','--port','{port}' "
        "-WorkingDirectory 'D:\\Orb' -WindowStyle Hidden "
        f"-RedirectStandardOutput '{CLONE_OUT_LOG}' -RedirectStandardError '{CLONE_ERR_LOG}'; "
        "Write-Output 'SPAWNED' }; "
        "$ready = $false; "
        "for ($i = 0; $i -lt 20; $i++) { "
        "  Start-Sleep -Seconds 1; "
        f"  try {{ $r = Invoke-WebRequest -Uri 'http://127.0.0.1:{port}/ping' -TimeoutSec 2 -UseBasicParsing; "
        "    if ($r.StatusCode -eq 200) { $ready = $true; Write-Output ('READY after ' + ($i+1) + 's'); break } } catch {} "
        "}; "
        "if (-not $ready) { Write-Output 'BOOT_TIMEOUT after 20s - recent stderr:'; "
        f"Get-Content '{CLONE_ERR_LOG}' -Tail 25 -ErrorAction SilentlyContinue }}"
    )


def _stop_script(port: int) -> str:
    return (
        f"$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*uvicorn*' -and $_.CommandLine -like '*{port}*' "
        f"-and $_.CommandLine -notlike '*8765*' }}; "
        "if ($procs) { $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; "
        "Write-Output ('STOPPED ' + @($procs).Count + ' process(es)') } "
        "else { Write-Output 'NOT_RUNNING' }"
    )


def _http_script(method: str, path: str, body: str, port: int) -> str:
    url = f"http://127.0.0.1:{port}{path}"
    parts = [
        "try { ",
        f"$r = Invoke-WebRequest -Uri {_ps_quote(url)} -Method {method} -TimeoutSec 20 -UseBasicParsing",
    ]
    if body and method in ("POST", "PUT", "PATCH"):
        parts.append(f" -Body {_ps_quote(body)} -ContentType 'application/json'")
    parts.append(
        "; Write-Output ('STATUS ' + [int]$r.StatusCode); "
        "Write-Output ($r.Content.Substring(0, [Math]::Min(1800, $r.Content.Length))) "
        "} catch { "
        "$code = try { [int]$_.Exception.Response.StatusCode } catch { 0 }; "
        "Write-Output ('STATUS ' + $code); "
        "$rs = try { (New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())).ReadToEnd() } catch { $_.Exception.Message }; "
        "Write-Output ($rs.Substring(0, [Math]::Min(1200, $rs.Length))) }"
    )
    return "".join(parts)


def _logs_script(lines: int) -> str:
    return (
        f"Write-Output '--- stdout (last {lines}) ---'; "
        f"Get-Content '{CLONE_OUT_LOG}' -Tail {lines} -ErrorAction SilentlyContinue; "
        f"Write-Output '--- stderr (last {lines}) ---'; "
        f"Get-Content '{CLONE_ERR_LOG}' -Tail {lines} -ErrorAction SilentlyContinue"
    )


async def execute_backend_tool(name: str, args: Dict[str, Any], profile: Any) -> str:
    """Execute one backend-driving tool through the sandbox shell. Never raises."""
    from app.pipeline_v2.sandbox_tools import run_shell

    try:
        if name == "wait":
            secs = min(float(args.get("seconds", 1)), 15.0)
            await asyncio.sleep(secs)
            return f"OK: waited {secs:.1f}s"

        port = int(args.get("port", 8000))
        if port == 8765:
            return "ERROR: port 8765 is the sandbox agent itself - off limits"

        if name == "backend_start":
            script, timeout = _start_script(port), 45
        elif name == "backend_stop":
            script, timeout = _stop_script(port), 20
        elif name == "http_request":
            method = str(args.get("method", "GET")).upper().strip()
            if method not in _ALLOWED_METHODS:
                return f"ERROR: method {method} not allowed"
            path = str(args.get("path", "/"))
            if not path.startswith("/"):
                path = "/" + path
            script, timeout = _http_script(method, path, str(args.get("body", "") or ""), port), 30
        elif name == "backend_logs":
            script, timeout = _logs_script(max(5, min(int(args.get("lines", 60)), 200))), 20
        else:
            return f"ERROR: unknown backend tool: {name}"

        res = await run_shell(script, timeout_sec=timeout, profile=profile)
        if not isinstance(res, dict):
            return "ERROR: sandbox shell returned nothing (VM down?)"
        out = (res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        if not out and err:
            return f"ERROR (stderr): {err[:800]}"
        return (out[:2500] or "(no output)") + (f"\n[stderr: {err[:300]}]" if err else "")

    except Exception as exc:
        logger.warning("[backend_tools] %s crashed: %s", name, exc)
        return f"ERROR: {name} crashed: {exc}"

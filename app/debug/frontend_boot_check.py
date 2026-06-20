# FILE: app/debug/frontend_boot_check.py
# Purpose: Read the SANDBOX CLONE's Electron/Vite frontend boot health as TEXT (not via
#          a screenshot). The Vite "Failed to resolve import @univerjs/presets" overlay
#          never reaches the backend astra.log and there is no frontend log file, so the
#          only text-reliable signal is: are the declared frontend deps actually installed
#          in the clone's node_modules, and is the Vite dev server (:5173) responding?
#          This is the primary frontend-error channel; the screenshot/OCR is the backup
#          (and it gets blocked by AV in the sandbox, which is exactly why this exists).
# Called-by: app.debug.sandbox_boot_tool.inspect_sandbox_boot
# Depends-on: app.debug.sandbox_console.visible_shell_run (Job 02 visible console),
#             app.sandbox.client.get_sandbox_client (call only -- PROTECTED, unmodified)
# Last-renovated: 2026-06-17 (created -- Job 03d frontend text probe)
"""Frontend boot text probe.

orb-desktop boots via `npm run electron:dev` = `concurrently "vite" "wait-on
http://localhost:5173 && electron ."`, so the renderer is served by Vite on :5173. When
an import like `@univerjs/presets` can't be resolved, Vite shows it as an OVERLAY in the
renderer (a screenshot, which AV blocks) -- it is NOT written to astra.log. But the root
cause IS text-checkable: the dependency is declared in package.json yet missing from the
clone's node_modules. This module checks exactly that (plus dev-server reachability) in
the clone, as TEXT, and emits a line the boot failure classifier maps to `npm install`.

Best-effort: never raises; returns a readable report string in all paths.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# orb-desktop's Vite dev server port (electron:dev waits on http://localhost:5173).
_DEV_PORT = 5173


def _frontend_dir_from_controller(client) -> Tuple[str, str]:
    """(frontend_dir, error). Derived from the controller's REPO_ROOT sibling
    'orb-desktop' -- no hardcoded drive. error non-empty => could not resolve."""
    try:
        repo_root = (getattr(client.health(), "repo_root", "") or "").rstrip("\\/")
    except Exception as e:  # pragma: no cover - needs a live controller
        return "", f"sandbox controller unreachable ({e})"
    if not repo_root:
        return "", "controller did not report REPO_ROOT"
    return os.path.join(os.path.dirname(repo_root), "orb-desktop"), ""


def _build_probe_snippet(frontend_dir: str, dev_port: int) -> str:
    """PowerShell that, in the clone frontend dir, checks node_modules + that every
    declared dependency is installed, then pings the Vite dev server. Emits one or more
    ``FE_PROBE ...`` status lines (parsed on the host). No interop / no DllImport -- AMSI-safe."""
    fe = (frontend_dir or "").replace("'", "''")
    snippet = r"""$ErrorActionPreference='Continue'
$root='__FE__'
$pkgPath = Join-Path $root 'package.json'
$nm = Join-Path $root 'node_modules'
if (-not (Test-Path -LiteralPath $pkgPath)) {
  Write-Output 'FE_PROBE nopkg'
} else {
  if (-not (Test-Path -LiteralPath $nm)) {
    Write-Output 'FE_PROBE nm_missing'
  } else {
    try {
      $pkg = Get-Content -LiteralPath $pkgPath -Raw | ConvertFrom-Json
      $deps = @()
      if ($pkg.dependencies) { $deps += $pkg.dependencies.PSObject.Properties.Name }
      if ($pkg.devDependencies) { $deps += $pkg.devDependencies.PSObject.Properties.Name }
      $missing = @()
      foreach ($d in $deps) { if ($d -and -not (Test-Path -LiteralPath (Join-Path $nm $d))) { $missing += $d } }
      if ($missing.Count -gt 0) { Write-Output ('FE_PROBE missing ' + ($missing -join ',')) } else { Write-Output 'FE_PROBE deps_ok' }
    } catch { Write-Output ('FE_PROBE pkgerr ' + $_.Exception.Message) }
  }
  try { $r = Invoke-WebRequest -Uri 'http://localhost:__PORT__' -TimeoutSec 4 -UseBasicParsing; Write-Output ('FE_PROBE dev ' + [int]$r.StatusCode) } catch { Write-Output 'FE_PROBE dev down' }
}
"""
    return snippet.replace("__FE__", fe).replace("__PORT__", str(int(dev_port)))


def _format(out: str, frontend_dir: str) -> str:
    lines = [l.strip() for l in (out or "").splitlines() if l.strip().startswith("FE_PROBE")]
    head = (
        f"FRONTEND (text probe of the clone's orb-desktop at {frontend_dir} -- read as "
        f"TEXT, no screenshot):"
    )
    if not lines:
        return head + "\n  (no probe output -- could not read the frontend in the clone)"

    report = [head]
    missing = []
    dev = "unknown"
    for l in lines:
        body = l[len("FE_PROBE"):].strip()
        if body == "nopkg":
            return head + (
                f"\n  no package.json at {frontend_dir} -- could not locate the Electron "
                "frontend in the clone."
            )
        if body == "nm_missing":
            report.append("  node_modules: MISSING")
            missing = ["(node_modules)"]
        elif body == "deps_ok":
            report.append("  declared dependencies: all present in node_modules")
        elif body.startswith("missing "):
            missing = [m for m in body[len("missing "):].split(",") if m]
            report.append("  declared dependencies NOT installed: " + ", ".join(missing))
        elif body.startswith("pkgerr"):
            report.append("  package.json read error: " + body[len("pkgerr"):].strip())
        elif body.startswith("dev "):
            v = body[len("dev "):].strip()
            dev = f"UP (HTTP {v})" if v.isdigit() else "DOWN"

    report.append(f"  Vite dev server (:{_DEV_PORT}): {dev}")

    # Emit a line the boot failure classifier maps to a remediation: a real declared
    # package missing -> "Failed to resolve import" (-> npm install); only node_modules
    # missing -> a generic "Cannot find module".
    real_missing = [m for m in missing if m and m != "(node_modules)"]
    if real_missing:
        report.append(
            f'  -> Failed to resolve import "{real_missing[0]}" -- declared in package.json '
            "but missing from the clone's node_modules (npm install needed)."
        )
    elif missing:  # only node_modules missing
        report.append("  -> Cannot find module: the clone's node_modules is missing (npm install needed).")
    elif dev == "DOWN":
        report.append("  -> the Vite dev server is not responding -- the frontend may not have started.")

    return "\n".join(report)


def probe_frontend_boot(client=None) -> str:
    """Text readout of the clone's Electron/Vite frontend boot health. Sync (the boot
    inspector runs it inline). Never raises -- all failures returned as readable text."""
    try:
        if client is None:
            from app.sandbox.client import get_sandbox_client
            client = get_sandbox_client()
    except Exception as e:
        return f"FRONTEND (text probe): sandbox unavailable ({e})."

    frontend_dir, err = _frontend_dir_from_controller(client)
    if err:
        return f"FRONTEND (text probe): {err}."

    try:
        from app.debug.sandbox_console import visible_shell_run
        res = visible_shell_run(
            _build_probe_snippet(frontend_dir, _DEV_PORT),
            cwd_target="REPO", timeout_seconds=40, client=client,
        )
        out = getattr(res, "stdout", "") or ""
    except Exception as e:
        return f"FRONTEND (text probe): probe command failed in the clone ({e})."

    return _format(out, frontend_dir)


__all__ = ["probe_frontend_boot"]

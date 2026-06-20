# FILE: app/debug/sandbox_console.py
# Purpose: Visible, LIVE-STREAMING equivalent of SandboxClient.shell_run. Runs a
#          command in the sandbox CLONE inside a thin wrapper that streams the
#          command + its output, line by line, into a persistent live-tail
#          transcript window ON the sandbox's interactive desktop -- so Taz watches
#          it happen in real time -- while still returning the full output + exit
#          code to the bridge. A non-protected wrapper that only CALLS the
#          (protected) SandboxClient -- it does NOT modify app/sandbox or controller.
# Called-by: sandbox self-heal paths -- app.debug.sandbox_screenshot (WI-3 capture),
#            app.debug.sandbox_boot_tool (WI-4 clone-log read + npm-install fix)
# Depends-on: app.sandbox.client.get_sandbox_client (call only -- PROTECTED, unmodified)
# Last-renovated: 2026-06-17 (live-streaming seal -- docs/debug-visibility Part A)
"""Visible sandbox console (live streaming).

The sandbox controller's /shell/run runs every command with capture_output (piped,
windowless) -- that single choice is why the work is invisible. We do NOT change
the controller. Instead:

  * ONE persistent visible PowerShell window is launched INSIDE the sandbox's
    interactive desktop session (verified interactive -- see docs/debug-visibility/
    AUDIT-2026-06-17.md), tailing a UTF-8 transcript with ``Get-Content -Wait``.
  * Each command runs through a wrapper that (a) writes the command line to the
    transcript BEFORE it runs (so it appears in the window immediately), (b) fans
    every output line out to the transcript AND passes it through to stdout, so the
    output streams into the visible window live while the bridge still captures it,
    and (c) preserves the command's exit code (``$LASTEXITCODE`` -> ``exit``) so the
    returned ShellResult.exit_code stays correct.

Why not Tee-Object / Start-Transcript: in Windows PowerShell 5.1 (what the sandbox
runs) Tee-Object has no -Encoding and emits UTF-16, which clashes with the UTF-8
transcript; Start-Transcript adds a noisy banner on every call. A per-line
ForEach-Object fan-out with explicit ``-Encoding UTF8`` is clean and 5.1-safe.

The model still receives the full ShellResult unchanged (exit code preserved;
stdout carries the merged out+err stream). Best-effort by design: a window/console
failure must never change the command's result -- if the transcript can't be
written the command still runs, its output is still captured and its exit code
still returned; only the visibility is lost.
"""
from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)

_SBX_LOG = r"C:\Orb\astra_console.log"

# Attempt the visible window at most ONCE per backend process. Set True UP FRONT (not
# only on success): the controller can be slow to detach a `cmd /c start`, so a failed
# launch used to time out (~25s) and re-spawn a window on EVERY command -- piling up
# windows and stalling each inspect. The per-command wrapper streams output to the
# transcript regardless of whether the window itself came up, so losing the window only
# costs live visibility, never the command result.
_window_tried = False


def _ensure_window(client) -> None:
    """Try ONCE per backend process to open a visible tail window in the sandbox.
    Best-effort; never retried (see the _window_tried note above)."""
    global _window_tried
    if _window_tried:
        return
    _window_tried = True
    try:
        # Ensure the transcript exists; APPEND a session header (not Set-Content) so a
        # leftover `-Wait` tail window holding the file can't block us with a write lock.
        client.shell_run(
            "if (-not (Test-Path -LiteralPath '" + _SBX_LOG + "')) { "
            "New-Item -ItemType File -Force -Path '" + _SBX_LOG + "' | Out-Null }; "
            "Add-Content -LiteralPath '" + _SBX_LOG + "' -Encoding UTF8 -Value "
            "('=== ASTRA sandbox console -- ' + (Get-Date).ToString('u') + ' ===')",
            timeout_seconds=8,
        )
        # Launch a detached visible tail window. The window TITLE comes from `start`'s
        # own title argument -- we do NOT set $Host.UI.RawUI.WindowTitle: that inner form
        # got mangled through the powershell->cmd->powershell quoting layers and errored
        # with CommandNotFoundException (live 2026-06-18), so it is dropped entirely.
        client.shell_run(
            'cmd /c start "ASTRA sandbox console (live)" powershell -NoProfile -NoExit '
            "-Command \"Get-Content -LiteralPath '" + _SBX_LOG + "' -Tail 40 -Wait\"",
            timeout_seconds=8,
        )
    except Exception as e:  # pragma: no cover - needs a live sandbox
        logger.warning("[sandbox_console] visible console unavailable (continuing without it): %s", e)


def _build_stream_wrapper(command: str) -> str:
    """PowerShell wrapper that runs ``command`` in the sandbox while streaming the
    command + its output into the visible transcript and preserving the exit code.

    The original command is base64-embedded (quoting-proof) and run via
    Invoke-Expression inside a child scope; every output line (incl. errors, merged
    with 2>&1) is appended to the UTF-8 transcript AND passed through to stdout, so
    the controller still captures the full output. The exit code is propagated with
    ``exit $code`` so ShellResult.exit_code matches the wrapped command.
    """
    b64 = base64.b64encode(command.encode("utf-8")).decode("ascii")
    wrapper = r"""$ErrorActionPreference='Continue'
$LOG='__LOG__'
$cmd=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__B64__'))
$ts=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
Add-Content -LiteralPath $LOG -Encoding UTF8 -Value ''
Add-Content -LiteralPath $LOG -Encoding UTF8 -Value ('=' * 72)
Add-Content -LiteralPath $LOG -Encoding UTF8 -Value ('[' + $ts + ']  SANDBOX PS> ' + $cmd)
Add-Content -LiteralPath $LOG -Encoding UTF8 -Value ('-' * 72)
& { Invoke-Expression $cmd } 2>&1 | ForEach-Object {
  $line = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { [string]$_ }
  Add-Content -LiteralPath $LOG -Encoding UTF8 -Value $line
  $_
}
$code = $LASTEXITCODE
$ErrorActionPreference='Continue'
if ($null -eq $code) { $code = 0 }
Add-Content -LiteralPath $LOG -Encoding UTF8 -Value ('Exit code: ' + $code)
exit $code
"""
    return wrapper.replace("__LOG__", _SBX_LOG).replace("__B64__", b64)


def visible_shell_run(command: str, cwd_target: str = "REPO",
                      timeout_seconds: int = 60, client=None):
    """Run ``command`` in the sandbox clone, VISIBLY and LIVE. Returns the
    ShellResult from SandboxClient.shell_run (exit code preserved; stdout carries the
    merged out+err stream). The visible console is best-effort.

    ``client`` is injectable for testing; it defaults to the live sandbox client.
    """
    if client is None:
        from app.sandbox.client import get_sandbox_client
        client = get_sandbox_client()

    _ensure_window(client)
    wrapper = _build_stream_wrapper(command)
    return client.shell_run(wrapper, cwd_target=cwd_target,
                            timeout_seconds=timeout_seconds)


__all__ = ["visible_shell_run"]

# FILE: app/pipeline_v2/verifier_agent/host_perception.py
# Purpose: Derek phase 6 — host-lane perception for greenfield desktop targets (launch, screenshot, windows, logs).
# Called-by: app.pipeline_v2.verifier_agent.eyes_judge
# Depends-on: stdlib only (PowerShell via subprocess)
# Last-renovated: 2026-07-04
"""
Perception primitives for apps built on the HOST (greenfield lane).

The Android lane has ADB perception (perception_tools.py); self-builds have
clone backend tools (backend_tools.py). Greenfield desktop apps had NOTHING
— this module gives the eyes hands-free perception on the host:

    launch_app        spawn the built app (non-blocking), tail its output
    capture_screen    PowerShell CopyFromScreen -> PNG in the evidence dir
    window_titles     visible top-level windows (is the app's window up?)
    send_keys         focus the spawned app by PID and inject keystrokes
    stop_app          terminate the spawned process tree

live11 (2026-07-05): host input ENABLED — Taz made the trust call
explicitly ("literally playing the game... physically hitting the
buttons"). Input only ever targets the PID the eyes themselves spawned,
via AppActivate — never arbitrary windows. Kill switch: ASTRA_EYES_INPUT=0.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def guess_entrypoint(project_root: str) -> Optional[List[str]]:
    """Best-effort entry command for a greenfield python app.

    Paths are ABSOLUTE — launch_app runs with cwd=project_root, so a
    relative script path would resolve against itself and double up.

    live12 (2026-07-05): the planner's standard layout puts the entry at
    src/main.py — the first full E2E run judged FAIL solely because only
    root-level names were checked. Search order: root names, src/ names,
    then a single-.py fallback in root then src/.
    """
    import sys
    root = Path(project_root).resolve()
    search_dirs = [root, root / "src"]
    for d in search_dirs:
        for name in ("main.py", "app.py", "game.py", "cli.py", "run.py"):
            if (d / name).is_file():
                return [sys.executable, str((d / name).resolve())]
    for d in search_dirs:
        if d.is_dir():
            candidates = sorted(d.glob("*.py"))
            if len(candidates) == 1:
                return [sys.executable, str(candidates[0].resolve())]
    return None


async def launch_app(
    project_root: str,
    cmd: Optional[List[str]] = None,
    settle_seconds: float = 4.0,
) -> Dict[str, Any]:
    """Spawn the app on the host, wait for it to settle, report its state.

    Returns {launched, pid, running, returncode, stdout, stderr, cmd}.
    A fast non-zero exit with a traceback is exactly the evidence the
    judge needs — capture it, never mask it.
    """
    cmd = cmd or guess_entrypoint(project_root)
    if not cmd:
        return {"launched": False, "error": f"no entrypoint found in {project_root}"}
    try:
        proc = subprocess.Popen(
            cmd, cwd=project_root,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except Exception as exc:
        return {"launched": False, "error": f"spawn failed: {exc}", "cmd": cmd}

    try:
        await asyncio.sleep(settle_seconds)
    except BaseException:
        # p7 review fix: never leak the spawned process if the settle wait
        # is cancelled — kill before propagating.
        try:
            proc.kill()
        except Exception:
            pass
        raise
    running = proc.poll() is None
    stdout = stderr = ""
    if not running:
        try:
            out_b, err_b = proc.communicate(timeout=5)
            stdout = (out_b or b"").decode("utf-8", errors="replace")[-3000:]
            stderr = (err_b or b"").decode("utf-8", errors="replace")[-3000:]
        except Exception:
            pass
    return {
        "launched": True, "pid": proc.pid, "running": running,
        "returncode": None if running else proc.returncode,
        "stdout": stdout, "stderr": stderr, "cmd": cmd, "_proc": proc,
    }


def collect_output(launch_info: Dict[str, Any]) -> Dict[str, str]:
    """Drain stdout/stderr tails from an exited process (best-effort)."""
    proc = launch_info.get("_proc")
    if proc is None:
        return {"stdout": "", "stderr": ""}
    try:
        out_b, err_b = proc.communicate(timeout=5)
        return {
            "stdout": (out_b or b"").decode("utf-8", errors="replace")[-3000:],
            "stderr": (err_b or b"").decode("utf-8", errors="replace")[-3000:],
        }
    except Exception:
        return {"stdout": "", "stderr": ""}


async def _window_title_for_pid(pid: int) -> str:
    """The process's main window title ('' when it has no window yet)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command",
            f"(Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue).MainWindowTitle",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return (stdout or b"").decode("utf-8", errors="replace").strip()
    except Exception as exc:
        logger.debug("[host_perception] window title probe: %s", exc)
        return ""


# live21 (2026-07-05): find a process's real top-level window via Win32
# EnumWindows — reliable and FOCUS-INDEPENDENT, unlike .NET MainWindowHandle
# which returned nothing for a pygame window sitting behind the user's active
# apps. Also foregrounds it so the screenshot is the GAME, not the desktop the
# user was working on (run #10 photographed the ASTRA app and the judge
# correctly failed "a Tetris game" it never saw). Emits: HWND|L,T,R,B|title
_FIND_WINDOW_PS = r"""
$ErrorActionPreference='SilentlyContinue'
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class W21 {
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
  public delegate bool EnumProc(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
  public struct RECT { public int L, T, R, B; }
}
"@
$target=[uint32]__PID__
$found=[IntPtr]::Zero
$cb=[W21+EnumProc]{ param($h,$p)
  if (-not [W21]::IsWindowVisible($h)) { return $true }
  $wp=[uint32]0; [void][W21]::GetWindowThreadProcessId($h,[ref]$wp)
  if ($wp -ne $target) { return $true }
  $r=New-Object W21+RECT; [void][W21]::GetWindowRect($h,[ref]$r)
  if (($r.R-$r.L) -lt 80 -or ($r.B-$r.T) -lt 80) { return $true }  # skip tool/tray windows
  $script:found=$h; return $false
}
[void][W21]::EnumWindows($cb,[IntPtr]::Zero)
if ($found -ne [IntPtr]::Zero) {
  if (__FOREGROUND__) { [void][W21]::ShowWindow($found,5); [void][W21]::BringWindowToTop($found); [void][W21]::SetForegroundWindow($found); Start-Sleep -Milliseconds 350 }
  $r=New-Object W21+RECT; [void][W21]::GetWindowRect($found,[ref]$r)
  $sb=New-Object System.Text.StringBuilder 256; [void][W21]::GetWindowText($found,$sb,256)
  "$([int64]$found)|$($r.L),$($r.T),$($r.R),$($r.B)|$($sb.ToString())"
}
"""


async def find_pid_window(pid: int, foreground: bool = False) -> Dict[str, Any]:
    """Find (optionally foreground) the visible top-level window owned by pid.

    Returns {found, hwnd, rect:(l,t,r,b), title} — found=False when the process
    has no qualifying window yet. focus-independent (Win32 EnumWindows)."""
    empty = {"found": False, "hwnd": 0, "rect": None, "title": ""}
    try:
        script = _FIND_WINDOW_PS.replace("__PID__", str(int(pid))).replace(
            "__FOREGROUND__", "$true" if foreground else "$false")
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", script,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        raw = (stdout or b"").decode("utf-8", errors="replace").strip()
        return parse_window_line(raw)
    except Exception as exc:
        logger.debug("[host_perception] find_pid_window: %s", exc)
        return empty


def parse_window_line(raw: str) -> Dict[str, Any]:
    """Parse 'HWND|L,T,R,B|title' — pure, unit-tested."""
    empty = {"found": False, "hwnd": 0, "rect": None, "title": ""}
    if not raw or "|" not in raw:
        return empty
    parts = raw.split("|", 2)
    if len(parts) < 2:
        return empty
    try:
        hwnd = int(parts[0].strip())
        nums = [int(x) for x in parts[1].split(",")]
        if hwnd == 0 or len(nums) != 4:
            return empty
        title = parts[2].strip() if len(parts) > 2 else ""
        return {"found": True, "hwnd": hwnd, "rect": tuple(nums), "title": title}
    except (ValueError, IndexError):
        return empty


async def _window_state_for_pid(pid: int) -> Dict[str, Any]:
    """live20: readiness by window EXISTENCE, not title text. A borderless /
    no-titlebar window (which the user explicitly asked for — no X button) has
    a real MainWindowHandle but an EMPTY MainWindowTitle, so title-only
    detection reported 'no window' on a working app. Returns
    {has_window: bool, title: str}."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command",
            f"$p=Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; "
            "if ($p) { \"$([int64]$p.MainWindowHandle)|$($p.MainWindowTitle)\" }",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        raw = (stdout or b"").decode("utf-8", errors="replace").strip()
        if "|" in raw:
            handle_s, title = raw.split("|", 1)
            try:
                has_window = int(handle_s.strip()) != 0
            except ValueError:
                has_window = False
            return {"has_window": has_window, "title": title.strip()}
    except Exception as exc:
        logger.debug("[host_perception] window state probe: %s", exc)
    return {"has_window": False, "title": ""}


async def wait_ready(
    launch_info: Dict[str, Any],
    timeout_s: Optional[float] = None,
    poll_s: float = 1.0,
    probe=None,
) -> Dict[str, Any]:
    """live13: wait for a REAL boot signal instead of assuming instant boot
    (Taz: "it can't just hit the boot button and assume it's there").

    Ready = process still alive AND it has a top-level window. live20: keyed
    on the window's EXISTENCE (MainWindowHandle != 0), NOT its title text — a
    borderless no-titlebar window (the user asked for no X button) has a real
    handle but an empty title, and title-only detection reported "no window"
    on a perfectly good app. Default timeout ASTRA_EYES_BOOT_TIMEOUT=40s
    (procedural audio synthesis needs headroom). Early exit with collected
    stdout/stderr when the process dies while we wait — that traceback is the
    judge's evidence. `probe` is injectable for tests: it may return a plain
    title string (legacy) or a {has_window, title} dict.
    """
    try:
        timeout = float(timeout_s if timeout_s is not None else os.getenv("ASTRA_EYES_BOOT_TIMEOUT", "40"))
    except ValueError:
        timeout = 40.0
    proc = launch_info.get("_proc")
    pid = launch_info.get("pid")
    start = time.time()

    while (time.time() - start) < timeout:
        if proc is not None and proc.poll() is not None:
            tails = collect_output(launch_info)
            return {
                "ready": False, "exited": True, "returncode": proc.poll(),
                "waited_s": round(time.time() - start, 1), "window_title": "",
                **tails,
            }
        has_window, title = False, ""
        if pid:
            # live21: prefer the Win32 EnumWindows finder (focus-independent);
            # probe override still honoured for tests.
            result = await (probe(pid) if probe else find_pid_window(pid))
            if isinstance(result, dict):
                has_window = bool(result.get("has_window") or result.get("found"))
                title = str(result.get("title") or "")
            else:  # legacy probe returning a title string
                title = str(result or "")
                has_window = bool(title)
        if has_window:
            return {
                "ready": True, "exited": False, "returncode": None,
                "waited_s": round(time.time() - start, 1), "window_title": title,
            }
        await asyncio.sleep(poll_s)

    return {
        "ready": False, "exited": False, "returncode": None,
        "waited_s": round(time.time() - start, 1), "window_title": "",
    }


def stop_app(launch_info: Dict[str, Any]) -> None:
    proc = launch_info.get("_proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except Exception as exc:
            logger.debug("[host_perception] stop_app: %s", exc)


_CAPTURE_TEMPLATE = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
$bitmap.Save('__OUT__')
$graphics.Dispose(); $bitmap.Dispose()
Write-Output "SCREENSHOT_OK $($screen.Width)x$($screen.Height)"
"""


# live21: capture a specific window's rectangle (the game), after it's been
# brought to the foreground. Falls back to a small margin around the rect so a
# borderless drop-shadow isn't clipped. Coordinates come from find_pid_window.
_CAPTURE_RECT_TEMPLATE = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$l=__L__; $t=__T__; $r=__R__; $b=__B__
$w=$r-$l; $h=$b-$t
if ($w -lt 1 -or $h -lt 1) { Write-Output "SCREENSHOT_FAIL bad-rect"; exit }
$bitmap = New-Object System.Drawing.Bitmap($w, $h)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($l, $t, 0, 0, (New-Object System.Drawing.Size($w,$h)))
$bitmap.Save('__OUT__')
$graphics.Dispose(); $bitmap.Dispose()
Write-Output "SCREENSHOT_OK ${w}x${h}"
"""


async def capture_screen(
    evidence_dir: str,
    label: str = "shot",
    rect: Optional[tuple] = None,
) -> Optional[str]:
    """Capture to <evidence_dir>/<label>_<n>.png. Returns the path or None.

    live21: when `rect`=(l,t,r,b) is given, capture ONLY that window rectangle
    (the game), not the whole desktop — a full-screen grab captured the user's
    active app instead of the game (run #10). Falls back to full screen when
    no rect is supplied."""
    os.makedirs(evidence_dir, exist_ok=True)
    n = len([f for f in os.listdir(evidence_dir) if f.endswith(".png")])
    out = str(Path(evidence_dir) / f"{label}_{n:03d}.png")
    if rect and len(rect) == 4:
        l, t, r, b = rect
        script = (_CAPTURE_RECT_TEMPLATE
                  .replace("__L__", str(int(l))).replace("__T__", str(int(t)))
                  .replace("__R__", str(int(r))).replace("__B__", str(int(b)))
                  .replace("__OUT__", out.replace("'", "''")))
    else:
        script = _CAPTURE_TEMPLATE.replace("__OUT__", out.replace("'", "''"))
    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", script,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
        if b"SCREENSHOT_OK" in (stdout or b"") and os.path.isfile(out):
            return out
        logger.warning("[host_perception] capture failed: %s", (stderr or stdout or b"")[:200])
    except Exception as exc:
        logger.warning("[host_perception] capture error: %s", exc)
    return None


_SENDKEYS_TEMPLATE = r"""
$wshell = New-Object -ComObject WScript.Shell
$focused = $wshell.AppActivate(__PID__)
Start-Sleep -Milliseconds 400
foreach ($k in @(__KEYS__)) {
  $wshell.SendKeys($k)
  Start-Sleep -Milliseconds __DELAY__
}
Write-Output "SENDKEYS_OK focused=$focused"
"""

# SendKeys tokens the interaction scripts are allowed to inject. Arrows and
# a few benign extras only — the eyes drive GAMES, they never type text.
_ALLOWED_KEY_TOKENS = {"{LEFT}", "{RIGHT}", "{UP}", "{DOWN}", "{ENTER}", " ", "p"}


def build_sendkeys_script(pid: int, keys: List[str], delay_ms: int = 120) -> Optional[str]:
    """Render the PowerShell for a key sequence. None if nothing valid.

    Pure + deterministic so tests can pin the exact injection surface:
    only whitelisted tokens, only the eyes' own PID.
    """
    safe = [k for k in keys if k in _ALLOWED_KEY_TOKENS]
    if not safe or not isinstance(pid, int) or pid <= 0:
        return None
    keys_ps = ",".join("'" + k.replace("'", "") + "'" for k in safe)
    return (
        _SENDKEYS_TEMPLATE
        .replace("__PID__", str(pid))
        .replace("__KEYS__", keys_ps)
        .replace("__DELAY__", str(max(30, int(delay_ms))))
    )


async def send_keys(pid: int, keys: List[str], delay_ms: int = 120) -> bool:
    """Focus the app we spawned (by PID) and inject a key sequence.

    Returns True when the injection script ran; the app may still ignore
    keys it doesn't handle — screenshots after the fact are the evidence.
    """
    script = build_sendkeys_script(pid, keys, delay_ms)
    if not script:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", script,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        ok = b"SENDKEYS_OK" in (stdout or b"")
        if not ok:
            logger.warning("[host_perception] send_keys failed: %s", (stderr or stdout or b"")[:200])
        return ok
    except Exception as exc:
        logger.warning("[host_perception] send_keys error: %s", exc)
        return False


async def window_titles() -> List[str]:
    """Visible top-level window titles on the host."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command",
            "Get-Process | Where-Object {$_.MainWindowTitle} | ForEach-Object {$_.MainWindowTitle}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        return [l.strip() for l in (stdout or b"").decode("utf-8", errors="replace").splitlines() if l.strip()]
    except Exception as exc:
        logger.debug("[host_perception] window_titles: %s", exc)
        return []


__all__ = [
    "guess_entrypoint", "launch_app", "stop_app", "capture_screen",
    "window_titles", "send_keys", "build_sendkeys_script",
    "wait_ready", "collect_output", "find_pid_window", "parse_window_line",
]

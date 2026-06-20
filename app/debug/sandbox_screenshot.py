# FILE: app/debug/sandbox_screenshot.py
# Purpose: The "visual eyes" of the sandbox self-heal loop. Captures a screenshot
#          of the SANDBOX desktop (the clone's Electron window) and vision/OCR-reads
#          it so the Debug brain can SEE how the app booted -- catching frontend
#          error overlays (e.g. the Vite "@univerjs/presets" import error) that the
#          backend log never shows.
# Called-by: app.debug.sandbox_boot_tool.inspect_sandbox_boot (WI-4)
# Depends-on: app.debug.sandbox_console.visible_shell_run (WI-2, visible capture),
#             app.sandbox.client.get_sandbox_client (call only -- PROTECTED, unmodified),
#             app.llm.gemini_vision.ask_about_image (same call read_image uses)
# Last-renovated: 2026-06-17 (created -- WI-3 sandbox-visual-selfheal)
"""Capture + vision-read the sandbox desktop.

How the bytes get out of the sandbox (the constraints that shape this):
  * The ONLY channels out of the (isolated) sandbox are the controller HTTP API.
  * ``SandboxClient.repo_file`` reads UTF-8 text, <=512 KB, ONLY under REPO_ROOT,
    and the controller DENIES any ``logs/``, ``cache/``, ``temp/`` ... path.
So we capture in the sandbox, downscale + JPEG-encode (small), base64 the bytes to a
TEXT file under a fresh ``_astra_selfheal/`` folder inside REPO_ROOT (not a denied
name, well under the size cap), then ``repo_file`` the base64 back and decode it on
the host. The capture command runs through the WI-2 ``visible_shell_run`` so Taz can
watch it; the base64 itself is NOT echoed (it goes to a file, read back over HTTP),
so the visible console stays clean -- it shows the command + a one-line status only.

Everything here is best-effort and NEVER raises: the function returns a human/agent
readable string in all paths (errors included) so the tool loop can read it.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Sandbox-side: where the capture snippet drops the base64 text (relative to
# REPO_ROOT so repo_file can read it; the folder name is NOT on the controller's
# deny list). The absolute sandbox path is built from health().repo_root at runtime.
_SBX_SELFHEAL_DIR = "_astra_selfheal"
_SBX_B64_NAME = "sandbox_screen.b64"
_SBX_B64_REL = f"{_SBX_SELFHEAL_DIR}/{_SBX_B64_NAME}"

# Host-side: where we drop the decoded image for the vision call (backend-writable
# data dir -- written by backend code, not via the gated run_command tool).
_HOST_IMG_DIR = r"D:\Orb\data\sandbox_selfheal"
_HOST_IMG_PATH = os.path.join(_HOST_IMG_DIR, "sandbox_screen.jpg")

# Keep the JPEG comfortably under the 512 KB repo_file cap (base64 ~ 1.34x bytes).
# 1280 px longest edge @ q60 is plenty to OCR an error overlay / stack trace.
_MAX_EDGE = 1280
_JPEG_QUALITY = 60

_DEFAULT_QUESTION = (
    "This is a screenshot of a Windows desktop running the ASTRA Electron app "
    "(a clone booted inside a Windows Sandbox for testing). Look at it carefully. "
    "Is the app running normally, or is something wrong -- e.g. an error overlay, a "
    "red error screen, a stack trace, a Vite/compiler error, a blank white/black "
    "window, or a crash dialog? If there is ANY error or failure text on screen, "
    "quote it VERBATIM, including any file paths, import names, module names and line "
    "numbers. If the app looks like it booted fine, say so and briefly describe what "
    "is visible."
)


# AMSI/Defender note: the sandbox runs Defender with real-time + AMSI script
# scanning. Two patterns in earlier versions got the WHOLE capture script blocked
# as "malicious content": (1) window-focus interop (SetForegroundWindow P/Invoke,
# later the WScript.Shell AppActivate COM call) -- a window-injection signature; and
# (2) an ImageCodecInfo/EncoderParameters JPEG-quality block -- a screen-grabber
# infostealer signature. Both are removed. We do a plain full-virtual-screen
# CopyFromScreen (no focus step needed -- it grabs everything regardless of
# foreground) and a default-quality ImageFormat::Jpeg save after downscaling.
def _build_capture_snippet(sandbox_abs_b64_path: str) -> str:
    """PowerShell that grabs the (virtual) screen, downscales, JPEG-encodes, and
    writes the base64 to ``sandbox_abs_b64_path`` inside the sandbox. Prints a single
    ``SCREENSHOT_OK ...`` / ``SCREENSHOT_ERR ...`` status line (no base64 to stdout,
    so the visible console stays readable)."""
    snippet = r"""$ErrorActionPreference='Stop'
try {
  Add-Type -AssemblyName System.Windows.Forms | Out-Null
  Add-Type -AssemblyName System.Drawing | Out-Null
  # NOTE: no window-focus step here. CopyFromScreen grabs the entire virtual
  # screen regardless of which window is foreground, so focusing the Electron
  # window buys nothing -- and the AppActivate/WScript.Shell focus pattern is an
  # AMSI window-injection signature that got the whole capture script blocked.
  $vs = [System.Windows.Forms.SystemInformation]::VirtualScreen
  $full = New-Object System.Drawing.Bitmap([int]$vs.Width, [int]$vs.Height)
  $g = [System.Drawing.Graphics]::FromImage($full)
  $g.CopyFromScreen([int]$vs.X, [int]$vs.Y, 0, 0, $full.Size)
  $g.Dispose()
  $maxEdge = __MAXEDGE__
  $scale = [Math]::Min(1.0, $maxEdge / [Math]::Max($full.Width, $full.Height))
  $tw = [Math]::Max(1, [int]($full.Width * $scale))
  $th = [Math]::Max(1, [int]($full.Height * $scale))
  $out = New-Object System.Drawing.Bitmap($tw, $th)
  $g2 = [System.Drawing.Graphics]::FromImage($out)
  $g2.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $g2.DrawImage($full, 0, 0, $tw, $th)
  $g2.Dispose(); $full.Dispose()
  # Plain default-quality JPEG save. The downscale above already keeps the file
  # small (well under the 512 KB repo_file cap), so we do NOT build a custom
  # EncoderParameters/ImageCodecInfo quality block -- that enumerate-encoders +
  # quality-control pattern is an AMSI infostealer signature and was the thing
  # blocking this capture in the sandbox.
  $ms = New-Object System.IO.MemoryStream
  $out.Save($ms, [System.Drawing.Imaging.ImageFormat]::Jpeg)
  $out.Dispose()
  $bytes = $ms.ToArray(); $ms.Dispose()
  $b64 = [Convert]::ToBase64String($bytes)
  $dir = Split-Path -Parent '__B64PATH__'
  if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  [System.IO.File]::WriteAllText('__B64PATH__', $b64)
  Write-Output ('SCREENSHOT_OK full={0}x{1} scaled={2}x{3} jpeg_bytes={4} b64_chars={5}' -f [int]$vs.Width,[int]$vs.Height,$tw,$th,$bytes.Length,$b64.Length)
} catch {
  Write-Output ('SCREENSHOT_ERR ' + $_.Exception.Message)
}"""
    return (
        snippet.replace("__MAXEDGE__", str(int(_MAX_EDGE)))
        .replace("__B64PATH__", sandbox_abs_b64_path)
    )


def _sandbox_abs_b64_path(repo_root: str) -> str:
    """Build the absolute Windows path inside the sandbox for the base64 drop,
    from the controller-reported REPO_ROOT (e.g. ``D:\\Orb``)."""
    base = (repo_root or r"D:\Orb").rstrip("\\/")
    return base + "\\" + _SBX_SELFHEAL_DIR + "\\" + _SBX_B64_NAME


def capture_and_read_sandbox_screen(question: Optional[str] = None, client=None) -> str:
    """Capture the sandbox desktop, vision/OCR-read it, and return a text report.

    Sync (the tool loop runs it via ``asyncio.to_thread``). Never raises -- all
    failures are returned as readable text so the agent can reason about them.
    """
    q = (question or "").strip() or _DEFAULT_QUESTION

    # 1. Sandbox client + REPO_ROOT (drives the in-sandbox drop path).
    try:
        if client is None:
            from app.sandbox.client import get_sandbox_client
            client = get_sandbox_client()
        health = client.health()
        repo_root = getattr(health, "repo_root", "") or r"D:\Orb"
    except Exception as e:
        return (
            "VISUAL: could not reach the sandbox controller to take a screenshot "
            f"({e}). Is the Windows Sandbox running?"
        )

    sandbox_b64_abs = _sandbox_abs_b64_path(repo_root)

    # 2. Capture in the sandbox, VISIBLY (WI-2). The base64 goes to a file, not stdout.
    try:
        from app.debug.sandbox_console import visible_shell_run
        result = visible_shell_run(
            _build_capture_snippet(sandbox_b64_abs),
            cwd_target="REPO",
            timeout_seconds=90,
            client=client,
        )
    except Exception as e:
        return f"VISUAL: screenshot capture command failed to run in the sandbox ({e})."

    stdout = (getattr(result, "stdout", "") or "").strip()
    stderr = (getattr(result, "stderr", "") or "").strip()
    if "SCREENSHOT_OK" not in stdout:
        detail = stdout or stderr or "no output"
        return f"VISUAL: screenshot capture did not succeed in the sandbox. Detail: {detail[:600]}"
    status_line = next((ln for ln in stdout.splitlines() if "SCREENSHOT_OK" in ln), stdout)

    # 3. Fetch the base64 back over HTTP (repo_file: text, <=512KB, under REPO_ROOT).
    try:
        fc = client.repo_file(_SBX_B64_REL)
        b64_text = (getattr(fc, "content", "") or "")
    except Exception as e:
        return (
            f"VISUAL: captured the screen ({status_line}) but could not retrieve the "
            f"image bytes from the sandbox ({e}). The base64 file is at "
            f"{sandbox_b64_abs} in the sandbox."
        )

    # 4. Decode + write the image on the host for the vision call.
    try:
        compact = "".join(b64_text.split())  # drop any whitespace/newlines
        img_bytes = base64.b64decode(compact)
        if not img_bytes:
            return f"VISUAL: retrieved an EMPTY screenshot payload ({status_line})."
        os.makedirs(_HOST_IMG_DIR, exist_ok=True)
        with open(_HOST_IMG_PATH, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        return f"VISUAL: failed to decode/save the screenshot image ({e}). {status_line}"

    # 5. Vision / OCR read (same backend stack as the read_image tool).
    try:
        from app.llm.gemini_vision import ask_about_image
        vres = ask_about_image(image_source=_HOST_IMG_PATH, user_question=q)
    except Exception as e:
        return (
            f"VISUAL: captured the screen ({status_line}) and saved it to "
            f"{_HOST_IMG_PATH}, but the vision read failed ({e})."
        )

    if vres.get("error"):
        provider = vres.get("provider", "unknown")
        return (
            f"VISUAL: captured the screen ({status_line}) but vision failed "
            f"({provider}): {vres['error']}. Image saved at {_HOST_IMG_PATH}."
        )

    answer = (vres.get("answer") or "").strip() or "(vision returned no description)"
    provider = vres.get("provider", "unknown")
    model = vres.get("model", "unknown")
    return (
        f"VISUAL READOUT of the sandbox Electron app ({status_line}; "
        f"image {_HOST_IMG_PATH}):\n[{provider}/{model}]\n{answer}"
    )


__all__ = ["capture_and_read_sandbox_screen"]

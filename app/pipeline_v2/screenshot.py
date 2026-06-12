# FILE: app/pipeline_v2/screenshot.py
# Purpose: Screenshot capture for visual verification.
# Called-by: app.pipeline_v2.checkout, app.pipeline_v2.verification
# Depends-on: app.pipeline_v2.sandbox_tools
# Last-renovated: 2026-06-11
"""
Screenshot capture for visual verification.

Takes a screenshot of the Windows Sandbox desktop via PowerShell,
saves it as PNG, and returns the base64-encoded image for sending
to a vision model.

v1.0 (2026-03-07): Initial implementation for ASTRA v2.1.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from app.pipeline_v2.sandbox_tools import run_shell

logger = logging.getLogger(__name__)

SCREENSHOT_PATH = r"D:\Orb\jobs\screenshot.png"

CAPTURE_SCRIPT = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
$bitmap.Save('SCREENSHOT_PATH')
$graphics.Dispose()
$bitmap.Dispose()
Write-Output "SCREENSHOT_OK $($screen.Width)x$($screen.Height)"
""".replace("SCREENSHOT_PATH", SCREENSHOT_PATH)


async def capture_screenshot() -> Tuple[Optional[str], str]:
    """Capture a screenshot of the sandbox desktop.

    Returns:
        (base64_png, screenshot_path) or (None, error_message)
    """
    logger.info("[screenshot] Capturing sandbox desktop...")

    result = await run_shell(CAPTURE_SCRIPT, timeout_sec=10)

    if "SCREENSHOT_OK" not in result["stdout"]:
        error = result["stderr"] or result["stdout"] or "Unknown error"
        logger.error("[screenshot] Capture failed: %s", error[:200])
        return None, f"Screenshot capture failed: {error[:200]}"

    # Read the PNG back as base64
    read_result = await run_shell(
        f"[Convert]::ToBase64String([IO.File]::ReadAllBytes('{SCREENSHOT_PATH}'))",
        timeout_sec=10,
    )

    b64 = read_result["stdout"].strip()
    if not b64 or len(b64) < 100:
        return None, "Screenshot file empty or unreadable"

    logger.info("[screenshot] Captured: %d bytes base64", len(b64))
    return b64, SCREENSHOT_PATH

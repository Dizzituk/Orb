# FILE: app/llm/image_output_dir.py
# Purpose: Resolve the output directory for generated images.
# Called-by: app.bridge.artifacts, app.llm.astra_filesystem_block, app.llm.chart_renderer, app.llm.image_gen (+3 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Resolve the output directory for generated images.

Single source of truth so image_gen.py, nano_banana.py, chart_renderer.py,
and main.py (the FastAPI static mount) all agree on where images live.

Resolution order:
1. IMAGE_OUTPUT_DIR env var \u2014 if set, used directly (no /images suffix).
2. Fallback: {ASTRA_OUTPUT_DIR or D:\\Orb\\output} / images

Set IMAGE_OUTPUT_DIR in .env to relocate generated images into a folder
that's already watched by file_watcher (so the agent's search_my_files /
search_disk_live tools can find them when posting to Meta etc.):

    IMAGE_OUTPUT_DIR=C:/Users/dizzi/OneDrive/Pictures/AstraPictures

Forward slashes are recommended in .env values to avoid backslash-escape
ambiguity. Python pathlib handles either correctly on Windows.

v1.0 (2026-04-25): Initial implementation. Decouples image output path
                    from D:\\Orb so the file_watcher can pick up generated
                    images without exposing D:\\Orb to user-facing tools.
"""
from __future__ import annotations

import os
from pathlib import Path


def get_image_output_dir() -> Path:
    """Return the resolved output directory for generated images.

    Does NOT create the directory \u2014 callers are responsible for mkdir.
    """
    explicit = os.getenv("IMAGE_OUTPUT_DIR")
    if explicit:
        return Path(explicit)
    base = os.getenv("ASTRA_OUTPUT_DIR", r"D:\Orb\output")
    return Path(base) / "images"


__all__ = ["get_image_output_dir"]

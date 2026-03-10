# FILE: app/llm/file_output.py
"""
File Output Service — saves generated content to disk during chat.

When ASTRA generates a file (HTML, document, etc.), this service:
1. Writes it to D:/Orb/output/ (or a configured output directory)
2. Returns the SSE event data for the frontend FileOutputCard

v2.5 (2026-03-09): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Output directory for generated files
OUTPUT_DIR = os.getenv("ASTRA_OUTPUT_DIR", r"D:\Orb\output")


def _ensure_output_dir() -> Path:
    """Ensure the output directory exists."""
    p = Path(OUTPUT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_generated_file(
    content: str,
    filename: str,
    file_type: str = "html",
    description: Optional[str] = None,
) -> dict:
    """Save generated content to the output directory.

    Args:
        content: The file content to write
        filename: The filename (e.g. "astra-website.html")
        file_type: Type for the FileOutputCard (html, pdf, docx, code, other)
        description: Optional description shown on the card

    Returns:
        Dict with path, filename, type, size, description — ready for SSE emission.
    """
    output_dir = _ensure_output_dir()

    # Sanitise filename
    safe_name = "".join(c for c in filename if c.isalnum() or c in ".-_ ").strip()
    if not safe_name:
        safe_name = f"output-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.{file_type}"

    filepath = output_dir / safe_name

    # Write the file
    filepath.write_text(content, encoding="utf-8")
    size = filepath.stat().st_size

    logger.info("[file_output] Saved %s (%d bytes) to %s", safe_name, size, filepath)

    return {
        "path": str(filepath),
        "filename": safe_name,
        "type": file_type,
        "size": size,
        "description": description or "",
    }


def sse_file_outputs(files: list) -> str:
    """Generate SSE event for file outputs."""
    return "data: " + json.dumps({"type": "file_outputs", "files": files}) + "\n\n"


__all__ = ["save_generated_file", "sse_file_outputs", "OUTPUT_DIR"]

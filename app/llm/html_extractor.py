# FILE: app/llm/html_extractor.py
"""
Extract and save HTML/code files from LLM chat responses.

Scans the full response for fenced code blocks containing complete HTML
documents (or other saveable content) and saves them to D:/Orb/output/.

Model-agnostic — works with any LLM response regardless of provider.

v2.5 (2026-03-09): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from app.llm.file_output import save_generated_file

logger = logging.getLogger(__name__)

# Regex to find fenced code blocks: ```html ... ``` or ```htm ... ```
_CODE_BLOCK_PATTERN = re.compile(
    r'```(html?|htm)\s*\n(.*?)```',
    re.DOTALL | re.IGNORECASE,
)

# Also catch unlabelled code blocks that contain <!DOCTYPE or <html
_UNLABELLED_HTML_PATTERN = re.compile(
    r'```\s*\n(<!DOCTYPE[^`]*?</html>\s*)```',
    re.DOTALL | re.IGNORECASE,
)


def _is_complete_html(content: str) -> bool:
    """Check if content looks like a complete HTML document (not a snippet)."""
    lower = content.strip().lower()
    return (
        ('<!doctype' in lower or '<html' in lower)
        and '</html>' in lower
        and len(content) > 500  # Must be substantial
    )


def _generate_filename(content: str) -> str:
    """Generate a meaningful filename from the HTML title or timestamp."""
    # Try to extract <title> content
    title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
        # Sanitise for filename
        safe = re.sub(r'[^\w\s-]', '', title).strip()
        safe = re.sub(r'\s+', '-', safe).lower()[:50]
        if safe:
            return f"{safe}.html"

    # Fallback: timestamp
    ts = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    return f"generated-{ts}.html"


def extract_and_save_html(full_response: str) -> List[dict]:
    """Scan an LLM response for HTML documents and save them.

    Returns a list of file output dicts ready for SSE emission.
    """
    saved_files: List[dict] = []

    # Pattern 1: Labelled HTML code blocks
    for match in _CODE_BLOCK_PATTERN.finditer(full_response):
        html_content = match.group(2).strip()
        if _is_complete_html(html_content):
            filename = _generate_filename(html_content)
            file_info = save_generated_file(
                content=html_content,
                filename=filename,
                file_type="html",
                description="Generated HTML page",
            )
            saved_files.append(file_info)
            logger.info("[html_extractor] Saved HTML from labelled block: %s", filename)

    # Pattern 2: Unlabelled code blocks containing full HTML
    if not saved_files:
        for match in _UNLABELLED_HTML_PATTERN.finditer(full_response):
            html_content = match.group(1).strip()
            if _is_complete_html(html_content):
                filename = _generate_filename(html_content)
                file_info = save_generated_file(
                    content=html_content,
                    filename=filename,
                    file_type="html",
                    description="Generated HTML page",
                )
                saved_files.append(file_info)
                logger.info("[html_extractor] Saved HTML from unlabelled block: %s", filename)

    # Pattern 3: Direct HTML in response (no code fences, but contains full document)
    # This catches cases where the LLM outputs raw HTML without backticks
    if not saved_files:
        # Look for <!DOCTYPE...></html> spans in the raw text
        raw_match = re.search(
            r'(<!DOCTYPE\s+html[^>]*>.*?</html>)',
            full_response,
            re.DOTALL | re.IGNORECASE,
        )
        if raw_match and _is_complete_html(raw_match.group(1)):
            html_content = raw_match.group(1).strip()
            filename = _generate_filename(html_content)
            file_info = save_generated_file(
                content=html_content,
                filename=filename,
                file_type="html",
                description="Generated HTML page",
            )
            saved_files.append(file_info)
            logger.info("[html_extractor] Saved HTML from raw content: %s", filename)

    return saved_files


__all__ = ["extract_and_save_html"]

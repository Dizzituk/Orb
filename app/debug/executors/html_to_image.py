# FILE: app/debug/executors/html_to_image.py
"""
HTML → PNG converter executor.

Uses PyMuPDF (fitz) end-to-end:
    HTML → fitz.Story → PDF page(s) → page.get_pixmap() → PNG

Why this rather than Playwright/Chromium:
    - PyMuPDF is already in requirements.txt; no new install needed.
    - fitz.Story handles the HTML the chat layer actually emits (tables,
      basic CSS, headers, lists, simple borders) cleanly and deterministically.
    - For complex CSS (gradients, flexbox, transforms) Story silently
      degrades but does not crash, so output is always valid PNG.
    - Renders take ~80–250ms — fast enough that the chat doesn't stall.

Limitations to flag honestly:
    - CSS gradients, flexbox, grid, transforms, animations are NOT rendered.
      For visually-styled output (premium quote cards, etc.) the chat layer
      should still go through gpt-image-2.
    - Custom web fonts are NOT loaded; fitz uses system + bundled fallbacks.
    - Output is pixel-rasterised, not vector — request DPI matters for
      sharpness on high-DPI displays.

v1.0 (2026-05-01): Initial implementation. Triggered by html_to_png tool.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# A4 portrait at 200 DPI produces ~1654x2339 px — print-quality and fits
# typical chart/report HTML well. Tweak via the tool params if needed.
_DEFAULT_PAGE_WIDTH_PT = 595.0   # A4 width in points (1pt = 1/72 in)
_DEFAULT_PAGE_HEIGHT_PT = 842.0  # A4 height in points
_DEFAULT_MARGIN_PT = 40.0
_DEFAULT_DPI = 200

# Cap PNG file size — 25 MB is plenty for any single chart and prevents
# runaway DPI requests from filling disk.
_MAX_OUTPUT_BYTES = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# Public executor (registered in action_executor.TOOL_HANDLERS)
# ---------------------------------------------------------------------------

async def execute_html_to_png(params: Dict[str, Any]) -> str:
    """Convert HTML (file path or inline content) to a PNG on disk.

    Params:
        source_path: Absolute path to an existing .html file. Takes priority
                     over `html_content` when both are provided.
        html_content: Inline HTML string. Used if source_path is missing.
        output_path: Absolute path for the PNG. Must end in .png. Required.
        page_width_pt: Optional page width in PDF points (default 595, A4).
        page_height_pt: Optional page height in PDF points (default 842, A4).
        dpi: Optional render DPI (default 200; higher = sharper/larger).
        page: Optional 1-based page index to render (default 1). Useful when
              the HTML flows over multiple pages and you only want one.

    Returns:
        Human-readable success/failure string for the LLM to relay.
    """
    source_path = params.get("source_path", "").strip()
    html_content = params.get("html_content", "")
    output_path = params.get("output_path", "").strip()

    if not output_path:
        return "Error: output_path is required and must end in .png"
    if not output_path.lower().endswith(".png"):
        return f"Error: output_path must end in .png, got: {output_path}"

    html, html_origin = _resolve_html_input(source_path, html_content)
    if html is None:
        return html_origin  # error string

    page_width = float(params.get("page_width_pt", _DEFAULT_PAGE_WIDTH_PT))
    page_height = float(params.get("page_height_pt", _DEFAULT_PAGE_HEIGHT_PT))
    dpi = int(params.get("dpi", _DEFAULT_DPI))
    page_index = max(1, int(params.get("page", 1))) - 1  # convert to 0-based

    if dpi < 50 or dpi > 400:
        return f"Error: dpi must be between 50 and 400, got: {dpi}"

    # Path validation — must land in an allowed user folder, just like
    # write_user_file. We piggyback on the same allowlist so we don't
    # accidentally introduce a way to write outside user areas.
    allow_err = _validate_output_path(output_path)
    if allow_err:
        return allow_err

    try:
        pix_bytes, dimensions = _render_html_to_png(
            html=html,
            page_width_pt=page_width,
            page_height_pt=page_height,
            margin_pt=_DEFAULT_MARGIN_PT,
            dpi=dpi,
            page_index=page_index,
        )
    except _StoryNotAvailable as e:
        return (
            f"Error: PyMuPDF Story API not available — {e}. "
            "This usually means an older PyMuPDF; upgrade to >=1.23 "
            "(pip install --upgrade pymupdf)."
        )
    except _RenderError as e:
        return f"Error rendering HTML to PNG: {e}"
    except Exception as e:  # noqa: BLE001
        logger.exception("[html_to_image] Unexpected render failure")
        return f"Error: unexpected failure during render: {e}"

    if len(pix_bytes) > _MAX_OUTPUT_BYTES:
        return (
            f"Error: rendered PNG would be {len(pix_bytes) // (1024 * 1024)} MB "
            f"(cap is {_MAX_OUTPUT_BYTES // (1024 * 1024)} MB). "
            "Lower the dpi parameter and retry."
        )

    # Write the bytes — PNG is binary so we must not go through write_text.
    target = Path(output_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pix_bytes)
    except OSError as e:
        return f"Error writing PNG: {e}"

    # Refresh manifest so the file is searchable immediately.
    try:
        from app.drive.manifest_scanner import index_single_file
        index_single_file(str(target))
    except Exception:
        pass  # non-fatal

    width_px, height_px = dimensions
    size_kb = len(pix_bytes) / 1024
    logger.info(
        "[html_to_image] Rendered %s (%dx%d, %.1f KB) from %s",
        output_path, width_px, height_px, size_kb, html_origin,
    )

    return (
        f"Successfully rendered HTML to PNG: {output_path} "
        f"({width_px}x{height_px}, {size_kb:.1f} KB) "
        f"[source: {html_origin}]"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StoryNotAvailable(RuntimeError):
    """fitz.Story missing — older PyMuPDF version."""


class _RenderError(RuntimeError):
    """Anything that goes wrong during the actual conversion."""


def _resolve_html_input(
    source_path: str, html_content: str,
) -> tuple[Optional[str], str]:
    """Resolve to (html_string, origin_label) or (None, error_message)."""
    if source_path:
        if not os.path.isfile(source_path):
            return None, f"Error: source_path does not exist: {source_path}"
        if not source_path.lower().endswith((".html", ".htm")):
            return None, (
                f"Error: source_path must point to an .html or .htm file, "
                f"got: {source_path}"
            )
        try:
            html = Path(source_path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return None, f"Error reading source_path: {e}"
        return html, f"file:{source_path}"

    if html_content and html_content.strip():
        return html_content, "inline_html_content"

    return None, (
        "Error: provide either source_path (path to existing HTML file) "
        "or html_content (inline HTML string)."
    )


def _validate_output_path(path: str) -> Optional[str]:
    """Return an error string if path is outside allowed user folders, else None."""
    try:
        from app.drive.file_utils import get_category_paths, is_safe_path
    except ImportError:
        return None  # If utils aren't loadable, fall back to OS-level write errors.

    target = Path(path)
    allowed_roots = list(get_category_paths().values())
    if not is_safe_path(target, allowed_roots):
        return (
            f"Access denied: {path} is outside allowed user folders. "
            "Use get_user_folders to list valid base paths, then write to "
            "Pictures/, Documents/, Desktop/, or ASTRA_OUTPUT."
        )
    return None


def _render_html_to_png(
    html: str,
    page_width_pt: float,
    page_height_pt: float,
    margin_pt: float,
    dpi: int,
    page_index: int,
) -> tuple[bytes, tuple[int, int]]:
    """Run the actual HTML → PNG conversion. Returns (png_bytes, (w_px, h_px))."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise _RenderError(f"PyMuPDF (fitz) not installed: {e}") from e

    if not hasattr(fitz, "Story"):
        raise _StoryNotAvailable(
            "fitz.Story missing on this PyMuPDF version"
        )

    try:
        story = fitz.Story(html=html)
    except Exception as e:  # noqa: BLE001
        raise _RenderError(f"failed to parse HTML: {e}") from e

    pdf_doc = fitz.Document()  # empty PDF in memory
    content_rect = fitz.Rect(
        margin_pt, margin_pt,
        page_width_pt - margin_pt, page_height_pt - margin_pt,
    )

    # Lay out the story across as many pages as it needs. Cap the page
    # count so a misformed HTML can't loop forever.
    pages_built = 0
    max_pages = 50
    while pages_built < max_pages:
        page = pdf_doc.new_page(width=page_width_pt, height=page_height_pt)
        more, _ = story.place(content_rect)
        story.draw(page)
        pages_built += 1
        if not more:
            break

    if page_index >= pdf_doc.page_count:
        raise _RenderError(
            f"requested page {page_index + 1} but render produced "
            f"only {pdf_doc.page_count} page(s)"
        )

    target_page = pdf_doc[page_index]
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = target_page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    dimensions = (pix.width, pix.height)

    # Free PyMuPDF resources eagerly — these can hold large buffers.
    pix = None
    pdf_doc.close()

    return png_bytes, dimensions


__all__ = ["execute_html_to_png"]

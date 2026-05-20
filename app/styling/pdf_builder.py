# FILE: app/styling/pdf_builder.py
"""
Build styled PDFs from a structured content list using ReportLab Platypus.

Same content schema as docx_builder:
  {"type": "heading", "level": 1, "text": "..."}
  {"type": "paragraph", "text": "..."}
  {"type": "list", "items": [...], "ordered": False}
  {"type": "table", "headers": [...], "rows": [...]}
  {"type": "rule"}
  {"type": "spacer"}
  {"type": "code", "text": "...", "language": "..."}

Cover page is added when theme.show_cover_page is True.
Page footer includes generated date and page numbers when configured.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_pdf(
    output_path: str,
    title: str,
    content: List[Dict[str, Any]],
    theme: Dict[str, Any],
    subtitle: Optional[str] = None,
    author: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a styled PDF. Returns {path, size_bytes, blocks_rendered}."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
        ListFlowable, ListItem, HRFlowable, Preformatted,
    )

    margin = theme.get("page_margin_cm", 2.0) * cm
    primary = colors.HexColor("#" + theme.get("colour_primary", "000000").lstrip("#"))
    text_col = colors.HexColor("#" + theme.get("colour_text", "1A1F2E").lstrip("#"))
    muted_col = colors.HexColor("#" + theme.get("colour_muted", "6B7280").lstrip("#"))
    rule_col = colors.HexColor("#" + theme.get("colour_rule", "E5E7EB").lstrip("#"))
    zebra_col = colors.HexColor("#" + theme.get("colour_zebra", "F8FAFC").lstrip("#"))
    header_fill = colors.HexColor("#" + theme.get("table_header_fill", "0B5FFF").lstrip("#"))
    header_text = colors.HexColor("#" + theme.get("table_header_text", "FFFFFF").lstrip("#"))

    # Map theme-requested fonts to ReportLab's built-in PostScript font
    # names so we don't need to register external TTFs. The theme's
    # font_body_single is the single-name fallback used by libraries that
    # can't parse a CSS stack. astra_minimal -> Times, astra_default ->
    # Helvetica.
    _theme_body_font = (theme.get("font_body_single") or "").lower()
    if "times" in _theme_body_font or "georgia" in _theme_body_font or "serif" in _theme_body_font:
        body_font = "Times-Roman"
        bold_font = "Times-Bold"
    else:
        body_font = "Helvetica"
        bold_font = "Helvetica-Bold"
    mono_font = "Courier"
    body_size = theme.get("body_size_pt", 11)
    small_size = theme.get("small_size_pt", 9)
    h_sizes = theme.get("heading_sizes_pt", [22, 16, 13, 11])

    # Styles
    styles = {
        "body": ParagraphStyle(
            "body", fontName=body_font, fontSize=body_size,
            textColor=text_col, leading=body_size * 1.4, spaceAfter=6,
        ),
        "muted": ParagraphStyle(
            "muted", fontName=body_font, fontSize=small_size,
            textColor=muted_col, leading=small_size * 1.4, alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "code", fontName=mono_font, fontSize=small_size,
            textColor=text_col, leading=small_size * 1.4,
            backColor=zebra_col, borderPadding=6, leftIndent=6, rightIndent=6,
            spaceBefore=4, spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1", fontName=bold_font, fontSize=h_sizes[0],
            textColor=primary, leading=h_sizes[0] * 1.2,
            spaceBefore=14, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2", fontName=bold_font, fontSize=h_sizes[1],
            textColor=text_col, leading=h_sizes[1] * 1.2,
            spaceBefore=12, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3", fontName=bold_font, fontSize=h_sizes[2],
            textColor=text_col, leading=h_sizes[2] * 1.2,
            spaceBefore=10, spaceAfter=4,
        ),
        "h4": ParagraphStyle(
            "h4", fontName=bold_font, fontSize=h_sizes[3],
            textColor=text_col, leading=h_sizes[3] * 1.2,
            spaceBefore=8, spaceAfter=4,
        ),
        "cover_title": ParagraphStyle(
            "cover_title", fontName=bold_font, fontSize=32,
            textColor=primary, leading=38, alignment=TA_CENTER,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontName=body_font, fontSize=14,
            textColor=muted_col, leading=18, alignment=TA_CENTER,
        ),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title=title, author=author or "ASTRA",
    )

    story: List[Any] = []

    # Cover page
    if theme.get("show_cover_page", False):
        story.append(Spacer(1, 6 * cm))
        story.append(Paragraph(_escape(title), styles["cover_title"]))
        story.append(Spacer(1, 0.5 * cm))
        if subtitle:
            story.append(Paragraph(_escape(subtitle), styles["cover_sub"]))
            story.append(Spacer(1, 1 * cm))
        meta_text = datetime.now().strftime("%d %B %Y")
        if author:
            meta_text = f"{_escape(author)}  |  {meta_text}"
        story.append(Paragraph(meta_text, styles["muted"]))
        story.append(PageBreak())
    else:
        # Inline title (minimal theme)
        story.append(Paragraph(_escape(title), styles["h1"]))
        if subtitle:
            story.append(Paragraph(_escape(subtitle), styles["body"]))
        story.append(Spacer(1, 0.3 * cm))

    # Body
    blocks_rendered = 0
    for block in content:
        try:
            btype = block.get("type", "paragraph")
            if btype == "heading":
                level = max(1, min(4, int(block.get("level", 1))))
                story.append(Paragraph(_escape(block.get("text", "")), styles[f"h{level}"]))
            elif btype == "paragraph":
                story.append(Paragraph(_escape(block.get("text", "")), styles["body"]))
            elif btype == "list":
                items = [ListItem(Paragraph(_escape(str(i)), styles["body"])) for i in block.get("items", [])]
                bullet = "1" if block.get("ordered") else "bullet"
                story.append(ListFlowable(items, bulletType=bullet, leftIndent=18, spaceAfter=6))
            elif btype == "table":
                _add_pdf_table(story, block, header_fill, header_text, rule_col, zebra_col,
                               theme.get("table_zebra", True), body_font, bold_font, body_size,
                               available_width_pt=(doc.pagesize[0] - 2 * margin))
            elif btype == "rule":
                story.append(HRFlowable(width="100%", thickness=0.5, color=rule_col,
                                        spaceBefore=6, spaceAfter=6))
            elif btype == "spacer":
                story.append(Spacer(1, 0.4 * cm))
            elif btype == "code":
                story.append(Preformatted(block.get("text", ""), styles["code"]))
            else:
                story.append(Paragraph(_escape(str(block)), styles["body"]))
            blocks_rendered += 1
        except Exception:
            continue

    # Footer with page numbers + generated date
    show_footer = theme.get("show_generated_footer", True) or theme.get("show_page_numbers", True)
    footer_fn = _make_footer(theme, muted_col, body_font, small_size) if show_footer else None

    if footer_fn:
        doc.build(story, onFirstPage=footer_fn, onLaterPages=footer_fn)
    else:
        doc.build(story)

    size = out.stat().st_size
    return {
        "path": str(out),
        "size_bytes": size,
        "blocks_rendered": blocks_rendered,
        "theme": theme.get("name", "unknown"),
    }


def _add_pdf_table(story, block, header_fill, header_text, rule_col, zebra_col,
                   use_zebra, body_font, bold_font, body_size,
                   available_width_pt=None):
    """Render a table block.

    - Cells are wrapped in Paragraph objects so text WRAPS instead of
      clipping when a column is narrow.
    - column_widths on the block is respected and treated as relative
      weights that sum to 1.0 of the page's printable width (values >1
      are taken as point widths directly).
    - When no column_widths are given we assign proportional widths
      based on the longest cell in each column, capped so nothing
      overflows the page.

    available_width_pt: printable width in points. Falls back to A4
    minus 2x 2cm margins (~495 pt) if not supplied.
    """
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT

    headers = block.get("headers", [])
    rows = block.get("rows", [])
    if not headers and not rows:
        return

    # Safe default printable width (A4 21cm - 2*2cm margins ~ 17cm -> 482 pt).
    if available_width_pt is None or available_width_pt <= 0:
        available_width_pt = 482.0

    body_style = ParagraphStyle(
        "_tbl_body", fontName=body_font, fontSize=body_size - 1,
        leading=(body_size - 1) * 1.25, alignment=TA_LEFT, wordWrap="LTR",
        textColor=colors.black,
    )
    header_style = ParagraphStyle(
        "_tbl_header", fontName=bold_font, fontSize=body_size - 1,
        leading=(body_size - 1) * 1.25, alignment=TA_LEFT, wordWrap="LTR",
        textColor=header_text,
    )

    def _cell(raw, style):
        # Escape HTML-specials for ReportLab Paragraph parsing.
        s = "" if raw is None else str(raw)
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(s, style)

    data = []
    if headers:
        data.append([_cell(h, header_style) for h in headers])
    for r in rows:
        data.append([_cell(c, body_style) for c in r])
    if not data:
        return

    n_cols = max(len(row) for row in data)

    # Resolve column widths.
    col_widths_cfg = block.get("column_widths") or []
    col_widths = None
    if col_widths_cfg and len(col_widths_cfg) == n_cols:
        try:
            values = [float(v) for v in col_widths_cfg]
            total = sum(values)
            if total <= 0:
                col_widths = None
            elif all(v <= 1.0 for v in values):
                # Treat as relative weights summing to the printable width.
                col_widths = [v / total * available_width_pt for v in values]
            elif total <= available_width_pt * 1.05:
                # Absolute point values that already fit.
                col_widths = values
            else:
                # Absolute values but too wide - rescale proportionally.
                col_widths = [v / total * available_width_pt for v in values]
        except (TypeError, ValueError):
            col_widths = None

    if col_widths is None:
        # Proportional auto-size based on the longest text in each column.
        raw_lens = []
        for ci in range(n_cols):
            longest = 1
            for row in ([headers] if headers else []) + rows:
                if ci < len(row):
                    cell_text = "" if row[ci] is None else str(row[ci])
                    longest = max(longest, min(60, len(cell_text)))
            raw_lens.append(longest)
        total = sum(raw_lens) or 1
        # Enforce a minimum weight so no column collapses to unreadable.
        min_weight = 0.5
        weights = [max(min_weight, ln / total * n_cols) for ln in raw_lens]
        w_total = sum(weights)
        col_widths = [w / w_total * available_width_pt for w in weights]

    t = Table(data, colWidths=col_widths, repeatRows=1 if headers else 0)
    style_cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, rule_col),
    ]
    if headers:
        style_cmds.append(("BACKGROUND", (0, 0), (-1, 0), header_fill))
        if use_zebra:
            for ri in range(1, len(data)):
                if ri % 2 == 0:
                    style_cmds.append(("BACKGROUND", (0, ri), (-1, ri), zebra_col))
    t.setStyle(TableStyle(style_cmds))
    story.append(t)


def _make_footer(theme, muted_col, font_name, font_size):
    show_gen = theme.get("show_generated_footer", True)
    show_page = theme.get("show_page_numbers", True)
    gen_date = datetime.now().strftime("%d %B %Y")

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, font_size)
        canvas.setFillColor(muted_col)
        parts = []
        if show_gen:
            parts.append(f"Generated {gen_date}")
        if show_page:
            parts.append(f"Page {doc.page}")
        text = "   |   ".join(parts)
        canvas.drawCentredString(doc.pagesize[0] / 2, 1.2 * 28.35, text)  # 1.2 cm from bottom
        canvas.restoreState()

    return _footer


def _escape(text: str) -> str:
    """Escape HTML-special chars for ReportLab Paragraph (which parses minimal HTML)."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

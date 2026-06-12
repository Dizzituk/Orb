# FILE: app/styling/html_builder.py
# Purpose: Build single-file styled HTML reports from a structured content list.
# Called-by: app.debug.executors.styled_files
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Build single-file styled HTML reports from a structured content list.

Same content schema as docx_builder/pdf_builder. Output is one self-contained
HTML file with embedded CSS - no external dependencies, prints cleanly,
respects prefers-color-scheme for dark mode.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_html(
    output_path: str,
    title: str,
    content: List[Dict[str, Any]],
    theme: Dict[str, Any],
    subtitle: Optional[str] = None,
    author: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a styled single-file HTML report. Returns {path, size_bytes, blocks_rendered}."""
    body_parts: List[str] = []
    blocks_rendered = 0

    # Header / cover
    body_parts.append('<header class="doc-header">')
    body_parts.append(f'  <h1 class="doc-title">{html.escape(title)}</h1>')
    if subtitle:
        body_parts.append(f'  <p class="doc-subtitle">{html.escape(subtitle)}</p>')
    meta_text = datetime.now().strftime("%d %B %Y")
    if author:
        meta_text = f"{html.escape(author)} &middot; {meta_text}"
    body_parts.append(f'  <p class="doc-meta">{meta_text}</p>')
    body_parts.append('</header>')

    # Body
    for block in content:
        try:
            btype = block.get("type", "paragraph")
            if btype == "heading":
                level = max(1, min(4, int(block.get("level", 1))))
                # H1 reserved for cover; body uses H2-H5
                tag = f"h{level + 1}"
                body_parts.append(f'<{tag}>{html.escape(block.get("text", ""))}</{tag}>')
            elif btype == "paragraph":
                body_parts.append(f'<p>{html.escape(block.get("text", ""))}</p>')
            elif btype == "list":
                tag = "ol" if block.get("ordered") else "ul"
                items = "".join(f'<li>{html.escape(str(i))}</li>' for i in block.get("items", []))
                body_parts.append(f'<{tag}>{items}</{tag}>')
            elif btype == "table":
                body_parts.append(_render_table(block))
            elif btype == "rule":
                body_parts.append('<hr>')
            elif btype == "spacer":
                body_parts.append('<div class="spacer"></div>')
            elif btype == "code":
                body_parts.append(f'<pre><code>{html.escape(block.get("text", ""))}</code></pre>')
            else:
                body_parts.append(f'<p>{html.escape(str(block))}</p>')
            blocks_rendered += 1
        except Exception:
            continue

    # Footer
    if theme.get("show_generated_footer", True):
        gen = datetime.now().strftime("%d %B %Y")
        body_parts.append(f'<footer class="doc-footer">Generated {gen}</footer>')

    css = _build_css(theme)
    document = _HTML_TEMPLATE.format(
        title_escaped=html.escape(title),
        css=css,
        body="\n".join(body_parts),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    size = out.stat().st_size

    return {
        "path": str(out),
        "size_bytes": size,
        "blocks_rendered": blocks_rendered,
        "theme": theme.get("name", "unknown"),
    }


def _render_table(block: Dict[str, Any]) -> str:
    headers = block.get("headers", [])
    rows = block.get("rows", [])
    parts = ['<table>']
    if headers:
        parts.append('<thead><tr>')
        for h in headers:
            parts.append(f'<th>{html.escape(str(h))}</th>')
        parts.append('</tr></thead>')
    if rows:
        parts.append('<tbody>')
        for r in rows:
            parts.append('<tr>')
            for c in r:
                parts.append(f'<td>{html.escape(str(c) if c is not None else "")}</td>')
            parts.append('</tr>')
        parts.append('</tbody>')
    parts.append('</table>')
    return "".join(parts)


def _hh(value: str) -> str:
    return f"#{(value or '').lstrip('#')}"


def _build_css(theme: Dict[str, Any]) -> str:
    primary = _hh(theme.get("colour_primary", "0B5FFF"))
    text = _hh(theme.get("colour_text", "1A1F2E"))
    muted = _hh(theme.get("colour_muted", "6B7280"))
    rule = _hh(theme.get("colour_rule", "E5E7EB"))
    bg = _hh(theme.get("colour_bg", "FFFFFF"))
    zebra = _hh(theme.get("colour_zebra", "F8FAFC"))
    header_fill = _hh(theme.get("table_header_fill", "0B5FFF"))
    header_text = _hh(theme.get("table_header_text", "FFFFFF"))
    body_font = theme.get("font_body", "Calibri, sans-serif")
    heading_font = theme.get("font_heading", "Calibri, sans-serif")
    mono_font = theme.get("font_mono", "Consolas, monospace")
    body_size = theme.get("body_size_pt", 11)
    h_sizes = theme.get("heading_sizes_pt", [22, 16, 13, 11])
    max_w = theme.get("max_width_px", 820)
    use_zebra = theme.get("table_zebra", True)
    zebra_rule = (
        f"tbody tr:nth-child(even) td {{ background: {zebra}; }}" if use_zebra else ""
    )

    return f"""
    :root {{
      color-scheme: light;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; padding: 0;
      background: {bg};
      color: {text};
      font-family: {body_font};
      font-size: {body_size}pt;
      line-height: 1.55;
    }}
    main {{
      max-width: {max_w}px;
      margin: 0 auto;
      padding: 48px 32px 64px;
    }}
    .doc-header {{
      border-bottom: 2px solid {primary};
      padding-bottom: 16px;
      margin-bottom: 28px;
    }}
    .doc-title {{
      font-family: {heading_font};
      font-size: {h_sizes[0]}pt;
      color: {primary};
      margin: 0 0 6px;
      font-weight: 700;
    }}
    .doc-subtitle {{
      color: {muted};
      font-size: {body_size + 1}pt;
      margin: 0 0 6px;
    }}
    .doc-meta {{
      color: {muted};
      font-size: {body_size - 2}pt;
      margin: 0;
    }}
    h2 {{ font-family: {heading_font}; color: {primary}; font-size: {h_sizes[1]}pt; margin: 28px 0 8px; }}
    h3 {{ font-family: {heading_font}; color: {text}; font-size: {h_sizes[2]}pt; margin: 22px 0 6px; }}
    h4, h5 {{ font-family: {heading_font}; color: {text}; font-size: {h_sizes[3]}pt; margin: 18px 0 4px; }}
    p {{ margin: 0 0 12px; }}
    ul, ol {{ margin: 0 0 14px; padding-left: 24px; }}
    li {{ margin-bottom: 4px; }}
    hr {{ border: none; border-top: 1px solid {rule}; margin: 22px 0; }}
    .spacer {{ height: 16px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 22px;
      font-size: {body_size - 1}pt;
    }}
    th, td {{
      border: 1px solid {rule};
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    thead th {{
      background: {header_fill};
      color: {header_text};
      font-weight: 600;
    }}
    {zebra_rule}
    pre {{
      background: {zebra};
      border: 1px solid {rule};
      border-radius: 4px;
      padding: 12px 14px;
      overflow-x: auto;
      font-family: {mono_font};
      font-size: {body_size - 2}pt;
      margin: 12px 0;
    }}
    code {{ font-family: {mono_font}; }}
    .doc-footer {{
      margin-top: 48px;
      padding-top: 16px;
      border-top: 1px solid {rule};
      color: {muted};
      font-size: {body_size - 2}pt;
      text-align: center;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{ color-scheme: dark; }}
      html, body {{ background: #0d1117; color: #e6edf3; }}
      .doc-subtitle, .doc-meta, .doc-footer {{ color: #8b949e; }}
      h3, h4, h5 {{ color: #e6edf3; }}
      th, td {{ border-color: #30363d; }}
      hr {{ border-top-color: #30363d; }}
      pre {{ background: #161b22; border-color: #30363d; }}
      tbody tr:nth-child(even) td {{ background: #161b22; }}
    }}
    @media print {{
      .doc-footer {{ position: fixed; bottom: 0; }}
      main {{ padding: 24px; }}
    }}
    """


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_escaped}</title>
<style>{css}</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""

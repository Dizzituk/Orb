# FILE: app/llm/chart_renderer.py
"""
Deterministic chart rendering via Plotly → PNG.

Takes structured chart data (extracted by LLM from research results)
and renders pixel-perfect, publication-quality charts.

Requires: pip install plotly kaleido

Zero API cost. Zero rate limits. Guaranteed-accurate data visualisation.

v1.0 (2026-03-20): Initial implementation.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.llm.image_output_dir import get_image_output_dir

logger = logging.getLogger(__name__)

# Chart type constants
BAR = "bar"
HORIZONTAL_BAR = "horizontal_bar"
LINE = "line"
PIE = "pie"
SCATTER = "scatter"
GROUPED_BAR = "grouped_bar"

# ASTRA brand-aligned colour palette
_COLOURS = [
    "#7c3aed",  # Purple (primary)
    "#3b82f6",  # Blue
    "#10b981",  # Green
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#8b5cf6",  # Light purple
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#14b8a6",  # Teal
    "#f97316",  # Orange
]


def _get_plotly():
    """Lazy import of plotly."""
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        logger.error("[chart_renderer] plotly not installed. Run: pip install plotly kaleido")
        return None


def render_chart(
    chart_data: dict,
    output_filename: Optional[str] = None,
    width: int = 1200,
    height: int = 700,
) -> Optional[dict]:
    """Render a chart from structured data and return file info.

    Args:
        chart_data: Dict with keys:
            - chart_type: str (bar, horizontal_bar, line, pie, scatter, grouped_bar)
            - title: str
            - labels: list[str] (x-axis labels / category names)
            - values: list[float] (y-axis values)
            - series: list[dict] (for grouped_bar/line: [{"name": str, "values": list}])
            - x_label: str (optional)
            - y_label: str (optional)
            - subtitle: str (optional)
            - source_note: str (optional, attribution line)
        output_filename: Override filename
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Dict with path, filename, size_bytes, base64_data, mime_type or None
    """
    go = _get_plotly()
    if not go:
        return None

    chart_type = chart_data.get("chart_type", BAR)
    title = chart_data.get("title", "Chart")
    labels = chart_data.get("labels", [])
    values = chart_data.get("values", [])
    series = chart_data.get("series", [])
    x_label = chart_data.get("x_label", "")
    y_label = chart_data.get("y_label", "")
    subtitle = chart_data.get("subtitle", "")
    source_note = chart_data.get("source_note", "")
    annotation_text = chart_data.get("annotation", "") or chart_data.get("annotation_text", "")

    # v1.1: Custom colours from user (overrides default palette)
    custom_colours = chart_data.get("colours") or chart_data.get("colors")
    if custom_colours and isinstance(custom_colours, list):
        colours = custom_colours
    else:
        colours = _COLOURS

    # v1.1: Social media size override (e.g. "1080x1080", "1080x1920")
    size_str = chart_data.get("size", "")
    if size_str and "x" in str(size_str):
        try:
            parts = str(size_str).lower().split("x")
            width = int(parts[0].strip())
            height = int(parts[1].strip())
            logger.info("[chart_renderer] Using custom size: %dx%d", width, height)
        except (ValueError, IndexError):
            pass  # Keep defaults

    try:
        fig = _build_figure(go, chart_type, labels, values, series, colours)
        _apply_theme(fig, title, subtitle, source_note, x_label, y_label, annotation_text)

        # Export to PNG bytes
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)

        b64 = base64.b64encode(img_bytes).decode("ascii")

        # Save to output directory (shared helper — see image_output_dir.py)
        output_dir = get_image_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_filename:
            h = hashlib.md5(title.encode()).hexdigest()[:8]
            ts = datetime.now(timezone.utc).strftime("%H%M%S")
            output_filename = f"chart-{h}-{ts}.png"

        filepath = output_dir / output_filename
        filepath.write_bytes(img_bytes)

        logger.info("[chart_renderer] Saved %s (%d bytes)", filepath, len(img_bytes))

        return {
            "path": str(filepath),
            "filename": output_filename,
            "size_bytes": len(img_bytes),
            "base64_data": b64,
            "mime_type": "image/png",
            "prompt": title,
            "text": None,
        }

    except Exception as e:
        logger.error("[chart_renderer] Failed: %s", e)
        return None


def _build_figure(go, chart_type, labels, values, series, colours=None):
    """Build the Plotly figure based on chart type. v1.1: custom colours."""
    fig = go.Figure()
    if not colours:
        colours = _COLOURS

    if chart_type == GROUPED_BAR and series:
        for i, s in enumerate(series):
            fig.add_trace(go.Bar(
                name=s.get("name", f"Series {i+1}"),
                x=labels,
                y=s.get("values", []),
                marker_color=colours[i % len(colours)],
            ))
        fig.update_layout(barmode="group")

    elif chart_type == HORIZONTAL_BAR:
        # v1.1: Handle both single values and multi-series horizontal bars
        if series:
            for i, s in enumerate(series):
                # Use series-specific colour if provided, else fall back to palette
                s_colour = s.get("colour", s.get("color", ""))
                bar_colour = s_colour if s_colour.startswith("#") else colours[i % len(colours)]
                s_values = s.get("values", [])
                fig.add_trace(go.Bar(
                    name=s.get("name", f"Series {i+1}"),
                    x=s_values,
                    y=labels,
                    orientation="h",
                    marker_color=bar_colour,
                    text=[f"{v}%" for v in s_values],
                    textposition="outside",
                    textfont_size=14,
                ))
            fig.update_layout(barmode="group")
        else:
            fig.add_trace(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colours[:len(labels)],
                text=[str(v) for v in values],
                textposition="outside",
                textfont_size=14,
            ))

    elif chart_type == LINE:
        if series:
            for i, s in enumerate(series):
                fig.add_trace(go.Scatter(
                    x=labels,
                    y=s.get("values", []),
                    mode="lines+markers",
                    name=s.get("name", f"Series {i+1}"),
                    line=dict(color=colours[i % len(colours)], width=3),
                    marker=dict(size=8),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=labels,
                y=values,
                mode="lines+markers",
                line=dict(color=colours[0], width=3),
                marker=dict(size=8),
            ))

    elif chart_type == PIE:
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker_colors=colours[:len(labels)],
            textinfo="label+percent",
            textfont_size=14,
        ))

    elif chart_type == SCATTER:
        fig.add_trace(go.Scatter(
            x=labels,
            y=values,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=12, color=colours[:len(values)]),
        ))

    else:  # Default: vertical bar
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker_color=colours[:len(labels)],
            text=[str(v) for v in values],
            textposition="outside",
            textfont_size=14,
        ))

    return fig


def _apply_theme(fig, title, subtitle, source_note, x_label, y_label, annotation_text=""):
    """Apply a clean dark theme consistent with ASTRA branding."""
    full_title = title
    if subtitle:
        full_title += f"<br><span style='font-size:14px;color:#a0a0a0'>{subtitle}</span>"

    annotations = []
    if source_note:
        annotations.append(dict(
            text=source_note,
            xref="paper", yref="paper",
            x=0, y=-0.12,
            showarrow=False,
            font=dict(size=10, color="#808080"),
        ))

    # v1.2: Right-side annotation for explanatory text
    if annotation_text:
        annotations.append(dict(
            text=annotation_text,
            xref="paper", yref="paper",
            x=1.02, y=0.5,    # Right of chart, vertically centred
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(size=13, color="#d0d0d0"),
            align="left",
            bordercolor="#3a3a5a",
            borderwidth=1,
            borderpad=10,
            bgcolor="rgba(26, 26, 46, 0.8)",
        ))

    fig.update_layout(
        title=dict(
            text=full_title,
            font=dict(size=22, color="#ffffff"),
            x=0.5,
            xanchor="center",
        ),
        font=dict(family="Inter, Arial, sans-serif", color="#e0e0e0"),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        xaxis=dict(
            title=x_label or None,
            gridcolor="#2a2a4a",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=y_label or None,
            gridcolor="#2a2a4a",
            tickfont=dict(size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="#2a2a4a",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(
            l=60,
            r=280 if annotation_text else 40,  # v1.2: widen right margin for annotation
            t=80,
            b=60 if not source_note else 80,
        ),
        annotations=annotations,
    )


__all__ = [
    "render_chart",
    "BAR", "HORIZONTAL_BAR", "LINE", "PIE", "SCATTER", "GROUPED_BAR",
]

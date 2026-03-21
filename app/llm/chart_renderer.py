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

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.getenv("ASTRA_OUTPUT_DIR", r"D:\Orb\output")

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

    try:
        fig = _build_figure(go, chart_type, labels, values, series)
        _apply_theme(fig, title, subtitle, source_note, x_label, y_label)

        # Export to PNG bytes
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)

        b64 = base64.b64encode(img_bytes).decode("ascii")

        # Save to output directory
        output_dir = Path(OUTPUT_DIR) / "images"
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


def _build_figure(go, chart_type, labels, values, series):
    """Build the Plotly figure based on chart type."""
    fig = go.Figure()

    if chart_type == GROUPED_BAR and series:
        for i, s in enumerate(series):
            fig.add_trace(go.Bar(
                name=s.get("name", f"Series {i+1}"),
                x=labels,
                y=s.get("values", []),
                marker_color=_COLOURS[i % len(_COLOURS)],
            ))
        fig.update_layout(barmode="group")

    elif chart_type == HORIZONTAL_BAR:
        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=_COLOURS[:len(labels)],
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
                    line=dict(color=_COLOURS[i % len(_COLOURS)], width=3),
                    marker=dict(size=8),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=labels,
                y=values,
                mode="lines+markers",
                line=dict(color=_COLOURS[0], width=3),
                marker=dict(size=8),
            ))

    elif chart_type == PIE:
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker_colors=_COLOURS[:len(labels)],
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
            marker=dict(size=12, color=_COLOURS[:len(values)]),
        ))

    else:  # Default: vertical bar
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker_color=_COLOURS[:len(labels)],
            text=[str(v) for v in values],
            textposition="outside",
            textfont_size=14,
        ))

    return fig


def _apply_theme(fig, title, subtitle, source_note, x_label, y_label):
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
        margin=dict(l=60, r=40, t=80, b=60 if not source_note else 80),
        annotations=annotations,
    )


__all__ = [
    "render_chart",
    "BAR", "HORIZONTAL_BAR", "LINE", "PIE", "SCATTER", "GROUPED_BAR",
]

# FILE: app/reports/renderer.py
# Purpose: Build a styled self-contained HTML report from a watcher ledger (chart + freshness + dated table).
# Called-by: app.reports.tools_registration
# Depends-on: app.styling.html_builder, app.styling.themes, app.llm.chart_renderer, app.watchers.framework
# Last-renovated: 2026-07-01
"""
Deterministic composition, no LLM: series -> Plotly PNG (embedded base64 so
the single file travels to the phone intact) -> html_builder blocks with a
freshness line ("data through <date>") and the dated table of readings.
Works for any watcher; anything tabular with (date, key, value) can reuse
build_series_report directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.reports.cache import get_reports_cache_dir, make_report_filename

logger = logging.getLogger(__name__)


def _chart_data_uri(title: str, unit: str, labels: List[str], series: List[dict]) -> Optional[str]:
    """Line chart PNG as a data URI; None when plotly is unavailable."""
    try:
        from app.llm.chart_renderer import render_chart

        rendered = render_chart(
            {
                "chart_type": "line",
                "title": title,
                "labels": labels,
                "series": series,
                "y_label": unit,
                "x_label": "date",
            },
            width=1200,
            height=600,
        )
        if rendered and rendered.get("base64_data"):
            return f"data:image/png;base64,{rendered['base64_data']}"
    except Exception as exc:
        logger.warning("[reports.renderer] chart render failed: %s", exc)
    return None


def build_series_report(
    slug: str,
    title: str,
    unit: str,
    by_key: Dict[str, List[dict]],
    latest: Dict[str, dict],
    max_table_rows: int = 120,
) -> Dict[str, Any]:
    """Render {series-per-key} into the cache. Returns {path, filename, title,
    data_through}. by_key: {key: [{date, value, source?, note?}, ...]} oldest-first."""
    from app.styling.html_builder import build_html
    from app.styling.themes import pick_theme

    all_dates = sorted({p["date"] for pts in by_key.values() for p in pts})
    data_through = all_dates[-1] if all_dates else "no data yet"

    chart_series = []
    for key, pts in sorted(by_key.items()):
        vals_by_date = {p["date"]: p["value"] for p in pts}
        chart_series.append({"name": key, "values": [vals_by_date.get(d) for d in all_dates]})
    chart_uri = _chart_data_uri(title, unit, all_dates, chart_series) if all_dates else None

    rendered_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    blocks: List[Dict[str, Any]] = [
        {"type": "paragraph", "text": f"Data through {data_through} — rendered {rendered_at}."},
    ]
    if chart_uri:
        blocks.append({"type": "image", "src": chart_uri, "alt": f"{title} — daily series", "caption": f"{title} ({unit})"})
    else:
        blocks.append({"type": "paragraph", "text": "(chart unavailable — plotly not installed or no data)"})

    blocks.append({"type": "heading", "level": 1, "text": "Latest"})
    blocks.append({
        "type": "table",
        "headers": ["Key", f"Latest ({unit})", "Change over window", "As of"],
        "rows": [
            [
                key,
                info.get("value") if info.get("value") is not None else "—",
                (f"{info['change_pct_over_window']:+.1f}%" if info.get("change_pct_over_window") is not None else "—"),
                info.get("date") or "—",
            ]
            for key, info in sorted(latest.items())
        ],
    })

    blocks.append({"type": "heading", "level": 1, "text": "Daily readings"})
    flat = [
        {"date": p["date"], "key": key, "value": p["value"], "note": p.get("note") or p.get("source") or ""}
        for key, pts in by_key.items()
        for p in pts
    ]
    flat.sort(key=lambda r: (r["date"], r["key"]), reverse=True)
    blocks.append({
        "type": "table",
        "headers": ["Date", "Key", f"Value ({unit})", "Source / note"],
        "rows": [
            [r["date"], r["key"], r["value"] if r["value"] is not None else "—", r["note"][:80]]
            for r in flat[:max_table_rows]
        ],
    })

    filename = make_report_filename(slug)
    output_path = str(get_reports_cache_dir() / filename)
    theme = pick_theme(f"{slug} report", "auto")
    build_html(output_path, title, blocks, theme, subtitle=f"ASTRA watcher report — data through {data_through}")
    return {"path": output_path, "filename": filename, "title": title, "data_through": data_through}


def render_watcher_report(db: Session, watcher_id: str, days: int = 30) -> Optional[Dict[str, Any]]:
    """Ledger -> report for one registered watcher. None if unknown watcher."""
    from app.watchers.framework import get_watcher, series_payload

    spec = get_watcher(watcher_id)
    if spec is None:
        return None
    payload = series_payload(db, spec, days)
    return build_series_report(
        slug=spec.watcher_id.replace("_", "-"),
        title=spec.title,
        unit=spec.unit,
        by_key=payload["series"],
        latest=payload["latest"],
    )

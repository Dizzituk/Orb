# FILE: app/orchestrator/scaffolds/templates.py
# Purpose: Scaffold Templates — pre-built code skeletons for common patterns.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Scaffold Templates — pre-built code skeletons for common patterns.

Each template function takes structured data (file path, props,
imports, exports, design tokens) and produces a TypeScript/TSX
skeleton with [LLM_FILL] markers where business logic belongs.

v1.0 (2026-03-01): Templates for view, grid, detail, data, css patterns.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def scaffold_view_component(
    file_path: str,
    component_name: str,
    frozen_imports: str = "",
    exports: Optional[List[str]] = None,
    props_interface: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
    architecture_hints: str = "",
) -> str:
    """Generate scaffold for a View/container component.

    View components are top-level containers that manage state,
    handle tab switching, and render child components conditionally.
    Pattern: InvestmentsView, FinanceView, LifestyleView, etc.
    """
    _imports = frozen_imports or f"import {{ useState, useCallback }} from 'react';"
    _css_class = _to_css_class(component_name)
    _props = props_interface or f"// No props — top-level view component"
    _token_vars = _extract_relevant_tokens(design_tokens, "view") if design_tokens else ""

    lines = [
        f"// {file_path}",
        f"/**",
        f" * [LLM_FILL: Component doc comment — describe purpose, layout, behaviour]",
        f" */",
        f"",
        _imports,
        f"",
    ]

    # Props interface (if needed)
    if "interface" not in _imports and "Props" not in _imports:
        lines.extend([
            f"interface {component_name}Props {{",
            f"  [LLM_FILL: Define props — callbacks, data, configuration]",
            f"}}",
            f"",
        ])

    lines.extend([
        f"export function {component_name}(/* [LLM_FILL: destructured props] */) {{",
        f"  // ── State ──────────────────────────────────────────",
        f"  [LLM_FILL: useState declarations for component state]",
        f"",
        f"  // ── Handlers ───────────────────────────────────────",
        f"  [LLM_FILL: useCallback handlers for user interactions]",
        f"",
        f"  // ── Render ─────────────────────────────────────────",
        f"  return (",
        f"    <div className=\"{_css_class}\">",
        f"      [LLM_FILL: Component JSX — header, conditional content, child components]",
        f"    </div>",
        f"  );",
        f"}}",
    ])

    return "\n".join(lines)


def scaffold_grid_component(
    file_path: str,
    component_name: str,
    frozen_imports: str = "",
    exports: Optional[List[str]] = None,
    props_interface: str = "",
    data_type: str = "",
    data_source: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate scaffold for a Grid/List component.

    Grid components map over a data array and render cards/items.
    Pattern: EducationCourseGrid, InvestmentsStocksGrid, etc.
    """
    _imports = frozen_imports or "import React from 'react';"
    _css_class = _to_css_class(component_name)
    _item_type = data_type or "[LLM_FILL: item type]"
    _data_var = data_source or "[LLM_FILL: data source]"

    lines = [
        f"// {file_path}",
        f"/**",
        f" * [LLM_FILL: Component doc comment — what items are displayed, selection behaviour]",
        f" */",
        f"",
        _imports,
        f"",
        f"interface {component_name}Props {{",
        f"  onSelect: (id: string) => void;",
        f"  [LLM_FILL: additional props — filters, search, sorting]",
        f"}}",
        f"",
        f"export function {component_name}({{ onSelect }}: {component_name}Props) {{",
        f"  return (",
        f"    <div className=\"{_css_class}\">",
        f"      {{{_data_var}.map((item: {_item_type}) => (",
        f"        <button",
        f"          key={{item.id}}",
        f"          className=\"{_css_class}__card\"",
        f"          onClick={{() => onSelect(item.id)}}",
        f"        >",
        f"          [LLM_FILL: Card content — title, description, metadata, CTA]",
        f"        </button>",
        f"      ))}}",
        f"    </div>",
        f"  );",
        f"}}",
    ]

    return "\n".join(lines)


def scaffold_detail_component(
    file_path: str,
    component_name: str,
    frozen_imports: str = "",
    exports: Optional[List[str]] = None,
    props_interface: str = "",
    data_type: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate scaffold for a Detail/Single-item component.

    Detail components display one selected item with full information
    and a back button. Pattern: EducationCourseDetail, StockDetail, etc.
    """
    _imports = frozen_imports or "import React from 'react';"
    _css_class = _to_css_class(component_name)
    _item_type = data_type or "[LLM_FILL: item type]"

    lines = [
        f"// {file_path}",
        f"/**",
        f" * [LLM_FILL: Component doc comment — what detail is shown, navigation]",
        f" */",
        f"",
        _imports,
        f"",
        f"interface {component_name}Props {{",
        f"  selectedId: string;",
        f"  onBack: () => void;",
        f"  [LLM_FILL: additional props]",
        f"}}",
        f"",
        f"export function {component_name}({{ selectedId, onBack }}: {component_name}Props) {{",
        f"  // ── Data lookup ────────────────────────────────────",
        f"  [LLM_FILL: Find the selected item from data source by ID]",
        f"",
        f"  // ── Guard ──────────────────────────────────────────",
        f"  if (!/* found item */) {{",
        f"    return <div className=\"{_css_class}--not-found\">Item not found</div>;",
        f"  }}",
        f"",
        f"  // ── Render ─────────────────────────────────────────",
        f"  return (",
        f"    <div className=\"{_css_class}\">",
        f"      <button className=\"{_css_class}__back\" onClick={{onBack}}>",
        f"        ← Back",
        f"      </button>",
        f"      [LLM_FILL: Detail content — heading, sections, nested lists]",
        f"    </div>",
        f"  );",
        f"}}",
    ]

    return "\n".join(lines)


def scaffold_data_module(
    file_path: str,
    frozen_imports: str = "",
    exports: Optional[List[str]] = None,
    architecture_hints: str = "",
) -> str:
    """Generate scaffold for a data/types module.

    Data modules define TypeScript interfaces and export mock/dummy
    data arrays. Pattern: education-data.ts, investments-data.ts, etc.
    """
    lines = [
        f"// {file_path}",
        f"/**",
        f" * [LLM_FILL: Module doc comment — what data types and mock data]",
        f" */",
        f"",
    ]

    if frozen_imports:
        lines.append(frozen_imports)
        lines.append("")

    lines.extend([
        f"// ── Type Definitions ─────────────────────────────────",
        f"",
        f"[LLM_FILL: TypeScript interfaces for the domain entities.",
        f" Each interface must have an `id: string` field.",
        f" Export all interfaces.]",
        f"",
        f"// ── Mock Data ──────────────────────────────────────",
        f"",
        f"[LLM_FILL: Export const arrays of mock data.",
        f" Use DUMMY_ prefix naming if appropriate.",
        f" Each array should have 3-6 items for demo purposes.",
        f" Data must match the interfaces above exactly.]",
    ])

    return "\n".join(lines)


def scaffold_css_module(
    file_path: str,
    component_name: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate scaffold for a CSS module.

    CSS modules define component styles using CSS variables from
    the design token registry. Pattern: education.css, investments.css.
    """
    _css_class = _to_css_class(component_name) if component_name else "[component-class]"

    # Build token reference comment
    token_ref = ""
    if design_tokens:
        cats = design_tokens.get("categories", {})
        colours = cats.get("Colours", {})
        spacing = cats.get("Spacing", {})
        if colours:
            top_colours = list(colours.items())[:8]
            token_ref += "/* Available colour tokens:\n"
            for name, val in top_colours:
                token_ref += f"   {name}: {val}\n"
            token_ref += "*/\n\n"
        if spacing:
            top_spacing = list(spacing.items())[:4]
            token_ref += "/* Available spacing tokens:\n"
            for name, val in top_spacing:
                token_ref += f"   {name}: {val}\n"
            token_ref += "*/\n\n"

    lines = [
        f"/* {file_path} */",
        f"/**",
        f" * [LLM_FILL: Module doc comment — which components are styled]",
        f" */",
        f"",
    ]

    if token_ref:
        lines.append(token_ref)

    lines.extend([
        f"/* ── Container ──────────────────────────────────────── */",
        f"",
        f".{_css_class} {{",
        f"  [LLM_FILL: Container styles — layout, padding, background]",
        f"  /* USE: var(--bg-panel), var(--radius-md), var(--border-primary) */",
        f"}}",
        f"",
        f"/* ── Cards / Items ─────────────────────────────────── */",
        f"",
        f".{_css_class}__card {{",
        f"  [LLM_FILL: Card styles — border, padding, hover state]",
        f"  /* USE: var(--bg-secondary), var(--border-primary), var(--radius-sm) */",
        f"}}",
        f"",
        f".{_css_class}__card:hover {{",
        f"  [LLM_FILL: Hover state — border colour, background shift]",
        f"  /* USE: var(--accent-subtle-border-hover), var(--bg-hover) */",
        f"}}",
        f"",
        f"/* ── Typography ─────────────────────────────────────── */",
        f"",
        f"[LLM_FILL: Title, description, metadata text styles",
        f" USE: var(--text-primary), var(--text-secondary), var(--text-muted)]",
        f"",
        f"/* ── Responsive ───────────────────────────────────── */",
        f"",
        f"[LLM_FILL: Grid/flex responsive breakpoints if needed]",
    ])

    return "\n".join(lines)


# ─── Helpers ───────────────────────────────────────────────────────

def _to_css_class(component_name: str) -> str:
    """Convert PascalCase component name to kebab-case CSS class.

    EducationCourseGrid -> education-course-grid
    InvestmentsView -> investments-view
    """
    # Insert hyphens before uppercase letters
    s = re.sub(r"([A-Z])", r"-\1", component_name).lower().lstrip("-")
    return s


def _extract_relevant_tokens(
    design_tokens: Optional[Dict[str, Any]],
    pattern: str,
) -> str:
    """Extract CSS variable references relevant to a pattern."""
    if not design_tokens:
        return ""

    cats = design_tokens.get("categories", {})
    relevant: List[str] = []

    if pattern in ("view", "grid", "detail"):
        # Layout tokens
        for name in ("--bg-panel", "--bg-secondary", "--border-primary",
                      "--radius-md", "--radius-sm", "--text-primary",
                      "--text-secondary", "--accent-purple"):
            colours = cats.get("Colours", {})
            spacing = cats.get("Spacing", {})
            if name in colours or name in spacing:
                relevant.append(name)

    if relevant:
        return f"/* Design tokens: {', '.join(relevant)} */"
    return ""

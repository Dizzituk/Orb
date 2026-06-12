# FILE: app/orchestrator/arch_template/token_registry.py
# Purpose: Design Token Registry — extracts CSS variables from the codebase.
# Called-by: app.llm.critical_pipeline.stream_handler, app.orchestrator.arch_template.token_cache
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Design Token Registry — extracts CSS variables from the codebase.

v1.0 (2026-03-01): Reads base.css, themes.css, fonts.css from the
sandbox and extracts all --var-name: value pairs into a structured
registry. Used by the architecture template engine to inject
deterministic styling constraints.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TOKEN_REGISTRY_BUILD_ID = "2026-03-01-v1.0-css-extraction"

# CSS files to scan for design tokens (in priority order)
_CSS_SOURCES = [
    "src/styles/base.css",
    "src/styles/themes.css",
    "src/styles/fonts.css",
    "src/index.css",
    "src/App.css",
]

# Regex to match CSS variable declarations: --var-name: value;
_CSS_VAR_RE = re.compile(
    r"(--[\w-]+)\s*:\s*([^;]+);"
)

# Category inference from variable name
_CATEGORY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Colours", re.compile(
        r"--(?:color|bg|background|border|text|accent|surface|shadow|"
        r"success|warning|error|danger|bubble|scrollbar|gradient)", re.I,
    )),
    ("Spacing", re.compile(r"--(?:spacing|gap|padding|margin|radius)", re.I)),
    ("Typography", re.compile(r"--(?:font|text-size|line-height|letter)", re.I)),
    ("Layout", re.compile(r"--(?:width|height|max-|min-|sidebar|header|nav)", re.I)),
    ("Animation", re.compile(r"--(?:transition|animation|duration|ease|delay)", re.I)),
    ("Z-Index", re.compile(r"--(?:z-|zindex)", re.I)),
]


def _categorise_variable(name: str) -> str:
    """Assign a CSS variable to a category based on its name."""
    for cat_name, pattern in _CATEGORY_PATTERNS:
        if pattern.match(name):
            return cat_name
    return "Other"


def extract_tokens_from_css(content: str) -> Dict[str, Dict[str, str]]:
    """Extract CSS variables from a CSS string.

    Returns:
        Dict of {category: {var_name: value}}, e.g.:
        {"Colours": {"--bg-primary": "#1a1a2e"}, ...}
    """
    categories: Dict[str, Dict[str, str]] = {}

    for match in _CSS_VAR_RE.finditer(content):
        var_name = match.group(1).strip()
        var_value = match.group(2).strip()
        category = _categorise_variable(var_name)

        if category not in categories:
            categories[category] = {}
        categories[category][var_name] = var_value

    return categories


def extract_font_families(content: str) -> List[str]:
    """Extract font-family declarations from CSS."""
    fonts: List[str] = []
    for match in re.finditer(r"font-family:\s*([^;]+);", content):
        family = match.group(1).strip().strip("'\"")
        if family and family not in fonts:
            fonts.append(family)
    return fonts


def build_token_registry(
    client: Any,
    frontend_base: str = r"D:\orb-desktop",
) -> Optional[Dict[str, Any]]:
    """Build a complete design token registry from the sandbox CSS files.

    Args:
        client: SandboxClient instance.
        frontend_base: Path to frontend repo in sandbox.

    Returns:
        Registry dict with 'categories' and 'fonts' keys, or None on error.
    """
    all_categories: Dict[str, Dict[str, str]] = {}
    all_fonts: List[str] = []

    for css_path in _CSS_SOURCES:
        abs_path = f"{frontend_base}\\{css_path}".replace("/", "\\")
        try:
            result = client._request(
                "POST", "/fs/contents",
                json_body={"paths": [abs_path], "max_file_size": 50000},
            )
            files = result.get("files", [])
            if not files or "error" in files[0]:
                continue

            content = files[0].get("content", "")
            if not content:
                continue

            cats = extract_tokens_from_css(content)
            for cat_name, tokens in cats.items():
                if cat_name not in all_categories:
                    all_categories[cat_name] = {}
                all_categories[cat_name].update(tokens)

            fonts = extract_font_families(content)
            for f in fonts:
                if f not in all_fonts:
                    all_fonts.append(f)

            logger.info(
                "[token_registry] Extracted %d tokens from %s",
                sum(len(v) for v in cats.values()), css_path,
            )

        except Exception as exc:
            logger.debug("[token_registry] Could not read %s: %s", css_path, exc)
            continue

    if not all_categories and not all_fonts:
        return None

    return {
        "categories": all_categories,
        "fonts": all_fonts,
    }


def build_token_registry_from_content(
    css_contents: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build token registry from pre-read CSS content.

    For use when CSS files have already been read (e.g. from evidence).

    Args:
        css_contents: {file_path: file_content} dict.

    Returns:
        Registry dict or None.
    """
    all_categories: Dict[str, Dict[str, str]] = {}
    all_fonts: List[str] = []

    for path, content in css_contents.items():
        cats = extract_tokens_from_css(content)
        for cat_name, tokens in cats.items():
            if cat_name not in all_categories:
                all_categories[cat_name] = {}
            all_categories[cat_name].update(tokens)

        fonts = extract_font_families(content)
        for f in fonts:
            if f not in all_fonts:
                all_fonts.append(f)

    if not all_categories and not all_fonts:
        return None

    return {
        "categories": all_categories,
        "fonts": all_fonts,
    }

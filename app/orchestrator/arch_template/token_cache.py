# FILE: app/orchestrator/arch_template/token_cache.py
# Purpose: Design Token Cache — persistent storage for extracted CSS tokens.
# Called-by: app.llm.critical_pipeline.stream_handler
# Depends-on: app.orchestrator.arch_template.token_registry, app.sandbox_fs
# Last-renovated: 2026-06-11
"""
Design Token Cache — persistent storage for extracted CSS tokens.

v1.0 (2026-03-01): Manages the token registry lifecycle:
  - Builds the registry from CSS files (via sandbox or host)
  - Saves to JSON alongside skeleton contracts in the job directory
  - Loads from cache on subsequent segment runs (avoid re-extraction)
  - Invalidates when CSS source files change (future: TTL/hash check)

This is the shared infrastructure that Jobs 4, 6, and 7 consume.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .token_registry import (
    build_token_registry_from_content,
    extract_font_families,
    extract_tokens_from_css,
)

logger = logging.getLogger(__name__)

TOKEN_CACHE_BUILD_ID = "2026-03-01-v1.0-shared-cache"

# Cache filename (stored in job_dir/segments/)
CACHE_FILENAME = "design_token_registry.json"

# Core CSS files to read from the frontend codebase
_CORE_CSS_PATHS = [
    r"src\styles\base.css",
    r"src\styles\themes.css",
    r"src\styles\fonts.css",
    r"src\styles\index.css",
]


def _read_css_from_host(frontend_base: str = r"D:\orb-desktop") -> Dict[str, str]:
    """Read CSS files directly from the host filesystem.

    This is the primary path when running in the orchestrator process
    (which has direct disk access). Falls back gracefully if files
    don't exist.
    """
    css_contents: Dict[str, str] = {}

    for rel_path in _CORE_CSS_PATHS:
        abs_path = os.path.join(frontend_base, rel_path)
        try:
            from app.sandbox_fs import sandbox_read_text
            _css = sandbox_read_text(abs_path)
            if _css is not None:
                css_contents[rel_path] = _css
                logger.debug("[token_cache] Read %s from sandbox (%d chars)", rel_path, len(_css))
        except Exception as exc:
            logger.debug("[token_cache] Failed to read %s: %s", rel_path, exc)
    return css_contents


def _read_css_from_sandbox(
    client: Any,
    frontend_base: str = r"D:\orb-desktop",
) -> Dict[str, str]:
    """Read CSS files from the sandbox via the SandboxClient.

    Fallback path when host filesystem isn't available.
    """
    css_contents: Dict[str, str] = {}

    paths_to_read = [
        os.path.join(frontend_base, p).replace("/", "\\")
        for p in _CORE_CSS_PATHS
    ]

    try:
        result = client._request(
            "POST", "/fs/contents",
            json_body={"paths": paths_to_read, "max_file_size": 100000},
        )
        for file_info in result.get("files", []):
            if "error" not in file_info and file_info.get("content"):
                # Map back to relative path
                full_path = file_info.get("path", "")
                for rel in _CORE_CSS_PATHS:
                    if full_path.replace("/", "\\").endswith(rel.replace("/", "\\")):
                        css_contents[rel] = file_info["content"]
                        break
    except Exception as exc:
        logger.warning("[token_cache] Sandbox CSS read failed: %s", exc)

    return css_contents


def build_and_cache_registry(
    job_dir: str,
    frontend_base: str = r"D:\orb-desktop",
    sandbox_client: Any = None,
) -> Optional[Dict[str, Any]]:
    """Build the design token registry and cache it to the job directory.

    Tries host filesystem first, falls back to sandbox. Saves the
    registry as JSON for subsequent segment runs to load.

    Args:
        job_dir: Job directory path (e.g. jobs/jobs/sg-xxxxx).
        frontend_base: Frontend repo root path.
        sandbox_client: Optional SandboxClient for sandbox reads.

    Returns:
        Token registry dict, or None if no CSS found.
    """
    start = time.time()

    # Try host first, then sandbox
    css_contents = _read_css_from_host(frontend_base)
    if not css_contents and sandbox_client:
        css_contents = _read_css_from_sandbox(sandbox_client, frontend_base)

    if not css_contents:
        logger.info("[token_cache] No CSS files found — skipping registry build")
        return None

    # Build the registry
    registry = build_token_registry_from_content(css_contents)
    if not registry:
        return None

    # Add metadata
    registry["_meta"] = {
        "build_id": TOKEN_CACHE_BUILD_ID,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_files": list(css_contents.keys()),
        "total_tokens": sum(len(t) for t in registry.get("categories", {}).values()),
        "build_ms": int((time.time() - start) * 1000),
    }

    # Extract theme variants
    themes = _extract_theme_names(css_contents)
    if themes:
        registry["themes"] = themes

    # Cache to disk
    cache_path = _get_cache_path(job_dir)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        logger.info(
            "[token_cache] Registry cached: %d tokens, %d categories, %d fonts → %s",
            registry["_meta"]["total_tokens"],
            len(registry.get("categories", {})),
            len(registry.get("fonts", [])),
            cache_path,
        )
    except Exception as exc:
        logger.warning("[token_cache] Cache write failed (non-fatal): %s", exc)

    return registry


def load_cached_registry(job_dir: str) -> Optional[Dict[str, Any]]:
    """Load a previously cached token registry from the job directory.

    Returns None if no cache exists.
    """
    cache_path = _get_cache_path(job_dir)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        logger.info(
            "[token_cache] Loaded cached registry: %d tokens",
            registry.get("_meta", {}).get("total_tokens", 0),
        )
        return registry
    except Exception as exc:
        logger.warning("[token_cache] Cache read failed: %s", exc)
        return None


def get_or_build_registry(
    job_dir: str,
    frontend_base: str = r"D:\orb-desktop",
    sandbox_client: Any = None,
) -> Optional[Dict[str, Any]]:
    """Load cached registry or build fresh if not cached.

    This is the main entry point for consumers (Jobs 4, 6, 7).
    """
    cached = load_cached_registry(job_dir)
    if cached:
        return cached

    return build_and_cache_registry(
        job_dir=job_dir,
        frontend_base=frontend_base,
        sandbox_client=sandbox_client,
    )


def get_token_value(
    registry: Dict[str, Any],
    token_name: str,
) -> Optional[str]:
    """Look up a single token value by name.

    Args:
        registry: Token registry dict.
        token_name: CSS variable name (e.g. '--bg-primary').

    Returns:
        The token value, or None if not found.
    """
    for tokens in registry.get("categories", {}).values():
        if token_name in tokens:
            return tokens[token_name]
    return None


def get_tokens_by_category(
    registry: Dict[str, Any],
    category: str,
) -> Dict[str, str]:
    """Get all tokens in a category.

    Args:
        registry: Token registry dict.
        category: Category name (e.g. 'Colours', 'Spacing').

    Returns:
        Dict of {token_name: value}, empty if category not found.
    """
    return registry.get("categories", {}).get(category, {})


def get_theme_overrides(
    registry: Dict[str, Any],
    theme_name: str,
) -> Dict[str, str]:
    """Get token overrides for a specific theme.

    Args:
        registry: Token registry dict.
        theme_name: Theme name (e.g. 'minimal').

    Returns:
        Dict of {token_name: override_value}, empty if theme not found.
    """
    return registry.get("themes", {}).get(theme_name, {})


def _get_cache_path(job_dir: str) -> str:
    """Get the cache file path for a job."""
    return os.path.join(job_dir, "segments", CACHE_FILENAME)


def _extract_theme_names(css_contents: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Extract theme variant names and their token overrides from themes.css."""
    import re

    themes: Dict[str, Dict[str, str]] = {}

    for path, content in css_contents.items():
        if "theme" not in path.lower():
            continue

        # Find [data-theme="xxx"] blocks
        # Pattern: [data-theme="name"] { ... }
        for block_match in re.finditer(
            r'\[data-theme="(\w+)"\]\s*\{([^}]+)\}',
            content,
            re.DOTALL,
        ):
            theme_name = block_match.group(1)
            block_content = block_match.group(2)
            tokens = {}

            for var_match in re.finditer(r'(--[\w-]+)\s*:\s*([^;]+);', block_content):
                tokens[var_match.group(1).strip()] = var_match.group(2).strip()

            if tokens:
                themes[theme_name] = tokens

    return themes

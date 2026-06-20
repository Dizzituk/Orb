# FILE: app/debug/orchestrator/arch_context.py
# Purpose: Build compact, truncated architecture-memory context for a spawned
#          sub-agent brief (WI-5) - the signatures of the brief's seed files plus
#          the backend's most-depended-on modules - so the agent starts oriented
#          instead of spending tool calls re-discovering structure.
# Called-by: app.debug.orchestrator.spawn_tool
# Depends-on: stdlib only (reads .architecture/*.json scans)
# Last-renovated: 2026-06-17
"""Scoped architecture memory for sub-agents.

Sources the cross-repo `.architecture` scans (IMPORT_GRAPH.json +
SIGNATURES_*.json). Deliberately bounded: only the seed files' symbol
signatures and a short hot-modules hint are injected, hard-capped to a few KB,
so a fan-out can't blow the model's context. Cached at module level so the
(potentially large) signature scans load once per process.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Repo roots that may carry a .architecture scan (cross-repo). Backend first
# (richest). Missing dirs are skipped gracefully.
_ARCH_DIRS = [
    Path("D:/Orb/.architecture"),
    Path("D:/orb-desktop/.architecture"),
    Path("D:/Astra Android Folder/Astra-Bridge/.architecture"),
]

_MAX_ARCH_CHARS = 3000
_MAX_SEED_FILES = 6
_MAX_SYMBOLS_PER_FILE = 14
_cache: Dict[str, Any] = {}


def _load_latest(pattern: str, arch_dir: Path) -> Optional[dict]:
    try:
        if not arch_dir.exists():
            return None
        files = sorted(arch_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return None
        return json.loads(files[0].read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        logger.debug("[arch_context] load %s in %s failed: %s", pattern, arch_dir, e)
        return None


def _signatures_by_file() -> Dict[str, list]:
    """Merged {abs_path: [symbol, ...]} across all repos' latest SIGNATURES scan."""
    if "sigs" in _cache:
        return _cache["sigs"]
    merged: Dict[str, list] = {}
    for d in _ARCH_DIRS:
        data = _load_latest("SIGNATURES_*.json", d)
        if data and isinstance(data.get("by_file"), dict):
            merged.update(data["by_file"])
    _cache["sigs"] = merged
    return merged


def _hot_modules() -> List[str]:
    if "hot" in _cache:
        return _cache["hot"]
    out: List[str] = []
    g = _load_latest("IMPORT_GRAPH.json", _ARCH_DIRS[0])
    if g:
        for m in (g.get("stats", {}).get("most_depended_on", []) or [])[:8]:
            mod = m.get("module")
            dep = m.get("dependents")
            if mod:
                out.append(f"{mod} ({dep} dependents)")
    _cache["hot"] = out
    return out


def _match_file(path: str, by_file: Dict[str, list]) -> Optional[str]:
    """Resolve a (possibly relative) context-file path to a by_file key."""
    if not path:
        return None
    p = path.replace("/", "\\").lower()
    for k in by_file:
        if k.lower() == p:
            return k
    for k in by_file:
        kl = k.lower()
        if kl.endswith("\\" + p) or p.endswith(kl) or kl.endswith(p):
            return k
    return None


def _fmt_symbols(syms: list, limit: int = _MAX_SYMBOLS_PER_FILE) -> str:
    lines: List[str] = []
    for s in syms[:limit]:
        kind = s.get("kind", "")
        name = s.get("name", "")
        sig = s.get("signature", "")
        ln = s.get("line", "")
        sig_part = sig if (sig and sig.startswith("(")) else ""
        lines.append(f"  {kind} {name}{sig_part}  :{ln}".rstrip())
    if len(syms) > limit:
        lines.append(f"  ... (+{len(syms) - limit} more)")
    return "\n".join(lines)


def build_arch_context(context_files: Optional[List[str]], target_project: str = "") -> Optional[str]:
    """Compact architecture memory for a brief: signatures of its seed files plus
    the backend's hot modules. Returns None if nothing useful is available."""
    try:
        by_file = _signatures_by_file()
        parts: List[str] = []

        hot = _hot_modules()
        if hot:
            parts.append("Most-depended-on backend modules:\n  " + "\n  ".join(hot))

        seen = 0
        for cf in (context_files or []):
            key = _match_file(cf, by_file)
            if not key:
                continue
            syms = by_file.get(key) or []
            if not syms:
                continue
            parts.append(f"Signatures for {key}:\n{_fmt_symbols(syms)}")
            seen += 1
            if seen >= _MAX_SEED_FILES:
                break

        if not parts:
            return None
        block = "[ARCHITECTURE MEMORY]\n" + "\n\n".join(parts)
        if len(block) > _MAX_ARCH_CHARS:
            block = block[:_MAX_ARCH_CHARS] + "\n... [arch memory truncated]"
        return block
    except Exception as e:
        logger.debug("[arch_context] build failed: %s", e)
        return None

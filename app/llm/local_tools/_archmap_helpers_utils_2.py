# Purpose: archmap helpers utils 2
# Called-by: app.llm.local_tools.archmap_helpers
# Depends-on: app.llm.local_tools.archmap_helpers
# Last-renovated: 2026-06-11
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional


def _read_controller_addr_txt(cache_root: Path) -> Optional[str]:
    addr_file = cache_root / "controller_addr.txt"
    try:
        if addr_file.exists():
            v = addr_file.read_text(encoding="utf-8", errors="replace").strip()
            return v or None
    except Exception:
        return None
    return None

def default_zobie_mapper_out_dir(start_file: Optional[str] = None) -> str:
    """Default mapper output dir.

    If legacy D:\\tools\\zobie_mapper\\out exists, keep using it.
    Otherwise store mapper artifacts under repo-local cache.
    """
    from .archmap_helpers import default_sandbox_cache_root
    legacy = Path(r"D:\tools\zobie_mapper\out")
    try:
        if legacy.exists():
            return str(legacy.resolve())
    except Exception:
        pass

    cache_root = Path(default_sandbox_cache_root(start_file))
    return str((cache_root / "zobie_mapper_out").resolve())

_UPDATE_ARCH_TRIGGER_SET = {
    "update architecture",
    "update arch",
    "update your architecture",
    "/update_architecture",
    "/update_arch",
}

_ARCHMAP_TRIGGER_SET = {
    "create architecture map",
    "arch map",
    "architecture map",
    "/arch_map",
    "/architecture_map",
    "/create_architecture_map",
}

ARCHMAP_PROVIDER = os.getenv("ORB_ARCHMAP_PROVIDER", "anthropic")

ARCHMAP_MAX_CONTINUATION_ROUNDS = 6

_DENY_FILE_PATTERNS = [
    r"(^|/)\.env$",               # Block .env only (allow .env.example)
    r"\.pem$",
    r"\.key$",
    r"\.pfx$",
    r"\.p12$",
    r"\.p8$",
    r"(^|/)\.git($|/)",
    r"(^|/)id_rsa($|/)",
    r"(^|/)known_hosts($|/)",
    r"(^|/)secrets?(/|$)",
    r"(^|/)credentials?(/|$)",
    r"(^|/)tokens?(/|$)",
    r"(^|/)api[_-]?keys?(/|$)",
]

def _safe_read_text(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes)
        return b.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<<failed to read {path}: {e}>>"

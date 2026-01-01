# FILE: app/llm/routing/local_actions.py
"""Prompt-triggered local actions for the router.

Contains:
- ZOBIE MAP: read-only repo mapper
- ARCH QUERY: architecture/signature queries via arch_query_service
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.llm.local_tools.archmap_helpers import default_controller_base_url
from app.llm.local_tools import arch_query
from urllib.request import Request, urlopen

from app.llm.schemas import JobType, LLMResult, LLMTask

# =============================================================================
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"
AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"


def _debug_log(msg: str):
    """Print debug message if ROUTER_DEBUG is enabled."""
    if ROUTER_DEBUG:
        print(f"[router-debug] {msg}")


# =============================================================================
# LOCAL ACTION: ARCH QUERY (Architecture/signature queries)
# =============================================================================

async def _maybe_handle_arch_query(task: LLMTask, original_message: str) -> Optional[LLMResult]:
    """Handle architecture query requests via arch_query_service."""
    msg_lower = original_message.lower()
    
    # Trigger patterns
    triggers = ["structure of", "signatures of", "signatures in", "find function", 
                "find class", "find method", "what's in", "whats in", "search for"]
    has_py = ".py" in msg_lower
    has_struct = any(w in msg_lower for w in ["structure", "signature", "function", "class", "method"])
    
    if not (any(t in msg_lower for t in triggers) or (has_py and has_struct)):
        return None
    
    if not arch_query.is_service_available():
        _debug_log("arch_query_service not available at localhost:8780")
        return None
    
    _debug_log(f"LOCAL ACTION: ARCH QUERY - {original_message[:80]}")
    
    try:
        result = arch_query.query_architecture(original_message)
        if result.startswith("Error:"):
            _debug_log(f"arch_query returned error: {result}")
            return None
        
        return LLMResult(
            content=result,
            provider="local",
            model="arch_query",
            finish_reason="stop",
            error_message=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response={"local_action": "arch_query"},
            job_type=JobType.TEXT_HEAVY,
            routing_decision={
                "job_type": "local.arch.query",
                "provider": "local",
                "model": "arch_query",
                "reason": "Architecture structure query",
            },
        )
    except Exception as e:
        _debug_log(f"arch_query exception: {e}")
        return None


# =============================================================================
# LOCAL ACTION: ZOBIE MAP (Prompt-triggered repo mapper; read-only)
# =============================================================================

_ZOBIE_MAP_TIMEOUT_SECS = int(os.getenv("ZOBIE_MAP_TIMEOUT_SECS", "30"))
_ZOBIE_MAP_DEFAULT_BASE = (os.getenv("ZOBIE_CONTROLLER_BASE") or os.getenv("ZOMBIE_CONTROLLER_BASE") or default_controller_base_url(__file__)).rstrip("/")
_ZOBIE_MAP_MAX_FILES_DEFAULT = int(os.getenv("ZOBIE_MAP_MAX_FILES", "200000"))

_ZOBIE_DENY_FILE_PATTERNS = [
    r"(^|/)\.env($|/)",
    r"\.pem$",
    r"\.key$",
    r"\.pfx$",
    r"\.p12$",
    r"secrets?",
    r"credentials?",
]

_ZOBIE_ANCHOR_BASENAMES = {
    "package.json", "main.js", "electron.js", "vite.config.js", "vite.config.ts",
    "requirements.txt", "pyproject.toml", "poetry.lock",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "README.md", "README.txt",
}

_ZOBIE_ANCHOR_PATH_HINTS = [
    "Orb-backend/main.py",
    "Orb-backend/router.py",
    "orb-desktop/package.json",
    "orb-desktop/main.js",
]


def _zobie_http_json(url: str) -> Any:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=_ZOBIE_MAP_TIMEOUT_SECS) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _zobie_is_denied_path(p: str) -> bool:
    p2 = p.replace("\\", "/").lower()
    return any(re.search(pat, p2) for pat in _ZOBIE_DENY_FILE_PATTERNS)


def _zobie_pick_anchor_files(tree_paths: List[str]) -> List[str]:
    anchors = set()

    tree_set = set(tree_paths)
    for hint in _ZOBIE_ANCHOR_PATH_HINTS:
        if hint in tree_set:
            anchors.add(hint)

    for p in tree_paths:
        if _zobie_is_denied_path(p):
            continue
        base = p.split("/")[-1]
        if base in _ZOBIE_ANCHOR_BASENAMES:
            anchors.add(p)

    for p in tree_paths:
        if _zobie_is_denied_path(p):
            continue
        base = p.split("/")[-1].lower()
        if base in {"main.py", "app.py", "server.py", "router.py", "main.ts", "main.jsx", "main.tsx"}:
            anchors.add(p)

    return sorted(list(anchors))[:80]


def _zobie_condensed_tree(paths: List[str], max_depth: int = 3) -> List[str]:
    out = set()
    for p in paths:
        if _zobie_is_denied_path(p):
            continue
        parts = p.split("/")
        for d in range(1, min(max_depth, len(parts)) + 1):
            out.add("/".join(parts[:d]))
    return sorted(out)


def _zobie_extract_signals(text: str, max_lines: int = 80) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    keep = []
    signal_re = re.compile(r"(error|exception|traceback|fail|warning|critical|todo|fixme)", re.I)
    for ln in lines:
        if signal_re.search(ln):
            keep.append(ln[:500])
        if len(keep) >= max_lines:
            break
    return "\n".join(keep)


def _zobie_default_out_dir() -> Path:
    out = os.getenv("ZOBIE_MAPPER_OUT_DIR") or os.getenv("ORB_ARCHMAP_OUT_DIR") or "D:/tools/zobie_mapper/out"
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _zobie_parse_command(message: str) -> Optional[Dict[str, Any]]:
    if not message:
        return None

    m = re.search(r"\bZOBIE\s+MAP\b(.*)$", message, flags=re.I | re.S)
    if not m:
        return None

    rest = (m.group(1) or "").strip()

    base_url = _ZOBIE_MAP_DEFAULT_BASE
    max_files = _ZOBIE_MAP_MAX_FILES_DEFAULT
    include_hashes = False

    if rest:
        parts = rest.split()
        if parts and (
            parts[0].startswith("http://")
            or parts[0].startswith("https://")
            or re.match(r"^\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}$", parts[0])
        ):
            tok = parts.pop(0)
            base_url = tok if tok.startswith("http") else f"http://{tok}"
            base_url = base_url.rstrip("/")

        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in {"max_files", "max"}:
                try:
                    max_files = int(v)
                except Exception:
                    pass
            elif k in {"hashes", "include_hashes"}:
                include_hashes = v.lower() in {"1", "true", "yes", "y"}

    return {"base_url": base_url, "max_files": max_files, "include_hashes": include_hashes}


def _zobie_run_map_sync(base_url: str, out_path: Path, max_files: int = 200000, include_hashes: bool = False) -> Dict[str, Any]:
    tree_url = f"{base_url}/repo/tree?max_files={max_files}"
    if include_hashes:
        tree_url += "&include_hashes=1"

    tree_data = _zobie_http_json(tree_url)
    tree_paths: List[str] = tree_data.get("paths") or []

    anchors = _zobie_pick_anchor_files(tree_paths)
    condensed = _zobie_condensed_tree(tree_paths, max_depth=3)

    files_meta: List[Dict[str, Any]] = []
    for p in anchors:
        try:
            data = _zobie_http_json(f"{base_url}/repo/file?path={p}")
        except Exception:
            continue
        content = data.get("content") or ""
        files_meta.append(
            {
                "path": p,
                "bytes": data.get("bytes"),
                "signals": _zobie_extract_signals(content),
            }
        )

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tree_txt = out_path / f"REPO_TREE_{stamp}.txt"
    map_md = out_path / f"ARCH_MAP_{stamp}.md"
    index_json = out_path / f"INDEX_{stamp}.json"

    tree_lines: List[str] = []
    tree_lines.append(f"Repo root (VM): {base_url}")
    tree_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    tree_lines.append("")
    tree_lines.append("CONDENSED TREE (depth<=3):")
    tree_lines.extend([f"- {p}" for p in condensed])
    tree_lines.append("")
    tree_lines.append("ANCHOR FILES:")
    tree_lines.extend([f"- {p}" for p in anchors])

    tree_txt.write_text("\n".join(tree_lines), encoding="utf-8", errors="replace")

    md_lines: List[str] = []
    md_lines.append("# Architecture Map (Zobie Mapper)")
    md_lines.append("")
    md_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    md_lines.append(f"Controller: {base_url}")
    md_lines.append("")
    md_lines.append("## Condensed Tree")
    md_lines.append("```")
    md_lines.extend(condensed)
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Anchor Files (signals only)")
    for fm in files_meta:
        md_lines.append(f"### {fm['path']}")
        md_lines.append(f"- bytes: {fm.get('bytes')}")
        sig = fm.get("signals") or ""
        if sig:
            md_lines.append("```")
            md_lines.append(sig)
            md_lines.append("```")
        else:
            md_lines.append("_No signals detected._")
        md_lines.append("")

    map_md.write_text("\n".join(md_lines), encoding="utf-8", errors="replace")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "controller": base_url,
        "max_files": max_files,
        "include_hashes": include_hashes,
        "tree_paths_count": len(tree_paths),
        "anchors": anchors,
        "condensed_tree": condensed,
        "tree_txt": str(tree_txt),
        "map_md": str(map_md),
    }
    index_json.write_text(json.dumps(payload, indent=2), encoding="utf-8", errors="replace")

    return payload


async def _maybe_handle_zobie_map(task: LLMTask, original_message: str) -> Optional[LLMResult]:
    parsed = _zobie_parse_command(original_message)
    if not parsed:
        return None

    base_url = parsed["base_url"]
    max_files = parsed["max_files"]
    include_hashes = parsed["include_hashes"]
    out_dir = _zobie_default_out_dir()

    if ROUTER_DEBUG:
        _debug_log("=" * 70)
        _debug_log("LOCAL ACTION: ZOBIE MAP")
        _debug_log(f"  Controller: {base_url}")
        _debug_log(f"  Out dir: {out_dir}")
        _debug_log(f"  max_files={max_files} include_hashes={include_hashes}")

    try:
        payload = _zobie_run_map_sync(base_url, out_dir, max_files=max_files, include_hashes=include_hashes)
        msg = (
            "ZOBIE MAP complete.\n\n"
            f"- Tree: {payload.get('tree_txt')}\n"
            f"- Map: {payload.get('map_md')}\n"
            f"- Index: {payload.get('index_json', payload.get('index'))}\n"
            f"- Anchors: {len(payload.get('anchors', []))}\n"
        )

        return LLMResult(
            content=msg,
            provider="local",
            model="zobie_mapper",
            finish_reason="stop",
            error_message=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response={"local_action": "zobie_map", "payload": payload},
            job_type=JobType.TEXT_HEAVY,
            routing_decision={
                "job_type": "local.zobie.map",
                "provider": "local",
                "model": "zobie_mapper",
                "reason": "Prompt trigger: ZOBIE MAP",
            },
        )
    except Exception as e:
        err = str(e)
        return LLMResult(
            content=f"ZOBIE MAP failed: {err}",
            provider="local",
            model="zobie_mapper",
            finish_reason="error",
            error_message=err,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response={"local_action": "zobie_map", "error": err},
            job_type=JobType.TEXT_HEAVY,
            routing_decision={
                "job_type": "local.zobie.map",
                "provider": "local",
                "model": "zobie_mapper",
                "reason": "Prompt trigger: ZOBIE MAP (failed)",
            },
        )


# =============================================================================
# MAIN DISPATCHER
# =============================================================================

async def maybe_handle_local_action(task: LLMTask, original_message: str) -> Optional[LLMResult]:
    """Try all local action handlers. Returns result if handled, None otherwise."""
    
    # Architecture queries (check first - more common)
    result = await _maybe_handle_arch_query(task, original_message)
    if result:
        return result
    
    # Zobie map
    result = await _maybe_handle_zobie_map(task, original_message)
    if result:
        return result
    
    return None

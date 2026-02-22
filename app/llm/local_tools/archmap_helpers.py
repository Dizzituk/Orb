# FILE: app/llm/local_tools/archmap_helpers.py
"""Helpers and configuration for architecture tools.

Commands:
- UPDATE ARCHITECTURE: Scan repo → store in Orb/.architecture/ (internal, no LLM)
- CREATE ARCHITECTURE MAP: Load .architecture/ → Claude Opus 4.5 → single markdown

v2.0 (2025-12): Split into two commands, Opus for map generation
v2.1 (2026-01): Added sentinel marker, continuation prompt, auto-resume support
"""

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from sqlalchemy.orm import Session
from app.memory import service as memory_service, schemas as memory_schemas

logger = logging.getLogger(__name__)

# =============================================================================
# PATH RESOLUTION (drive-agnostic defaults; env vars still override)
# =============================================================================

def _env_first(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v and str(v).strip():
            return str(v).strip()
    return None


def get_repo_root(start_file: Optional[str] = None) -> Path:
    """Best-effort backend repo root discovery (no hard-coded drive letters)."""
    env_root = _env_first("ORB_REPO_ROOT", "ORB_BACKEND_ROOT", "ZOMBIEORB_REPO_ROOT", "ZOBIEORB_REPO_ROOT")
    if env_root:
        try:
            return Path(env_root).expanduser().resolve()
        except Exception:
            pass

    start = Path(start_file).resolve() if start_file else Path(__file__).resolve()

    markers = (".git", "pyproject.toml", "requirements.txt")
    for p in (start,) + tuple(start.parents):
        try:
            if any((p / m).exists() for m in markers):
                return p
        except Exception:
            pass

        # If we're inside .../app/... assume repo root is parent of app/
        try:
            if p.name == "app" and (p / "__init__.py").exists():
                return p.parent
        except Exception:
            pass

    # Fallback: relative to this module location (repo/app/llm/local_tools/...)
    try:
        return start.parents[3]
    except Exception:
        return start


def default_architecture_dir(start_file: Optional[str] = None) -> str:
    rr = get_repo_root(start_file)
    return str((rr / ".architecture").resolve())


def default_archmap_output_dir(start_file: Optional[str] = None) -> str:
    # Keep map output repo-local by default (promotion-safe).
    return default_architecture_dir(start_file)


def default_sandbox_cache_root(start_file: Optional[str] = None) -> str:
    rr = get_repo_root(start_file)
    return str((rr / "_sandbox_cache").resolve())


def _read_controller_addr_txt(cache_root: Path) -> Optional[str]:
    addr_file = cache_root / "controller_addr.txt"
    try:
        if addr_file.exists():
            v = addr_file.read_text(encoding="utf-8", errors="replace").strip()
            return v or None
    except Exception:
        return None
    return None


def default_controller_base_url(start_file: Optional[str] = None) -> str:
    """Controller base URL resolution.

    Precedence:
    1) ORB_ZOBIE_CONTROLLER_URL (and legacy aliases)
    2) <repo_root>/_sandbox_cache/controller_addr.txt (written by start_controller.ps1)
    3) Historical fallback (kept for backwards compatibility)
    """
    env_url = _env_first(
        "ORB_ZOBIE_CONTROLLER_URL",
        "ZOBIE_CONTROLLER_BASE",
        "ZOMBIE_CONTROLLER_BASE",
        "ZOBIE_CONTROLLER_URL",
        "ZOMBIE_CONTROLLER_URL",
    )
    if env_url:
        return env_url.rstrip("/")

    cache_root = Path(default_sandbox_cache_root(start_file))
    txt = _read_controller_addr_txt(cache_root)
    if txt:
        return txt.rstrip("/")

    return "http://192.168.250.2:8765"


def default_zobie_mapper_out_dir(start_file: Optional[str] = None) -> str:
    """Default mapper output dir.

    If legacy D:\\tools\\zobie_mapper\\out exists, keep using it.
    Otherwise store mapper artifacts under repo-local cache.
    """
    legacy = Path(r"D:\tools\zobie_mapper\out")
    try:
        if legacy.exists():
            return str(legacy.resolve())
    except Exception:
        pass

    cache_root = Path(default_sandbox_cache_root(start_file))
    return str((cache_root / "zobie_mapper_out").resolve())


# =============================================================================
# TRIGGER SETS
# =============================================================================

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

# =============================================================================
# STORAGE PATHS
# =============================================================================

# Internal architecture storage (fixed filenames, no timestamps)
# Default: Orb/.architecture/ (relative to repo root)
ARCHITECTURE_DIR = os.getenv("ORB_ARCHITECTURE_DIR") or default_architecture_dir(__file__)

# Output for human-readable map
ARCHMAP_OUTPUT_DIR = os.getenv("ORB_ARCHMAP_OUTPUT_DIR") or default_archmap_output_dir(__file__)
ARCHMAP_OUTPUT_FILE = "ARCHITECTURE_MAP.md"  # Single file, always overwritten

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# UPDATE ARCHITECTURE: No LLM needed (just scan)

# CREATE ARCHITECTURE MAP: Use Claude Opus 4.5
ARCHMAP_PROVIDER = os.getenv("ORB_ARCHMAP_PROVIDER", "anthropic")
ARCHMAP_MODEL = os.getenv("ORB_ARCHMAP_MODEL", "claude-opus-4-5-20251101")

# Fallback if Opus unavailable
ARCHMAP_FALLBACK_PROVIDER = os.getenv("ORB_ARCHMAP_FALLBACK_PROVIDER", "openai")
ARCHMAP_FALLBACK_MODEL = os.getenv("ORB_ARCHMAP_FALLBACK_MODEL", "gpt-5-mini")

# =============================================================================
# SCAN CONFIGURATION
# =============================================================================

# Controller base URL (sandbox controller)
ARCHMAP_CONTROLLER_BASE_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL") or default_controller_base_url(__file__)
ARCHMAP_CONTROLLER_TIMEOUT_SEC = int(os.getenv("ORB_ARCHMAP_CONTROLLER_TIMEOUT_SEC", "30"))

# Zobie mapper script
ZOBIE_MAPPER_SCRIPT = os.getenv("ORB_ZOBIE_MAPPER_SCRIPT", r"D:\tools\zobie_mapper\zobie_map.py")
ZOBIE_MAPPER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_MAPPER_TIMEOUT_SEC", "180"))

# =============================================================================
# LLM CALL CONFIGURATION
# =============================================================================

# NOTE: These are legacy defaults. Prefer using get_archmap_config() from stage_models.py
# which reads ARCHMAP_MAX_OUTPUT_TOKENS (60000 default) and ARCHMAP_TIMEOUT_SECONDS (300 default)
ARCHMAP_MAX_TOKENS = int(os.getenv("ORB_ARCHMAP_MAX_TOKENS", "16000"))
ARCHMAP_TEMPERATURE = float(os.getenv("ORB_ARCHMAP_TEMPERATURE", "0.7"))

# =============================================================================
# COMPLETION SENTINEL & CONTINUATION CONFIG
# =============================================================================

# Sentinel marker - THE ground truth for output completion
ARCHMAP_SENTINEL = "<!-- ARCHMAP_END -->"

# Maximum continuation rounds before giving up
ARCHMAP_MAX_CONTINUATION_ROUNDS = 6

# Characters to include from end of partial file for continuation context
ARCHMAP_CONTINUATION_CONTEXT_CHARS = 2000

# =============================================================================
# DENY PATTERNS (security)
# =============================================================================

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


def _is_denied_repo_path(p: str) -> bool:
    p2 = (p or "").replace("\\", "/").lower()
    return any(re.search(pat, p2) for pat in _DENY_FILE_PATTERNS)


# =============================================================================
# ARCHITECTURE DIRECTORY HELPERS
# =============================================================================

def get_architecture_dir() -> Path:
    """Get or create the .architecture directory."""
    arch_dir = Path(ARCHITECTURE_DIR).resolve()
    arch_dir.mkdir(parents=True, exist_ok=True)
    return arch_dir


def get_architecture_file(name: str) -> Path:
    """Get path to a specific architecture file."""
    return get_architecture_dir() / name


def architecture_exists() -> bool:
    """Check if architecture data exists."""
    manifest = get_architecture_file("manifest.json")
    files = get_architecture_file("files.json")
    return manifest.exists() and files.exists()


def load_architecture_manifest() -> Dict[str, Any]:
    """Load the architecture manifest."""
    manifest_path = get_architecture_file("manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load manifest: {e}")
        return {}


def load_architecture_files() -> Dict[str, Any]:
    """Load the files.json (main architecture data)."""
    files_path = get_architecture_file("files.json")
    if not files_path.exists():
        return {}
    try:
        with open(files_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load files.json: {e}")
        return {}


def load_architecture_enums() -> Dict[str, Any]:
    """Load enums.json."""
    enums_path = get_architecture_file("enums.json")
    if not enums_path.exists():
        return {}
    try:
        with open(enums_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load enums.json: {e}")
        return {}


def load_architecture_routes() -> Dict[str, Any]:
    """Load routes.json."""
    routes_path = get_architecture_file("routes.json")
    if not routes_path.exists():
        return {}
    try:
        with open(routes_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load routes.json: {e}")
        return {}


def load_architecture_imports() -> Dict[str, Any]:
    """Load imports.json."""
    imports_path = get_architecture_file("imports.json")
    if not imports_path.exists():
        return {}
    try:
        with open(imports_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load imports.json: {e}")
        return {}


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _safe_read_text(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes)
        return b.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<<failed to read {path}: {e}>>"


def _controller_http_json(url: str) -> Dict[str, Any]:
    """stdlib-only JSON fetch."""
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=ARCHMAP_CONTROLLER_TIMEOUT_SEC) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}") from e
    except URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Bad JSON from {url}: {e}") from e


# =============================================================================
# ARCHITECTURE MAP PROMPT
# =============================================================================

ARCHMAP_SYSTEM_PROMPT = """You are an expert software architect creating a detailed architecture map.

Your task is to analyze the provided codebase data and produce a comprehensive, human-readable architecture document.

RULES:
1. Use ONLY the provided data - never invent files, functions, or routes
2. Be precise about file paths, function names, and line numbers
3. Explain the purpose and relationships between components
4. Highlight important patterns, entry points, and data flows
5. Note any potential issues or areas of concern
6. Use clear markdown formatting with headers, lists, and code references

OUTPUT FORMAT:
- Start with an executive summary
- Document each major subsystem
- Include route tables where relevant
- Show key data flows
- List important enums and their purposes
- Note entry points and boot sequences

CRITICAL: When the architecture map is FULLY COMPLETE, you MUST end with this exact marker on its own line:
<!-- ARCHMAP_END -->

Do NOT include this marker until you have finished documenting ALL subsystems.
"""

ARCHMAP_USER_PROMPT_TEMPLATE = """Create a detailed architecture map for the Orb/ASTRA system.

## Architecture Data

### Manifest
```json
{manifest}
```

### File Structure Summary
Total files: {file_count}

Key files and their symbols:
{files_summary}

### Enums
{enums_summary}

### Routes
{routes_summary}

### Import Graph
{imports_summary}

---

Generate a comprehensive architecture map document. Be thorough and precise.

REMEMBER: When fully complete, end with exactly:
<!-- ARCHMAP_END -->
"""


ARCHMAP_CONTINUATION_PROMPT_TEMPLATE = """You are continuing an architecture map markdown document that was cut off.

The document ends at this point (last ~2000 characters):
---BEGIN PARTIAL CONTENT---
{partial_content}
---END PARTIAL CONTENT---

{heading_hint}

INSTRUCTIONS:
1. Continue EXACTLY from where the text ends above
2. Do NOT repeat anything already written
3. Do NOT restart sections or headings that have already been started
4. Maintain consistent formatting (markdown headers, lists, code blocks)
5. When the architecture map is FULLY COMPLETE, end with this exact marker on its own line:
<!-- ARCHMAP_END -->

Continue the document now:
"""


# =============================================================================
# CONTINUATION HELPERS
# =============================================================================

def extract_last_heading(content: str) -> Optional[str]:
    """Extract the last markdown heading (## or ###) from content.
    
    Helps continuation resume cleanly without duplicating structure.
    """
    # Match ## or ### headings
    heading_pattern = re.compile(r'^(#{2,3})\s+(.+?)\s*$', re.MULTILINE)
    matches = list(heading_pattern.finditer(content))
    if matches:
        last_match = matches[-1]
        level = last_match.group(1)
        text = last_match.group(2)
        return f"{level} {text}"
    return None


def has_sentinel(content: str) -> bool:
    """Check if content contains the completion sentinel."""
    return ARCHMAP_SENTINEL in content


def build_continuation_prompt(partial_content: str) -> str:
    """Build a continuation prompt from partial content.
    
    Includes:
    - Last ~2000 characters of content
    - Last detected markdown heading (if any)
    """
    # Get last N characters
    context_chars = min(ARCHMAP_CONTINUATION_CONTEXT_CHARS, len(partial_content))
    tail_content = partial_content[-context_chars:] if context_chars > 0 else ""
    
    # Extract last heading for context
    last_heading = extract_last_heading(partial_content)
    if last_heading:
        heading_hint = f"The last section heading was: {last_heading}"
    else:
        heading_hint = "(No section heading detected in the partial content)"
    
    return ARCHMAP_CONTINUATION_PROMPT_TEMPLATE.format(
        partial_content=tail_content,
        heading_hint=heading_hint,
    )


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_archmap_prompt(
    manifest: Dict[str, Any],
    files: Dict[str, Any],
    enums: Dict[str, Any],
    routes: Dict[str, Any],
    imports: Dict[str, Any],
) -> str:
    """Build the user prompt for architecture map generation."""
    
    # Files summary (limited to prevent token overflow)
    files_lines = []
    for path, data in list(files.items())[:100]:
        classes = data.get("classes", [])
        functions = data.get("functions", [])
        files_lines.append(f"\n#### {path}")
        if classes:
            files_lines.append("Classes:")
            for c in classes[:10]:
                files_lines.append(f"  - {c.get('name')} (line {c.get('line')}): {c.get('docstring', '')[:100]}")
        if functions:
            files_lines.append("Functions:")
            for f in functions[:15]:
                files_lines.append(f"  - {f.get('name')}{f.get('signature', '()')} (line {f.get('line')})")
    
    # Enums summary
    enums_lines = []
    for enum_key, enum_data in list(enums.items())[:50]:
        members = enum_data.get("members", [])
        member_names = [m.get("name") for m in members[:10]]
        enums_lines.append(f"- {enum_key}: {', '.join(member_names)}")
    
    # Routes summary
    routes_lines = []
    for route_key, route_data in list(routes.items())[:80]:
        file = route_data.get("file", "")
        func = route_data.get("function", "")
        routes_lines.append(f"- {route_key} → {file}::{func}")
    
    # Imports summary (just counts)
    imports_lines = []
    for file_path, import_data in list(imports.items())[:50]:
        imports_from = import_data.get("imports_from", [])
        imported_by = import_data.get("imported_by", [])
        imports_lines.append(f"- {file_path}: imports {len(imports_from)}, imported by {len(imported_by)}")
    
    return ARCHMAP_USER_PROMPT_TEMPLATE.format(
        manifest=json.dumps(manifest, indent=2)[:2000],
        file_count=len(files),
        files_summary="\n".join(files_lines)[:30000],
        enums_summary="\n".join(enums_lines) if enums_lines else "(none)",
        routes_summary="\n".join(routes_lines) if routes_lines else "(none)",
        imports_summary="\n".join(imports_lines) if imports_lines else "(none)",
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Trigger sets
    "_UPDATE_ARCH_TRIGGER_SET",
    "_ARCHMAP_TRIGGER_SET",
    # Paths
    "ARCHITECTURE_DIR",
    "ARCHMAP_OUTPUT_DIR",
    "ARCHMAP_OUTPUT_FILE",
    # Model config
    "ARCHMAP_PROVIDER",
    "ARCHMAP_MODEL",
    "ARCHMAP_FALLBACK_PROVIDER",
    "ARCHMAP_FALLBACK_MODEL",
    "ARCHMAP_MAX_TOKENS",
    "ARCHMAP_TEMPERATURE",
    # Sentinel & continuation
    "ARCHMAP_SENTINEL",
    "ARCHMAP_MAX_CONTINUATION_ROUNDS",
    "ARCHMAP_CONTINUATION_CONTEXT_CHARS",
    # Scan config
    "ARCHMAP_CONTROLLER_BASE_URL",
    "ZOBIE_MAPPER_SCRIPT",
    "ZOBIE_MAPPER_TIMEOUT_SEC",
    # Functions
    "get_architecture_dir",
    "get_architecture_file",
    "architecture_exists",
    "load_architecture_manifest",
    "load_architecture_files",
    "load_architecture_enums",
    "load_architecture_routes",
    "load_architecture_imports",
    "build_archmap_prompt",
    "build_continuation_prompt",
    "extract_last_heading",
    "has_sentinel",
    "ARCHMAP_SYSTEM_PROMPT",
    "ARCHMAP_CONTINUATION_PROMPT_TEMPLATE",
    "_is_denied_repo_path",
    "_safe_read_text",
    "_controller_http_json",
]

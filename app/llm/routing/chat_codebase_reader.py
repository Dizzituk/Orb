# FILE: app/llm/routing/chat_codebase_reader.py
# Purpose: Read-only codebase access for trusted models in standard chat.
# Called-by: app.bridge.capability_layer, app.llm.routing.chat_routing
# Depends-on: app.memory.router, app.rag.retrieval.arch_search, app.sandbox.client
# Last-renovated: 2026-06-11
"""
Read-only codebase access for trusted models in standard chat.

When a trusted model (Opus, Gemini 3.1) fires in the standard chat window,
this module provides pre-gathered codebase context by reading files from
the sandbox — NOT the host filesystem. All reads go through the sandbox
bridge (SandboxClient.repo_tree / repo_file) to maintain blast radius.

The module:
1. Uses RAG search to identify relevant file paths from the user's message
2. Reads those files via the sandbox client (read-only)
3. Returns formatted context for injection into the system prompt

v1.0 (2026-03-01): Initial implementation.

BUILD_ID: 2026-03-01-v1.0-chat-codebase-reader
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================

# Models allowed to use codebase read access
TRUSTED_MODELS: Set[str] = {
    "claude-opus-4-6",
    "claude-opus-4-5-20250929",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-2.5-pro",
}

# Maximum files to read per chat turn
MAX_FILES_PER_TURN = 8

# Maximum chars per file to include in context
MAX_CHARS_PER_FILE = 8000

# Maximum total chars for codebase context block
MAX_TOTAL_CHARS = 30000

# Feature flag
CHAT_CODEBASE_READ_ENABLED = (
    os.environ.get("ASTRA_CHAT_CODEBASE_READ", "1") == "1"
)

# Repo roots on host (read-only fallback when sandbox doesn't have the file)
# The sandbox clones D:\Orb (backend) but NOT D:\orb-desktop (frontend).
# For chat read-only context, we fall back to host reads for frontend files.
BACKEND_ROOT = os.environ.get("ASTRA_BACKEND_ROOT", r"D:\Orb")
FRONTEND_ROOT = os.environ.get("ASTRA_FRONTEND_ROOT", r"D:\orb-desktop")


def is_chat_codebase_read_enabled() -> bool:
    """Check if the feature is enabled."""
    return CHAT_CODEBASE_READ_ENABLED


def is_trusted_model(model: str) -> bool:
    """Check if the model is allowed codebase read access."""
    return model in TRUSTED_MODELS


# =========================================================================
# RAG-based file discovery
# =========================================================================

def _find_relevant_paths(
    message: str,
    db: Any,
    limit: int = MAX_FILES_PER_TURN,
) -> List[str]:
    """Use RAG to find file paths relevant to the user's message.

    Queries the architecture search index to find files related to
    what the user is talking about. Returns normalised POSIX paths
    suitable for the sandbox client.
    """
    paths: List[str] = []

    # Strategy 1: RAG architecture search
    try:
        from app.rag.retrieval.arch_search import search_architecture
        response = search_architecture(db, query=message, top_k=limit + 4)
        for result in response.results:
            # ArchSearchResult carries canonical_path (".path" never existed —
            # this strategy silently AttributeError'd until 2026-06-12)
            p = (result.canonical_path or "").replace("\\", "/")
            if p and p not in paths:
                paths.append(p)
    except Exception as e:
        logger.debug("[chat_codebase] RAG arch search failed: %s", e)

    # Strategy 2: Memory router (may surface experience patterns)
    try:
        from app.memory.router import memory_router
        mem_results = memory_router.query(
            text=message,
            project_id="astra-core",
            domains=["codebase"],
            limit=5,
            min_relevance=0.4,
        )
        for r in mem_results:
            if hasattr(r, "path") and r.path:
                p = r.path.replace("\\", "/")
                if p not in paths:
                    paths.append(p)
    except Exception as e:
        logger.debug("[chat_codebase] Memory router failed: %s", e)

    # Strategy 3: Keyword extraction — pick up explicit file mentions
    # e.g. "look at EducationTab.tsx" or "the education tab component"
    try:
        paths = _extract_mentioned_files(message, paths)
    except Exception as e:
        logger.debug("[chat_codebase] Keyword extraction failed: %s", e)

    # Strategy 4: Concept-based fallback — when RAG has no embeddings
    # for the frontend, use keyword-to-file mapping
    if not paths:
        try:
            concept_paths = _find_concept_paths(message)
            for cp in concept_paths:
                if cp not in paths:
                    paths.append(cp)
        except Exception as e:
            logger.debug("[chat_codebase] Concept mapping failed: %s", e)

    return paths[:limit]


def _extract_mentioned_files(
    message: str,
    existing_paths: List[str],
) -> List[str]:
    """Extract explicitly mentioned filenames from the message.

    Looks for patterns like "EducationTab.tsx", "main.py", etc.
    and tries to match them against known extensions.
    """
    import re

    # Match filename-like patterns (word.ext)
    file_pattern = re.compile(
        r'\b([\w\-]+\.(?:tsx?|jsx?|py|css|html|json|md))\b',
        re.IGNORECASE,
    )
    mentioned = file_pattern.findall(message)

    for filename in mentioned:
        # Only add if not already covered by a path
        already_covered = any(filename in p for p in existing_paths)
        if not already_covered:
            existing_paths.append(filename)

    return existing_paths


# =========================================================================
# Concept-based file discovery (fallback when RAG has no embeddings)
# =========================================================================

# Maps UI/feature keywords to frontend file paths.
# These are tried when RAG returns nothing.
_CONCEPT_MAP: Dict[str, List[str]] = {
    # Core layout
    "sidebar": [
        "src/components/sidebar/Sidebar.tsx",
        "src/styles/components/sidebar.css",
    ],
    "app": ["src/App.tsx"],
    "routing": ["src/App.tsx"],
    "theme": [
        "src/styles/themes.css",
        "src/styles/base.css",
        "src/hooks/useTheme.ts",
    ],
    "style": [
        "src/styles/themes.css",
        "src/styles/base.css",
        "src/styles/index.css",
    ],
    "colour": ["src/styles/themes.css", "src/styles/base.css"],
    "color": ["src/styles/themes.css", "src/styles/base.css"],
    "font": ["src/styles/fonts.css", "src/styles/base.css"],
    "css": ["src/styles/index.css", "src/styles/base.css"],
    # Feature domains (existing)
    "finance": [
        "src/components/finance/FinanceView.tsx",
        "src/styles/components/finance.css",
    ],
    "investment": [
        "src/components/investments/InvestmentsView.tsx",
        "src/styles/components/investments.css",
    ],
    "lifestyle": [
        "src/components/lifestyle/LifestyleView.tsx",
        "src/styles/components/lifestyle.css",
    ],
    "fitness": [
        "src/components/lifestyle/LifestyleView.tsx",
        "src/components/lifestyle/tabs/FitnessTab.tsx",
    ],
    "build": [
        "src/components/builds/BuildsView.tsx",
        "src/styles/components/jobs.css",
    ],
    "content": [
        "src/components/content/ContentView.tsx",
    ],
    "social": [
        "src/components/social-media/SocialMediaView.tsx",
    ],
    "debug": [
        "src/components/debug/DebugView.tsx",
        "src/styles/components/debug.css",
    ],
    "settings": [
        "src/components/settings/SettingsPage.tsx",
        "src/styles/components/settings.css",
    ],
    # Jobs / placeholder pages (education, etc.)
    "education": [
        "src/components/jobs/JobPage.tsx",
        "src/styles/components/jobs.css",
        "src/App.tsx",
    ],
    "job": [
        "src/components/jobs/JobPage.tsx",
        "src/styles/components/jobs.css",
    ],
    "tab": [
        "src/App.tsx",
        "src/components/sidebar/JobTypeGrid.tsx",
    ],
    "placeholder": [
        "src/components/jobs/JobPage.tsx",
        "src/components/content/PlaceholderTab.tsx",
    ],
    "coming soon": [
        "src/components/jobs/JobPage.tsx",
    ],
    # Chat/input
    "chat": [
        "src/components/ChatWindow.tsx",
        "src/components/chat-panel/ChatPanel.tsx",
        "src/styles/components/chat.css",
    ],
    "voice": [
        "src/components/VoiceInput.tsx",
        "src/components/SpeakButton.tsx",
    ],
    # Backend
    "pipeline": ["app/orchestrator/segment_loop.py"],
    "weaver": ["app/orchestrator/weaver.py"],
    "overwatcher": ["app/overwatcher/architecture_executor/orchestrator.py"],
    "sandbox": ["sandbox_controller/main.py"],
    "api": ["app/main.py", "src/services/api.ts"],
}


def _find_concept_paths(message: str) -> List[str]:
    """Find relevant file paths based on concept keywords in the message.

    This is the fallback when RAG search returns nothing (e.g. no
    embeddings indexed for the frontend). Scans the message for
    known UI/feature keywords and returns the associated file paths.
    """
    msg_lower = message.lower()
    found: List[str] = []
    seen: Set[str] = set()

    for keyword, paths in _CONCEPT_MAP.items():
        if keyword in msg_lower:
            for p in paths:
                if p not in seen:
                    found.append(p)
                    seen.add(p)

    # Always include App.tsx for UI-related queries (it has the routing)
    ui_signals = {
        "front", "ui", "design", "layout", "component", "view",
        "screen", "page", "tab", "panel",
    }
    if any(s in msg_lower for s in ui_signals):
        if "src/App.tsx" not in seen:
            found.append("src/App.tsx")

    return found[:MAX_FILES_PER_TURN]


# =========================================================================
# Sandbox file reading
# =========================================================================

def _to_absolute_path(rel_path: str) -> List[str]:
    """Convert a relative path to candidate absolute paths.

    RAG returns relative paths like 'src/App.tsx' or 'app/main.py'.
    The sandbox /fs/contents endpoint needs absolute paths.
    We try both repo roots to find the file.
    """
    candidates = []

    # Already absolute
    if os.path.isabs(rel_path) or rel_path.startswith("D:"):
        return [rel_path.replace("/", os.sep)]

    # Normalise forward slashes in rel_path to OS separator
    rel_norm = rel_path.replace("/", os.sep)

    # Frontend paths (src/, public/, package.json etc.)
    if rel_path.startswith("src/") or rel_path.startswith("public/"):
        candidates.append(os.path.join(FRONTEND_ROOT, rel_norm))
    # Backend paths (app/, main.py, etc.)
    elif rel_path.startswith("app/") or rel_path == "main.py":
        candidates.append(os.path.join(BACKEND_ROOT, rel_norm))
    else:
        # Try frontend first (more likely for UI queries), then backend
        candidates.append(os.path.join(FRONTEND_ROOT, rel_norm))
        candidates.append(os.path.join(BACKEND_ROOT, rel_norm))

    return candidates


def _read_files_via_sandbox_fs(
    paths: List[str],
) -> List[Dict[str, str]]:
    """Read files via the sandbox /fs/contents endpoint.

    Uses the sandbox's filesystem read endpoint which accepts absolute
    paths and checks against ALLOWED_FS_ROOTS (includes all of D:\\).
    This is the correct way to read BOTH backend (D:\\Orb) and
    frontend (D:\\orb-desktop) files through the sandbox.

    Returns list of {path, content, size} dicts.
    """
    try:
        from app.sandbox.client import get_sandbox_client
    except ImportError:
        logger.debug("[chat_codebase] Sandbox client not available")
        return []

    try:
        client = get_sandbox_client()
        if not client.is_connected():
            logger.debug("[chat_codebase] Sandbox not connected")
            return []
    except Exception as e:
        logger.debug("[chat_codebase] Sandbox connect failed: %s", e)
        return []

    # Build list of absolute paths to request
    abs_paths: List[str] = []
    path_map: Dict[str, str] = {}  # abs -> original relative for display

    for rel_path in paths:
        candidates = _to_absolute_path(rel_path)
        for c in candidates:
            norm = c.replace("/", os.sep)
            abs_paths.append(norm)
            path_map[norm] = rel_path

    if not abs_paths:
        return []

    # Call /fs/contents via the sandbox client's HTTP transport
    try:
        response = client._request(
            "POST",
            "/fs/contents",
            json_body={
                "paths": abs_paths[:MAX_FILES_PER_TURN * 2],
                "max_file_size": MAX_CHARS_PER_FILE * 2,
                "include_line_numbers": False,
            },
        )
    except Exception as e:
        logger.debug("[chat_codebase] /fs/contents request failed: %s", e)
        return []

    # Parse response
    results: List[Dict[str, str]] = []
    total_chars = 0
    seen_rel: Set[str] = set()  # Dedupe by relative path

    for file_data in response.get("files", []):
        if total_chars >= MAX_TOTAL_CHARS:
            break

        if file_data.get("error"):
            continue

        abs_p = file_data.get("path", "")
        rel_p = path_map.get(abs_p, abs_p)
        content = file_data.get("content", "")
        size = file_data.get("size_bytes", 0)

        if rel_p in seen_rel:
            continue
        seen_rel.add(rel_p)

        # Truncate large content
        if len(content) > MAX_CHARS_PER_FILE:
            content = content[:MAX_CHARS_PER_FILE] + (
                f"\n\n... [truncated at {MAX_CHARS_PER_FILE} chars, "
                f"full file is {len(file_data.get('content', ''))} chars]"
            )

        results.append({
            "path": rel_p,
            "content": content,
            "size": size,
        })
        total_chars += len(content)

    return results


def _resolve_partial_filenames(
    paths: List[str],
) -> List[str]:
    """Resolve partial filenames (e.g. 'EducationTab.tsx') to full paths.

    Uses the sandbox /fs/tree endpoint to scan both D:\\Orb and
    D:\\orb-desktop for matching filenames.
    """
    # Separate full paths from bare filenames
    full_paths = []
    bare_names = []
    for p in paths:
        if "/" in p or "\\" in p:
            full_paths.append(p)
        else:
            bare_names.append(p.lower())

    if not bare_names:
        return full_paths

    try:
        from app.sandbox.client import get_sandbox_client
        client = get_sandbox_client()
        if not client.is_connected():
            return full_paths + [n for n in bare_names]

        # Scan both repos via /fs/tree (single call, both roots)
        try:
            tree_resp = client._request(
                "POST",
                "/fs/tree",
                json_body={
                    "roots": [FRONTEND_ROOT, BACKEND_ROOT],
                    "max_files": 5000,
                },
            )
            for entry in tree_resp.get("files", []):
                if not bare_names:
                    break
                entry_path = entry.get("path", "")
                entry_name = (
                    entry_path.rsplit("\\", 1)[-1]
                    .rsplit("/", 1)[-1]
                    .lower()
                )
                if entry_name in bare_names:
                    if entry_path not in full_paths:
                        full_paths.append(entry_path)
                        bare_names.remove(entry_name)
        except Exception as e:
            logger.debug("[chat_codebase] Tree scan failed: %s", e)

    except Exception as e:
        logger.debug("[chat_codebase] Partial resolution failed: %s", e)

    return full_paths


# =========================================================================
# Context formatting
# =========================================================================

def _format_codebase_context(
    files: List[Dict[str, str]],
) -> str:
    """Format read files into a context block for injection.

    Returns a clean, structured block that the model can reference.
    The preamble instructs the model to USE these files rather than
    trying to explore the filesystem manually.
    """
    if not files:
        return ""

    lines = [
        "[CODEBASE CONTEXT — read-only from sandbox]",
        "",
        "",
        "IMPORTANT: The source files below were pre-loaded for you.",
        "You ALREADY HAVE the codebase. Do NOT:",
        "  - Call execute_command or shell commands to explore files",
        "  - Say 'let me look at the codebase' or 'give me a moment to dig'",
        "  - Generate fake tool_call JSON blocks",
        "Instead, reference the code below directly in your response.",
        "Cite specific lines, patterns, variable names, and structures",
        "from these files to ground your analysis.",
        "",
        "Environment: Windows sandbox. Repos on D: drive.",
        "  Backend:  D:\\Orb (Python/FastAPI)",
        "  Frontend: D:\\orb-desktop (React/TypeScript/Vite)",
        "",
        f"Files loaded: {len(files)}",
        "",
    ]

    for f in files:
        path = f["path"]
        content = f["content"]
        lines.append(f"--- {path} ({f['size']} bytes) ---")
        lines.append(content)
        lines.append("")

    lines.append("[/CODEBASE CONTEXT]")
    return "\n".join(lines)


# =========================================================================
# Main entry point
# =========================================================================

def gather_codebase_context(
    message: str,
    model: str,
    db: Any,
) -> str:
    """Gather codebase context for a trusted model in standard chat.

    Call this from handle_chat_mode BEFORE building the system prompt.
    Returns formatted context string, or empty string if:
    - Feature is disabled
    - Model is not trusted
    - Sandbox is unavailable
    - No relevant files found

    All reads go through the sandbox bridge (read-only).
    """
    if not CHAT_CODEBASE_READ_ENABLED:
        return ""

    if not is_trusted_model(model):
        return ""

    start = time.time()

    # Step 1: Find relevant file paths via RAG + keyword extraction
    paths = _find_relevant_paths(message, db)
    if not paths:
        logger.debug("[chat_codebase] No relevant paths found")
        return ""

    # Step 2: Resolve bare filenames to full paths via sandbox tree
    paths = _resolve_partial_filenames(paths)
    if not paths:
        logger.debug("[chat_codebase] No resolvable paths")
        return ""

    # Step 3: Read files via sandbox /fs/contents (read-only, both repos)
    files = _read_files_via_sandbox_fs(paths)
    if not files:
        logger.debug("[chat_codebase] No files readable from sandbox")
        return ""

    # Step 4: Format as context block
    context = _format_codebase_context(files)

    elapsed = int((time.time() - start) * 1000)
    logger.info(
        "[chat_codebase] Gathered %d file(s) in %dms for model %s",
        len(files), elapsed, model,
    )
    print(
        f"[CHAT_CODEBASE] {len(files)} file(s) read from sandbox "
        f"in {elapsed}ms (paths searched: {len(paths)})"
    )

    return context


__all__ = [
    "gather_codebase_context",
    "is_chat_codebase_read_enabled",
    "is_trusted_model",
    "TRUSTED_MODELS",
]

# FILE: app/memory/working_set.py
"""
Project working set — per-session, per-project cache of files the
conversation has actually touched.

Solves three problems at once:

1. Cross-model handoff.  Gemini writes a file in one turn; GPT-mini
   handles the next turn.  Without shared state, GPT-mini has no idea
   the file exists.  The working set is injected into every turn's
   system prompt regardless of which model runs, so the file is in
   front of every model that picks up the conversation.

2. Read-before-answer.  The model used to answer queries about a
   known file from chat history snippets, missing rows that had
   scrolled out.  When the file's contents are inlined into the
   system prompt, "what is my total mileage" has the answer in
   front of the model with zero tool calls.

3. Reference disambiguation.  "The dashboard" is meaningless on its
   own.  Inside the Work project, with one dashboard.html in the
   working set, it resolves cleanly.  Cross-project bleed is
   prevented because the store is keyed by project_id.

Design choices:
- In-memory only (per-session).  Aliases reset on backend restart or
  session change, which matches the design intent — the user should
  confirm referents in a fresh session, never inherit them.
- Size-bounded.  Up to MAX_FILES tracked per project, of which the
  MAX_INLINED most recently touched have their contents auto-injected.
  Older files stay as path+metadata and require an explicit
  read_user_file to load.
- Stale-safe.  Cached contents are re-read from disk when the file's
  mtime changes.  Files deleted externally are dropped on next access.

v1.0 (2026-05-24): Initial implementation.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# How many distinct files we'll track per project.  Older files fall
# off the working set when this is exceeded (oldest touch wins).
MAX_FILES = 5

# Of those tracked files, how many get their CONTENTS inlined into the
# system prompt.  The rest are referenced by path + metadata only,
# requiring an explicit read_user_file call.
MAX_INLINED = 3

# Hard per-file size cap for inlining.  Files larger than this are
# tracked but never inlined, regardless of MAX_INLINED.  30KB chosen
# because it's roughly 7-8K tokens, a reasonable per-file budget for
# the system prompt context.
MAX_INLINE_BYTES = 30 * 1024

# Reference patterns trigger lookup against the working set.  Kept
# small and deliberate — we want strict matching, not greedy.
# Each entry maps a regex-friendly pattern (case-insensitive) to a
# list of filename glob fragments that are candidates for that pattern.
# These are HINTS used during disambiguation, not auto-loaders.
REFERENCE_HINTS = {
    "the dashboard":  ["dashboard"],
    "the html":       [".html"],
    "the spreadsheet": [".xlsx", ".xls", ".csv"],
    "the log":        ["log", "incidents"],
    "the doc":        [".docx", ".doc", ".md"],
}


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class WorkingFile:
    """One file the conversation has touched in this project."""
    path: str
    last_touched: float = field(default_factory=time.time)
    last_action: str = "read"   # 'read' | 'wrote' | 'created' | 'edited'
    last_model: str = ""        # which model touched it (informational)
    size_bytes: int = 0
    mtime: float = 0.0
    cached_content: Optional[str] = None
    cached_at_mtime: float = 0.0  # mtime when content was cached

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    def is_fresh(self) -> bool:
        """True if the file still exists and our cached content matches
        the file's current mtime on disk.  Stale or vanished returns False."""
        try:
            current_mtime = os.path.getmtime(self.path)
        except (FileNotFoundError, OSError):
            return False
        return abs(current_mtime - self.cached_at_mtime) < 0.001

    def refresh(self) -> bool:
        """Re-read size/mtime/content from disk if changed.  Returns
        True on success, False if the file is gone."""
        try:
            stat = os.stat(self.path)
        except (FileNotFoundError, OSError):
            return False
        self.size_bytes = stat.st_size
        self.mtime = stat.st_mtime
        # Re-cache content if eligible and changed
        if (
            stat.st_size <= MAX_INLINE_BYTES
            and abs(self.mtime - self.cached_at_mtime) > 0.001
        ):
            try:
                with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                    self.cached_content = f.read()
                self.cached_at_mtime = self.mtime
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("[working_set] Re-read failed for %s: %s", self.path, e)
                self.cached_content = None
        return True


@dataclass
class ProjectWorkingSet:
    """All the working-set state for one project."""
    project_id: int
    files: Dict[str, WorkingFile] = field(default_factory=dict)
    # session-scoped aliases: "the dashboard" -> "/path/to/dashboard.html"
    # Built up through user confirmation.  Reset on new session.
    aliases: Dict[str, str] = field(default_factory=dict)
    canonical_folder: Optional[str] = None  # e.g. C:/Users/dizzi/OneDrive/Documents/Work/


# =============================================================================
# STORE (in-memory, thread-safe)
# =============================================================================

_lock = threading.RLock()
_store: Dict[int, ProjectWorkingSet] = {}


def _hydrate_from_disk(project_id: int) -> ProjectWorkingSet:
    """Load persisted state from disk and rehydrate into a live
    ProjectWorkingSet.  Stale-checks each file as it loads — files
    that no longer exist on disk, or whose mtime has moved, are
    dropped (refresh() returns False for missing files; for changed
    mtime it re-reads content).  This means after restart the working
    set self-heals: deleted files vanish, changed files get fresh
    cached content."""
    from app.memory._working_set_persistence import load as _load
    payload = _load(project_id)
    ws = ProjectWorkingSet(project_id=project_id)
    if not payload:
        return ws
    ws.canonical_folder = payload.get("canonical_folder")
    ws.aliases = dict(payload.get("aliases") or {})
    for entry in payload.get("files") or []:
        try:
            path = entry["path"]
            wf = WorkingFile(
                path=path,
                last_touched=float(entry.get("last_touched") or 0.0),
                last_action=entry.get("last_action") or "read",
                last_model=entry.get("last_model") or "",
                size_bytes=int(entry.get("size_bytes") or 0),
                mtime=float(entry.get("mtime") or 0.0),
            )
            # Re-stat the file and re-cache content.  Drops the entry
            # if the file is gone, which is the right behaviour — we
            # don't carry ghosts across restarts.
            if wf.refresh():
                ws.files[path] = wf
            else:
                logger.info(
                    "[working_set] Dropped persisted ghost on rehydrate: %s",
                    path,
                )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("[working_set] Skipping bad persisted entry: %s", e)
            continue
    if ws.files or ws.aliases:
        logger.info(
            "[working_set] Rehydrated project=%d: %d file(s), %d alias(es)",
            project_id, len(ws.files), len(ws.aliases),
        )
    return ws


def _save(ws: ProjectWorkingSet) -> None:
    """Best-effort persist.  Errors swallowed inside the persistence
    module so they never break a tool call."""
    try:
        from app.memory._working_set_persistence import save
        save(ws)
    except Exception as e:
        logger.debug("[working_set] save failed (non-fatal): %s", e)


def get_or_create(project_id: int) -> ProjectWorkingSet:
    """Get the working set for a project, creating it if needed.

    v2.0 (2026-05-24): On first access for a project this process
    hasn't seen yet, hydrate from disk so state survives backend
    restarts.  This is what makes cross-restart file handoff work."""
    with _lock:
        ws = _store.get(project_id)
        if ws is None:
            ws = _hydrate_from_disk(project_id)
            _store[project_id] = ws
        return ws


def reset_project(project_id: int) -> None:
    """Drop all working-set state for a project (e.g. on session end).
    v2.0: Also drops the persisted file on disk."""
    with _lock:
        _store.pop(project_id, None)
    try:
        from app.memory._working_set_persistence import delete
        delete(project_id)
    except Exception as e:
        logger.debug("[working_set] delete failed (non-fatal): %s", e)


# =============================================================================
# REGISTRATION (called from tool execution hooks)
# =============================================================================

def register_file(
    project_id: int,
    path: str,
    action: str = "read",
    model: str = "",
) -> None:
    """Record that a tool touched a file in this project.  Reads
    size/mtime/content from disk.  Idempotent — calling twice on the
    same file just bumps last_touched."""
    if not project_id or not path:
        return
    # Normalise the path so 'C:/foo/bar.html' and 'C:\foo\bar.html'
    # don't double-register.
    norm = os.path.normpath(path)
    with _lock:
        ws = get_or_create(project_id)
        existing = ws.files.get(norm)
        if existing is None:
            existing = WorkingFile(path=norm)
            ws.files[norm] = existing
        existing.last_touched = time.time()
        existing.last_action = action
        if model:
            existing.last_model = model
        # Pull fresh size/mtime/content
        if not existing.refresh():
            # File doesn't exist (or vanished between register and stat).
            # Drop it — we don't keep ghosts.
            ws.files.pop(norm, None)
            logger.debug("[working_set] Dropped ghost file: %s", norm)
            return
        # Enforce MAX_FILES cap, evicting oldest touch.
        if len(ws.files) > MAX_FILES:
            oldest = min(ws.files.values(), key=lambda f: f.last_touched)
            ws.files.pop(oldest.path, None)
            logger.debug("[working_set] Evicted oldest: %s", oldest.path)
        logger.info(
            "[working_set] Registered %s (action=%s, project=%d, size=%dB)",
            norm, action, project_id, existing.size_bytes,
        )
        # v2.0: Persist after mutation so restart-survival is built in.
        _save(ws)


# =============================================================================
# CONTEXT VAR (so action_executor can call register_file without
# threading project_id through every tool function signature)
# =============================================================================

_current_project_id: ContextVar[Optional[int]] = ContextVar(
    "working_set_current_project_id", default=None,
)
_current_model: ContextVar[str] = ContextVar(
    "working_set_current_model", default="",
)


def set_current(project_id: Optional[int], model: str = "") -> None:
    """Called at the start of a chat turn so subsequent tool calls
    auto-register against the right project."""
    _current_project_id.set(project_id)
    _current_model.set(model)


def get_current_project_id() -> Optional[int]:
    return _current_project_id.get()


def get_current_model() -> str:
    return _current_model.get()


# =============================================================================
# ALIAS RESOLUTION (used by reference resolver in prompt_builders)
# =============================================================================

def set_alias(project_id: int, phrase: str, path: str) -> None:
    """Record that 'the dashboard' resolves to a specific file in this
    project for the rest of this session.
    v2.0: Persists across restarts."""
    with _lock:
        ws = get_or_create(project_id)
        ws.aliases[phrase.lower().strip()] = os.path.normpath(path)
        logger.info(
            "[working_set] Alias set: '%s' -> %s (project=%d)",
            phrase, path, project_id,
        )
        _save(ws)


def get_alias(project_id: int, phrase: str) -> Optional[str]:
    """Return the path this phrase resolves to in this project, or
    None if no alias has been established.
    v2.0: Uses get_or_create so persisted aliases hydrate from disk."""
    with _lock:
        ws = get_or_create(project_id)
        return ws.aliases.get(phrase.lower().strip())


# =============================================================================
# CONTEXT BLOCK BUILDER (used by prompt_builders to inject into prompts)
# =============================================================================

def build_context_block(project_id: int) -> str:
    """Build the system-prompt block for this project's working set.

    Top MAX_INLINED files get their contents inlined.  Remaining files
    appear as path + metadata only.  Stale entries (deleted on disk)
    are pruned as a side effect.

    Returns an empty string if there are no files to inject."""
    if not project_id:
        return ""
    with _lock:
        # v2.0: get_or_create triggers disk hydration on first access.
        ws = get_or_create(project_id)
        if not ws.files:
            return ""
        # Refresh and prune ghosts in one pass.
        live: List[WorkingFile] = []
        dead: List[str] = []
        for wf in ws.files.values():
            if not wf.refresh():
                dead.append(wf.path)
                continue
            live.append(wf)
        for p in dead:
            ws.files.pop(p, None)
        if not live:
            return ""
        # Sort by last_touched descending.  Most-recently-touched
        # files get content inlining priority.
        live.sort(key=lambda f: f.last_touched, reverse=True)
        return _format_block(live, ws.canonical_folder)


def _format_block(
    files: List[WorkingFile],
    canonical_folder: Optional[str],
) -> str:
    """Render the working-set context block.  Inlines content for the
    top MAX_INLINED files under MAX_INLINE_BYTES; rest are path+meta."""
    lines: List[str] = []
    lines.append("")
    lines.append("## Files in this project (working set)")
    lines.append("")
    lines.append(
        "These files have been touched in the current conversation. "
        "Their contents are loaded below — answer questions about them "
        "by reading from this context directly, do not ask the user to "
        "paste content that is already here."
    )
    if canonical_folder:
        lines.append(
            f"\nThis project's home folder is: `{canonical_folder}`"
        )
    lines.append("")
    inlined = 0
    for wf in files:
        size_kb = wf.size_bytes / 1024.0
        meta = (
            f"- `{wf.path}` "
            f"({size_kb:.1f} KB, last {wf.last_action}"
            + (f" by {wf.last_model}" if wf.last_model else "")
            + ")"
        )
        lines.append(meta)
        if (
            inlined < MAX_INLINED
            and wf.size_bytes <= MAX_INLINE_BYTES
            and wf.cached_content is not None
        ):
            lines.append("")
            lines.append(f"  --- BEGIN {wf.filename} ---")
            lines.append(wf.cached_content)
            lines.append(f"  --- END {wf.filename} ---")
            lines.append("")
            inlined += 1
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def snapshot(project_id: int) -> dict:
    """Debug snapshot of a project's working set state.
    v2.0: Uses get_or_create so disk-persisted state is hydrated."""
    with _lock:
        ws = get_or_create(project_id)
        if ws is None:
            return {"project_id": project_id, "files": [], "aliases": {}}
        return {
            "project_id": project_id,
            "canonical_folder": ws.canonical_folder,
            "files": [
                {
                    "path": wf.path,
                    "size_bytes": wf.size_bytes,
                    "last_action": wf.last_action,
                    "last_model": wf.last_model,
                    "last_touched": wf.last_touched,
                    "cached": wf.cached_content is not None,
                }
                for wf in ws.files.values()
            ],
            "aliases": dict(ws.aliases),
        }

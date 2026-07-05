# FILE: app/llm/greenfield_autoscope.py
# Purpose: Auto-scope a greenfield job straight from the Weaver output (name + location).
# Called-by: app.llm._sg_target_injection
# Depends-on: app.pipeline_v2.build_targets, app.pipeline_v2.target_registry
# Last-renovated: 2026-07-04
"""
Greenfield AUTO-SCOPE (v1.0, 2026-07-04).

When the Weaver classifies a job as greenfield_new_app AND its output already
names the project and an explicit target folder (it usually does — e.g.
"Target folder/location: C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's
Tetris"), stopping the pipeline to demand a manual "let's start a new project"
conversation is pure ceremony. This leaf does what that conversation would do,
deterministically, from the weave itself:

    extract name + root  ->  create the folder (visibly, as the user asked)
    -> register a dynamic BuildTargetProfile (persisted, JOB 1)
    -> set the ProjectSession  ->  stamp BuildProject.build_target_id (JOB 4)
    -> hand the profile back for constraints_hint injection

Extraction is DETERMINISTIC (regex + filters), never LLM. If the weave lacks a
usable location, return None and the caller falls back to the v12.0
"No Build Target" scoping stop — auto-scope never guesses a location.

v1.1 (2026-07-04 evening, Taz directives after the live Tetris stop):
  1. VERBAL user-folder paths resolve deterministically — "Documents/Games/
     Tazza's Tetris" resolves via Windows' own User Shell Folders registry
     (which knows about OneDrive redirection; "forgetting the OneDrive" was
     the recurring failure). Still zero guessing: the mapping is Windows',
     and unknown tokens fall back to the scoping stop as before.
  2. An existing POPULATED target folder no longer refuses — the build
     carries on and leaves the existing files untouched (they are not in any
     segment's file_scope, and the scaffold's non-destructive guard already
     demotes existing files to MODIFY). Building INTO a user-folder ROOT
     itself (e.g. Documents) is still rejected.

Stack default is python/pip/generic (same as the scoping flow's "something
else" branch); SpecGate/implementer refine the concrete stack downstream.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)

# Windows path candidates: drive letter + at least one separator, then any
# run of path-safe chars (spaces and apostrophes allowed — "Tazza's Tetris").
# The tempered token stops BEFORE a second drive anchor so two paths on one
# line ("...from C:\\A\\Sprites into C:\\B\\Tetris") extract as TWO candidates
# instead of one merged garbage root (review finding, 2026-07-04).
_WIN_PATH_RE = re.compile(r"[A-Za-z]:[\\/](?:(?![A-Za-z]:[\\/])[^\n<>:\"|?*])+")

# Trailing sentence punctuation to strip from a matched path. Deliberately
# does NOT include the apostrophe (legitimate inside folder names) and
# handles ")" separately — "Foo (v2)" must keep its closing paren.
_TRAILING_JUNK = " \t.,;:!?`\""

# Tiered line hints (review finding, 2026-07-04): STRONG lines name the
# project destination; WEAK lines merely mention a folder-ish word (asset /
# source folders often do). Lines with NO hint word are ignored entirely —
# an incidental path ("reuse the sprite pack in C:\\...\\Downloads\\sprites")
# must never become the project root.
_STRONG_HINTS = ("target", "destination", "project folder", "new folder")
_WEAK_HINTS = ("folder", "location", "path", "director", "created")

# The path regex runs to end-of-line, so a prose-embedded path ("...\\Foo and
# then open it up") swallows the rest of the sentence. Real project folder
# names are short — reject blobs instead of creating absurd folders.
_MAX_NAME_WORDS = 3
_MAX_NAME_CHARS = 60

# Trailing connector words left behind when the tempered regex stops before a
# second path ("...\\Sprites into" -> "...\\Sprites").
_TRAILING_CONNECTORS = frozenset(
    ("into", "in", "to", "and", "from", "at", "under", "inside", "on",
     "then", "or", "the", "a", "an", "of", "for", "with")
)

_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}

# Never auto-scope into system locations, whatever the weave says (checked
# drive-independently against the path with the drive prefix removed).
_FORBIDDEN_TAILS = ("windows", "program files", "program files (x86)", "programdata")

# v1.1: well-known user folders -> User Shell Folders registry value names.
# Windows' OWN mapping (honours OneDrive Known Folder Move) — deterministic.
_KNOWN_FOLDER_REG = {
    "documents": "Personal",
    "my documents": "Personal",
    "desktop": "Desktop",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "My Pictures",
    "music": "My Music",
    "videos": "My Video",
}

# Relative location starting at a known user folder, e.g.
# "Documents/Games/Tazza's Tetris" or "Desktop\\New App". Requires at least
# one segment AFTER the known folder — the folder root itself is never a
# project destination.
_RELATIVE_KNOWN_RE = re.compile(
    r"\b(my documents|documents|desktop|downloads|pictures|music|videos)"
    r"[\\/]((?:(?![A-Za-z]:[\\/])[^\n<>:\"|?*])+)",
    re.IGNORECASE,
)


def _known_folder_root(token: str) -> Optional[str]:
    """Resolve a well-known user folder via the User Shell Folders registry.

    This is how "Documents" becomes C:\\Users\\dizzi\\OneDrive\\Documents on
    a OneDrive-redirected machine — Windows' registry says so, we never
    guess. Returns None (-> scoping stop fallback) when unresolvable.
    """
    value_name = _KNOWN_FOLDER_REG.get(token.lower().strip())
    if not value_name:
        return None
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as key:
            raw, _ = winreg.QueryValueEx(key, value_name)
        path = os.path.expandvars(str(raw))
        return path if os.path.isdir(path) else None
    except Exception as exc:
        logger.debug("[greenfield_autoscope] known-folder lookup failed for %s: %s", token, exc)
        return None


def _protected_user_roots_lower() -> List[str]:
    """Roots a project may live UNDER but never BE: the user profile and the
    known folders themselves (building directly into Documents is never
    what anyone meant)."""
    roots: List[str] = []
    profile_dir = os.path.expanduser("~")
    if profile_dir and profile_dir != "~":
        roots.append(profile_dir.replace("\\", "/").rstrip("/").lower())
    for token in ("documents", "desktop", "downloads", "pictures", "music", "videos"):
        resolved = _known_folder_root(token)
        if resolved:
            roots.append(resolved.replace("\\", "/").rstrip("/").lower())
    return roots


def _builtin_roots_lower() -> List[str]:
    """Normalized roots of the built-in targets — never auto-scope into these."""
    try:
        from app.pipeline_v2.target_registry import ALL_PROFILES, BUILTIN_PROFILE_IDS
        return [
            p.project_root.replace("\\", "/").rstrip("/").lower()
            for pid, p in ALL_PROFILES.items()
            if pid in BUILTIN_PROFILE_IDS
        ]
    except Exception:
        return ["d:/orb", "d:/orb-desktop"]


def _slugify(name: str) -> str:
    """Project name -> slug id (same rule as project_scoping_stream._slugify)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")
    return slug or "untitled"


# v1.2 (live5): trailing DESCRIPTOR words are prose about the location, not
# part of the folder name — the live Tetris run minted "Tazza's Tetris
# location" from "...only the named Documents/Games/Tazza's Tetris location."
_TRAILING_DESCRIPTORS = frozenset(
    ("location", "folder", "directory", "dir", "area", "path", "destination", "place")
)


def _clean_candidate(raw: str) -> str:
    # v1.2 (live5): chat rendering turns ' into curly ’ — normalise so the
    # extracted path matches the REAL folder on disk (Tazza's, not Tazza’s).
    raw = raw.replace("’", "'").replace("‘", "'")
    cleaned = raw.strip().rstrip(_TRAILING_JUNK).rstrip("\\/")
    # Strip trailing brackets only when UNBALANCED sentence punctuation, not
    # part of the name ("...Games\\Foo (v2)" keeps it, "[see C:\\X\\Foo]" loses it).
    while cleaned and cleaned[-1] in _BRACKET_PAIRS and (
        cleaned.count(_BRACKET_PAIRS[cleaned[-1]]) < cleaned.count(cleaned[-1])
    ):
        cleaned = cleaned[:-1].rstrip(_TRAILING_JUNK).rstrip("\\/")
    # Drop trailing connector words the tempered regex left behind.
    words = cleaned.split(" ")
    while len(words) > 1 and words[-1].lower() in _TRAILING_CONNECTORS:
        words.pop()
    return " ".join(words).rstrip(_TRAILING_JUNK).rstrip("\\/")


def _is_forbidden_root(norm_lower: str) -> bool:
    """System locations are never greenfield destinations (any drive)."""
    tail = norm_lower[3:] if len(norm_lower) > 3 and norm_lower[1] == ":" else norm_lower
    return any(tail == f or tail.startswith(f + "/") for f in _FORBIDDEN_TAILS)


def extract_greenfield_target(weaver_text: str) -> Optional[Dict[str, str]]:
    """Deterministically extract {name, root, slug} from Weaver output.

    Returns None when no usable EXTERNAL location is present — paths inside
    the built-in target repos (D:/Orb, D:/orb-desktop, the Android app roots)
    are rejected, as are bare drive roots.
    """
    if not weaver_text:
        return None

    builtin_roots = _builtin_roots_lower()
    protected_roots = _protected_user_roots_lower()
    strong: List[str] = []
    weak: List[str] = []
    seen: set = set()

    for line in weaver_text.split("\n"):
        line_lower = line.lower()
        if any(h in line_lower for h in _STRONG_HINTS):
            bucket = strong
        elif any(h in line_lower for h in _WEAK_HINTS):
            bucket = weak
        else:
            continue  # incidental path on a non-location line — never adopt

        # v1.1: prose arrows ("Documents → Games → Foo") normalise to path
        # separators before matching — arrows never occur in real paths.
        # Spaced slashes collapse too ("Documents / Games" -> "Documents/Games");
        # the space-on-both-sides requirement leaves real paths untouched.
        line_sane = line.replace("→", "/").replace("›", "/").replace(" > ", "/")
        line_sane = re.sub(r"\s+/\s+", "/", line_sane)

        raw_candidates: List[str] = list(_WIN_PATH_RE.findall(line_sane))

        # v1.1: verbal user-folder locations ("Documents/Games/Tazza's
        # Tetris") resolve through Windows' User Shell Folders mapping —
        # OneDrive redirection included. Absolute matches are preferred:
        # only lines WITHOUT a drive-anchored path get the relative pass.
        if not raw_candidates:
            for m in _RELATIVE_KNOWN_RE.finditer(line_sane):
                base = _known_folder_root(m.group(1))
                if not base:
                    continue
                raw_candidates.append(
                    base.replace("\\", "/").rstrip("/") + "/" + m.group(2)
                )

        for match in raw_candidates:
            candidate = _clean_candidate(match)
            # Resolve '.'/'..' segments BEFORE the safety filters —
            # 'D:\\Projects\\..\\Orb\\x' must not sneak past the repo check
            # (path-traversal review finding, 2026-07-04).
            norm = os.path.normpath(candidate).replace("\\", "/")
            # Bare drive roots / single-segment paths are not project folders.
            if len(norm) < 4 or norm[1] != ":" or "/" not in norm[3:]:
                continue
            norm_lower = norm.lower()
            if norm_lower in seen:
                continue
            if any(
                norm_lower == root or norm_lower.startswith(root + "/")
                for root in builtin_roots
            ):
                continue  # existing codebase — not a greenfield destination
            if _is_forbidden_root(norm_lower):
                logger.warning(
                    "[greenfield_autoscope] Rejected system location: %s", norm,
                )
                continue
            if norm_lower in protected_roots:
                logger.warning(
                    "[greenfield_autoscope] Rejected user-folder ROOT as "
                    "destination (projects live UNDER it, never AS it): %s", norm,
                )
                continue
            seen.add(norm_lower)
            bucket.append(norm)

    for candidate in strong + weak:
        parent, _, name = candidate.rstrip("/").rpartition("/")
        name = name.strip()
        # v1.2 (live5): drop trailing descriptor words (prose, not name).
        words = name.split(" ")
        while len(words) > 1 and words[-1].lower().rstrip(_TRAILING_JUNK) in _TRAILING_DESCRIPTORS:
            words.pop()
        name = " ".join(words).rstrip(_TRAILING_JUNK)
        if not name or name.lower() in _TRAILING_DESCRIPTORS:
            continue  # nothing left but prose — not a real folder
        if len(name) < 2:
            continue
        if len(name) > _MAX_NAME_CHARS or len(name.split()) > _MAX_NAME_WORDS:
            # Prose-embedded path swallowed the rest of the sentence — not a
            # real folder name. Skip rather than create an absurd folder.
            logger.warning(
                "[greenfield_autoscope] Rejected implausible folder name: %r", name,
            )
            continue
        root = f"{parent}/{name}" if parent else name
        return {"name": name, "root": root, "slug": _slugify(name)}
    return None


def _deconflict_slug(slug: str, norm_root: str) -> str:
    """Never let an auto-scoped slug shadow an existing target (review
    finding, 2026-07-04: a weave folder named 'Driver Copilot' slugs to the
    BUILT-IN id 'driver-copilot' and would hijack that whole project).

    Same slug + same root = idempotent re-scope, keep it. Built-in ID or a
    differently-rooted existing profile -> deterministic suffix derived from
    the root (stable across restarts, no randomness).
    """
    try:
        from app.pipeline_v2.target_registry import get_profile, BUILTIN_PROFILE_IDS
        existing = get_profile(slug)
        if slug not in BUILTIN_PROFILE_IDS and (
            existing is None
            or existing.project_root.replace("\\", "/").rstrip("/").lower() == norm_root
        ):
            return slug
    except Exception:
        return slug
    suffix = hashlib.sha1(norm_root.encode("utf-8")).hexdigest()[:6]
    deconflicted = f"{slug}-{suffix}"
    logger.warning(
        "[greenfield_autoscope] Slug '%s' collides with an existing target — "
        "using '%s'", slug, deconflicted,
    )
    return deconflicted


def build_greenfield_profile(name: str, root: str, slug: str) -> BuildTargetProfile:
    """Dynamic profile for a fresh standalone project (python/pip default —
    mirrors the scoping flow's 'something else' branch)."""
    return BuildTargetProfile(
        project_id=slug,
        project_name=name,
        project_root=root.replace("\\", "/"),
        language="python",
        build_system="pip",
        framework="generic",
        source_root="src",
        package_name=slug,
        architecture_pattern="modular",
        file_extension=".py",
    )


def autoscope_greenfield_job(
    db: Session,
    chat_project_id: int,
    weaver_text: str,
) -> Optional[BuildTargetProfile]:
    """Full auto-scope. Returns the registered profile, or None to fall back
    to the manual-scoping stop. Never raises."""
    try:
        target = extract_greenfield_target(weaver_text)
        if not target:
            logger.info(
                "[greenfield_autoscope] No usable location in weave — "
                "falling back to scoping stop",
            )
            return None

        name, root = target["name"], target["root"]
        norm_root = root.rstrip("/").lower()
        slug = _deconflict_slug(target["slug"], norm_root)

        # 0. Existing POPULATED folder: carry on, leave the files be (Taz
        # directive 2026-07-04 evening, replacing the earlier hard refusal —
        # "if there's already files in the target folder, ignore them, leave
        # them be, carry on"). Existing files are outside every segment's
        # file_scope and the scaffold's non-destructive guard demotes any
        # overlap to MODIFY, so nothing here gets clobbered. The user-folder
        # ROOT guard in extraction still blocks the dangerous case.
        root_os = root.replace("/", os.sep)
        _existing_items = 0
        try:
            if os.path.isdir(root_os):
                _existing_items = len(os.listdir(root_os))
        except OSError:
            pass
        if _existing_items:
            logger.info(
                "[greenfield_autoscope] Target folder exists with %d item(s) — "
                "leaving them untouched and carrying on: %s", _existing_items, root,
            )

        # 1. Create the folder — visibly, exactly as greenfield weaves ask.
        os.makedirs(root_os, exist_ok=True)

        # 2. Register (+ persist via JOB 1) the dynamic build target.
        from app.pipeline_v2.target_registry import register_profile
        profile = build_greenfield_profile(name, root, slug)
        register_profile(profile)

        # 3. Durable BuildProject stamp BEFORE the session (review finding,
        # 2026-07-04): with the session still unset, get_or_create's
        # stale-target check is a no-op, so we stamp the SAME row the stage
        # wrapper is tracking instead of archiving it mid-stage and splitting
        # ledger/spec/extraction across two rows.
        try:
            from app.builds.pipeline_bridge import get_or_create_build_project
            bp = get_or_create_build_project(db, chat_project_id, brief=name)
            if bp is not None and bp.build_target_id != slug:
                bp.build_target_id = slug
                bp.target_path = profile.project_root
                db.commit()
        except Exception as bp_err:
            logger.warning("[greenfield_autoscope] BuildProject stamp failed: %s", bp_err)

        # 4. Session last — matching slugs mean later get_or_create calls
        # see session == build_target_id and never archive the row.
        try:
            from app.shared_context.project_session import get_project_session
            get_project_session(str(chat_project_id)).finish_scoping(
                project_id=slug, project_name=name,
                project_type="desktop", project_root=profile.project_root,
            )
        except Exception as session_err:
            logger.warning("[greenfield_autoscope] Session set failed: %s", session_err)

        logger.info(
            "[greenfield_autoscope] AUTO-SCOPED '%s' (%s) at %s", name, slug, root,
        )
        _note = f" (existing folder — {_existing_items} item(s) left untouched)" if _existing_items else ""
        print(f"[greenfield_autoscope] AUTO-SCOPED: {name} ({slug}) at {root}{_note}")
        return profile
    except Exception as e:
        logger.warning("[greenfield_autoscope] Auto-scope failed (non-fatal): %s", e)
        return None

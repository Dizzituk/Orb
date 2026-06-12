# Purpose: Dependency Presence Gate (v1.0)
# Called-by: app.pot_spec.grounded._simple_create_utils_17, app.pot_spec.grounded._simple_create_utils_18
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Dependency Presence Gate (v1.0)
================================
Deterministic gate that walks a project's build files and answers:
"Is library X actually declared in this project?"

Used by SpecGate to filter LLM-proposed file content that imports
libraries the target project does not have. Without this gate, the LLM
hallucinates Room/WorkManager/Retrofit on Android projects that have
none of those declared, even with correct evidence in the prompt.

Public API:
    walk_project(project_root)        -> dict[str, str]   (filename -> content)
    get_declared_dependencies(root)   -> set[str]         (lowercased dep ids)
    is_dependency_available(root, name) -> bool
    scan_proposed_content(content, declared) -> list[str] (missing libs)

v1.0 (2026-04-10): Initial implementation. Empirically required after
Astra Bridge SpecGate run hallucinated Room with build.gradle.kts in
the prompt context.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


# Build files we know how to parse, in order of preference per ecosystem
_BUILD_FILES = (
    # Android / Kotlin / Java (Gradle)
    "app/build.gradle.kts",
    "build.gradle.kts",
    "app/build.gradle",
    "build.gradle",
    "settings.gradle.kts",
    "settings.gradle",
    "gradle/libs.versions.toml",
    # JS / TS
    "package.json",
    # Python
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    # Rust
    "Cargo.toml",
    # Go
    "go.mod",
    # Swift
    "Package.swift",
    "Podfile",
    # .NET
    "*.csproj",
)


# Library detection markers used in proposed file content.
# Each entry: (regex_in_content, declared_dep_substring, friendly_name)
LIBRARY_MARKERS: Tuple[Tuple[str, str, str], ...] = (
    # Android persistence / background work
    (r"androidx\.room\b", "androidx.room", "Room (androidx.room)"),
    (r"@Entity\b|@Dao\b|RoomDatabase\b", "androidx.room", "Room annotations"),
    # v1.1: Prose markers — catch markdown-style descriptions, not just code imports.
    # The LLM's analysis text is prose, not Kotlin source, so code-only patterns miss.
    (r"\bRoom\s+(?:DB|database|entity|DAO|dao|Dao|runtime|compiler)\b", "androidx.room", "Room (prose mention)"),
    (r"\b(?:Room|room)Database\b", "androidx.room", "RoomDatabase (prose mention)"),
    (r"\bWorkManager\s+(?:setup|integration|scheduler|worker|task|job)\b", "androidx.work", "WorkManager (prose mention)"),
    (r"\bCoroutineWorker\b|\bListenableWorker\b", "androidx.work", "WorkManager worker class"),
    (r"\bRetrofit\s+(?:client|builder|service|instance|setup)\b", "retrofit2", "Retrofit (prose mention)"),
    (r"\bHilt\s+(?:module|setup|DI|integration|annotation)\b", "dagger.hilt", "Hilt (prose mention)"),
    (r"\bOkHttp(?:Client)?\s+(?:builder|client|interceptor|setup)\b", "okhttp3", "OkHttp (prose mention)"),
    (r"androidx\.work\b|WorkManager\b|CoroutineWorker\b", "androidx.work", "WorkManager (androidx.work)"),
    # Android networking / DI
    (r"retrofit2\b|Retrofit\.Builder", "retrofit2", "Retrofit"),
    (r"okhttp3\b|OkHttpClient", "okhttp3", "OkHttp"),
    (r"dagger\.hilt|@HiltAndroidApp|@AndroidEntryPoint", "dagger.hilt", "Hilt (dagger.hilt)"),
    (r"javax\.inject|@Inject\b", "javax.inject", "javax.inject"),
    # Compose / Coroutines (almost always present, low false-positive risk)
    (r"androidx\.datastore", "androidx.datastore", "DataStore"),
    (r"kotlinx\.serialization\b|@Serializable", "kotlinx.serialization", "kotlinx.serialization"),
)


def _read_safe(path: str, max_bytes: int = 200_000) -> str:
    """Read a file as text, swallowing all errors. Return '' on failure."""
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("[dep_presence] read failed for %s: %s", path, e)
        return ""


@lru_cache(maxsize=64)
def walk_project(project_root: str) -> Tuple[Tuple[str, str], ...]:
    """
    Walk a project root and return all known build-file contents.
    Cached per project_root for the life of the process.

    Returns a tuple of (relative_filename, content) so it remains hashable.
    """
    if not project_root or not os.path.isdir(project_root):
        return ()
    found: List[Tuple[str, str]] = []
    for rel in _BUILD_FILES:
        if "*" in rel:
            # Glob pattern (e.g. *.csproj) — walk one level deep only
            try:
                for entry in os.listdir(project_root):
                    if entry.lower().endswith(rel.lstrip("*").lower()):
                        full = os.path.join(project_root, entry)
                        content = _read_safe(full)
                        if content:
                            found.append((entry, content))
            except Exception:
                pass
            continue
        full = os.path.join(project_root, rel.replace("/", os.sep))
        if os.path.exists(full):
            content = _read_safe(full)
            if content:
                found.append((rel, content))
    if found:
        logger.info("[dep_presence] %s: walked %d build file(s)", project_root, len(found))
    return tuple(found)


# Regex used to extract dependency identifiers from gradle / package.json /
# pyproject / requirements / Cargo / etc. We don't try to be perfect — we
# extract substrings that callers can use with `in` checks.
_DEP_ID_PAT = re.compile(
    r"""
    [\"']                       # opening quote
    ([a-zA-Z0-9_.\-/@]{3,200})  # the dep id itself
    (?:[:\"'])                  # version separator or closing quote
    """,
    re.VERBOSE,
)


def get_declared_dependencies(project_root: str) -> Set[str]:
    """
    Return the set of declared dependency identifiers across all build files
    in the project root. Lowercased for case-insensitive matching.
    The set contains substrings — callers should use `any(needle in dep ...)`.
    """
    declared: Set[str] = set()
    for _, content in walk_project(project_root):
        lowered = content.lower()
        # Strip block / line comments to reduce noise
        lowered = re.sub(r"/\*.*?\*/", " ", lowered, flags=re.DOTALL)
        lowered = re.sub(r"(?m)^\s*//.*$", " ", lowered)
        lowered = re.sub(r"(?m)^\s*#.*$", " ", lowered)
        for m in _DEP_ID_PAT.finditer(lowered):
            dep = m.group(1).strip()
            if len(dep) >= 3 and not dep.startswith(("http", ".", "/")):
                declared.add(dep)
    return declared


def is_dependency_available(project_root: str, needle: str) -> bool:
    """Return True if `needle` appears as a substring of any declared dep."""
    needle = needle.lower().strip()
    if not needle:
        return False
    declared = get_declared_dependencies(project_root)
    return any(needle in dep for dep in declared)


def scan_proposed_content(content: str, project_root: str) -> List[Tuple[str, str]]:
    """
    Scan proposed file content for library markers. Return a list of
    (friendly_name, missing_dep_id) for each library used in the content
    that is NOT declared in the target project.

    Empty list = no missing deps. Non-empty = SpecGate should drop or
    flag the proposed file.
    """
    if not content or not project_root:
        return []
    missing: List[Tuple[str, str]] = []
    declared = get_declared_dependencies(project_root)
    for pattern, dep_substring, friendly in LIBRARY_MARKERS:
        if re.search(pattern, content):
            if not any(dep_substring.lower() in d for d in declared):
                missing.append((friendly, dep_substring))
    # Dedupe preserving order
    seen = set()
    out: List[Tuple[str, str]] = []
    for item in missing:
        if item[1] not in seen:
            seen.add(item[1])
            out.append(item)
    return out
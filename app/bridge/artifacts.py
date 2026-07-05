# FILE: app/bridge/artifacts.py
"""
Bridge artifact handling — parsing markers from message content and
resolving them to safe file paths.

When a bridge-originated turn produces a downloadable artifact (a generated
image, a created document, etc.), the handler appends a marker of the form
[ASTRA_ARTIFACT:<kind>:<filename>] to the assistant's reply content. The
marker is stored verbatim in the message DB so historical messages can be
re-parsed on demand, but it is stripped from any text shown to the user.

Two separate paths consume this module:
  * The send-side: capability_layer / router strip the markers from the
    reply text before sending it to the phone, but emit structured
    artifact refs alongside the cleaned text.
  * The receive-side: the /bridge/artifacts/<kind>/<filename> endpoint
    uses resolve_artifact_path() as the safety boundary that prevents
    path traversal or access outside the kind's allowed base directory.

The marker syntax is deliberately ugly so it cannot be confused with
anything an LLM would naturally emit. Filenames are constrained to the
charset produced by image_gen and nano_banana (alnum + dash + dot).

v1.0 (2026-05-25): Initial implementation.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Allowed kinds. Adding a new one requires both a code change here and an
# Android-side handler — the chip renderer keys off the kind for the icon.
KIND_IMAGE = "image"
KIND_DOCUMENT = "document"
KIND_AUDIO = "audio"
KIND_VIDEO = "video"  # 2026-07-02: shorts/reels delivered to chat like images

# Marker: [ASTRA_ARTIFACT:<kind>:<filename>]. Filename charset is intentionally
# narrow — alphanumerics, dot, dash, underscore. No spaces, slashes, or
# anything that could be exploited for path traversal. Generators that
# produce artifacts MUST emit filenames within this charset.
ARTIFACT_MARKER_RE = re.compile(
    r"\[ASTRA_ARTIFACT:(?P<kind>image|document|audio|video):(?P<filename>[A-Za-z0-9._-]+)\]"
)


@dataclass
class ArtifactRef:
    """Reference to a downloadable artifact attached to an assistant turn."""
    kind: str
    filename: str
    size_bytes: int = 0
    mime_type: str = "application/octet-stream"


def extract_artifacts(content: str) -> list[ArtifactRef]:
    """Parse all artifact markers from a message content string.

    Returns refs in document order. Duplicates are preserved — the caller
    decides whether to deduplicate. Empty list if the content has no
    markers (the overwhelmingly common case for non-artifact messages).
    """
    if not content or "[ASTRA_ARTIFACT:" not in content:
        return []
    refs: list[ArtifactRef] = []
    for m in ARTIFACT_MARKER_RE.finditer(content):
        refs.append(ArtifactRef(kind=m.group("kind"), filename=m.group("filename")))
    return refs


def strip_artifacts(content: str) -> str:
    """Return the content with all artifact markers removed.

    Also collapses any whitespace runs introduced by marker removal so the
    user-visible text reads cleanly. The original content (with markers) is
    what goes into the DB; this is only used for what the phone displays
    and what TTS speaks.
    """
    if not content or "[ASTRA_ARTIFACT:" not in content:
        return content
    cleaned = ARTIFACT_MARKER_RE.sub("", content)
    # Collapse 3+ newlines into 2, then trim trailing whitespace per line.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


# ─── Base directory resolution ──────────────────────────────────
# Map artifact kind to the on-disk root it can be served from. Images
# live in the configured IMAGE_OUTPUT_DIR (currently OneDrive/Pictures/
# AstraPictures). Documents / audio kinds reserved for future use.
def _get_image_dir() -> Path:
    # Lazy import: image_output_dir reads env vars at import time, and
    # we want to defer that until first call so test envs can override.
    from app.llm.image_output_dir import get_image_output_dir
    return Path(get_image_output_dir()).resolve()


def _get_documents_dir() -> Path:
    # Documents = rendered reports (v1.1, 2026-07-01). Same lazy-import
    # rationale as images: the cache dir env is read at call time.
    from app.reports.cache import get_reports_cache_dir
    return Path(get_reports_cache_dir()).resolve()


def _get_video_dir() -> Path:
    # Video = rendered shorts/reels (2026-07-02). Flat delivery dir so the
    # per-job subfolder structure doesn't break the flat-filename artifact
    # model; shorts_delivery copies the final mp4 here under a unique name.
    explicit = os.getenv("SHORTS_DELIVERY_DIR")
    base = Path(explicit) if explicit else Path(
        os.getenv("ASTRA_OUTPUT_DIR", r"D:\Orb\data\content\output")
    ) / "shorts_delivery"
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def _get_base_dir(kind: str) -> Optional[Path]:
    """Return the on-disk root directory artifacts of this kind live in.

    Returns None for unknown kinds — the artifact endpoint treats that
    as 404 rather than leaking which kinds exist server-side.
    """
    if kind == KIND_IMAGE:
        return _get_image_dir()
    if kind == KIND_DOCUMENT:
        # v1.1 (2026-07-01): documents wired to the reports cache
        # (REPORTS_CACHE_DIR). Same traversal safety as images below.
        return _get_documents_dir()
    if kind == KIND_VIDEO:
        # v1.2 (2026-07-02): shorts/reels served from the flat delivery dir.
        return _get_video_dir()
    # Future: audio has an output directory elsewhere in the codebase but
    # isn't wired through bridge yet.
    return None


def resolve_artifact_path(kind: str, filename: str) -> Optional[Path]:
    """Safe-path resolution for the bridge artifact endpoint.

    Returns the absolute Path if and only if:
      1. The kind is known and has a base directory.
      2. The filename passes the ARTIFACT_MARKER_RE filename charset
         (already constrained, but we re-check defensively in case the
         endpoint is called directly without going through marker parsing).
      3. The resolved path is inside the base directory (path traversal
         attempts via .. or symlinks pointing outside are rejected).
      4. The file actually exists and is a regular file.

    Returns None on any failure. The caller (the FastAPI endpoint) maps
    None to HTTP 404 without distinguishing failure reasons — we don't
    want to leak whether a path traversal was attempted vs. just a missing
    file.
    """
    # Charset re-check. Slashes, dots-dots, null bytes — none of it should
    # reach Path() but defence in depth is cheap.
    if not filename or not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
        logger.warning("[artifacts] Rejected filename (charset): %r", filename)
        return None
    base = _get_base_dir(kind)
    if base is None:
        logger.warning("[artifacts] Unknown kind: %r", kind)
        return None
    try:
        candidate = (base / filename).resolve()
    except (OSError, ValueError) as e:
        logger.warning("[artifacts] Path resolve failed: %s", e)
        return None
    # Defence-in-depth: confirm the resolved path is inside base after
    # resolving symlinks. base.resolve() was already called in _get_base_dir.
    try:
        candidate.relative_to(base)
    except ValueError:
        logger.warning("[artifacts] Path traversal blocked: %r -> %r", filename, candidate)
        return None
    if not candidate.is_file():
        return None
    return candidate


def enrich_artifact_ref(ref: ArtifactRef) -> ArtifactRef:
    """Fill in size_bytes and mime_type by stat-ing the file on disk.

    Returns the same ref with fields populated, or with zeros / default
    MIME if the file isn't resolvable. The endpoint is the authority on
    whether a file is actually servable; this is purely metadata for the
    client's chip display.
    """
    path = resolve_artifact_path(ref.kind, ref.filename)
    if path is None:
        return ref
    try:
        ref.size_bytes = path.stat().st_size
    except OSError:
        pass
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        ref.mime_type = mime
    return ref

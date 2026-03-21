# FILE: app/content/video_pipeline/local_scanner.py
"""
Local Asset Scanner — indexes stock footage from the local file system.

Extracts keyframes using FFmpeg, generates Gemini embeddings,
and stores vectors in a local search index for the asset resolver.
"""
import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/indexes/local_stock")
KEYFRAME_DIR = Path("data/content/video_pipeline/keyframes")
METADATA_FILE = INDEX_DIR / "metadata.json"

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def get_scan_dirs() -> List[str]:
    """Get directories to scan from env config."""
    dirs_str = os.getenv("LOCAL_STOCK_DIRS", "")
    if not dirs_str:
        return []
    return [d.strip() for d in dirs_str.split(",") if d.strip()]


def check_ffmpeg() -> bool:
    """Check FFmpeg availability."""
    return shutil.which("ffmpeg") is not None


def discover_videos(scan_dirs: List[str]) -> List[Dict[str, Any]]:
    """
    Walk directories and find all video files.
    Returns list of dicts with path, filename, size, modified time.
    """
    videos = []
    for base_dir in scan_dirs:
        base = Path(base_dir)
        if not base.exists():
            logger.warning(f"[local_scanner] Directory not found: {base_dir}")
            continue

        for path in base.rglob("*"):
            if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
                stat = path.stat()
                videos.append({
                    "path": str(path),
                    "filename": path.name,
                    "folder": str(path.parent),
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "extension": path.suffix.lower(),
                })

    logger.info(
        f"[local_scanner] Discovered {len(videos)} videos "
        f"across {len(scan_dirs)} directories"
    )
    return videos


def extract_keyframes(
    video_path: str,
    num_frames: int = 4,
) -> List[str]:
    """
    Extract evenly-spaced keyframes from a video using FFmpeg.
    Returns list of saved frame file paths.
    """
    if not check_ffmpeg():
        logger.warning("[local_scanner] FFmpeg not available, skipping keyframes")
        return []

    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    # Get video duration first
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-show_entries",
        "format=duration", "-of", "csv=p=0", video_path,
    ]
    try:
        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=30,
        )
        duration = float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        logger.warning(f"[local_scanner] Cannot probe duration: {video_path}")
        return []

    if duration <= 0:
        return []

    # Extract frames at evenly spaced intervals
    frame_paths = []
    interval = duration / (num_frames + 1)

    for i in range(num_frames):
        timestamp = interval * (i + 1)
        frame_file = KEYFRAME_DIR / f"{video_name}_frame{i}.jpg"

        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1", "-q:v", "3",
            str(frame_file),
        ]
        try:
            subprocess.run(
                cmd, capture_output=True, timeout=30,
                check=True,
            )
            if frame_file.exists():
                frame_paths.append(str(frame_file))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.warning(
                f"[local_scanner] Frame extraction failed at {timestamp}s: {e}"
            )

    return frame_paths


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Get video metadata (duration, resolution, codec) via ffprobe."""
    if not check_ffmpeg():
        return {}

    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration:stream=width,height,codec_name",
        "-of", "json", video_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [{}])
        video_stream = next(
            (s for s in streams if s.get("width")), streams[0] if streams else {}
        )
        return {
            "duration_s": float(fmt.get("duration", 0)),
            "width": video_stream.get("width", 0),
            "height": video_stream.get("height", 0),
            "codec": video_stream.get("codec_name", ""),
        }
    except Exception:
        return {}


def load_metadata_index() -> Dict[str, Any]:
    """Load the existing metadata index from disk."""
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return {"videos": {}, "last_scan": None}


def save_metadata_index(index: Dict[str, Any]) -> None:
    """Save the metadata index to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(
        json.dumps(index, indent=2, default=str),
        encoding="utf-8",
    )


async def scan_and_index(
    force_rescan: bool = False,
) -> Dict[str, Any]:
    """
    Full scan workflow:
    1. Discover videos in configured directories
    2. Extract keyframes for new/modified files
    3. Get metadata for each video
    4. Save to metadata index

    Embedding generation is a separate step (requires Gemini API).
    """
    scan_dirs = get_scan_dirs()
    if not scan_dirs:
        return {"status": "no_dirs", "message": "LOCAL_STOCK_DIRS not configured"}

    existing_index = load_metadata_index()
    existing_videos = existing_index.get("videos", {})

    discovered = discover_videos(scan_dirs)

    new_count = 0
    updated_count = 0
    skipped_count = 0

    for video_info in discovered:
        path = video_info["path"]
        existing = existing_videos.get(path)

        # Skip if not modified since last scan (unless forced)
        if (
            not force_rescan
            and existing
            and existing.get("modified_at") == video_info["modified_at"]
        ):
            skipped_count += 1
            continue

        # Extract metadata
        meta = get_video_metadata(path)
        video_info.update(meta)

        # Extract keyframes
        num_frames = int(os.getenv("LOCAL_STOCK_KEYFRAMES", "4"))
        keyframes = extract_keyframes(path, num_frames)
        video_info["keyframe_paths"] = keyframes

        existing_videos[path] = video_info

        if existing:
            updated_count += 1
        else:
            new_count += 1

    # Save updated index
    from datetime import datetime, timezone
    existing_index["videos"] = existing_videos
    existing_index["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_metadata_index(existing_index)

    summary = {
        "status": "complete",
        "total_indexed": len(existing_videos),
        "new": new_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "scan_dirs": scan_dirs,
    }
    logger.info(f"[local_scanner] Scan complete: {summary}")
    return summary


def search_local(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search the local index by keyword matching.

    This is the simple text-based search. When embeddings are
    available, this will be replaced with vector similarity search.
    For now, matches against filename and folder path.
    """
    index = load_metadata_index()
    videos = index.get("videos", {})

    query_lower = query.lower()
    query_terms = query_lower.split()

    scored = []
    for path, info in videos.items():
        searchable = (
            f"{info.get('filename', '')} "
            f"{info.get('folder', '')} "
        ).lower()

        score = sum(1 for term in query_terms if term in searchable)
        if score > 0:
            scored.append((score, info))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [info for _, info in scored[:max_results]]

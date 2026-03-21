# FILE: app/content/production/silence_snap.py
"""
Silence Snap — find clean cut points in audio.

Uses FFmpeg's silencedetect to find silence gaps near a target
timestamp, allowing shorts and clips to be cut at natural speech
pauses instead of mid-word.

Approach: progressive loosening. Start with tight detection
parameters and widen if nothing is found. This ensures we get
the best (cleanest) silence gap when available, but still find
a workable gap in AI-generated speech where pauses are short.

Usage:
    from app.content.production.silence_snap import snap_to_silence

    # Find the nearest silence gap near 42.5 seconds
    clean_end = snap_to_silence("video.mp4", target_seconds=42.5)
    # Returns e.g. 43.1 — the start of the nearest silence gap
"""
import logging
import subprocess
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Progressive detection tiers — try tight first, loosen if needed
DETECTION_TIERS = [
    # (threshold_db, min_silence_duration, description)
    (-30, 0.15, "clean pause"),
    (-25, 0.10, "soft pause"),
    (-20, 0.08, "breath gap"),
    (-18, 0.05, "micro gap"),
]


def detect_silences(
    video_path: str,
    start_seconds: float = 0,
    end_seconds: Optional[float] = None,
    threshold_db: int = -30,
    min_duration: float = 0.15,
) -> List[Tuple[float, float]]:
    """
    Detect silence gaps in a video's audio track.

    Returns list of (start, end) tuples for each silence gap.
    """
    cmd = ["ffmpeg"]

    # Only scan the relevant portion
    if start_seconds > 0:
        cmd.extend(["-ss", str(start_seconds)])
    cmd.extend(["-i", video_path])
    if end_seconds:
        duration = end_seconds - max(start_seconds, 0)
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-",
    ])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        # silencedetect outputs to stderr
        output = result.stderr

        silences = []
        silence_start = None

        for line in output.split("\n"):
            if "silence_start:" in line:
                try:
                    val = float(
                        line.split("silence_start:")[1].strip().split()[0]
                    )
                    # Adjust for the -ss offset
                    silence_start = val + max(start_seconds, 0)
                except (ValueError, IndexError):
                    continue
            elif "silence_end:" in line and silence_start is not None:
                try:
                    val = float(
                        line.split("silence_end:")[1].strip().split()[0]
                    )
                    silence_end = val + max(start_seconds, 0)
                    silences.append((silence_start, silence_end))
                    silence_start = None
                except (ValueError, IndexError):
                    continue

        return silences

    except subprocess.TimeoutExpired:
        logger.warning("[silence_snap] FFmpeg timeout scanning audio")
        return []
    except Exception as e:
        logger.error(f"[silence_snap] Failed to detect silences: {e}")
        return []


def snap_to_silence(
    video_path: str,
    target_seconds: float,
    search_window: float = 4.0,
    direction: str = "nearest",
) -> float:
    """
    Find the nearest silence gap to a target timestamp.

    Uses progressive loosening: tries strict detection first,
    then relaxes thresholds until a gap is found.

    Args:
        video_path: Path to the video file
        target_seconds: The approximate cut point
        search_window: How many seconds before/after to search
        direction: 'nearest' (default), 'after' (only look forward),
                   'before' (only look backward)

    Returns:
        The snapped timestamp (start of the nearest silence gap).
        If no silence found after all tiers, returns the original.
    """
    scan_start = max(0, target_seconds - search_window)
    scan_end = target_seconds + search_window

    # Try each detection tier, from strict to loose
    for threshold_db, min_dur, tier_name in DETECTION_TIERS:
        silences = detect_silences(
            video_path,
            start_seconds=scan_start,
            end_seconds=scan_end,
            threshold_db=threshold_db,
            min_duration=min_dur,
        )

        if not silences:
            logger.debug(
                f"[silence_snap] No {tier_name} found near "
                f"{target_seconds:.1f}s (threshold={threshold_db}dB)"
            )
            continue

        # Find the closest silence to our target
        best = _find_best_silence(
            silences, target_seconds, direction,
        )

        if best is not None:
            logger.info(
                f"[silence_snap] Snapped {target_seconds:.1f}s → "
                f"{best:.1f}s (Δ{best - target_seconds:+.1f}s) "
                f"[{tier_name}]"
            )
            return best

    # All tiers exhausted — nothing found
    logger.warning(
        f"[silence_snap] No silence found near {target_seconds:.1f}s "
        f"after all tiers, using original timestamp"
    )
    return target_seconds


def _find_best_silence(
    silences: List[Tuple[float, float]],
    target: float,
    direction: str,
) -> Optional[float]:
    """Pick the closest silence gap to the target timestamp."""
    best = None
    best_distance = float("inf")

    for s_start, s_end in silences:
        # Use the midpoint of the silence gap as the cut point
        # This gives the cleanest cut — right in the middle of
        # the pause rather than at the edge of speech
        cut_point = (s_start + s_end) / 2

        if direction == "after" and cut_point < target:
            continue
        if direction == "before" and cut_point > target:
            continue

        distance = abs(cut_point - target)
        if distance < best_distance:
            best_distance = distance
            best = cut_point

    return best


def snap_both_ends(
    video_path: str,
    start_seconds: float,
    end_seconds: float,
    search_window: float = 4.0,
) -> Tuple[float, float]:
    """
    Snap both the start and end of a clip to silence gaps.

    For the start: looks backward (cut just before speech begins).
    For the end: looks forward (cut just after speech ends).
    """
    snapped_start = snap_to_silence(
        video_path, start_seconds,
        search_window=search_window,
        direction="before",
    )
    snapped_end = snap_to_silence(
        video_path, end_seconds,
        search_window=search_window,
        direction="after",
    )

    # Safety: ensure we didn't create a negative or tiny duration
    if snapped_end <= snapped_start + 3.0:
        logger.warning(
            f"[silence_snap] Snapped range too short or invalid "
            f"({snapped_start:.1f}-{snapped_end:.1f}), "
            f"using originals"
        )
        return start_seconds, end_seconds

    return snapped_start, snapped_end

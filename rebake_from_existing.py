"""
Re-bake from existing job — ZERO API calls.

Reads job a7324295-32a state, uses existing TTS + video assets,
runs bake + FFmpeg assembly only. No script analysis, no director,
no TTS generation, no clip downloads, no Gemini, no HeyGen.

Pure local FFmpeg operations. Tests the bake + assembly pipeline
end-to-end with real production assets.
"""
import asyncio
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

sys.path.insert(0, "D:\\Orb")

from app.content.video_pipeline.bake_segment import (
    bake_broll_segment, pad_avatar_segment, _probe_duration,
    _probe_video_duration,
)

JOB_DIR = Path("D:/Orb/data/content/video_pipeline/jobs/a7324295-32a")
OUTPUT_DIR = Path("D:/Orb/data/content/output/rebake_test_v8")
BAKED_DIR = OUTPUT_DIR / "baked_clips"
PEXELS_DIR = "D:/Orb/data/content/video_pipeline/downloads/pexels"
MUSIC_PATH = "D:/Orb/data/content/video_pipeline/music/alchemy_ambient_scifi.mp3"


def load_state():
    return json.loads((JOB_DIR / "state.json").read_text())


async def rebake_all():
    """Re-bake all segments from existing assets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BAKED_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    segments = state["scene_plan"]["segments"]
    assets = {a["segment_id"]: a for a in state["resolved_plan"]["assets"]}
    tts_dir = JOB_DIR / "tts"

    used = set()
    results = []
    baked_paths = []  # ordered list for final assembly

    print(f"Re-baking {len(segments)} segments from job a7324295-32a")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    for i, seg in enumerate(segments):
        sid = seg["segment_id"]
        asset = assets.get(sid)
        if not asset or not asset.get("file_path"):
            print(f"  SKIP [{i:02d}] {sid}: no asset")
            continue

        video_path = asset["file_path"]
        if not os.path.exists(video_path):
            print(f"  SKIP [{i:02d}] {sid}: video missing")
            continue

        output_path = str(BAKED_DIR / f"baked_{i:03d}.mp4")

        if seg.get("requires_avatar"):
            print(f"  [{i:02d}] {sid}: AVATAR")
            ok = pad_avatar_segment(video_path, output_path)
            dur = _probe_duration(output_path) if ok else 0
            status = "OK" if ok else "FAIL"
            print(f"       -> {dur:.1f}s {status}")
            results.append({"i": i, "sid": sid, "type": "avatar",
                           "ok": ok, "dur": dur, "status": status})
        else:
            audio_path = str(tts_dir / f"{sid}.mp3")
            if not os.path.exists(audio_path):
                print(f"  SKIP [{i:02d}] {sid}: no TTS")
                continue

            clip_b = asset.get("metadata", {}).get("clip_b_path", "")
            audio_dur = _probe_duration(audio_path)

            print(f"  [{i:02d}] {sid}: BROLL {audio_dur:.1f}s")
            ok = bake_broll_segment(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                segment_index=i,
                filler_clips_dir=PEXELS_DIR,
                used_video_paths=used,
                clip_b_path=clip_b if clip_b and os.path.exists(str(clip_b)) else "",
            )

            if ok and os.path.exists(output_path):
                vid_dur = _probe_video_duration(output_path)
                expected = audio_dur + 0.75
                gap = expected - vid_dur
                status = "OK" if gap < 0.5 else f"FREEZE gap={gap:.1f}s"
                print(f"       -> {vid_dur:.1f}s {status}")
                results.append({"i": i, "sid": sid, "type": "broll",
                               "ok": True, "dur": vid_dur, "audio": audio_dur,
                               "status": status})
            else:
                print(f"       -> BAKE FAILED")
                results.append({"i": i, "sid": sid, "type": "broll",
                               "ok": False, "dur": 0, "status": "FAIL"})

        if os.path.exists(output_path):
            baked_paths.append(output_path)

    # Summary
    print("\n" + "=" * 60)
    ok_count = sum(1 for r in results if "OK" in str(r["status"]))
    freeze_count = sum(1 for r in results if "FREEZE" in str(r["status"]))
    print(f"BAKE: {ok_count} OK, {freeze_count} FREEZE, "
          f"{len(results) - ok_count - freeze_count} FAIL")

    if freeze_count > 0:
        print(f"\n*** {freeze_count} FREEZES — skipping assembly ***")
        return

    # ── ASSEMBLY ──
    print(f"\nAssembling {len(baked_paths)} clips into final video...")

    concat_out = OUTPUT_DIR / "concat.mp4"

    # Use concat FILTER — handles mixed frame rates correctly
    inputs = []
    filter_parts = []
    for i, p in enumerate(baked_paths):
        inputs.extend(["-i", os.path.abspath(p)])
        filter_parts.append(f"[{i}:v][{i}:a]")

    filter_str = (
        "".join(filter_parts)
        + f"concat=n={len(baked_paths)}:v=1:a=1[outv][outa]"
    )

    r = subprocess.run(
        ["ffmpeg", "-y",
         *inputs,
         "-filter_complex", filter_str,
         "-map", "[outv]", "-map", "[outa]",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", "-b:a", "192k",
         "-r", "25",
         str(concat_out)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        print(f"Concat failed: {r.stderr[-300:]}")

    # Add music
    final_out = OUTPUT_DIR / "final_video.mp4"
    if os.path.exists(MUSIC_PATH):
        print("Adding background music...")
        r = subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(concat_out),
             "-i", MUSIC_PATH,
             "-filter_complex",
             "[1:a]volume=0.08[music];"
             "[0:a]asplit[voice][sc];"
             "[music][sc]sidechaincompress=threshold=0.03:ratio=10:"
             "attack=500:release=3000[ducked];"
             "[voice][ducked]amix=inputs=2:duration=first:"
             "dropout_transition=2[out]",
             "-map", "0:v", "-map", "[out]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", str(final_out)],
            capture_output=True, text=True, timeout=180,
        )
        if r.returncode != 0:
            print(f"Music mix failed: {r.stderr[-200:]}")
            shutil.copy(str(concat_out), str(final_out))
    else:
        print("No music file — using concat as final")
        shutil.copy(str(concat_out), str(final_out))

    final_dur = _probe_duration(str(final_out))
    final_size = os.path.getsize(str(final_out)) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"FINAL VIDEO: {final_out}")
    print(f"Duration: {final_dur:.1f}s ({final_dur/60:.1f} min)")
    print(f"Size: {final_size:.1f} MB")
    print(f"*** ASSEMBLY COMPLETE ***")


if __name__ == "__main__":
    asyncio.run(rebake_all())









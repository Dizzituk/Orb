"""
Stress test - randomised segment durations.
Uses existing clips but fakes different audio lengths
to test the bake handles everything from 3s to 60s.
"""
import asyncio, json, os, sys, random, shutil, subprocess
from pathlib import Path
sys.path.insert(0, "D:\\Orb")
from app.content.video_pipeline.bake_segment import (
    bake_broll_segment, pad_avatar_segment, _probe_duration,
)

JOB_DIR = Path("D:/Orb/data/content/video_pipeline/jobs/7aa5584c-eec")
OUTPUT_DIR = Path("D:/Orb/data/content/output/stress_test")
BAKED_DIR = OUTPUT_DIR / "baked_clips"
PEXELS_DIR = "D:/Orb/data/content/video_pipeline/downloads/pexels"
TEMP_AUDIO = OUTPUT_DIR / "fake_audio"

# Stress test durations - deliberately varied
TEST_DURATIONS = [3.5, 5.2, 8.0, 12.4, 15.7, 20.1, 25.3, 30.0, 35.8, 42.5, 50.0, 58.0]

def make_silent_audio(duration_s, output_path):
    """Generate a silent audio file of exact duration."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", f"{duration_s:.3f}",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(output_path),
    ], capture_output=True, timeout=15)
    return os.path.exists(output_path)

async def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BAKED_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO.mkdir(parents=True, exist_ok=True)

    # Get all available video clips
    state = json.loads((JOB_DIR / "state.json").read_text())
    assets = state["resolved_plan"]["assets"]
    broll_assets = [a for a in assets if not any(
        s.get("requires_avatar") for s in state["scene_plan"]["segments"]
        if s["segment_id"] == a["segment_id"]
    )]

    used = set()
    ok_count = freeze_count = 0
    results = []

    random.seed(42)  # reproducible

    for i, target_dur in enumerate(TEST_DURATIONS):
        # Pick a random clip as the primary
        asset = broll_assets[i % len(broll_assets)]
        video_path = asset["file_path"]
        clip_b = asset.get("metadata", {}).get("clip_b_path", "")

        # Generate silent audio of target duration
        audio_path = str(TEMP_AUDIO / f"stress_{i:03d}.mp3")
        make_silent_audio(target_dur, audio_path)
        output_path = str(BAKED_DIR / f"stress_{i:03d}.mp4")

        print(f"  [{i:02d}] TARGET: {target_dur}s")
        ok = bake_broll_segment(
            video_path=video_path, audio_path=audio_path,
            output_path=output_path, segment_index=i,
            filler_clips_dir=PEXELS_DIR, used_video_paths=used,
            clip_b_path=clip_b if clip_b and os.path.exists(str(clip_b)) else "",
        )

        if ok and os.path.exists(output_path):
            vid_dur = _probe_duration(output_path)
            expected = target_dur + 0.75
            gap = expected - vid_dur
            if gap < 0.5:
                print(f"       -> {vid_dur:.1f}s OK (target was {expected:.1f}s)")
                ok_count += 1
                results.append(f"OK  {target_dur}s -> {vid_dur:.1f}s")
            else:
                print(f"       -> {vid_dur:.1f}s FREEZE gap={gap:.1f}s (target was {expected:.1f}s)")
                freeze_count += 1
                results.append(f"FREEZE {target_dur}s -> {vid_dur:.1f}s (gap {gap:.1f}s)")
        else:
            print(f"       -> BAKE FAILED")
            freeze_count += 1
            results.append(f"FAIL {target_dur}s")

    print(f"\n{'='*50}")
    print(f"STRESS TEST RESULTS")
    print(f"{'='*50}")
    for r in results:
        icon = "V" if r.startswith("OK") else "X"
        print(f"  {icon} {r}")
    print(f"\nPASSED: {ok_count}/{len(TEST_DURATIONS)}")
    print(f"FREEZE: {freeze_count}/{len(TEST_DURATIONS)}")
    if freeze_count == 0:
        print("\n*** STRESS TEST PASSED - ALL DURATIONS CLEAN ***")

asyncio.run(run())

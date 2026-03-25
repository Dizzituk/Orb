"""
Standalone bake test - skips ALL API calls.
Uses existing assets from job 7aa5584c-eec.
Zero API calls. Zero cost. Pure FFmpeg testing.
"""
import asyncio, json, os, sys
from pathlib import Path
sys.path.insert(0, "D:\\Orb")
from app.content.video_pipeline.bake_segment import (
    bake_broll_segment, pad_avatar_segment, _probe_duration,
)

JOB_DIR = Path("D:/Orb/data/content/video_pipeline/jobs/7aa5584c-eec")
OUTPUT_DIR = Path("D:/Orb/data/content/output/bake_test")
BAKED_DIR = OUTPUT_DIR / "baked_clips"
PEXELS_DIR = "D:/Orb/data/content/video_pipeline/downloads/pexels"

async def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BAKED_DIR.mkdir(parents=True, exist_ok=True)
    state = json.loads((JOB_DIR / "state.json").read_text())
    segments = state["scene_plan"]["segments"]
    assets = {a["segment_id"]: a for a in state["resolved_plan"]["assets"]}
    tts_dir = JOB_DIR / "tts"
    used = set()
    ok_count = freeze_count = 0

    for i, seg in enumerate(segments):
        sid = seg["segment_id"]
        asset = assets.get(sid)
        if not asset or not asset.get("file_path") or not os.path.exists(asset["file_path"]):
            print(f"  SKIP {sid}: no asset"); continue
        video_path = asset["file_path"]
        output_path = str(BAKED_DIR / f"baked_{i:03d}.mp4")

        if seg.get("requires_avatar"):
            print(f"  [{i:02d}] {sid}: AVATAR")
            ok = pad_avatar_segment(video_path, output_path)
            dur = _probe_duration(output_path) if ok else 0
            print(f"       -> {dur:.1f}s {'OK' if ok else 'FAIL'}")
            if ok: ok_count += 1
        else:
            audio_path = str(tts_dir / f"{sid}.mp3")
            if not os.path.exists(audio_path):
                print(f"  SKIP {sid}: no TTS audio"); continue
            clip_b = asset.get("metadata", {}).get("clip_b_path", "")
            audio_dur = _probe_duration(audio_path)
            print(f"  [{i:02d}] {sid}: BROLL {audio_dur:.1f}s audio")
            ok = bake_broll_segment(
                video_path=video_path, audio_path=audio_path,
                output_path=output_path, segment_index=i,
                filler_clips_dir=PEXELS_DIR, used_video_paths=used,
                clip_b_path=clip_b if clip_b and os.path.exists(clip_b) else "",
            )
            if ok and os.path.exists(output_path):
                vid_dur = _probe_duration(output_path)
                gap = (audio_dur + 0.75) - vid_dur
                if gap < 0.5:
                    print(f"       -> {vid_dur:.1f}s OK")
                    ok_count += 1
                else:
                    print(f"       -> {vid_dur:.1f}s FREEZE gap={gap:.1f}s")
                    freeze_count += 1
            else:
                print(f"       -> BAKE FAILED")
                freeze_count += 1

    print(f"\n{'='*50}")
    print(f"PASSED: {ok_count}  FREEZE: {freeze_count}")
    if freeze_count == 0: print("*** ALL CLEAN - NO FREEZES ***")

asyncio.run(run())

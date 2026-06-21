# FILE: app/content/video_pipeline/orchestrator.py
# Purpose: Pipeline Orchestrator — chains all stages together.
# Called-by: app.content.video_pipeline.router
# Depends-on: app.content.video_pipeline.pipeline_job, app.content.video_pipeline.narration_bake, app.content.video_pipeline.models (+10 more)
# Last-renovated: 2026-06-21
"""
Pipeline Orchestrator — chains all stages together.

Entry point for script-to-video generation. Manages:
1. Script analysis
2. Style resolution
3. Asset resolution (tiered cascade)
4. Clip verification (relevance gate)
5. Narration generation (TTS)
6. Audio-first bake (equal-part filling)
7. FFmpeg assembly
8. QA gate

Emits SSE progress events for the frontend.

BATCH 4 split: the PipelineJob state container moved to pipeline_job.py and the
audio-first bake helper moved to narration_bake.py; both are re-exported below so
this module's public surface is unchanged.
"""
import os
import logging
from typing import Optional, Callable, Awaitable, AsyncGenerator
from pathlib import Path

from sqlalchemy.orm import Session

from app.content.video_pipeline.models import (
    PipelineJobRequest, PipelineStageUpdate,
    ScenePlan, ResolvedPlan, StyleProfile,
)
from app.content.video_pipeline.pipeline_job import (
    PipelineJob, JOBS_DIR, TTS_OUTPUT_DIR,
)
from app.content.video_pipeline.narration_bake import _bake_narration_into_clips

logger = logging.getLogger(__name__)


async def run_pipeline(
    db: Session,
    request: PipelineJobRequest,
    event_callback: Optional[Callable[[PipelineStageUpdate], Awaitable]] = None,
) -> PipelineJob:
    """
    Execute the full script-to-video pipeline.

    Returns:
        PipelineJob with results
    """
    job = PipelineJob(request)
    job.status = "running"

    async def emit(stage, status, message="", **data):
        event = job.record_event(stage, status, message, **data)
        if event_callback:
            await event_callback(event)
        logger.info(f"[pipeline:{job.job_id}] {stage}: {status} — {message}")

    try:
        await emit("pipeline", "started", f"Job {job.job_id} started")

        # ── STAGE 1: Style Resolution ──
        job.current_stage = "style_resolution"
        await emit("style_resolution", "started")

        from app.content.video_pipeline.style_resolver import get_style_profile
        job.style_profile = get_style_profile(db, request.style_profile_id)
        await emit("style_resolution", "complete",
                    f"Profile: {job.style_profile.profile_id or 'defaults'}")

        # ── STAGE 2: Script Analysis ──
        job.current_stage = "script_analysis"
        await emit("script_analysis", "started", "Analyzing script with GPT-5.4")

        from app.content.video_pipeline.script_analyzer import analyze_script
        job.scene_plan = await analyze_script(
            script_text=request.script_text,
            title=request.title,
            target_platform=request.target_platform,
            target_duration_s=request.target_duration_s,
            style_profile=job.style_profile,
        )
        await emit(
            "script_analysis", "complete",
            f"{job.scene_plan.total_segments} segments, "
            f"~{job.scene_plan.estimated_total_duration_s:.0f}s",
        )
        if (
            job.scene_plan.title
            and job.scene_plan.title.lower() not in ("untitled video", "untitled", "")
            and request.title in ("Untitled Video", "", None)
        ):
            request.title = job.scene_plan.title

        job.save_state()

        # ── STAGE 2.5: Director Pre-Production Review ──
        job.current_stage = "director_review"
        await emit("director_review", "started", "Director reviewing scene plan")

        from app.content.video_pipeline.director import (
            run_director_review, apply_director_notes,
        )
        direction = await run_director_review(
            scene_plan=job.scene_plan,
            style_profile=job.style_profile,
            target_platform=request.target_platform,
        )
        if direction:
            job.scene_plan = apply_director_notes(job.scene_plan, direction)
            hook = direction.get("hook_assessment", "unknown")
            score = direction.get("overall_quality_score", "?")
            ai_count = len(direction.get("ai_budget_segments", []))
            await emit(
                "director_review", "complete",
                f"Hook: {hook}, score: {score}/10, AI segments: {ai_count}",
            )
        else:
            await emit("director_review", "complete", "Director skipped")

        job.save_state()

        # ── STAGE 3: Narration Generation (TTS) ──
        job.current_stage = "narration"
        await emit("narration", "started", "Generating narration audio")

        from app.content.video_pipeline.narration import generate_narration

        tts_dir = job.job_dir / "tts"
        tts_dir.mkdir(exist_ok=True)
        narration_map = await generate_narration(
            job.scene_plan, str(tts_dir), emit,
        )
        await emit("narration", "complete", f"{len(narration_map)} audio files")

        # ═══════════════════════════════════════════════════
        # RENDER → VERIFY → BAKE → QA LOOP
        # ═══════════════════════════════════════════════════
        from app.content.video_pipeline.asset_resolver import resolve_assets
        from app.content.video_pipeline.edl_builder import build_edl
        from app.content.production.edit_engine import execute_edl, check_ffmpeg
        from app.content.video_pipeline.director import run_qa_gate
        from app.content.video_pipeline.clip_verifier import verify_clips

        QA_PASS_THRESHOLD = 8
        MAX_FIX_CYCLES = 5
        SCENE_GAP_S = 0.75

        if not check_ffmpeg():
            await emit("assembly", "error", "FFmpeg not installed")
            job.error = "FFmpeg not available"
            job.status = "failed"
            job.save_state()
            return job

        qa_score = 0
        output_path = None
        prev_clip_map = {}
        stuck_segments: set = set()
        actually_used_clips: set = set()

        for render_pass in range(1, MAX_FIX_CYCLES + 2):
            pass_label = (
                "Initial render" if render_pass == 1
                else f"Fix cycle {render_pass - 1}/{MAX_FIX_CYCLES}"
            )

            # ── STAGE 4: Asset Resolution ──
            job.current_stage = "asset_resolution"
            await emit(
                "asset_resolution", "started",
                f"[{pass_label}] Resolving footage for each segment",
            )

            async def asset_progress(msg, pct):
                await emit("asset_resolution", "progress", msg, progress_pct=pct)

            job.resolved_plan = await resolve_assets(
                scene_plan=job.scene_plan,
                max_budget_usd=request.max_budget_usd,
                progress_callback=asset_progress,
                force_ai_segments=stuck_segments if stuck_segments else None,
                style_profile=job.style_profile,
            )
            job.total_cost_usd = job.resolved_plan.total_cost_usd
            await emit(
                "asset_resolution", "complete",
                f"Resolved {len(job.resolved_plan.assets)} assets, "
                f"cost: ${job.total_cost_usd:.2f}",
            )
            job.save_state()

            # ── Futility check ──
            if prev_clip_map and render_pass > 1:
                new_clip_map = {}
                for asset in job.resolved_plan.assets:
                    if asset.segment_id in prev_clip_map:
                        new_clip_map[asset.segment_id] = (
                            os.path.abspath(asset.file_path)
                            if asset.file_path else ""
                        )
                stuck = [
                    sid for sid, old_path in prev_clip_map.items()
                    if new_clip_map.get(sid) == old_path
                ]
                if stuck and len(stuck) == len(prev_clip_map):
                    logger.info(
                        f"[pipeline:{job.job_id}] Futility guard: "
                        f"all {len(stuck)} flagged segments returned same clips."
                    )
                    await emit(
                        "qa_gate", "complete",
                        f"QA score: {qa_score}/10 — no new clips. Best effort.",
                    )
                    break
                elif stuck:
                    stuck_segments.update(stuck)

            # ── STAGE 4.5: Clip Verification (Relevance Gate) ──
            SKIP_VERIFICATION = False
            verification_results = {}
            failed_verification = set()

            if SKIP_VERIFICATION:
                await emit(
                    "clip_verification", "complete",
                    f"[{pass_label}] Verification SKIPPED (testing bake)",
                )
            else:
                job.current_stage = "clip_verification"
                await emit(
                    "clip_verification", "started",
                    f"[{pass_label}] Verifying clip relevance with Gemini",
                )

                verify_batch = []
                asset_map = {a.segment_id: a for a in job.resolved_plan.assets}

                for seg in job.scene_plan.segments:
                    if seg.requires_avatar:
                        continue
                    asset = asset_map.get(seg.segment_id)
                    if not asset or not asset.file_path:
                        continue
                    if not os.path.exists(asset.file_path):
                        continue

                    clip_desc = (
                        asset.metadata.get("clip_description", "")
                        or seg.visual_description
                    )

                    verify_batch.append({
                        "segment_id": seg.segment_id,
                        "script_text": seg.script_text,
                        "clip_path": asset.file_path,
                        "clip_description": clip_desc,
                    })

                if verify_batch:
                    verification_results = await verify_clips(verify_batch)

                for sid, result in verification_results.items():
                    if not result.get("passed", True):
                        failed_verification.add(sid)

                passed_count = len(verification_results) - len(failed_verification)
                await emit(
                    "clip_verification", "complete",
                    f"Verified {len(verification_results)} clips: "
                    f"{passed_count} passed, {len(failed_verification)} failed",
                )

                if failed_verification:
                    from app.content.video_pipeline.asset_resolver import (
                        _try_fal_ai, _try_pexels, _try_pixabay,
                        _analyze_and_index, BudgetTracker,
                    )
                    ai_budget = BudgetTracker(request.max_budget_usd - job.total_cost_usd)
                    used_clips = set()
                    for a in job.resolved_plan.assets:
                        if a.file_path:
                            used_clips.add(os.path.abspath(a.file_path))

                    retry_count = 0
                    ai_count = 0
                    await emit(
                        "clip_verification", "progress",
                        f"Retrying {len(failed_verification)} failed clips "
                        f"(stock retry → Pixabay → AI generation)",
                    )

                    for sid in failed_verification:
                        seg = next(
                            (s for s in job.scene_plan.segments if s.segment_id == sid),
                            None,
                        )
                        if not seg:
                            continue

                        score = verification_results[sid].get("relevance_score", 0)
                        reason = verification_results[sid].get("reason", "")
                        logger.info(
                            f"[pipeline:{job.job_id}] Verification FAIL {sid} "
                            f"(score={score}) — retrying stock first"
                        )

                        replacement = None

                        retry_asset = await _try_pexels(seg, used_clips)
                        if retry_asset and retry_asset.file_path:
                            rv = await verify_clips([{
                                "segment_id": sid,
                                "script_text": seg.script_text,
                                "clip_path": retry_asset.file_path,
                                "clip_description": seg.visual_description,
                            }])
                            rs = rv.get(sid, {}).get("relevance_score", 0)
                            if rs >= 75:
                                replacement = retry_asset
                                retry_count += 1
                                logger.info(
                                    f"[pipeline:{job.job_id}] Pexels retry "
                                    f"PASSED for {sid}: {rs}/100"
                                )

                        if not replacement:
                            retry_asset = await _try_pixabay(seg, used_clips)
                            if retry_asset and retry_asset.file_path:
                                rv = await verify_clips([{
                                    "segment_id": sid,
                                    "script_text": seg.script_text,
                                    "clip_path": retry_asset.file_path,
                                    "clip_description": seg.visual_description,
                                }])
                                rs = rv.get(sid, {}).get("relevance_score", 0)
                                if rs >= 75:
                                    replacement = retry_asset
                                    retry_count += 1
                                    logger.info(
                                        f"[pipeline:{job.job_id}] Pixabay retry "
                                        f"PASSED for {sid}: {rs}/100"
                                    )

                        if not replacement:
                            logger.info(
                                f"[pipeline:{job.job_id}] Stock retries failed "
                                f"for {sid} — escalating to AI generation"
                            )
                            ai_asset = await _try_fal_ai(seg, ai_budget)
                            if ai_asset and ai_asset.file_path and os.path.exists(ai_asset.file_path):
                                replacement = ai_asset
                                ai_count += 1

                        if replacement:
                            if replacement.file_path:
                                used_clips.add(os.path.abspath(replacement.file_path))
                            for i, existing in enumerate(job.resolved_plan.assets):
                                if existing.segment_id == sid:
                                    job.resolved_plan.assets[i] = replacement
                                    job.total_cost_usd += replacement.cost_usd
                                    logger.info(
                                        f"[pipeline:{job.job_id}] Replaced {sid} "
                                        f"with {replacement.source.value} clip"
                                    )
                                    break
                        else:
                            stuck_segments.add(sid)
                            logger.warning(
                                f"[pipeline:{job.job_id}] All retries failed "
                                f"for {sid}, using original clip"
                            )

                    await emit(
                        "clip_verification", "progress",
                        f"Retry results: {retry_count} stock replacements, "
                        f"{ai_count} AI generated, "
                        f"{len(failed_verification) - retry_count - ai_count} unchanged",
                    )

            job.scene_plan.metadata[f"verification_pass_{render_pass}"] = {
                sid: {
                    "score": r.get("relevance_score", 0),
                    "passed": r.get("passed", True),
                }
                for sid, r in verification_results.items()
            }

            # ── STAGE 4.6: Avatar Compositing ──
            from app.content.production.edit_engine import (
                composite_avatar_on_background,
            )

            composite_dir = job.job_dir / "composited"
            composite_dir.mkdir(exist_ok=True)

            avatar_has_bg = os.getenv(
                "HEYGEN_AVATAR_NEEDS_COMPOSITING", "false"
            ).lower() not in ("true", "1", "yes")

            for asset in job.resolved_plan.assets:
                if asset.source.value != "heygen":
                    continue
                if not asset.file_path or not os.path.exists(asset.file_path):
                    continue
                if avatar_has_bg:
                    continue

                comp_path = str(composite_dir / f"comp_{asset.segment_id}.mp4")
                bg_path = os.getenv("HEYGEN_AVATAR_BG_PATH", "")
                success = composite_avatar_on_background(
                    avatar_path=asset.file_path,
                    output_path=comp_path,
                    background_path=bg_path or None,
                )
                if success and os.path.exists(comp_path):
                    asset.file_path = comp_path

            # ── STAGE 5: Storyboard Ready ──
            await emit(
                "storyboard_ready", "complete",
                f"[{pass_label}] Storyboard ready",
                segments=[a.model_dump() for a in job.resolved_plan.assets],
                total_cost=job.total_cost_usd,
            )

            # ── STAGE 6: EDL Construction ──
            job.current_stage = "edl_construction"
            await emit("edl_construction", "started", "Building edit timeline")

            edl = build_edl(
                scene_plan=job.scene_plan,
                resolved_plan=job.resolved_plan,
                narration_audio_paths=narration_map,
                style=job.style_profile,
                output_format=request.target_platform,
            )
            await emit(
                "edl_construction", "complete",
                f"{len(edl.segments)} EDL segments",
            )

            # ── Background music selection ──
            music_dir = Path("data/content/video_pipeline/music")
            if music_dir.exists():
                import random as _music_rng
                tracks = (
                    list(music_dir.glob("*.mp3"))
                    + list(music_dir.glob("*.wav"))
                    + list(music_dir.glob("*.ogg"))
                    + list(music_dir.glob("*.flac"))
                    + list(music_dir.glob("*.m4a"))
                )
                if tracks:
                    chosen = _music_rng.choice(tracks)
                    edl.background_music_path = str(chosen)
                    edl.music_volume = 0.08

            # ── STAGE 6.5: Audio-First Bake ──
            if narration_map:
                await emit(
                    "assembly", "started",
                    f"[{pass_label}] Audio-first bake: equal-part clip filling",
                )
                baked_count, actually_used_clips = await _bake_narration_into_clips(
                    edl=edl,
                    narration_map=narration_map,
                    scene_plan=job.scene_plan,
                    job_dir=str(job.job_dir),
                    emit=emit,
                    resolved_plan=job.resolved_plan,
                )
                await emit(
                    "assembly", "progress",
                    f"Baked {baked_count} clips ({len(actually_used_clips)} source clips used)",
                )

            # ── STAGE 7: FFmpeg Assembly ──
            job.current_stage = "assembly"
            await emit(
                "assembly", "progress",
                f"[{pass_label}] Assembling video with FFmpeg",
            )

            output_path = execute_edl(edl)
            if not output_path:
                job.error = "FFmpeg assembly failed"
                job.status = "failed"
                await emit("assembly", "error", "FFmpeg assembly returned no output")
                job.save_state()
                return job

            job.output_path = output_path
            await emit(
                "assembly", "complete",
                f"[{pass_label}] Video saved: {output_path}",
            )

            # ── STAGE 8: Post-Production QA Gate ──
            job.current_stage = "qa_gate"
            await emit(
                "qa_gate", "started",
                f"[{pass_label}] Director reviewing assembled video",
            )

            import subprocess as _sp

            summary_segments = []
            for i, seg in enumerate(edl.segments):
                edl_dur = round(seg.end_seconds - seg.start_seconds, 1)
                seg_info = {
                    "index": i,
                    "source": seg.source_path.split(os.sep)[-1],
                    "edl_duration_s": edl_dur,
                    "type": seg.segment_type,
                    "has_caption": bool(seg.caption_text),
                }

                if os.path.exists(seg.source_path):
                    try:
                        probe = _sp.run(
                            [
                                "ffprobe", "-v", "quiet",
                                "-show_entries", "format=duration",
                                "-of", "csv=p=0",
                                os.path.abspath(seg.source_path),
                            ],
                            capture_output=True, text=True, timeout=10,
                        )
                        actual_dur = round(float(probe.stdout.strip()), 1)
                        seg_info["actual_source_duration_s"] = actual_dur
                    except Exception:
                        pass

                # Add verification score if available
                latest_verify = job.scene_plan.metadata.get(
                    f"verification_pass_{render_pass}", {}
                )
                seg_id = (
                    job.scene_plan.segments[i].segment_id
                    if i < len(job.scene_plan.segments) else ""
                )
                if seg_id in latest_verify:
                    seg_info["relevance_score"] = latest_verify[seg_id].get("score", 0)

                summary_segments.append(seg_info)

            assembly_summary = {
                "output_path": output_path,
                "total_segments": len(edl.segments),
                "total_duration_s": edl.total_duration,
                "total_cost_usd": job.total_cost_usd,
                "format": request.target_platform,
                "render_pass": render_pass,
                "segments": summary_segments,
            }

            try:
                qa_result = await run_qa_gate(
                    scene_plan=job.scene_plan,
                    assembly_summary=assembly_summary,
                )
            except Exception as qa_err:
                logger.warning(
                    f"[pipeline:{job.job_id}] QA gate failed: {qa_err}. "
                    f"Skipping QA — video is saved."
                )
                await emit(
                    "qa_gate", "complete",
                    f"QA skipped (error: {str(qa_err)[:80]}). Video saved.",
                )
                break

            qa_score = qa_result.get("overall_score", 5)
            qa_issues = qa_result.get("issues", [])

            job.scene_plan.metadata["qa_result"] = qa_result
            job.scene_plan.metadata[f"qa_result_pass_{render_pass}"] = qa_result

            if qa_score >= QA_PASS_THRESHOLD:
                # Extract shorts suggestions if present
                shorts = qa_result.get("suggested_shorts", [])
                if shorts:
                    job.scene_plan.metadata["suggested_shorts"] = shorts
                    logger.info(
                        f"[qa_gate] Found {len(shorts)} suggested shorts"
                    )
                    await emit(
                        "qa_gate", "progress",
                        f"Found {len(shorts)} potential YouTube Shorts",
                    )
                await emit(
                    "qa_gate", "complete",
                    f"QA PASSED — score: {qa_score}/10.",
                )
                break

            # ── QA failed — apply fixes ──
            if render_pass > MAX_FIX_CYCLES:
                await emit(
                    "qa_gate", "complete",
                    f"QA score: {qa_score}/10 after {MAX_FIX_CYCLES} fixes. Best effort.",
                )
                break

            flagged_ids = set()
            for issue in qa_issues:
                seg_id = issue.get("segment_id", "")
                if seg_id:
                    flagged_ids.add(seg_id)
                    stuck_segments.add(seg_id)

            await emit(
                "qa_gate", "progress",
                f"QA score: {qa_score}/10 — {len(flagged_ids)} segments "
                f"flagged for AI regeneration",
            )

            prev_clip_map = {}
            if job.resolved_plan:
                for asset in job.resolved_plan.assets:
                    if asset.segment_id in flagged_ids:
                        prev_clip_map[asset.segment_id] = (
                            os.path.abspath(asset.file_path)
                            if asset.file_path else ""
                        )

        # ── Thumbnail Generation ──
        if job.output_path and os.path.exists(job.output_path):
            thumb_path = str(Path(job.output_path).parent / "thumbnail.png")
            title = (
                getattr(job, "title", "")
                or getattr(job.request, "title", "")
                or (job.scene_plan.title if job.scene_plan else "")
                or "Untitled"
            )
            from app.content.video_pipeline.thumbnail import generate_ai_thumbnail
            thumb_ok = await generate_ai_thumbnail(
                scene_plan=job.scene_plan,
                edl=edl,
                title=title,
                output_path=thumb_path,
            )
            if not thumb_ok:
                from app.content.production.thumbnail_gen import (
                    generate_thumbnail as gen_thumb,
                )
                gen_thumb(
                    video_path=job.output_path,
                    title=title,
                    output_path=thumb_path,
                )

        # ── Mark ONLY actually-used clips for 90-day cooldown ──
        # Only clips that made it into the final baked output get
        # marked. Clips that were downloaded but not used (e.g.
        # rejected by verification) stay available for future runs.
        if actually_used_clips:
            from app.content.video_pipeline.asset_library import mark_clip_used
            marked = 0
            for clip_path in actually_used_clips:
                if os.path.exists(clip_path):
                    mark_clip_used(clip_path)
                    marked += 1
            logger.info(
                f"[pipeline:{job.job_id}] Cooldown: marked {marked} "
                f"source clips (90-day reuse block)"
            )

        # ── DONE ──
        job.status = "complete"
        await emit(
            "pipeline", "complete",
            f"Pipeline finished — QA score: {qa_score}/10, "
            f"render passes: {render_pass}",
            output_path=job.output_path,
            total_cost=job.total_cost_usd,
        )

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.exception(f"[pipeline:{job.job_id}] Pipeline failed: {e}")
        await emit("pipeline", "error", str(e))

    job.save_state()
    return job

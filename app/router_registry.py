# FILE: app/router_registry.py
# Purpose: Register all FastAPI routers + static mounts onto the app (extracted from main.py).
# Called-by: main
# Depends-on: (all app.* routers — see imports below)
# Last-renovated: 2026-06-21
"""
Router registration for the Orb backend.

BATCH 4 split: the ~80 include_router(...) calls + their guarded try/except imports
were lifted verbatim out of main.py into register_routers(app). main.py calls this
once at module load. Returns the briefing-availability flag the startup hook needs.
The static/output mounts stay in main.py (they resolve paths via main.py's __file__).
"""
import os
from fastapi import Depends
from app.auth import require_auth

from app.auth.router import router as auth_router
from app.memory.api_router import router as memory_router
from app.llm.stream_router import router as stream_router
from app.llm.telemetry_router import router as telemetry_router
from app.llm.web_search_router import router as web_search_router

# Briefing system (morning news digests)
try:
    from app.briefing.briefing_router import router as briefing_router
    _BRIEFING_AVAILABLE = True
except ImportError as e:
    _BRIEFING_AVAILABLE = False
    print(f"[main] Briefing system not available: {e}")
from app.embeddings.router import router as embeddings_router
from app.introspection.router import router as introspection_router
from app.astra_memory.router import router as astra_memory_router
from app.rag.router import router as rag_router
from app.shared_context.router import router as shared_context_router
from app.content.router import router as content_router
from app.content.scout_router import router as content_scout_router
from app.content.production_router import router as content_production_router
from app.content.distribution_router import router as content_distribution_router
from app.content.distribution.youtube_router import router as youtube_router
from app.content.project_router import router as content_project_router
from app.content.style_router import router as content_style_router
from app.content.item_router import router as content_item_router
from app.content.stream_router import router as content_stream_router
import app.content.project_models  # noqa: F401 — register Content Hub tables with Base

# Video Pipeline (script-to-video production)
try:
    from app.content.video_pipeline.router import router as video_pipeline_router
    from app.content.video_pipeline.style_resolver import StyleProfileRecord  # noqa: F401 — register table
    _VIDEO_PIPELINE_AVAILABLE = True
except ImportError as e:
    _VIDEO_PIPELINE_AVAILABLE = False
    print(f"[main] Video pipeline not available: {e}")
from app.content.engagement.router import router as engagement_router
import app.content.engagement.models  # noqa: F401 — register Engagement tables with Base
from app.builds.router import router as builds_router
import app.builds.models  # noqa: F401 — register Build Projects tables with Base
import app.builds.messages  # noqa: F401 — register Build Project Messages table with Base
from app.education.router import router as education_router
import app.education.models  # noqa: F401 — register Education tables with Base
import app.web_automation.models  # noqa: F401 — register Web Automation tables with Base
import app.content.distribution.browser_analytics.models  # noqa: F401 — register ChannelAnalytics table with Base
import app.learning.models  # noqa: F401 — register Course* tables with Base
from app.settings.router import router as settings_router
from app.transparency.router import router as transparency_router
import app.transparency.models  # noqa: F401 — register Transparency tables with Base
import app.vehicle.models  # noqa: F401 — register Vehicle (OBD2 van) tables with Base

# Import refactored endpoints
from app.endpoints import router as endpoints_router
from app.endpoints.ambient import router as ambient_router
from app.voice_ambient.router import router as voice_ambient_router

# Import voice/transcription routers (safe - don't crash app if deps missing)
try:
    from app.routers.transcribe import router as transcribe_router
    _TRANSCRIBE_AVAILABLE = True
except ImportError as e:
    _TRANSCRIBE_AVAILABLE = False
    print(f"[main] Transcribe router not available: {e}")

try:
    from app.routers.audio_stream import router as audio_stream_router
    _AUDIO_STREAM_AVAILABLE = True
except ImportError as e:
    _AUDIO_STREAM_AVAILABLE = False
    print(f"[main] Audio stream router not available: {e}")


def register_routers(app):
    """Register every router + guarded router import onto `app`. Returns _BRIEFING_AVAILABLE."""
    app.include_router(auth_router)
    app.include_router(memory_router)
    app.include_router(stream_router)
    app.include_router(telemetry_router)
    app.include_router(web_search_router)
    if _BRIEFING_AVAILABLE:
        app.include_router(briefing_router)
    app.include_router(embeddings_router)
    app.include_router(astra_memory_router)
    app.include_router(shared_context_router, dependencies=[Depends(require_auth)])
    app.include_router(content_router)
    app.include_router(content_scout_router)
    app.include_router(content_production_router)
    from app.content.production.file_router import router as content_file_router
    app.include_router(content_file_router)
    app.include_router(content_distribution_router)
    app.include_router(youtube_router)
    app.include_router(engagement_router)
    app.include_router(content_project_router)
    app.include_router(content_style_router)
    app.include_router(content_item_router)
    app.include_router(content_stream_router)
    if _VIDEO_PIPELINE_AVAILABLE:
        app.include_router(video_pipeline_router)
        print("[startup] Video Pipeline: [OK] registered")
    app.include_router(builds_router)
    app.include_router(education_router)

    # v11.0: Project Registry
    try:
        from app.project_registry.api_router import router as project_registry_router
        app.include_router(project_registry_router, dependencies=[Depends(require_auth)])
    except Exception as _pr_err:
        print(f'[main.py] Project registry router not available: {_pr_err}')
    app.include_router(settings_router)
    app.include_router(transparency_router)

    from app.investments.router import router as investments_router
    app.include_router(investments_router)
    from app.investments.chat_router import router as investments_chat_router
    app.include_router(investments_chat_router)
    from app.finance.work_router import router as finance_work_router
    app.include_router(finance_work_router)

    # Vehicle module (OBD2 van health & mileage, 2026-06-12)
    try:
        from app.vehicle.router import router as vehicle_router
        app.include_router(vehicle_router)
        print("[startup] Vehicle: [OK] registered")
    except Exception as _vehicle_err:
        print(f"[startup] Vehicle: [WARN] {_vehicle_err}")
    from app.lifestyle.router import router as lifestyle_router
    app.include_router(lifestyle_router)
    # Personal food product library — Job 4 memory roadmap (2026-06-10)
    try:
        from app.lifestyle.product_router import router as lifestyle_products_router
        app.include_router(lifestyle_products_router)
        print("[startup] Lifestyle products: [OK] registered")
    except Exception as e:
        print(f"[startup] Lifestyle products: [WARN] {e}")
    # Nutrition day-copy — voice-first diary reuse (2026-06-11)
    try:
        from app.lifestyle.nutrition_copy import router as nutrition_copy_router
        app.include_router(nutrition_copy_router)
        print("[startup] Nutrition copy-day: [OK] registered")
    except Exception as e:
        print(f"[startup] Nutrition copy-day: [WARN] {e}")
    # Energy engine — context-aware burn + weekly ledger (2026-06-11)
    try:
        from app.lifestyle.energy_router import router as energy_router
        app.include_router(energy_router)
        print("[startup] Energy engine: [OK] registered")
    except Exception as e:
        print(f"[startup] Energy engine: [WARN] {e}")
    # ASTRA Drive — local file system management
    try:
        from app.drive.router import router as astra_drive_router
        app.include_router(astra_drive_router, dependencies=[Depends(require_auth)])
        print("[startup] Drive: [OK] registered")
    except ImportError as e:
        print(f"[startup] Drive not available: {e}")
    app.include_router(endpoints_router)
    app.include_router(ambient_router)
    app.include_router(voice_ambient_router)
    app.include_router(rag_router)

    try:
        from app.endpoints.cost_dashboard import router as cost_dashboard_router
        app.include_router(cost_dashboard_router, tags=["Cost Dashboard"])
    except ImportError as e:
        print(f"[startup] Cost dashboard not available: {e}")

    try:
        from app.debug.debug_chat import router as debug_chat_router
        app.include_router(debug_chat_router, tags=["Debug Assistant"], dependencies=[Depends(require_auth)])
        print("[startup] Debug Assistant: [OK] registered")
    except ImportError as e:
        print(f"[startup] Debug Assistant not available: {e}")

    try:
        from app.debug.project_router import router as debug_project_router
        app.include_router(debug_project_router, tags=["Debug Projects"], dependencies=[Depends(require_auth)])
        print("[startup] Debug Projects: [OK] registered")
    except ImportError as e:
        print(f"[startup] Debug Projects not available: {e}")


    # Screen Recordings (standalone — used by Debug and future modules)
    try:
        from app.debug.recordings_router import router as recordings_router
        app.include_router(recordings_router, dependencies=[Depends(require_auth)], tags=["Recordings"])
        print("[startup] Recordings: [OK] registered")
    except Exception as e:
        print(f"[startup] Recordings not available: {e}")

    # Debug File Upload (images + text for Gemini multimodal)
    try:
        from app.debug.file_upload_router import router as debug_upload_router
        app.include_router(debug_upload_router, dependencies=[Depends(require_auth)], tags=["Debug File Upload"])
        print("[startup] Debug File Upload: [OK] registered")
    except Exception as e:
        print(f"[startup] Debug File Upload not available: {e}")
    try:
        from app.debug.feedback import router as debug_feedback_router
        app.include_router(debug_feedback_router, dependencies=[Depends(require_auth)], tags=["Debug Feedback"])
        print("[startup] Debug Feedback: [OK] registered")
    except Exception as e:
        print(f"[startup] Debug Feedback not available: {e}")
    try:
        from app.debug.orchestrator.endpoint import router as debug_orchestrator_router
        app.include_router(debug_orchestrator_router, dependencies=[Depends(require_auth)], tags=["Debug Orchestrator"])
        print("[startup] Debug Orchestrator: [OK] registered")
    except Exception as e:
        print(f"[startup] Debug Orchestrator not available: {e}")
    app.include_router(introspection_router, tags=["Introspection"], dependencies=[Depends(require_auth)])

    if _TRANSCRIBE_AVAILABLE:
        app.include_router(transcribe_router)
    if _AUDIO_STREAM_AVAILABLE:
        app.include_router(audio_stream_router)

    if os.getenv("ORB_ENABLE_PHASE4", "false").lower() == "true":
        try:
            from app.jobs.router import router as jobs_router
            from app.artefacts.router import router as artefacts_router

            app.include_router(jobs_router, prefix="/jobs", tags=["Phase 4 Jobs"], dependencies=[Depends(require_auth)])
            app.include_router(artefacts_router, prefix="/artefacts", tags=["Phase 4 Artefacts"], dependencies=[Depends(require_auth)])
            print("[startup] Phase 4 routers registered successfully")
        except ImportError as e:
            print(f"[startup] WARNING: Phase 4 import failed: {e}")

    # Image generation (Nano Banana / GPT Image)
    try:
        from app.llm.image_router import router as image_gen_router
        app.include_router(image_gen_router, dependencies=[Depends(require_auth)])
        print("[startup] Image generation: [OK] registered")
    except ImportError as e:
        print(f"[startup] Image generation: [WARN] {e}")

    # Optimize Tab — codebase analysis and improvement engine
    try:
        from app.optimize.router import router as optimize_router
        app.include_router(optimize_router, tags=["Optimize"], dependencies=[Depends(require_auth)])
        print("[startup] Optimize: [OK] registered")
    except Exception as e:
        print(f"[startup] Optimize: [WARN] {e}")

    # Self-Model & Evolution Layer — capability awareness, user understanding, suggestions
    try:
        from app.self_model.router import router as self_model_router
        app.include_router(self_model_router, tags=["Self-Model"], dependencies=[Depends(require_auth)])
        print("[startup] Self-Model: [OK] registered")
    except Exception as e:
        print(f"[startup] Self-Model: [WARN] {e}")


    # Self-Model identity store (Tier 1 hard facts) — separate router
    try:
        from app.self_model.identity_router import router as identity_router
        app.include_router(identity_router, tags=["Self-Model-Identity"], dependencies=[Depends(require_auth)])
        print("[startup] Self-Model Identity: [OK] registered")
    except Exception as e:
        print(f"[startup] Self-Model Identity: [WARN] {e}")

    # Self-Model proposal review router (Phase 3 audit, 2026-05-02).
    # In shadow mode this surfaces what the arbiter WOULD have done; in enforce
    # mode (Phase 4 onward) this is the path Taz uses to accept/reject proposed
    # Tier 1 writes. Kept in its own try/except so a failure here is visible
    # at boot rather than masked by the identity router's success.
    try:
        from app.self_model.proposal_review_router import router as proposal_review_router
        app.include_router(proposal_review_router, tags=["Self-Model-Proposals"], dependencies=[Depends(require_auth)])
        print("[startup] Self-Model Proposals (arbiter): [OK] registered")
    except Exception as e:
        print(f"[startup] Self-Model Proposals (arbiter): [WARN] {e}")

    # Self-Model fragments / themes (Phase 7 — long-term behavioural memory)
    try:
        from app.self_model.fragments.router import router as fragments_router
        app.include_router(fragments_router, tags=["Self-Model-Fragments"], dependencies=[Depends(require_auth)])
        print("[startup] Self-Model Fragments: [OK] registered")
    except Exception as e:
        print(f"[startup] Self-Model Fragments: [WARN] {e}")

    # Web Automation — Chromium-driven interaction with sites that lack APIs
    # (Coursera, TikTok/IG/FB publishing, WordPress admin). Electron shell
    # polls /web_automation/pending-action/* and executes primitives in
    # persistent session partitions.
    try:
        from app.web_automation import router as web_automation_router, register_web_tools
        app.include_router(web_automation_router)  # auth is per-endpoint; Electron polling is local-trusted
        _wa_tool_count = register_web_tools()
        print(f"[startup] Web Automation: [OK] registered ({_wa_tool_count} LLM tools)")
    except Exception as e:
        print(f"[startup] Web Automation: [WARN] {e}")

    # Browser Analytics — scrapes insights pages inside the logged-in
    # WebContentsView sessions (Meta Business Suite, TikTok Studio, YouTube
    # Studio) to fill the Insights dashboard. Phase 1 is recon-only (dumps
    # page text to disk for selector discovery). Later phases add parsers,
    # DB writes, and scheduled pulls.
    try:
        from app.content.distribution.browser_analytics import router as browser_analytics_router
        app.include_router(browser_analytics_router)
        print("[startup] Browser Analytics: [OK] registered (recon mode)")
    except Exception as e:
        print(f"[startup] Browser Analytics: [WARN] {e}")
    # Bridge API (Android companion app)
    try:
        from app.bridge.router import router as bridge_router
        app.include_router(bridge_router)
        from app.bridge.dashboards import router as bridge_dashboards_router
        app.include_router(bridge_dashboards_router)
        from app.bridge.tts_proxy import router as bridge_tts_router
        app.include_router(bridge_tts_router)
        from app.bridge.log_uploads import router as bridge_logs_router
        app.include_router(bridge_logs_router)
        from app.bridge.missed_replies import router as bridge_missed_replies_router
        app.include_router(bridge_missed_replies_router)
        from app.bridge.tts_audio import router as bridge_tts_audio_router
        app.include_router(bridge_tts_audio_router)
        from app.bridge.vehicle import router as bridge_vehicle_router
        app.include_router(bridge_vehicle_router)
        print("[startup] Bridge API: [OK] registered (+ TTS proxy, + missed-replies, + tts-audio cache, + vehicle)")
    except ImportError as _bridge_err:
        print(f"[startup] Bridge API: [WARN] {_bridge_err}")

    # ASTRA Sentinel (network security monitor, Phase 1 2026-06-12)
    try:
        from app.sentinel.router import router as sentinel_router
        from app.sentinel.router import debug_router as sentinel_debug_router
        app.include_router(sentinel_router)
        app.include_router(sentinel_debug_router)
        from app.bridge.sentinel_alerts import router as bridge_sentinel_alerts_router
        app.include_router(bridge_sentinel_alerts_router)
        print("[startup] Sentinel: [OK] registered (+ localhost debug inject, + bridge alerts feed)")
    except Exception as _sentinel_err:
        print(f"[startup] Sentinel: [WARN] {_sentinel_err}")

    # Reminders (2026-07-01): one-shot reminders that fire on desktop + phone
    # alike. Core API + phone-facing bridge feed registered together.
    try:
        from app.reminders import models as _reminders_models  # noqa: F401 — register Reminder table with Base
        from app.reminders.router import router as reminders_router
        app.include_router(reminders_router)
        from app.bridge.reminders_feed import router as bridge_reminders_router
        app.include_router(bridge_reminders_router)
        print("[startup] Reminders: [OK] registered (+ bridge upcoming/due/ack feed)")
    except Exception as _reminders_err:
        print(f"[startup] Reminders: [WARN] {_reminders_err}")

    # ASTRA Room — scene director (2026-06-12): LLM-composed SceneDocs pushed to the
    # Unity renderer over /scene/ws; compose is Bearer-auth'd, renderer endpoints
    # are local-trusted (see app/scene_director/router.py header).
    try:
        from app.scene_director.router import router as scene_director_router
        app.include_router(scene_director_router)
        print("[startup] Scene Director (ASTRA Room): [OK] registered")
    except Exception as _scene_err:
        print(f"[startup] Scene Director: [WARN] {_scene_err}")

    # Documents — Univer editor pane seam (2026-06-12): file ⇄ snapshot conversion
    # (xlsx/docx/csv/md) + the editor action channel that lets chat tools read and
    # edit whatever is open in the desktop's command-centre Editor tab.
    try:
        from app.documents import router as documents_router
        app.include_router(documents_router)
        print("[startup] Documents (editor pane): [OK] registered")
    except Exception as _documents_err:
        print(f"[startup] Documents: [WARN] {_documents_err}")

    # ASTRA presence (2026-06-13, Room v2): in-memory orb-state broadcast so the Room
    # orb (and any surface) can reflect ASTRA's live state — WS /astra/ws + POST
    # /astra/state. Local-trusted like the Room's /scene/* endpoints.
    try:
        from app.astra_presence.router import router as astra_presence_router
        app.include_router(astra_presence_router)
        print("[startup] ASTRA Presence: [OK] registered")
    except Exception as _presence_err:
        print(f"[startup] ASTRA Presence: [WARN] {_presence_err}")

    # ASTRA Room voice (2026-06-13, Room v2): local-trusted POST /scene/ask — text in,
    # ASTRA's spoken reply out (audio/mpeg + X-Full-Text). Reuses the bridge chat brain
    # (run_astra_chat) + TTS helpers without bridge auth (see scene_director/voice.py).
    try:
        from app.scene_director.voice import router as scene_voice_router
        app.include_router(scene_voice_router)
        print("[startup] Scene Voice (Room /ask): [OK] registered")
    except Exception as _scene_voice_err:
        print(f"[startup] Scene Voice: [WARN] {_scene_voice_err}")

    try:
        from app.cloud.router import router as cloud_router
        app.include_router(cloud_router)
        print("[startup] Cloud (Proton Drive): [OK] registered")
    except Exception as _cloud_err:
        print(f"[startup] Cloud: [WARN] {_cloud_err}")

    try:
        from app.email_service.router import router as email_router
        app.include_router(email_router)
        print("[startup] Email (Proton Mail): [OK] registered")
    except Exception as _email_err:
        print(f"[startup] Email: [WARN] {_email_err}")

    return _BRIEFING_AVAILABLE

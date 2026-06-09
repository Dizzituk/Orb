"""
Orb Backend - FastAPI Application
Version: 0.17.0

v0.17.0 Changes (Refactor):
- Refactored into multiple files for maintainability
- Endpoints moved to app/endpoints/
- Helpers moved to app/helpers/
- BUG FIX: File attachment context ordering (current file now appears first)

v0.16.0 Changes (Log Introspection):
- Added read-only log introspection feature
- GET /introspection/logs/last - Last completed job logs
- GET /introspection/logs - Time-based log query
- GET /introspection/logs/{job_id} - Specific job logs
- LLM-powered log summarization

Previous versions: See git history
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # Load .env from project root before anything reads os.getenv
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.db import init_db, get_db
from app.auth import require_auth, is_auth_configured
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

app = FastAPI(
    title="Orb Assistant",
    version="0.17.0",
    description="Personal AI assistant with multi-LLM orchestration and semantic search",
)


# ============================================================================
# CORS
# ============================================================================

# Suppress noisy access log entries (thumbnails, polling, OPTIONS)
import logging as _logging

class _QuietAccessFilter(_logging.Filter):
    """Suppress high-frequency polling endpoints from the live console.

    Philosophy: the live console exists for things a human should notice —
    errors, real user actions (POST/DELETE/PATCH), important state changes,
    and warnings. UI poll loops do not qualify. They still flow to the file
    log (uvicorn writes access entries through Python logging, and the file
    handler set up in app.logging_config does not apply this filter), so
    debugging is preserved — we just stop firehose-ing the eye.

    Filter rules, in order:
      1. Errors / non-2xx responses ALWAYS pass through (visibility on failure).
      2. CORS preflights (OPTIONS) are silently dropped — never informative.
      3. Successful GETs on known polling endpoints are silently dropped.
      4. Everything else passes (POST/DELETE/PATCH = real user intent).
    """

    # GET endpoints hit by the frontend or Electron on a timer. Each of
    # these has a comment so future-me knows why it's quiet — and can
    # un-mute it cheaply if the polling cadence ever becomes a real signal.
    _QUIET_GET_PATHS = (
        '/drive/thumbnail',              # image previews — noisy and useless
        '/bridge/pending-navigation',    # Android bridge poll
        '/ping',                          # health checks (Chatterbox, Bridge)
        '/web_automation/pending-action', # Electron long-poll, ~5x/5s
        '/web_automation/sessions',       # Web tab list refresh
        '/api/cost/tally',                # cost meter UI poll
        '/auth/status',                   # auth header refresh
        '/auth/check',                    # auth header refresh
        '/memory/projects',               # left-nav project list refresh
        '/drive/categories',              # Drive sidebar refresh
        '/drive/storage-stats',           # Drive header stats
        '/drive/files',                   # Drive listing refresh
    )

    def filter(self, record: _logging.LogRecord) -> bool:
        msg = record.getMessage()

        # CORS preflights — never useful in the live stream.
        if 'OPTIONS' in msg:
            return False

        # Anything other than a clean success: let it through. uvicorn's
        # access format ends with the status reason, e.g. '" 200 OK' or
        # '" 304 Not Modified'. Treat both as "successful + boring".
        is_clean_success = (' 200 OK' in msg) or (' 304 Not Modified' in msg)
        if not is_clean_success:
            return True

        # Successful GET on a known polling path — mute.
        for path in self._QUIET_GET_PATHS:
            if f'"GET {path}' in msg:
                return False

        return True

_logging.getLogger('uvicorn.access').addFilter(_QuietAccessFilter())

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "file://",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
def on_startup():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/files", exist_ok=True)

    import asyncio
    _loop = asyncio.get_event_loop()
    _original_handler = _loop.get_exception_handler()

    def _suppress_loop_closed(loop, context):
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return
        if _original_handler:
            _original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    _loop.set_exception_handler(_suppress_loop_closed)

    try:
        from app.logging_config import setup_file_logging
        _log_path = setup_file_logging()
        print(f"[startup] File logging: [OK] {_log_path}")
    except Exception as e:
        print(f"[startup] File logging: [WARN] failed to init: {e}")

    print("[startup] Initializing encryption...")
    from app.crypto import require_master_key_or_exit, is_master_key_initialized
    require_master_key_or_exit()

    if is_master_key_initialized():
        print("[startup] Database encryption: [OK] master key active")

    init_db()

    print("[startup] Checking authentication...")
    if is_auth_configured():
        print("[startup] Password authentication: [OK] configured")
    else:
        print("[startup] Password authentication: [X] NOT CONFIGURED")
        print("[startup] Call POST /auth/setup to set a password")

    print("[startup] Checking environment variables...")
    if os.getenv("GOOGLE_API_KEY"):
        print("[startup] GOOGLE_API_KEY: [OK] set (enables vision + web search)")
    else:
        print("[startup] GOOGLE_API_KEY: [X] NOT SET - vision and web search will fail")

    if os.getenv("OPENAI_API_KEY"):
        print("[startup] OPENAI_API_KEY: [OK] set (enables chat + embeddings)")
    else:
        print("[startup] OPENAI_API_KEY: [X] NOT SET - chat and semantic search will fail")

    if os.getenv("ANTHROPIC_API_KEY"):
        print("[startup] ANTHROPIC_API_KEY: [OK] set")
    else:
        print("[startup] ANTHROPIC_API_KEY: [X] NOT SET")

    print("[startup] Checking Phase 4 status...")
    phase4_enabled = os.getenv("ORB_ENABLE_PHASE4", "false").lower() == "true"
    if phase4_enabled:
        print("[startup] Phase 4 Job System: [OK] ENABLED")
    else:
        print("[startup] Phase 4 Job System: [X] DISABLED")

    try:
        from app.astra_memory.indexer import run_full_index
        from app.db import SessionLocal
        _db = SessionLocal()
        _results = run_full_index(_db)
        print(f"[startup] ASTRA memory indexed: {sum(_results.values())} records")
        _db.close()
    except Exception as e:
        print(f"[startup] ASTRA memory indexing skipped: {e}")
    try:
        from app.settings.service import sync_all_to_env
        from app.db import SessionLocal
        _sdb = SessionLocal()
        _key_count = sync_all_to_env(_sdb)
        if _key_count > 0:
            print(f"[startup] Settings: synced {_key_count} API keys from DB")
        else:
            print("[startup] Settings: no DB-stored API keys (using .env)")
        _sdb.close()
    except Exception as e:
        print(f"[startup] Settings sync skipped: {e}")

    # v0.14.0: Boot-time filesystem scan — awareness of all personal files
    # Positioned AFTER settings sync so GOOGLE_API_KEY is available for
    # the background content indexer (Tier 2) which needs Gemini embeddings.
    try:
        from app.drive.boot_scan import run_boot_scan
        from app.db import SessionLocal
        _scan_db = SessionLocal()
        _scan_result = run_boot_scan(_scan_db)
        _manifest = _scan_result.get("manifest", {})
        if isinstance(_manifest, dict) and "total" in _manifest:
            _new = _manifest.get("new", 0)
            _mod = _manifest.get("modified", 0)
            _del = _manifest.get("deleted", 0)
            _ms = _manifest.get("duration_ms", 0)
            print(f"[startup] Drive scan: {_manifest['total']} files ({_new} new, {_mod} modified, {_del} deleted) in {_ms}ms")
        elif isinstance(_manifest, dict) and "error" in _manifest:
            print(f"[startup] Drive scan: [WARN] {_manifest['error']}")
        _scan_db.close()
    except Exception as e:
        print(f"[startup] Drive scan skipped: {e}")

    # ── Live file watcher: keeps drive_file_manifest in sync between boots ──
    try:
        from app.drive.file_watcher import start_file_watcher
        _watch_result = start_file_watcher()
        _wstatus = _watch_result.get("status", "unknown")
        _wpaths = _watch_result.get("paths", [])
        if _wstatus == "running":
            print(f"[startup] File watcher: [OK] watching {len(_wpaths)} path(s)")
        elif _wstatus == "already_running":
            print("[startup] File watcher: [OK] already running")
        else:
            print(f"[startup] File watcher: [WARN] status={_wstatus}")
    except Exception as e:
        print(f"[startup] File watcher skipped: {e}")

    try:
        from app.content.seed import seed_content_data
        from app.db import SessionLocal
        _cdb = SessionLocal()
        _seed_result = seed_content_data(_cdb)
        if _seed_result.get("series_created", 0) > 0:
            print(f"[startup] Content pipeline: seeded {_seed_result['series_created']} series")
        else:
            print("[startup] Content pipeline: [OK] data present")
        _cdb.close()
    except Exception as e:
        print(f"[startup] Content pipeline seed skipped: {e}")

    try:
        from app.investments.scheduler import start_scheduler as start_investments_scheduler
        start_investments_scheduler()
        print("[startup] Investments scheduler: [OK] 08:00 + 18:00 UK")
    except Exception as e:
        print(f"[startup] Investments scheduler: [WARN] {e}")

    # ── ASTRA memory decay scheduler ──
    # Recomputes preference confidence, expires low-confidence preferences,
    # cleans stale hot index entries on a daily timer. Defined since v2.0
    # but pre-2026-05-02 was never actually started at boot — the only
    # path was a manual POST to /astra-memory/decay/scheduler/start that
    # did not survive a process restart, so decay was effectively off.
    if os.getenv("ASTRA_DECAY_SCHEDULER_ENABLED", "true").lower() not in ("false", "0", "no"):
        try:
            from app.astra_memory.decay_job import start_decay_scheduler_background
            _decay_started = start_decay_scheduler_background(loop=_loop, interval_hours=24.0)
            if _decay_started:
                print("[startup] Decay scheduler: [OK] 24h interval")
            else:
                print("[startup] Decay scheduler: [WARN] not started (already running or no loop)")
        except Exception as e:
            print(f"[startup] Decay scheduler: [WARN] {e}")
    else:
        print("[startup] Decay scheduler: [SKIP] ASTRA_DECAY_SCHEDULER_ENABLED=false")

    # v3.0: Take a fresh investments snapshot on startup if stale/missing
    try:
        from app.db import SessionLocal
        from app.investments.models import PortfolioSnapshot
        from datetime import datetime, timezone, timedelta
        _inv_db = SessionLocal()
        _latest = _inv_db.query(PortfolioSnapshot).order_by(PortfolioSnapshot.captured_at.desc()).first()
        _stale = not _latest or (datetime.now(timezone.utc) - _latest.captured_at.replace(tzinfo=timezone.utc)) > timedelta(hours=12)
        if _stale:
            import asyncio
            from app.investments.service import take_snapshot
            try:
                asyncio.get_event_loop().run_until_complete(take_snapshot(_inv_db))
                print("[startup] Investments: [OK] fresh snapshot taken")
            except Exception as _snap_err:
                print(f"[startup] Investments snapshot: [WARN] {_snap_err}")
        else:
            _age = datetime.now(timezone.utc) - _latest.captured_at.replace(tzinfo=timezone.utc)
            print(f"[startup] Investments: [OK] snapshot {_age.seconds // 3600}h old")
        _inv_db.close()
    except Exception as e:
        print(f"[startup] Investments startup snapshot skipped: {e}")

    try:
        from app.finance.seed import seed_finance_data
        from app.db import SessionLocal
        _fdb = SessionLocal()
        _fin_result = seed_finance_data(_fdb)
        if _fin_result.get("categories_created", 0) > 0:
            print(f"[startup] Finance: seeded {_fin_result['categories_created']} categories")
        else:
            print("[startup] Finance: [OK] data present")
        _fdb.close()
    except Exception as e:
        print(f"[startup] Finance seed skipped: {e}")

    try:
        from app.lifestyle.seed import seed_lifestyle_data
        from app.db import SessionLocal
        _ldb = SessionLocal()
        _life_result = seed_lifestyle_data(_ldb)
        if _life_result.get("goals_created", 0) > 0:
            print(f"[startup] Lifestyle: seeded {_life_result['goals_created']} default goals")
        else:
            print("[startup] Lifestyle: [OK] data present")
        _ldb.close()
    except Exception as e:
        print(f"[startup] Lifestyle seed skipped: {e}")

    # Web Automation — seed default sessions so ASTRA has something to drive
    try:
        from app.web_automation import seed_sessions as _seed_web_sessions
        from app.db import SessionLocal
        _wdb = SessionLocal()
        _web_seed_result = _seed_web_sessions(_wdb)
        if _web_seed_result.get("created", 0) > 0:
            print(f"[startup] Web Automation: seeded {_web_seed_result['created']} session(s)")
        else:
            print(f"[startup] Web Automation: [OK] {_web_seed_result.get('total', 0)} session definitions present")
        _wdb.close()
    except Exception as e:
        print(f"[startup] Web Automation seed skipped: {e}")

    try:
        from app.translation.confidence_graduation import run_graduation
        _grad_count = run_graduation()
        if _grad_count > 0:
            print(f"[startup] Confidence graduation: {_grad_count} rule(s) graduated to Tier 0")
        else:
            print("[startup] Confidence graduation: no new candidates")
    except Exception as e:
        print(f"[startup] Confidence graduation skipped: {e}")

    try:
        from app.builds.service import recover_stale_running_stages
        from app.db import SessionLocal
        _bdb = SessionLocal()
        _reset_count = recover_stale_running_stages(_bdb)
        if _reset_count > 0:
            print(f"[startup] Builds recovery: reset {_reset_count} stale running stage(s)")
        _bdb.close()
    except Exception as e:
        print(f"[startup] Builds recovery skipped: {e}")

    if _BRIEFING_AVAILABLE:
        try:
            from app.briefing.briefing_scheduler import start_scheduler_background
            import asyncio
            loop = asyncio.get_event_loop()
            start_scheduler_background(loop)
            print("[startup] Briefing scheduler: [OK] background task started")
        except Exception as e:
            print(f"[startup] Briefing scheduler: [WARN] {e}")


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
from app.lifestyle.router import router as lifestyle_router
app.include_router(lifestyle_router)
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
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve generated images (image generation pipeline output).
# Path resolved via shared helper — honours IMAGE_OUTPUT_DIR if set,
# else falls back to legacy ASTRA_OUTPUT_DIR/images. Setting
# IMAGE_OUTPUT_DIR to a file_watcher-watched folder lets the agent's
# search tools find generated images when posting to Meta etc.
from app.llm.image_output_dir import get_image_output_dir
_output_images_dir = str(get_image_output_dir())
os.makedirs(_output_images_dir, exist_ok=True)
app.mount("/output/images", StaticFiles(directory=_output_images_dir), name="output_images")
print(f"[startup] Output images: [OK] serving {_output_images_dir} at /output/images")


@app.get("/")
def read_index():
    return {"status": "ok", "version": "0.17.0"}


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/health")
def health():
    """Health endpoint for Astra Bridge companion app."""
    return {"status": "ok", "service": "astra-backend", "version": "2.2"}


# Bridge API (Android companion app)
try:
    from app.bridge.router import router as bridge_router
    app.include_router(bridge_router)
    from app.bridge.dashboards import router as bridge_dashboards_router
    app.include_router(bridge_dashboards_router)
    from app.bridge.tts_proxy import router as bridge_tts_router
    app.include_router(bridge_tts_router)
    from app.bridge.missed_replies import router as bridge_missed_replies_router
    app.include_router(bridge_missed_replies_router)
    print("[startup] Bridge API: [OK] registered (+ TTS proxy, + missed-replies)")
except ImportError as _bridge_err:
    print(f"[startup] Bridge API: [WARN] {_bridge_err}")

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


@app.post("/admin/reset-flow/{project_id}")
def reset_flow(project_id: int, auth=Depends(require_auth)):
    from app.llm.spec_flow_state import set_flow_state, SpecFlowStage, _FLOW_STATES
    state = _FLOW_STATES.get(project_id)
    if not state:
        return {"error": f"No flow state for project {project_id}"}
    old_stage = state.stage.value
    state.stage = SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM
    set_flow_state(state)
    return {
        "reset": True,
        "project_id": project_id,
        "old_stage": old_stage,
        "new_stage": state.stage.value,
        "weaver_output_preserved": bool(state.weaver_job_description),
        "vision_context_preserved": bool(state.weaver_vision_context),
    }


@app.get("/providers")
def list_providers(auth=Depends(require_auth)):
    from app.llm.clients import check_provider_availability
    return check_provider_availability()


@app.get("/job-types")
def list_job_types(auth=Depends(require_auth)):
    from app.llm import JobType
    return {"job_types": [{"value": jt.value, "name": jt.name} for jt in JobType]}






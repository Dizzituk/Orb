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
from app.content.engagement.router import router as engagement_router
import app.content.engagement.models  # noqa: F401 — register Engagement tables with Base
from app.builds.router import router as builds_router
import app.builds.models  # noqa: F401 — register Build Projects tables with Base
import app.builds.messages  # noqa: F401 — register Build Project Messages table with Base
from app.education.router import router as education_router
import app.education.models  # noqa: F401 — register Education tables with Base
from app.settings.router import router as settings_router
from app.transparency.router import router as transparency_router
import app.transparency.models  # noqa: F401 — register Transparency tables with Base

# Import refactored endpoints
from app.endpoints import router as endpoints_router

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
from app.finance.router import router as finance_router
app.include_router(finance_router)
from app.finance.drive_router import router as finance_drive_router
app.include_router(finance_drive_router)
from app.finance.budget_router import router as finance_budget_router
app.include_router(finance_budget_router)
from app.finance.credit_card_router import router as finance_cc_router
app.include_router(finance_cc_router)
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

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def read_index():
    return {"status": "ok", "version": "0.17.0"}


@app.get("/ping")
def ping():
    return {"status": "ok"}


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





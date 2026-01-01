# FILE: main.py
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
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.db import init_db, get_db
from app.auth import require_auth, is_auth_configured
from app.auth.router import router as auth_router
from app.memory.router import router as memory_router
from app.llm.stream_router import router as stream_router
from app.llm.telemetry_router import router as telemetry_router
from app.llm.web_search_router import router as web_search_router
from app.embeddings.router import router as embeddings_router, search_router as embeddings_search_router
from app.introspection.router import router as introspection_router
from app.astra_memory.router import router as astra_memory_router

# Import refactored endpoints
from app.endpoints import router as endpoints_router


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

    # ASTRA Memory: Auto-index on startup
    try:
        from app.astra_memory.indexer import run_full_index
        from app.db import SessionLocal
        _db = SessionLocal()
        _results = run_full_index(_db)
        print(f"[startup] ASTRA memory indexed: {sum(_results.values())} records")
        _db.close()
    except Exception as e:
        print(f"[startup] ASTRA memory indexing skipped: {e}")


# ============================================================================
# ROUTERS
# ============================================================================

app.include_router(auth_router)
app.include_router(memory_router)
app.include_router(stream_router)
app.include_router(telemetry_router)
app.include_router(web_search_router)
app.include_router(embeddings_router)
app.include_router(embeddings_search_router)
app.include_router(astra_memory_router)

# Refactored endpoints (chat, chat_with_attachments, direct_llm)
app.include_router(endpoints_router)

# Log introspection (read-only, requires auth)
app.include_router(
    introspection_router,
    tags=["Introspection"],
    dependencies=[Depends(require_auth)]
)

# Phase 4 conditional routers
if os.getenv("ORB_ENABLE_PHASE4", "false").lower() == "true":
    try:
        from app.jobs.router import router as jobs_router
        from app.artefacts.router import router as artefacts_router
        
        app.include_router(
            jobs_router,
            prefix="/jobs",
            tags=["Phase 4 Jobs"],
            dependencies=[Depends(require_auth)]
        )
        app.include_router(
            artefacts_router,
            prefix="/artefacts",
            tags=["Phase 4 Artefacts"],
            dependencies=[Depends(require_auth)]
        )
        print("[startup] Phase 4 routers registered successfully")
    except ImportError as e:
        print(f"[startup] WARNING: Phase 4 import failed: {e}")


# ============================================================================
# STATIC FILES
# ============================================================================

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================

@app.get("/")
def read_index():
    """Serve index page or health check."""
    return {"status": "ok", "version": "0.17.0"}


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return {"status": "ok"}


# ============================================================================
# PROTECTED ENDPOINTS
# ============================================================================

@app.get("/providers")
def list_providers(auth = Depends(require_auth)):
    """List available LLM providers."""
    from app.llm.clients import check_provider_availability
    return check_provider_availability()


@app.get("/job-types")
def list_job_types(auth = Depends(require_auth)):
    """List available job types."""
    from app.llm import JobType
    return {
        "job_types": [
            {"value": jt.value, "name": jt.name}
            for jt in JobType
        ]
    }

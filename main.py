# Purpose: Orb Backend - FastAPI Application
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.artefacts.router, app.astra_memory.decay_job, app.astra_memory.indexer, app.astra_memory.router (+94 more)
# Last-renovated: 2026-06-12
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

# Tightened 2026-07-02 (was allow_methods=["*"] / allow_headers=["*"]):
# methods = verbs the renderer actually issues (PUT: 2 call sites, PATCH: 3);
# headers = what the app sends (X-Auth-Token: financeApi.ts; X-Idempotency-Key
# + X-Original-Filename: Bridge uploads; X-Astra-Local: Electron local-trust).
# "file://" stays: main.js production mode loads dist/index.html via loadFile.
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
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Auth-Token",
        "X-Idempotency-Key",
        "X-Astra-Local",
        "X-Original-Filename",
    ],
)

# Firewall — one-way sandbox isolation (ACTIVATED 2026-07-02; was dead code).
# Added AFTER CORS so it wraps outermost and rejects blocked sources before
# anything else runs. Client-IP resolution is spoof-resistant: XFF/X-Real-IP
# only count from trusted proxy peers (app/security/client_ip.py). The
# on_startup hook logs "Firewall middleware: ACTIVE" and
# tests/test_firewall_header_trust.py asserts this wiring exists, so it can
# never silently regress to dead code again.
from app.security.firewall import FirewallMiddleware
app.add_middleware(FirewallMiddleware)


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
def on_startup():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/files", exist_ok=True)

    # DEREK phase 2 (2026-07-04): fail-loud stage-role check. An unresolvable
    # role logs an unmissable ERROR naming the exact env key (it does not
    # kill the backend — chat must boot; the pipeline errors at dispatch).
    try:
        from app.llm.stage_roles import validate_stage_roles_at_startup
        _sr_failures = validate_stage_roles_at_startup()
        print(
            "[startup] Stage roles: "
            + ("ALL RESOLVED" if _sr_failures == 0 else f"{_sr_failures} UNRESOLVED - see ERROR log")
        )
    except Exception as _sr_exc:
        print(f"[startup] Stage-role validation unavailable: {_sr_exc}")

    # Assert the sandbox-isolation firewall is wired (see add_middleware above).
    # If this line ever stops printing ACTIVE, the one-way isolation is OFF.
    _fw_active = any(
        getattr(m, "cls", None) is not None and m.cls.__name__ == "FirewallMiddleware"
        for m in app.user_middleware
    )
    if _fw_active:
        print("[startup] Firewall middleware: ACTIVE (sandbox 192.168.250.0/24 blocked, XFF spoof-proof)")
    else:
        print("[startup] Firewall middleware: [X] NOT REGISTERED — one-way sandbox isolation is OFF")

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
    # Fail closed if bcrypt is missing (2026-07-02) — never silently
    # downgrade password hashing to SHA256 again.
    from app.auth.config import assert_strong_hash_available
    assert_strong_hash_available()
    print("[startup] Password hashing: [OK] bcrypt available (fail-closed check)")
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

    # Capability manifest (2026-06-12): regenerate the self-description when
    # the registry hash changed (tools/handlers/models), so "what are you?"
    # always retrieves a current answer.
    try:
        from app.self_model.capability_manifest import regenerate_if_stale
        from app.db import SessionLocal
        _mdb = SessionLocal()
        _regen = regenerate_if_stale(_mdb)
        print(f"[startup] Capability manifest: "
              f"{'regenerated' if _regen else '[OK] up to date'}")
        _mdb.close()
    except Exception as e:
        print(f"[startup] Capability manifest skipped: {e}")
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

    # Lifestyle scheduler — Jobs 4+5 memory roadmap (2026-06-10): nudge
    # checkpoints (13:30/19:15 local) + nightly habit learning (02:40).
    try:
        from app.lifestyle.scheduler import start_lifestyle_scheduler_background
        if start_lifestyle_scheduler_background(loop=_loop):
            print("[startup] Lifestyle scheduler: [OK] nudges + nightly habit learning")
        else:
            print("[startup] Lifestyle scheduler: [WARN] not started (disabled, running, or no loop)")
    except Exception as e:
        print(f"[startup] Lifestyle scheduler: [WARN] {e}")

    # Reminders (2026-07-01): 30s poll for punctual desktop firing; the
    # phone gets its punctuality from a local exact alarm, not this loop.
    try:
        from app.reminders.scheduler import start_reminder_scheduler_background
        if start_reminder_scheduler_background(loop=_loop):
            print("[startup] Reminder scheduler: [OK] 30s due-reminder polling")
        else:
            print("[startup] Reminder scheduler: [WARN] not started (disabled, running, or no loop)")
    except Exception as e:
        print(f"[startup] Reminder scheduler: [WARN] {e}")

    # ASTRA Sentinel Phase 1 (2026-06-12): network security monitor — collect
    # every 30s from the elevated agent (127.0.0.1:8771), daily retention prune
    # + baseline maintenance. Degrades to agent_offline if the agent is down.
    try:
        from app.sentinel.scheduler import start_sentinel_scheduler_background
        if start_sentinel_scheduler_background(loop=_loop):
            print("[startup] Sentinel scheduler: [OK] 30s collect + daily prune")
        else:
            print("[startup] Sentinel scheduler: [WARN] not started (disabled, running, or no loop)")
    except Exception as e:
        print(f"[startup] Sentinel scheduler: [WARN] {e}")

    # Idle governor (2026-07-01): activity-based background work. Drains the
    # persistent idle-task ledger (repo map, watchers, deep research) after
    # IDLE_MINUTES of chat silence; boot catch-up re-queues anything due.
    # All task LLM use is locked to the local lane (background_local).
    try:
        from app.idle.governor import start_idle_governor_background
        if start_idle_governor_background(loop=_loop):
            print("[startup] Idle governor: [OK] drains task ledger when idle")
        else:
            print("[startup] Idle governor: [WARN] not started (disabled, running, or no loop)")
    except Exception as e:
        print(f"[startup] Idle governor: [WARN] {e}")

    # LANE E (2026-07-02): 4080 VRAM orchestrator — restores persisted state
    # (GAMING survives restarts; interrupted INGEST recovers to INTERACTIVE)
    # and converges residents in a background thread (never blocks startup).
    try:
        import threading as _gpu_threading
        from app.gpu.orchestrator import get_orchestrator
        _gpu_orch = get_orchestrator()
        # boot_bringup (not plain converge): staggers + verifies the text
        # embedder's cold-start behind Nat/Chatterbox so it can't lose the
        # 4080 VRAM race and strand embeddings (2026-07-03 incident).
        _gpu_threading.Thread(target=_gpu_orch.boot_bringup, daemon=True).start()
        print(f"[startup] GPU orchestrator: [OK] state={_gpu_orch.current_state()}")
    except Exception as e:
        print(f"[startup] GPU orchestrator: [WARN] {e}")

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
        if _web_seed_result.get("failed", 0) > 0:
            print(f"[startup] Web Automation: {_web_seed_result['failed']} session seed(s) FAILED, "
                  f"{_web_seed_result.get('created', 0)} created — see log")
        elif _web_seed_result.get("created", 0) > 0:
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

# ============================================================================
# ROUTER REGISTRATION — extracted to app/router_registry.py (BATCH 4)
# ============================================================================
from app.router_registry import register_routers
_BRIEFING_AVAILABLE = register_routers(app)

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

# Rendered reports cache (2026-07-01): the desktop reports window opens these
# URLs fullscreen; the Bridge fetches the same files as document artifacts
# via /bridge/artifacts/document/<filename>. REPORTS_CACHE_DIR overrides.
try:
    from app.reports.cache import get_reports_cache_dir
    _reports_dir = str(get_reports_cache_dir())
    app.mount("/output/reports", StaticFiles(directory=_reports_dir), name="output_reports")
    print(f"[startup] Reports cache: [OK] serving {_reports_dir} at /output/reports")
except Exception as e:
    print(f"[startup] Reports cache: [WARN] {e}")


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






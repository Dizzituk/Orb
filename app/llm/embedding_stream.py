# FILE: app/llm/embedding_stream.py
"""
Streaming handlers for embedding commands.

Commands:
- "embedding status" → Show current embedding stats
- "generate embeddings" → Trigger manual embedding job

v1.1 (2026-01): Added schema migration check, better error handling
v1.0 (2026-01): Initial implementation
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional
from sqlite3 import OperationalError as SQLite3OperationalError

from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError

from app.llm.audit_logger import RoutingTrace
from app.db import get_db_session, ensure_embedding_schema

logger = logging.getLogger(__name__)


# =============================================================================
# SSE HELPERS
# =============================================================================

def _sse_token(content: str) -> str:
    return "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"


def _sse_error(error: str) -> str:
    return "data: " + json.dumps({"type": "error", "error": error}) + "\n\n"


def _sse_done(
    *,
    provider: str = "local",
    model: str = "embedding_manager",
    total_length: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    meta: Optional[dict] = None,
) -> str:
    payload = {
        "type": "done",
        "provider": provider,
        "model": model,
        "total_length": total_length,
        "success": success,
    }
    if error:
        payload["error"] = error
    if meta:
        payload["meta"] = meta
    return "data: " + json.dumps(payload) + "\n\n"


# =============================================================================
# EMBEDDING STATUS COMMAND
# =============================================================================

async def generate_embedding_status_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream embedding status report.
    
    Triggered by: "Astra, command: embedding status"
    """
    try:
        # v1.1: Ensure schema is migrated before accessing embedding columns
        ensure_embedding_schema()
        
        from app.rag.jobs.embedding_job import (
            format_embedding_status_report,
            get_embedding_stats,
        )
        
        # Generate report
        report = format_embedding_status_report(db)
        
        yield _sse_token(report)
        
        # Get stats for meta
        stats = get_embedding_stats(db)
        
        yield _sse_done(
            success=True,
            total_length=len(report),
            meta=stats,
        )
        
    except ImportError as e:
        logger.error(f"[embedding_status] Import error: {e}")
        yield _sse_error(f"Embedding module not available: {e}")
        yield _sse_done(success=False, error=str(e))
    
    except (SQLAlchemyOperationalError, SQLite3OperationalError) as e:
        # v1.1: Schema mismatch - provide helpful message
        error_str = str(e)
        logger.error(f"[embedding_status] Schema error: {e}")
        
        if "no such column" in error_str.lower():
            yield _sse_error(
                "⚠️ **Database Schema Out of Date**\n\n"
                "The embedding columns are missing from arch_code_chunks table.\n\n"
                "**To fix:**\n"
                "1. Restart the backend server (init_db will run migration)\n"
                "2. Or run: `from app.db import ensure_embedding_schema; ensure_embedding_schema()`\n"
            )
        else:
            yield _sse_error(f"Database error: {e}")
        
        yield _sse_done(success=False, error="schema_mismatch")
        
    except Exception as e:
        logger.exception(f"[embedding_status] Error: {e}")
        yield _sse_error(f"Failed to get embedding status: {e}")
        yield _sse_done(success=False, error=str(e))


# =============================================================================
# GENERATE EMBEDDINGS COMMAND
# =============================================================================

async def generate_embeddings_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Trigger embedding generation (manual command).
    
    Triggered by: "Astra, command: generate embeddings"
    
    This queues the background job and returns immediately.
    For synchronous (blocking) embedding, use the HTTP endpoint.
    """
    try:
        # v1.1: Ensure schema is migrated before accessing embedding columns
        ensure_embedding_schema()
        
        from app.rag.jobs.embedding_job import (
            queue_embedding_job,
            get_embedding_status,
            get_embedding_stats,
            EMBEDDING_AUTO_ENABLED,
        )
        
        yield _sse_token("🔄 Checking embedding status...\n\n")
        
        # Get current stats
        stats = get_embedding_stats(db)
        
        yield _sse_token(f"📊 Current state:\n")
        yield _sse_token(f"   Total chunks: {stats['total_chunks']}\n")
        yield _sse_token(f"   Embedded: {stats['embedded_chunks']} ({stats['embedding_pct']}%)\n")
        yield _sse_token(f"   Pending: {stats['pending_chunks']}\n\n")
        
        # Check if already running
        status = get_embedding_status()
        if status.running:
            yield _sse_token(f"⚠️ Embedding job already running!\n")
            yield _sse_token(f"   Current tier: {status.current_tier}\n")
            yield _sse_token(f"   Progress: {status.embedded_chunks}/{status.total_chunks}\n")
            yield _sse_done(
                success=True,
                meta={"already_running": True, **stats},
            )
            return
        
        # Check if anything to embed
        if stats['pending_chunks'] == 0:
            yield _sse_token("✅ All chunks already embedded! Nothing to do.\n")
            yield _sse_done(
                success=True,
                meta={"nothing_to_embed": True, **stats},
            )
            return
        
        # Queue the job
        yield _sse_token(f"🚀 Queueing embedding job for {stats['pending_chunks']} chunks...\n\n")
        
        # Create a session factory for the background thread
        # get_db_session() returns a Session directly (not a generator)
        queued = queue_embedding_job(get_db_session)
        
        if queued:
            yield _sse_token("✅ Embedding job queued!\n\n")
            yield _sse_token("The job will run in the background with priority ordering:\n")
            yield _sse_token("   1. Tier1_Critical: Entry points, routers, local_tools\n")
            yield _sse_token("   2. Tier2_High: Pipeline (spec_gate, overwatcher, weaver)\n")
            yield _sse_token("   3. Tier3_Medium: Services, models, schemas\n")
            yield _sse_token("   4. Tier4_Low: Handlers, utils, clients\n")
            yield _sse_token("   5. Tier5_Normal: Everything else\n\n")
            yield _sse_token("Use `embedding status` to check progress.\n")
            yield _sse_token("Semantic search will work as soon as Tier1 completes (~30s).\n")
        else:
            if not EMBEDDING_AUTO_ENABLED:
                yield _sse_token("⚠️ Auto-embedding is disabled (ORB_EMBEDDING_AUTO=false)\n")
                yield _sse_token("Set environment variable to enable.\n")
            else:
                yield _sse_token("⚠️ Could not queue job (may already be running)\n")
        
        yield _sse_done(
            success=queued,
            meta={"queued": queued, **stats},
        )
        
    except ImportError as e:
        logger.error(f"[generate_embeddings] Import error: {e}")
        yield _sse_error(f"Embedding module not available: {e}")
        yield _sse_done(success=False, error=str(e))
    
    except (SQLAlchemyOperationalError, SQLite3OperationalError) as e:
        # v1.1: Schema mismatch - provide helpful message
        error_str = str(e)
        logger.error(f"[generate_embeddings] Schema error: {e}")
        
        if "no such column" in error_str.lower():
            yield _sse_error(
                "⚠️ **Database Schema Out of Date**\n\n"
                "The embedding columns are missing from arch_code_chunks table.\n\n"
                "**To fix:**\n"
                "1. Restart the backend server (init_db will run migration)\n"
                "2. Or run: `from app.db import ensure_embedding_schema; ensure_embedding_schema()`\n"
            )
        else:
            yield _sse_error(f"Database error: {e}")
        
        yield _sse_done(success=False, error="schema_mismatch")
        
    except Exception as e:
        logger.exception(f"[generate_embeddings] Error: {e}")
        yield _sse_error(f"Failed to start embedding job: {e}")
        yield _sse_done(success=False, error=str(e))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "generate_embedding_status_stream",
    "generate_embeddings_stream",
]

# FILE: app/rag/jobs/embedding_job.py
# Purpose: Background embedding job with incremental + priority support.
# Called-by: app.llm.embedding_stream, app.llm.local_tools.zobie.rag_helpers, app.llm.local_tools.zobie.streams.archmap_full, app.llm.local_tools.zobie.streams.scan_sandbox (+6 more)
# Depends-on: app.rag.jobs._embedding_batch, app.rag.jobs._embedding_job_utils_6, app.rag.jobs._embedding_job_utils_7, app.rag.jobs._embedding_job_utils_8 (+1 more)
# Last-renovated: 2026-06-11
"""
Background embedding job with incremental + priority support.

Features:
- Priority-based: High-value chunks (routers, pipelines) embedded first
- Incremental: Only embeds NEW or CHANGED chunks (via content_hash)
- Non-blocking: Runs in background thread
- Observable: Logs progress, writes status to DB
- Controllable: Can be disabled via env var, triggered manually
- Rate-limited: Batches API calls to avoid spikes

Priority Tiers (refined v1.1):
1. CRITICAL: main.py, stream_router, translation_routing, local_tools, intents
2. HIGH: spec_gate, overwatcher, weaver, critical_pipeline, rag/pipeline, memory/architecture
3. MEDIUM: models, services, schemas, repositories, db.py
4. LOW: handlers, clients, utils
5. NORMAL: everything else

v1.2 (2026-01): SQLite lock contention fix
  - Session-per-batch pattern (prevents poisoned session cascade)
  - Retry with exponential backoff on "database is locked" errors
  - Fresh session for each batch write
v1.1 (2026-01): Refined priority patterns per Taz's spec
v1.0 (2026-01): Initial implementation
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable

from sqlalchemy.orm import Session
from app.rag.jobs._embedding_job_utils_6 import ARCHITECTURE_OUTPUT_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_RATE_LIMIT_DELAY, SQLITE_LOCK_INITIAL_BACKOFF, SQLITE_LOCK_MAX_BACKOFF, SQLITE_LOCK_MAX_RETRIES, _is_sqlite_lock_error, run_embedding_job_sync
from app.rag.jobs._embedding_job_utils_7 import EMBEDDING_AUTO_ENABLED, EMBEDDING_MODEL, STATUS_FILE, classify_chunk_priority, compute_content_hash, format_embedding_status_report, get_embedding_stats, queue_embedding_job
from app.rag.jobs._embedding_job_utils_8 import EmbeddingPriority, get_embedding_status

logger = logging.getLogger(__name__)

# Priority patterns and chunk grouping in _embedding_priority.py
from app.rag.jobs._embedding_priority import PRIORITY_PATTERNS, get_chunks_by_priority


# =============================================================================
# STATUS TRACKING
# =============================================================================

@dataclass
class EmbeddingJobStatus:
    """Current embedding job status."""
    running: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    total_chunks: int = 0
    embedded_chunks: int = 0
    skipped_chunks: int = 0  # Already embedded, unchanged
    failed_chunks: int = 0
    pending_chunks: int = 0  # Remaining to embed
    
    current_tier: Optional[str] = None
    current_batch: int = 0
    total_batches: int = 0
    
    last_error: Optional[str] = None
    model_used: str = EMBEDDING_MODEL
    
    # Per-tier counts (for queue visibility)
    tier_counts: Dict[str, int] = field(default_factory=dict)
    tier_progress: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "running": self.running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_chunks": self.total_chunks,
            "embedded_chunks": self.embedded_chunks,
            "skipped_chunks": self.skipped_chunks,
            "failed_chunks": self.failed_chunks,
            "pending_chunks": self.pending_chunks,
            "progress_pct": round(100 * self.embedded_chunks / max(self.total_chunks, 1), 1),
            "current_tier": self.current_tier,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "last_error": self.last_error,
            "model_used": self.model_used,
            "tier_counts": self.tier_counts,
            "tier_progress": self.tier_progress,
        }
    
    def save_to_file(self):
        """Write status to disk for observability."""
        try:
            STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATUS_FILE, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write embedding status file: {e}")
    
    @classmethod
    def load_from_file(cls) -> Optional["EmbeddingJobStatus"]:
        """Load status from disk."""
        try:
            if STATUS_FILE.exists():
                with open(STATUS_FILE) as f:
                    data = json.load(f)
                status = cls()
                status.running = data.get("running", False)
                if data.get("started_at"):
                    status.started_at = datetime.fromisoformat(data["started_at"])
                if data.get("completed_at"):
                    status.completed_at = datetime.fromisoformat(data["completed_at"])
                status.total_chunks = data.get("total_chunks", 0)
                status.embedded_chunks = data.get("embedded_chunks", 0)
                status.skipped_chunks = data.get("skipped_chunks", 0)
                status.failed_chunks = data.get("failed_chunks", 0)
                status.pending_chunks = data.get("pending_chunks", 0)
                status.current_tier = data.get("current_tier")
                status.last_error = data.get("last_error")
                status.model_used = data.get("model_used", EMBEDDING_MODEL)
                status.tier_counts = data.get("tier_counts", {})
                status.tier_progress = data.get("tier_progress", {})
                return status
        except Exception as e:
            logger.warning(f"Failed to load embedding status file: {e}")
        return None


# Global status (thread-safe via GIL for simple reads)
_current_status = EmbeddingJobStatus()
_job_lock = threading.Lock()


# =============================================================================
# EMBEDDING JOB
# =============================================================================

class EmbeddingJob:
    """
    Background embedding job runner.
    
    Usage:
        job = EmbeddingJob(db_session_factory)
        job.run_async()  # Non-blocking
        # or
        job.run_sync()   # Blocking (for manual trigger)
    """
    
    def __init__(
        self,
        db_session_factory: Callable[[], Session],
        scan_id: Optional[int] = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        rate_limit_delay: float = EMBEDDING_RATE_LIMIT_DELAY,
    ):
        """
        Args:
            db_session_factory: Callable that returns a new DB session
            scan_id: Optional scan_id to limit embedding scope
            batch_size: Chunks per API call
            rate_limit_delay: Seconds between batches
        """
        self.db_session_factory = db_session_factory
        self.scan_id = scan_id
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self._stop_requested = False
    
    def run_async(self):
        """Start embedding job in background thread."""
        global _current_status
        
        with _job_lock:
            if _current_status.running:
                print("[embedding_job] run_async: Job already running, skipping")
                logger.warning("[embedding_job] Job already running, skipping")
                return False
        
        print("[embedding_job] run_async: Starting background thread...")
        thread = threading.Thread(target=self._run_internal, daemon=True, name="EmbeddingJobThread")
        thread.start()
        print(f"[embedding_job] run_async: Thread started (name={thread.name}, daemon={thread.daemon})")
        logger.info("[embedding_job] Background job started")
        return True
    
    def run_sync(self) -> EmbeddingJobStatus:
        """Run embedding job synchronously (blocking)."""
        return self._run_internal()
    
    def stop(self):
        """Request job stop (will complete current batch)."""
        self._stop_requested = True
        logger.info("[embedding_job] Stop requested")
    
    def _run_internal(self) -> EmbeddingJobStatus:
        """Main job logic."""
        global _current_status
        
        with _job_lock:
            _current_status = EmbeddingJobStatus(
                running=True,
                started_at=datetime.utcnow(),
                model_used=EMBEDDING_MODEL,
            )
        
        # Use both logger and print for maximum visibility
        def _log(msg: str):
            logger.info(msg)
            print(msg)
        
        _log("[embedding_job] " + "=" * 50)
        _log("[embedding_job] EMBEDDING JOB THREAD STARTED")
        _log(f"[embedding_job] scan_id filter: {self.scan_id or 'None (all chunks)'}")
        _log(f"[embedding_job] model: {EMBEDDING_MODEL}")
        _log(f"[embedding_job] batch_size: {self.batch_size}")
        _log(f"[embedding_job] status_file: {STATUS_FILE}")
        _log("[embedding_job] " + "=" * 50)
        _current_status.save_to_file()
        
        db = None  # v1.2: Initialize to avoid UnboundLocalError in finally
        try:
            _log("[embedding_job] Creating DB session...")
            db = self.db_session_factory()
            _log("[embedding_job] DB session created successfully")
            
            # Get chunks needing embedding, grouped by priority
            _log("[embedding_job] Querying chunks from ArchCodeChunk table...")
            chunks_by_tier = self._get_chunks_by_priority(db)
            
            # Calculate totals and log queue
            total_to_embed = sum(len(chunks) for chunks in chunks_by_tier.values())
            _current_status.total_chunks = total_to_embed
            _current_status.pending_chunks = total_to_embed
            
            _log(f"[embedding_job] Found {total_to_embed} chunks needing embedding")
            
            if total_to_embed == 0:
                _log("[embedding_job] NO CHUNKS TO EMBED - job finishing immediately")
                _log("[embedding_job] This could mean:")
                _log("[embedding_job]   1. All chunks are already embedded")
                _log("[embedding_job]   2. No ArchCodeChunk rows exist in the database")
                _log("[embedding_job]   3. scan_id filter excluded all chunks")
            
            # Build tier counts for visibility
            tier_names = {
                EmbeddingPriority.TIER1_CRITICAL: "Tier1_Critical",
                EmbeddingPriority.TIER2_HIGH: "Tier2_High",
                EmbeddingPriority.TIER3_MEDIUM: "Tier3_Medium",
                EmbeddingPriority.TIER4_LOW: "Tier4_Low",
                EmbeddingPriority.TIER5_NORMAL: "Tier5_Normal",
            }
            
            for tier, chunks in chunks_by_tier.items():
                tier_name = tier_names.get(tier, tier.name)
                _current_status.tier_counts[tier_name] = len(chunks)
                _current_status.tier_progress[tier_name] = {
                    "total": len(chunks),
                    "done": 0,
                    "failed": 0,
                }
            
            # Log queue breakdown
            queue_parts = [f"{tier_names[t]}={len(c)}" for t, c in chunks_by_tier.items() if c]
            logger.info(f"[embedding_job] Embedding queue: {' '.join(queue_parts)}")
            logger.info(f"[embedding_job] Total to embed: {total_to_embed} chunks")
            
            _current_status.save_to_file()
            
            # v1.2: Close the query session before processing batches
            # Each batch will create its own fresh session
            db.close()
            _log("[embedding_job] Query session closed, processing batches with fresh sessions")
            
            # Process by priority (TIER1 first, then TIER2, etc.)
            for tier in EmbeddingPriority:
                if self._stop_requested:
                    logger.info("[embedding_job] Stop requested, exiting")
                    break
                
                chunks = chunks_by_tier.get(tier, [])
                if not chunks:
                    continue
                
                # v1.2: Extract chunk IDs for session-per-batch pattern
                chunk_ids = [c.id for c in chunks]
                
                tier_name = tier_names.get(tier, tier.name)
                _current_status.current_tier = tier_name
                logger.info(f"[embedding_job] Processing {tier_name} ({len(chunk_ids)} chunks)")
                
                # v1.2: Pass chunk_ids, not chunk objects (session-per-batch)
                self._process_chunks(chunk_ids, tier, tier_name)
            
            _current_status.completed_at = datetime.utcnow()
            _current_status.running = False
            _current_status.pending_chunks = 0
            
            duration = (_current_status.completed_at - _current_status.started_at).total_seconds()
            
            # Use _log helper for completion message
            _log("[embedding_job] " + "=" * 50)
            _log("[embedding_job] EMBEDDING JOB COMPLETE")
            _log(f"[embedding_job] Duration: {duration:.1f}s")
            _log(f"[embedding_job] Embedded: {_current_status.embedded_chunks}")
            _log(f"[embedding_job] Skipped: {_current_status.skipped_chunks}")
            _log(f"[embedding_job] Failed: {_current_status.failed_chunks}")
            _log("[embedding_job] " + "=" * 50)
            
        except Exception as e:
            import traceback
            error_tb = traceback.format_exc()
            logger.exception(f"[embedding_job] Fatal error: {e}")
            print(f"[embedding_job] FATAL ERROR: {e}")
            print(f"[embedding_job] Traceback:\n{error_tb}")
            _current_status.last_error = f"{e}\n{error_tb}"
            _current_status.running = False
            _current_status.completed_at = datetime.utcnow()
        
        finally:
            # v1.2: DB session is now closed before batch processing starts
            # This finally block handles the case where we failed before closing it
            try:
                if db is not None:
                    db.close()
            except Exception:
                pass  # db might already be closed or not exist
            
            _current_status.save_to_file()
            print(f"[embedding_job] Job finished. Status saved to {STATUS_FILE}")
            print(f"[embedding_job] Final stats: embedded={_current_status.embedded_chunks}, failed={_current_status.failed_chunks}")
        
        return _current_status
    
    def _get_chunks_by_priority(self, db: Session) -> Dict[EmbeddingPriority, List]:
        """Get chunks needing embedding, grouped by priority. Delegates to _embedding_priority."""
        return get_chunks_by_priority(db, self.scan_id)
    
    def _process_chunks(
        self,
        chunk_ids: List[int],
        tier: EmbeddingPriority,
        tier_name: str,
    ):
        """
        Process a list of chunks in batches.
        
        v1.2: Changed to session-per-batch pattern to prevent poisoned session cascade.
              Now takes chunk_ids instead of chunk objects, creates fresh session per batch.
        """
        total_batches = (len(chunk_ids) + self.batch_size - 1) // self.batch_size
        _current_status.total_batches = total_batches
        
        for batch_idx in range(0, len(chunk_ids), self.batch_size):
            if self._stop_requested:
                break
            
            batch_ids = chunk_ids[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            _current_status.current_batch = batch_num
            
            done_in_tier = _current_status.tier_progress[tier_name]["done"]
            total_in_tier = _current_status.tier_progress[tier_name]["total"]
            
            logger.info(
                f"[embedding_job] {tier_name} {done_in_tier + len(batch_ids)}/{total_in_tier} "
                f"(batch {batch_num}/{total_batches})"
            )
            
            # v1.2: Fresh session per batch - prevents poisoned session cascade
            try:
                embedded_count = self._embed_batch_with_retry(batch_ids)
                _current_status.tier_progress[tier_name]["done"] += embedded_count
                _current_status.pending_chunks -= embedded_count
                
            except Exception as e:
                logger.error(f"[embedding_job] Batch {batch_num} failed after retries: {e}")
                print(f"[embedding_job] Batch {batch_num} FAILED: {e}")
                _current_status.failed_chunks += len(batch_ids)
                _current_status.tier_progress[tier_name]["failed"] += len(batch_ids)
                _current_status.pending_chunks -= len(batch_ids)
                _current_status.last_error = str(e)
            
            _current_status.save_to_file()
            
            # Rate limiting
            if self.rate_limit_delay > 0 and batch_idx + self.batch_size < len(chunk_ids):
                time.sleep(self.rate_limit_delay)
    
    def _embed_batch(self, db: Session, chunks: List) -> int:
        """Embed a batch of chunks. Delegates to _embedding_batch module."""
        from app.rag.jobs._embedding_batch import embed_batch
        return embed_batch(db, chunks, _current_status)

    def _embed_batch_with_retry(self, chunk_ids: List[int]) -> int:
        """Embed batch with SQLite lock retry. Delegates to _embedding_batch module."""
        from app.rag.jobs._embedding_batch import embed_batch_with_retry
        return embed_batch_with_retry(chunk_ids, self.db_session_factory, _current_status)


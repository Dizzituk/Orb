# FILE: app/rag/jobs/embedding_job.py
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

import os
import re
import json
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


# =============================================================================
# SQLITE RETRY CONFIGURATION (v1.2)
# =============================================================================

SQLITE_LOCK_MAX_RETRIES = 10
SQLITE_LOCK_INITIAL_BACKOFF = 0.25  # seconds
SQLITE_LOCK_MAX_BACKOFF = 8.0  # seconds


def _is_sqlite_lock_error(exc: Exception) -> bool:
    """Check if exception is a SQLite database lock error."""
    error_str = str(exc).lower()
    return "database is locked" in error_str or "database_is_locked" in error_str


# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment flags
EMBEDDING_AUTO_ENABLED = os.getenv("ORB_EMBEDDING_AUTO", "true").lower() == "true"
EMBEDDING_MODEL = os.getenv("ORB_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("ORB_EMBEDDING_BATCH_SIZE", "20"))
EMBEDDING_RATE_LIMIT_DELAY = float(os.getenv("ORB_EMBEDDING_RATE_DELAY", "0.5"))  # seconds between batches

# Status file location - use consistent path with zobie_tools
# IMPORTANT: Must match FULL_ARCHMAP_OUTPUT_DIR in zobie_tools.py
ARCHITECTURE_OUTPUT_DIR = Path(r"D:\Orb\.architecture")
STATUS_FILE = ARCHITECTURE_OUTPUT_DIR / "embedding_status.json"


# =============================================================================
# PRIORITY CLASSIFICATION (v1.1 - refined patterns)
# =============================================================================

class EmbeddingPriority(Enum):
    """Embedding priority tiers."""
    TIER1_CRITICAL = 1   # Entry points, routers, dispatch - semantic search useful fast
    TIER2_HIGH = 2       # Pipeline core - spec gate, overwatcher, weaver
    TIER3_MEDIUM = 3     # Services, models, DB - infrastructure queries
    TIER4_LOW = 4        # Handlers, utils, clients
    TIER5_NORMAL = 5     # Everything else


# Priority patterns (regex on file_path) - order matters, first match wins
PRIORITY_PATTERNS: Dict[EmbeddingPriority, List[re.Pattern]] = {
    # ==========================================================================
    # TIER 1: CRITICAL - Entry points, routing, dispatch
    # Semantic search useful for "how does routing work?" immediately
    # ==========================================================================
    EmbeddingPriority.TIER1_CRITICAL: [
        re.compile(r"main\.py$", re.IGNORECASE),
        re.compile(r"stream_router", re.IGNORECASE),
        re.compile(r"translation_routing", re.IGNORECASE),
        re.compile(r"app[/\\]translation[/\\]intents\.py$", re.IGNORECASE),
        re.compile(r"app[/\\]translation[/\\]tier0_rules\.py$", re.IGNORECASE),
        re.compile(r"app[/\\]translation[/\\]modes\.py$", re.IGNORECASE),
        re.compile(r"app[/\\]llm[/\\]local_tools[/\\]", re.IGNORECASE),  # Tool dispatch is critical
        re.compile(r"app[/\\]llm[/\\]streaming\.py$", re.IGNORECASE),
        re.compile(r"uvicorn|gunicorn|startup", re.IGNORECASE),
    ],
    
    # ==========================================================================
    # TIER 2: HIGH - Pipeline core (spec gate, overwatcher, memory backbone)
    # Answers "how does the pipeline work?" within ~1 minute
    # ==========================================================================
    EmbeddingPriority.TIER2_HIGH: [
        re.compile(r"spec_gate", re.IGNORECASE),
        re.compile(r"overwatcher", re.IGNORECASE),
        re.compile(r"critical_pipeline", re.IGNORECASE),
        re.compile(r"weaver", re.IGNORECASE),
        re.compile(r"app[/\\]rag[/\\]pipeline\.py$", re.IGNORECASE),  # RAG pipeline
        re.compile(r"app[/\\]rag[/\\]answerer\.py$", re.IGNORECASE),  # RAG answerer
        re.compile(r"app[/\\]memory[/\\].*architecture", re.IGNORECASE),  # Architecture models
        re.compile(r"app[/\\]memory[/\\]service\.py$", re.IGNORECASE),  # Memory service
        re.compile(r"job_classifier", re.IGNORECASE),
        re.compile(r"astra_memory", re.IGNORECASE),
    ],
    
    # ==========================================================================
    # TIER 3: MEDIUM - Services, models, schemas, DB infrastructure
    # Answers "what data models exist?" "how is DB structured?"
    # ==========================================================================
    EmbeddingPriority.TIER3_MEDIUM: [
        re.compile(r"models\.py$", re.IGNORECASE),
        re.compile(r"schemas\.py$", re.IGNORECASE),
        re.compile(r"service\.py$", re.IGNORECASE),
        re.compile(r"services[/\\]", re.IGNORECASE),
        re.compile(r"repository", re.IGNORECASE),
        re.compile(r"app[/\\]db\.py$", re.IGNORECASE),
        re.compile(r"database", re.IGNORECASE),
        re.compile(r"app[/\\]embeddings[/\\]", re.IGNORECASE),
    ],
    
    # ==========================================================================
    # TIER 4: LOW - Handlers, utilities, clients
    # Supporting infrastructure
    # ==========================================================================
    EmbeddingPriority.TIER4_LOW: [
        re.compile(r"handler", re.IGNORECASE),
        re.compile(r"client", re.IGNORECASE),
        re.compile(r"util", re.IGNORECASE),
        re.compile(r"helper", re.IGNORECASE),
        re.compile(r"config\.py$", re.IGNORECASE),
    ],
    
    # TIER 5: NORMAL - No patterns, catch-all
}


def classify_chunk_priority(file_path: str) -> EmbeddingPriority:
    """
    Classify a chunk's embedding priority based on file path.
    
    Returns highest matching priority (TIER1 > TIER2 > ... > TIER5).
    First match wins.
    """
    if not file_path:
        return EmbeddingPriority.TIER5_NORMAL
    
    # Normalize path separators
    normalized = file_path.replace("\\", "/")
    
    for priority in [
        EmbeddingPriority.TIER1_CRITICAL,
        EmbeddingPriority.TIER2_HIGH,
        EmbeddingPriority.TIER3_MEDIUM,
        EmbeddingPriority.TIER4_LOW,
    ]:
        for pattern in PRIORITY_PATTERNS.get(priority, []):
            if pattern.search(normalized):
                return priority
    
    return EmbeddingPriority.TIER5_NORMAL


def compute_content_hash(chunk) -> str:
    """
    Compute content hash for change detection.
    
    Hash is based on: chunk_name + signature + docstring
    """
    content = f"{chunk.chunk_name or ''}|{chunk.signature or ''}|{chunk.docstring or ''}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


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


def get_embedding_status() -> EmbeddingJobStatus:
    """Get current embedding job status."""
    # Try to load from file if not running (for persistence across restarts)
    if not _current_status.running:
        loaded = EmbeddingJobStatus.load_from_file()
        if loaded:
            return loaded
    return _current_status


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
        """
        Get chunks needing embedding, grouped by priority.
        
        INCREMENTAL: Only returns chunks that:
        - Have no embedding (embedded=False)
        - OR have stale embedding (content_hash != embedded_content_hash)
        
        Never re-embeds unchanged chunks regardless of tier.
        """
        from app.rag.models import ArchCodeChunk
        from sqlalchemy import func
        
        # First, count total chunks in DB for diagnostics
        total_in_db = db.query(func.count(ArchCodeChunk.id)).scalar() or 0
        logger.info(f"[embedding_job] Total ArchCodeChunk rows in DB: {total_in_db}")
        print(f"[embedding_job] Total ArchCodeChunk rows in DB: {total_in_db}")
        
        # Query all chunks (optionally filtered by scan_id)
        query = db.query(ArchCodeChunk)
        if self.scan_id:
            logger.info(f"[embedding_job] Filtering by scan_id={self.scan_id}")
            print(f"[embedding_job] Filtering by scan_id={self.scan_id}")
            query = query.filter(ArchCodeChunk.scan_id == self.scan_id)
        else:
            logger.info("[embedding_job] No scan_id filter - querying ALL chunks")
            print("[embedding_job] No scan_id filter - querying ALL chunks")
        
        all_chunks = query.all()
        logger.info(f"[embedding_job] Chunks after filter: {len(all_chunks)}")
        print(f"[embedding_job] Chunks after filter: {len(all_chunks)}")
        
        # First pass: compute content hashes for chunks that don't have them
        hash_updates = 0
        for chunk in all_chunks:
            if not chunk.content_hash:
                chunk.content_hash = compute_content_hash(chunk)
                hash_updates += 1
        
        if hash_updates > 0:
            db.commit()
            logger.info(f"[embedding_job] Computed {hash_updates} missing content hashes")
        
        # Group by priority, filtering for those needing embedding
        result: Dict[EmbeddingPriority, List] = {p: [] for p in EmbeddingPriority}
        
        already_embedded = 0
        needs_reembed = 0
        
        for chunk in all_chunks:
            # Check if needs embedding (INCREMENTAL logic)
            if chunk.embedded:
                # Already embedded - check if content changed
                if chunk.content_hash and chunk.embedded_content_hash:
                    if chunk.content_hash == chunk.embedded_content_hash:
                        # Up to date, skip
                        already_embedded += 1
                        continue
                    else:
                        # Content changed, needs re-embed
                        needs_reembed += 1
                else:
                    # No hash comparison possible, skip
                    already_embedded += 1
                    continue
            
            # Needs embedding - classify priority
            priority = classify_chunk_priority(chunk.file_path or "")
            result[priority].append(chunk)
        
        logger.info(f"[embedding_job] Already embedded (unchanged): {already_embedded}")
        logger.info(f"[embedding_job] Needs re-embed (changed): {needs_reembed}")
        
        return result
    
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
        """
        Embed a batch of chunks.
        
        Returns: number of successfully embedded chunks
        """
        print(f"[embedding_job] _embed_batch: processing {len(chunks)} chunks")
        logger.info(f"[embedding_job] _embed_batch: processing {len(chunks)} chunks")
        
        try:
            from app.llm.clients import get_embeddings
            print("[embedding_job] Imported get_embeddings successfully")
        except ImportError as e:
            print(f"[embedding_job] Failed to import get_embeddings: {e}")
            logger.error(f"[embedding_job] Failed to import get_embeddings: {e}")
            raise
        
        try:
            from app.embeddings.service import store_embedding
            print("[embedding_job] Imported store_embedding successfully")
        except ImportError as e:
            print(f"[embedding_job] Failed to import store_embedding: {e}")
            logger.error(f"[embedding_job] Failed to import store_embedding: {e}")
            raise
        
        # Build texts for embedding
        texts = []
        for chunk in chunks:
            # Build embeddable text: name + signature + docstring
            text_parts = []
            if chunk.chunk_name:
                text_parts.append(f"{chunk.chunk_type}: {chunk.chunk_name}")
            if chunk.signature:
                text_parts.append(f"Signature: {chunk.signature}")
            if chunk.docstring:
                text_parts.append(f"Description: {chunk.docstring[:500]}")
            if chunk.file_path:
                text_parts.append(f"File: {chunk.file_path}")
            
            texts.append("\n".join(text_parts))
        
        # Call embedding API
        print(f"[embedding_job] Calling OpenAI embeddings API for {len(texts)} texts...")
        logger.info(f"[embedding_job] Calling get_embeddings API for {len(texts)} texts...")
        try:
            vectors = get_embeddings(texts, model=EMBEDDING_MODEL)
            print(f"[embedding_job] SUCCESS: Got {len(vectors)} vectors from API")
            logger.info(f"[embedding_job] Got {len(vectors)} vectors from API")
        except Exception as e:
            print(f"[embedding_job] API call FAILED: {e}")
            logger.error(f"[embedding_job] get_embeddings API call failed: {e}")
            raise
        
        # Store embeddings
        print(f"[embedding_job] Storing {len(vectors)} embeddings in DB...")
        now = datetime.utcnow()
        success_count = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Store in embeddings table
                store_embedding(
                    db=db,
                    project_id=0,  # Architecture embeddings use project_id=0
                    source_type="arch_code_chunk",
                    source_id=chunk.id,
                    content=texts[i],
                    embedding=vectors[i],
                )
                
                # Update chunk status
                chunk.embedded = True
                chunk.embedding_model = EMBEDDING_MODEL
                chunk.embedded_at = now
                chunk.embedded_content_hash = chunk.content_hash
                
                _current_status.embedded_chunks += 1
                success_count += 1
                
            except Exception as e:
                print(f"[embedding_job] Failed to store chunk {chunk.id}: {e}")
                logger.error(f"[embedding_job] Failed to store embedding for chunk {chunk.id}: {e}")
                _current_status.failed_chunks += 1
        
        db.commit()
        print(f"[embedding_job] Batch complete: {success_count}/{len(chunks)} stored, total embedded: {_current_status.embedded_chunks}")
        return success_count
    
    def _embed_batch_with_retry(self, chunk_ids: List[int]) -> int:
        """
        Embed a batch of chunks with SQLite lock retry and fresh session per batch.
        
        v1.2: Added to fix "database is locked" cascade errors.
        
        Key behaviors:
        - Creates fresh session for this batch (prevents poisoned session cascade)
        - Retries with exponential backoff on SQLite lock errors
        - Properly closes session in all cases
        
        Returns: number of successfully embedded chunks
        """
        from sqlalchemy.exc import OperationalError
        from app.rag.models import ArchCodeChunk
        
        backoff = SQLITE_LOCK_INITIAL_BACKOFF
        last_error = None
        
        for attempt in range(1, SQLITE_LOCK_MAX_RETRIES + 1):
            db = None
            try:
                # Fresh session for this batch
                db = self.db_session_factory()
                
                # Re-fetch chunks by ID (they're detached from previous session)
                chunks = db.query(ArchCodeChunk).filter(
                    ArchCodeChunk.id.in_(chunk_ids)
                ).all()
                
                if len(chunks) != len(chunk_ids):
                    logger.warning(
                        f"[embedding_job] Fetched {len(chunks)}/{len(chunk_ids)} chunks "
                        f"(some may have been deleted)"
                    )
                
                if not chunks:
                    logger.warning("[embedding_job] No chunks found for batch, skipping")
                    return 0
                
                # Delegate to existing _embed_batch logic
                result = self._embed_batch(db, chunks)
                return result
                
            except OperationalError as e:
                if _is_sqlite_lock_error(e):
                    last_error = e
                    logger.warning(
                        f"[embedding_job] SQLite lock on attempt {attempt}/{SQLITE_LOCK_MAX_RETRIES}, "
                        f"backing off {backoff:.2f}s"
                    )
                    print(
                        f"[embedding_job] SQLite lock (attempt {attempt}), "
                        f"retrying in {backoff:.2f}s..."
                    )
                    
                    # Rollback and close current session before retry
                    if db:
                        try:
                            db.rollback()
                        except Exception:
                            pass
                        try:
                            db.close()
                        except Exception:
                            pass
                        db = None
                    
                    time.sleep(backoff)
                    backoff = min(backoff * 2, SQLITE_LOCK_MAX_BACKOFF)
                else:
                    # Non-lock OperationalError, re-raise
                    raise
                    
            except Exception as e:
                # Non-retryable error
                logger.error(f"[embedding_job] Non-retryable error in batch: {e}")
                raise
                
            finally:
                # Always close session if it exists
                if db:
                    try:
                        db.close()
                    except Exception:
                        pass
        
        # Exhausted retries
        error_msg = f"SQLite lock persisted after {SQLITE_LOCK_MAX_RETRIES} retries: {last_error}"
        logger.error(f"[embedding_job] {error_msg}")
        print(f"[embedding_job] {error_msg}")
        _current_status.last_error = error_msg
        raise RuntimeError(error_msg)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def queue_embedding_job(
    db_session_factory: Callable[[], Session],
    scan_id: Optional[int] = None,
) -> bool:
    """
    Queue an embedding job to run in background.
    
    Args:
        db_session_factory: Callable returning new DB session
        scan_id: Optional scan_id to scope embeddings (None = all pending chunks)
        
    Returns:
        True if job was queued, False if already running or disabled
    """
    global _current_status
    
    print(f"[embedding_job] queue_embedding_job called with scan_id={scan_id}")
    logger.info(f"[embedding_job] queue_embedding_job called with scan_id={scan_id}")
    
    if not EMBEDDING_AUTO_ENABLED:
        print("[embedding_job] Auto embedding disabled (ORB_EMBEDDING_AUTO=false)")
        logger.info("[embedding_job] Auto embedding disabled (ORB_EMBEDDING_AUTO=false)")
        return False
    
    # v1.2: Check for stale running status from crashed previous run
    # If status file says running but started > 30 min ago, assume crashed
    if _current_status.running:
        if _current_status.started_at:
            elapsed = (datetime.utcnow() - _current_status.started_at).total_seconds()
            if elapsed > 1800:  # 30 minutes
                print(f"[embedding_job] Stale running status detected (started {elapsed/60:.1f}min ago), resetting")
                logger.warning(f"[embedding_job] Stale running status detected (started {elapsed/60:.1f}min ago), resetting")
                _current_status.running = False
                _current_status.last_error = "Previous job timed out or crashed"
                _current_status.save_to_file()
            else:
                print(f"[embedding_job] Job already running (started {elapsed:.1f}s ago), skipping")
                logger.info(f"[embedding_job] Job already running (started {elapsed:.1f}s ago), not queueing another")
                return False
        else:
            print("[embedding_job] Job already running, skipping")
            logger.info("[embedding_job] Job already running, not queueing another")
            return False
    
    print(f"[embedding_job] Creating EmbeddingJob with scan_id={scan_id}")
    logger.info(f"[embedding_job] Queueing embedding job (scan_id={scan_id})")
    job = EmbeddingJob(db_session_factory, scan_id=scan_id)
    result = job.run_async()
    print(f"[embedding_job] run_async() returned: {result}")
    return result


def run_embedding_job_sync(
    db_session_factory: Callable[[], Session],
    scan_id: Optional[int] = None,
) -> EmbeddingJobStatus:
    """
    Run embedding job synchronously (for manual trigger).
    
    Returns:
        Final job status
    """
    job = EmbeddingJob(db_session_factory, scan_id=scan_id)
    return job.run_sync()


def get_embedding_stats(db: Session) -> Dict[str, Any]:
    """Get embedding statistics from DB."""
    from app.rag.models import ArchCodeChunk
    
    total = db.query(func.count(ArchCodeChunk.id)).scalar() or 0
    embedded = db.query(func.count(ArchCodeChunk.id)).filter(
        ArchCodeChunk.embedded == True
    ).scalar() or 0
    
    # Get status (from file if not running)
    status = get_embedding_status()
    
    return {
        "total_chunks": total,
        "embedded_chunks": embedded,
        "pending_chunks": total - embedded,
        "embedding_pct": round(100 * embedded / max(total, 1), 1),
        "job_running": status.running,
        "current_tier": status.current_tier,
        "last_run": status.completed_at.isoformat() if status.completed_at else None,
        "last_error": status.last_error,
        "tier_counts": status.tier_counts,
    }


def format_embedding_status_report(db: Session) -> str:
    """
    Format a human-readable embedding status report.
    
    Returns text suitable for streaming to user.
    """
    stats = get_embedding_stats(db)
    status = get_embedding_status()
    
    lines = [
        "📊 **Embedding Status**",
        "",
        f"**Total chunks:** {stats['total_chunks']}",
        f"**Embedded:** {stats['embedded_chunks']} ({stats['embedding_pct']}%)",
        f"**Pending:** {stats['pending_chunks']}",
        "",
    ]
    
    if status.running:
        lines.extend([
            f"🔄 **Currently running:** {status.current_tier}",
            f"   Batch {status.current_batch}/{status.total_batches}",
            "",
        ])
    elif status.completed_at:
        duration = ""
        if status.started_at:
            dur_sec = (status.completed_at - status.started_at).total_seconds()
            duration = f" ({dur_sec:.1f}s)"
        lines.extend([
            f"✅ **Last run:** {status.completed_at.strftime('%Y-%m-%d %H:%M:%S')}{duration}",
            f"   Embedded: {status.embedded_chunks} | Failed: {status.failed_chunks}",
            "",
        ])
    
    if status.tier_counts:
        lines.append("**Queue breakdown:**")
        for tier, count in status.tier_counts.items():
            progress = status.tier_progress.get(tier, {})
            done = progress.get("done", 0)
            if count > 0:
                lines.append(f"   • {tier}: {done}/{count}")
        lines.append("")
    
    if status.last_error:
        lines.extend([
            f"⚠️ **Last error:** {status.last_error}",
            "",
        ])
    
    lines.append(f"**Model:** {status.model_used}")
    lines.append(f"**Auto-embed:** {'enabled' if EMBEDDING_AUTO_ENABLED else 'disabled'}")
    
    return "\n".join(lines)

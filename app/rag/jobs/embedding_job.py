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
from app.rag.jobs._embedding_job_utils import ARCHITECTURE_OUTPUT_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_RATE_LIMIT_DELAY, SQLITE_LOCK_INITIAL_BACKOFF, SQLITE_LOCK_MAX_BACKOFF, SQLITE_LOCK_MAX_RETRIES, _is_sqlite_lock_error, run_embedding_job_sync
from app.rag.jobs._embedding_job_utils import EMBEDDING_AUTO_ENABLED, EMBEDDING_MODEL, STATUS_FILE, classify_chunk_priority, compute_content_hash, format_embedding_status_report, get_embedding_stats, queue_embedding_job
from app.rag.jobs._embedding_job_utils import EmbeddingPriority, get_embedding_status
from app.rag.jobs._embedding_job_utils import EmbeddingJob

logger = logging.getLogger(__name__)


# =============================================================================
# SQLITE RETRY CONFIGURATION (v1.2)
# =============================================================================


# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment flags

# Status file location - use consistent path with zobie_tools
# IMPORTANT: Must match FULL_ARCHMAP_OUTPUT_DIR in zobie_tools.py


# =============================================================================
# PRIORITY CLASSIFICATION (v1.1 - refined patterns)
# =============================================================================


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


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

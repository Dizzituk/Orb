from __future__ import annotations
import hashlib
import logging
import os
from app.rag.jobs._embedding_job_utils import ARCHITECTURE_OUTPUT_DIR
from app.rag.jobs.embedding_job import EmbeddingJob, EmbeddingJobStatus, EmbeddingPriority, _current_status, get_embedding_status, logger
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Any, Callable, Dict, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_current_status = EmbeddingJobStatus()


EMBEDDING_AUTO_ENABLED = os.getenv("ORB_EMBEDDING_AUTO", "true").lower() == "true"

EMBEDDING_MODEL = os.getenv("ORB_EMBEDDING_MODEL", "text-embedding-3-small")

STATUS_FILE = ARCHITECTURE_OUTPUT_DIR / "embedding_status.json"

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

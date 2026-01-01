# FILE: app/astra_memory/decay_job.py
"""
Periodic Confidence Decay Job for ASTRA Memory System.

Runs on schedule to:
1. Recompute confidence scores as evidence ages
2. Expire very low confidence preferences
3. Clean up stale hot index entries
4. Generate metrics/stats

Spec §7.2: Periodic confidence decay job
"""

from __future__ import annotations

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.astra_memory.preference_models import (
    PreferenceRecord,
    PreferenceEvidence,
    HotIndex,
    MemoryRecordConfidence,
    RecordStatus,
)
from app.astra_memory.confidence_scoring import (
    recompute_preference_confidence,
    batch_recompute_confidence,
)
from app.astra_memory.confidence_config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# DECAY JOB CONFIGURATION
# =============================================================================

class DecayJobConfig:
    """Configuration for decay job."""
    
    # Confidence thresholds
    expire_below_confidence: float = 0.15  # Expire preferences below this
    dispute_threshold: float = 0.3  # Mark as disputed below this
    
    # Batch sizes
    preference_batch_size: int = 100
    hot_index_batch_size: int = 500
    
    # Stale thresholds
    hot_index_stale_days: int = 90  # Remove entries not accessed in X days
    
    # Job limits
    max_preferences_per_run: int = 1000
    max_expirations_per_run: int = 50


DEFAULT_DECAY_CONFIG = DecayJobConfig()


# =============================================================================
# CORE DECAY FUNCTIONS
# =============================================================================

def run_confidence_decay(
    db: Session,
    config: Optional[DecayJobConfig] = None,
) -> Dict[str, Any]:
    """
    Run the full confidence decay job.
    
    Returns:
        Dict with job statistics
    """
    config = config or DEFAULT_DECAY_CONFIG
    start_time = datetime.now(timezone.utc)
    
    logger.info("[decay_job] Starting confidence decay run")
    
    results = {
        "started_at": start_time.isoformat(),
        "preferences_recomputed": 0,
        "preferences_expired": 0,
        "preferences_disputed": 0,
        "hot_index_cleaned": 0,
        "errors": [],
    }
    
    try:
        # 1. Recompute all preference confidences
        recompute_result = _recompute_all_preferences(db, config)
        results["preferences_recomputed"] = recompute_result["recomputed"]
        results["errors"].extend(recompute_result.get("errors", []))
        
        # 2. Expire very low confidence preferences
        expire_result = _expire_low_confidence(db, config)
        results["preferences_expired"] = expire_result["expired"]
        results["preferences_disputed"] = expire_result["disputed"]
        
        # 3. Clean stale hot index entries
        clean_result = _clean_stale_hot_index(db, config)
        results["hot_index_cleaned"] = clean_result["cleaned"]
        
        db.commit()
        
    except Exception as e:
        logger.error(f"[decay_job] Job failed: {e}")
        results["errors"].append(str(e))
        db.rollback()
    
    end_time = datetime.now(timezone.utc)
    results["completed_at"] = end_time.isoformat()
    results["duration_seconds"] = (end_time - start_time).total_seconds()
    
    logger.info(
        f"[decay_job] Complete: recomputed={results['preferences_recomputed']}, "
        f"expired={results['preferences_expired']}, "
        f"disputed={results['preferences_disputed']}, "
        f"cleaned={results['hot_index_cleaned']}, "
        f"duration={results['duration_seconds']:.2f}s"
    )
    
    return results


def _recompute_all_preferences(
    db: Session,
    config: DecayJobConfig,
) -> Dict[str, Any]:
    """Recompute confidence for all active preferences."""
    result = {"recomputed": 0, "errors": []}
    
    # Get all active preferences
    preferences = db.query(PreferenceRecord).filter(
        PreferenceRecord.status == RecordStatus.ACTIVE
    ).limit(config.max_preferences_per_run).all()
    
    for pref in preferences:
        try:
            recompute_preference_confidence(db, pref.id)
            result["recomputed"] += 1
        except Exception as e:
            result["errors"].append(f"Preference {pref.id}: {e}")
    
    logger.debug(f"[decay_job] Recomputed {result['recomputed']} preferences")
    return result


def _expire_low_confidence(
    db: Session,
    config: DecayJobConfig,
) -> Dict[str, Any]:
    """Expire or mark disputed preferences with very low confidence."""
    result = {"expired": 0, "disputed": 0}
    
    # Find low confidence preferences
    low_confidence = db.query(PreferenceRecord).filter(
        PreferenceRecord.status == RecordStatus.ACTIVE,
        PreferenceRecord.confidence < config.dispute_threshold,
    ).limit(config.max_expirations_per_run).all()
    
    for pref in low_confidence:
        if pref.confidence < config.expire_below_confidence:
            # Very low - expire it
            pref.status = RecordStatus.EXPIRED
            result["expired"] += 1
            logger.info(f"[decay_job] Expired preference {pref.namespace}.{pref.key} (confidence={pref.confidence:.2f})")
        else:
            # Low but not critical - mark disputed
            pref.status = RecordStatus.DISPUTED
            result["disputed"] += 1
            logger.debug(f"[decay_job] Disputed preference {pref.namespace}.{pref.key} (confidence={pref.confidence:.2f})")
    
    return result


def _clean_stale_hot_index(
    db: Session,
    config: DecayJobConfig,
) -> Dict[str, Any]:
    """Remove stale hot index entries."""
    result = {"cleaned": 0}
    
    stale_cutoff = datetime.now(timezone.utc) - timedelta(days=config.hot_index_stale_days)
    
    # Find stale entries (not updated recently)
    stale_entries = db.query(HotIndex).filter(
        HotIndex.updated_at < stale_cutoff,
    ).limit(config.hot_index_batch_size).all()
    
    for entry in stale_entries:
        db.delete(entry)
        result["cleaned"] += 1
    
    if result["cleaned"] > 0:
        logger.info(f"[decay_job] Cleaned {result['cleaned']} stale hot index entries")
    
    return result


# =============================================================================
# DECAY METRICS
# =============================================================================

def get_decay_metrics(db: Session) -> Dict[str, Any]:
    """Get current decay/confidence metrics."""
    
    # Preference stats by status
    status_counts = dict(
        db.query(
            PreferenceRecord.status,
            func.count(PreferenceRecord.id)
        ).group_by(PreferenceRecord.status).all()
    )
    
    # Confidence distribution
    confidence_buckets = {
        "very_low": 0,   # < 0.2
        "low": 0,        # 0.2 - 0.4
        "medium": 0,     # 0.4 - 0.6
        "high": 0,       # 0.6 - 0.8
        "very_high": 0,  # > 0.8
    }
    
    preferences = db.query(PreferenceRecord).filter(
        PreferenceRecord.status == RecordStatus.ACTIVE
    ).all()
    
    for pref in preferences:
        conf = pref.confidence or 0
        if conf < 0.2:
            confidence_buckets["very_low"] += 1
        elif conf < 0.4:
            confidence_buckets["low"] += 1
        elif conf < 0.6:
            confidence_buckets["medium"] += 1
        elif conf < 0.8:
            confidence_buckets["high"] += 1
        else:
            confidence_buckets["very_high"] += 1
    
    # Evidence stats
    evidence_count = db.query(func.count(PreferenceEvidence.id)).scalar() or 0
    
    # Hot index stats
    hot_index_count = db.query(func.count(HotIndex.id)).scalar() or 0
    
    return {
        "preferences_by_status": {str(k): v for k, v in status_counts.items()},
        "confidence_distribution": confidence_buckets,
        "total_evidence_records": evidence_count,
        "hot_index_entries": hot_index_count,
    }


# =============================================================================
# SCHEDULED JOB WRAPPER
# =============================================================================

class DecayJobScheduler:
    """
    Manages scheduled execution of decay job.
    
    Can be started as a background task or triggered manually.
    """
    
    def __init__(
        self,
        interval_hours: float = 24,
        config: Optional[DecayJobConfig] = None,
    ):
        self.interval_hours = interval_hours
        self.config = config or DEFAULT_DECAY_CONFIG
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[Dict[str, Any]] = None
    
    async def start(self):
        """Start the scheduled job."""
        if self._running:
            logger.warning("[decay_job] Scheduler already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[decay_job] Scheduler started (interval={self.interval_hours}h)")
    
    async def stop(self):
        """Stop the scheduled job."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[decay_job] Scheduler stopped")
    
    async def _run_loop(self):
        """Main scheduling loop."""
        while self._running:
            try:
                # Run the job
                await self._execute_job()
                
                # Wait for next interval
                await asyncio.sleep(self.interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[decay_job] Scheduler error: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes
    
    async def _execute_job(self):
        """Execute the decay job."""
        from app.db import SessionLocal
        
        db = SessionLocal()
        try:
            self._last_result = run_confidence_decay(db, self.config)
            self._last_run = datetime.now(timezone.utc)
        finally:
            db.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "interval_hours": self.interval_hours,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_result": self._last_result,
        }
    
    def run_now(self) -> Dict[str, Any]:
        """Run job immediately (synchronous)."""
        from app.db import SessionLocal
        
        db = SessionLocal()
        try:
            result = run_confidence_decay(db, self.config)
            self._last_run = datetime.now(timezone.utc)
            self._last_result = result
            return result
        finally:
            db.close()


# Global scheduler instance
_scheduler: Optional[DecayJobScheduler] = None


def get_scheduler() -> DecayJobScheduler:
    """Get or create the global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = DecayJobScheduler()
    return _scheduler


def run_decay_now() -> Dict[str, Any]:
    """Convenience function to run decay job immediately."""
    return get_scheduler().run_now()


__all__ = [
    "DecayJobConfig",
    "DEFAULT_DECAY_CONFIG",
    "run_confidence_decay",
    "get_decay_metrics",
    "DecayJobScheduler",
    "get_scheduler",
    "run_decay_now",
]

# FILE: app/experience/calibration.py
"""
Confidence Calibration — Phase 8 of the Unified Memory System.

Tracks whether injected patterns actually helped, adjusts confidence
scores based on real outcomes, implements monthly decay, and generates
calibration reports.

Section 8 of the Unified Memory System v3.0 spec.

Usage:
    # After a job completes, record which patterns helped:
    calibrate_from_job(db, job_id, job_outcome, injected_pattern_ids)

    # Monthly maintenance:
    run_monthly_decay(db)

    # Generate calibration report (every 10 jobs):
    report = generate_calibration_report(db)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func

from .models import ExperiencePattern
from .experience_store import (
    validate_pattern,
    contradict_pattern,
    record_usefulness,
    apply_monthly_decay,
    CONFIDENCE_SKIP_THRESHOLD,
    CONFIDENCE_DEMOTE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# JOB OUTCOME CALIBRATION
# =============================================================================

def calibrate_from_job(
    db: Session,
    job_id: str,
    job_outcome: str,
    injected_pattern_ids: Optional[List[int]] = None,
) -> Dict[str, int]:
    """
    After a job completes, update confidence scores for patterns
    that were injected during the run.

    Args:
        db: SQLAlchemy session
        job_id: The completed job ID
        job_outcome: "pass", "fail", or "partial"
        injected_pattern_ids: IDs of patterns that were injected
            (from retrieval tracking)

    Returns:
        Stats: {"boosted": N, "penalised": N}
    """
    stats = {"boosted": 0, "penalised": 0}

    if not injected_pattern_ids:
        # No patterns were injected — nothing to calibrate
        # But we can still learn from the journal
        injected_pattern_ids = _get_injected_patterns_for_job(db, job_id)

    if not injected_pattern_ids:
        return stats

    was_successful = job_outcome in ("pass", "complete")

    for pid in injected_pattern_ids:
        record_usefulness(db, pid, was_useful=was_successful)
        if was_successful:
            stats["boosted"] += 1
        else:
            stats["penalised"] += 1

    db.commit()

    logger.info(
        f"[calibration] Job {job_id} ({job_outcome}): "
        f"boosted {stats['boosted']}, penalised {stats['penalised']} patterns"
    )

    return stats


def _get_injected_patterns_for_job(
    db: Session,
    job_id: str,
) -> List[int]:
    """
    Get pattern IDs that were injected during a job.

    Uses the last_injected_at timestamp — patterns injected since
    the job started are considered part of this job.
    """
    # Get patterns that were recently injected (within last 2 hours)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
    patterns = db.query(ExperiencePattern.id).filter(
        ExperiencePattern.last_injected_at >= cutoff,
        ExperiencePattern.times_injected > 0,
    ).all()

    return [p.id for p in patterns]


# =============================================================================
# MONTHLY DECAY
# =============================================================================

def run_monthly_decay(db: Session) -> Dict[str, Any]:
    """
    Apply monthly confidence decay to all patterns.

    Unused patterns slowly lose confidence, ensuring the system
    doesn't accumulate stale knowledge. High-confidence patterns
    that are regularly validated resist decay.

    Should be called monthly (or at system startup if >30 days since last).
    """
    stats = apply_monthly_decay(db)

    # Also generate a snapshot of pattern health
    health = _assess_pattern_health(db)
    stats.update(health)

    logger.info(f"[calibration] Monthly decay: {stats}")
    return stats


def _assess_pattern_health(db: Session) -> Dict[str, int]:
    """Quick health check of the pattern database."""
    total = db.query(func.count(ExperiencePattern.id)).scalar() or 0
    high_conf = db.query(func.count(ExperiencePattern.id)).filter(
        ExperiencePattern.confidence >= 0.8
    ).scalar() or 0
    medium_conf = db.query(func.count(ExperiencePattern.id)).filter(
        ExperiencePattern.confidence >= 0.5,
        ExperiencePattern.confidence < 0.8,
    ).scalar() or 0
    low_conf = db.query(func.count(ExperiencePattern.id)).filter(
        ExperiencePattern.confidence >= CONFIDENCE_SKIP_THRESHOLD,
        ExperiencePattern.confidence < 0.5,
    ).scalar() or 0
    dead = db.query(func.count(ExperiencePattern.id)).filter(
        ExperiencePattern.confidence < CONFIDENCE_SKIP_THRESHOLD
    ).scalar() or 0

    return {
        "total": total,
        "high_confidence": high_conf,
        "medium_confidence": medium_conf,
        "low_confidence": low_conf,
        "dead_patterns": dead,
    }


# =============================================================================
# CALIBRATION REPORT
# =============================================================================

def generate_calibration_report(db: Session) -> Dict[str, Any]:
    """
    Generate a comprehensive calibration report.

    Intended to run every 10 jobs to track whether the memory system
    is improving pipeline performance.

    Returns a dict with:
    - pattern_health: Distribution of confidence levels
    - injection_stats: How often patterns are injected and useful
    - category_performance: Which categories produce the most useful patterns
    - recommendations: Actions to take
    """
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pattern_health": _assess_pattern_health(db),
        "injection_stats": _injection_statistics(db),
        "category_performance": _category_performance(db),
        "recommendations": [],
    }

    # Generate recommendations
    recs = []
    stats = report["injection_stats"]

    if stats["total_injected"] > 0:
        useful_rate = stats["total_useful"] / stats["total_injected"]
        if useful_rate < 0.5:
            recs.append(
                f"Low utility rate ({useful_rate:.0%}). "
                f"Consider tightening retrieval filters."
            )
        elif useful_rate > 0.8:
            recs.append(
                f"High utility rate ({useful_rate:.0%}). "
                f"Memory system is performing well."
            )

    health = report["pattern_health"]
    if health["dead_patterns"] > health["total"] * 0.3:
        recs.append(
            f"{health['dead_patterns']} dead patterns "
            f"({health['dead_patterns']/max(health['total'],1):.0%}). "
            f"Consider archiving old patterns."
        )

    if health["total"] == 0:
        recs.append("No patterns yet. Run more jobs to build the experience database.")

    report["recommendations"] = recs

    logger.info(f"[calibration] Report generated: {len(recs)} recommendations")
    return report


def _injection_statistics(db: Session) -> Dict[str, Any]:
    """Statistics about pattern injection and utility."""
    total_injected = db.query(
        func.sum(ExperiencePattern.times_injected)
    ).scalar() or 0

    total_useful = db.query(
        func.sum(ExperiencePattern.times_useful)
    ).scalar() or 0

    never_injected = db.query(func.count(ExperiencePattern.id)).filter(
        ExperiencePattern.times_injected == 0,
        ExperiencePattern.confidence >= CONFIDENCE_SKIP_THRESHOLD,
    ).scalar() or 0

    return {
        "total_injected": int(total_injected),
        "total_useful": int(total_useful),
        "utility_rate": (
            round(total_useful / total_injected, 3)
            if total_injected > 0 else 0.0
        ),
        "never_injected": never_injected,
    }


def _category_performance(db: Session) -> Dict[str, Dict[str, Any]]:
    """Performance breakdown by pattern category."""
    categories = db.query(
        ExperiencePattern.category,
        func.count(ExperiencePattern.id),
        func.avg(ExperiencePattern.confidence),
        func.sum(ExperiencePattern.times_injected),
        func.sum(ExperiencePattern.times_useful),
    ).group_by(ExperiencePattern.category).all()

    result = {}
    for cat, count, avg_conf, total_inj, total_use in categories:
        result[cat] = {
            "count": count,
            "avg_confidence": round(float(avg_conf or 0), 3),
            "total_injected": int(total_inj or 0),
            "total_useful": int(total_use or 0),
            "utility_rate": (
                round(float(total_use or 0) / float(total_inj), 3)
                if total_inj else 0.0
            ),
        }

    return result

# FILE: app/astra_memory/router.py
"""
ASTRA Memory API Router

Provides endpoints for:
- Triggering hot index population
- Querying index stats
- Managing preferences via API
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/astra-memory",
    tags=["astra-memory"],
    dependencies=[Depends(require_auth)],
)


# =============================================================================
# SCHEMAS
# =============================================================================

class IndexStats(BaseModel):
    total: int
    by_type: Dict[str, int]


class IndexResult(BaseModel):
    projects: int
    notes: int
    messages: int
    jobs: int
    global_prefs: int
    total: int


class PreferenceCreate(BaseModel):
    key: str
    value: Any
    strength: str = "default"  # soft, default, hard_rule
    applies_to: Optional[str] = None


class PreferenceOut(BaseModel):
    preference_key: str
    preference_value: Any
    strength: str
    confidence: float
    applies_to: Optional[str]
    status: str


class RetrievalRequest(BaseModel):
    message: str
    depth: Optional[str] = None  # D0, D1, D2, D3, D4


class RetrievalResponse(BaseModel):
    depth: str
    records_retrieved: int
    token_estimate: int
    preferences_applied: List[str]
    records: List[Dict[str, Any]]


# =============================================================================
# INDEX ENDPOINTS
# =============================================================================

@router.post("/index", response_model=IndexResult)
def trigger_indexing(db: Session = Depends(get_db)):
    """
    Trigger hot index population.
    
    Indexes projects, notes, messages, and jobs into the hot index
    for fast retrieval.
    """
    from app.astra_memory.indexer import run_full_index
    
    try:
        results = run_full_index(db, cleanup_first=True)
        return IndexResult(
            projects=results.get("projects", 0),
            notes=results.get("notes", 0),
            messages=results.get("messages", 0),
            jobs=results.get("jobs", 0),
            global_prefs=results.get("global_prefs", 0),
            total=sum(results.values()),
        )
    except Exception as e:
        logger.error(f"[astra-memory] Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/stats", response_model=IndexStats)
def get_index_stats(db: Session = Depends(get_db)):
    """Get hot index statistics."""
    from app.astra_memory.indexer import get_index_stats as _get_stats
    
    stats = _get_stats(db)
    return IndexStats(
        total=stats.get("total", 0),
        by_type=stats.get("by_type", {}),
    )


# =============================================================================
# PREFERENCE ENDPOINTS
# =============================================================================

@router.post("/preferences", response_model=PreferenceOut)
def create_preference(
    data: PreferenceCreate,
    db: Session = Depends(get_db),
):
    """Create a new preference."""
    from app.astra_memory.preference_service import create_preference, create_hard_rule
    from app.astra_memory.preference_models import PreferenceStrength
    
    try:
        if data.strength == "hard_rule":
            pref = create_hard_rule(
                db=db,
                preference_key=data.key,
                preference_value=data.value,
                applies_to=data.applies_to,
            )
        else:
            strength = PreferenceStrength(data.strength) if data.strength else PreferenceStrength.DEFAULT
            pref = create_preference(
                db=db,
                preference_key=data.key,
                preference_value=data.value,
                strength=strength,
                applies_to=data.applies_to,
            )
        
        return PreferenceOut(
            preference_key=pref.preference_key,
            preference_value=pref.preference_value,
            strength=pref.strength.value,
            confidence=pref.confidence,
            applies_to=pref.applies_to,
            status=pref.status.value,
        )
    except Exception as e:
        logger.error(f"[astra-memory] Create preference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preferences", response_model=List[PreferenceOut])
def list_preferences(
    component: Optional[str] = Query(None, description="Filter by component"),
    db: Session = Depends(get_db),
):
    """List active preferences."""
    from app.astra_memory.preference_service import get_preferences_for_component
    from app.astra_memory.preference_models import PreferenceRecord, RecordStatus
    
    if component:
        prefs = get_preferences_for_component(db, component)
    else:
        prefs = db.query(PreferenceRecord).filter(
            PreferenceRecord.status == RecordStatus.ACTIVE
        ).all()
    
    return [
        PreferenceOut(
            preference_key=p.preference_key,
            preference_value=p.preference_value,
            strength=p.strength.value,
            confidence=p.confidence,
            applies_to=p.applies_to,
            status=p.status.value,
        )
        for p in prefs
    ]


@router.get("/preferences/{key}", response_model=PreferenceOut)
def get_preference(key: str, db: Session = Depends(get_db)):
    """Get a specific preference by key."""
    from app.astra_memory.preference_service import get_preference
    
    pref = get_preference(db, key)
    if not pref:
        raise HTTPException(status_code=404, detail="Preference not found")
    
    return PreferenceOut(
        preference_key=pref.preference_key,
        preference_value=pref.preference_value,
        strength=pref.strength.value,
        confidence=pref.confidence,
        applies_to=pref.applies_to,
        status=pref.status.value,
    )


@router.delete("/preferences/{key}")
def delete_preference(key: str, db: Session = Depends(get_db)):
    """Expire (soft-delete) a preference."""
    from app.astra_memory.preference_service import expire_preference
    
    pref = expire_preference(db, key)
    if not pref:
        raise HTTPException(status_code=404, detail="Preference not found")
    
    return {"status": "expired", "key": key}


# =============================================================================
# RETRIEVAL ENDPOINTS
# =============================================================================

@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve_memory(
    request: RetrievalRequest,
    db: Session = Depends(get_db),
):
    """
    Retrieve memory context for a message.
    
    Useful for testing retrieval without making an LLM call.
    """
    from app.astra_memory import retrieve_for_query, IntentDepth
    
    depth_override = None
    if request.depth:
        try:
            depth_override = IntentDepth(request.depth)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid depth: {request.depth}")
    
    result = retrieve_for_query(
        db=db,
        user_message=request.message,
        depth_override=depth_override,
    )
    
    return RetrievalResponse(
        depth=result.depth.value,
        records_retrieved=result.records_expanded,
        token_estimate=result.token_estimate,
        preferences_applied=result.preferences_applied,
        records=[
            {
                "type": r.record_type,
                "id": r.record_id,
                "title": r.title,
                "content": r.content[:200],
                "relevance": r.relevance_score,
            }
            for r in result.records
        ],
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Check ASTRA memory system health."""
    from app.astra_memory.indexer import get_index_stats
    from app.crypto import is_encryption_ready
    
    stats = get_index_stats(db)
    
    return {
        "status": "ok",
        "encryption_ready": is_encryption_ready(),
        "hot_index_count": stats.get("total", 0),
        "index_by_type": stats.get("by_type", {}),
    }


# =============================================================================
# SUMMARY PYRAMID ENDPOINTS
# =============================================================================

class PyramidGenerateRequest(BaseModel):
    project_id: Optional[int] = None
    force: bool = False


class PyramidStats(BaseModel):
    total: int
    by_type: Dict[str, int]
    coverage: Dict[str, int]


class PyramidGenerateResult(BaseModel):
    messages: Dict[str, int]
    notes: Dict[str, int]
    documents: Dict[str, int]


@router.post("/pyramids/generate", response_model=PyramidGenerateResult)
def generate_pyramids(
    request: PyramidGenerateRequest,
    db: Session = Depends(get_db),
):
    """
    Generate summary pyramids for artifacts.
    
    Creates L0 (sentence), L1 (bullets), L2 (paragraphs) summaries
    using LLM for messages, notes, and documents.
    
    Args:
        project_id: Optional filter by project
        force: Regenerate even if content unchanged
    """
    from app.astra_memory.pyramid_generator import run_pyramid_generation
    
    try:
        results = run_pyramid_generation(
            db,
            project_id=request.project_id,
            force=request.force,
        )
        return PyramidGenerateResult(**results)
    except Exception as e:
        logger.error(f"[pyramids] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pyramids/stats", response_model=PyramidStats)
def pyramid_stats(db: Session = Depends(get_db)):
    """Get summary pyramid statistics."""
    from app.astra_memory.pyramid_generator import get_pyramid_stats
    
    stats = get_pyramid_stats(db)
    return PyramidStats(**stats)


@router.post("/pyramids/generate-one")
def generate_single_pyramid(
    artifact_type: str = Query(..., description="Type: message, note, document"),
    artifact_id: str = Query(..., description="ID of the artifact"),
    db: Session = Depends(get_db),
):
    """Generate pyramid for a single artifact."""
    from app.astra_memory.pyramid_generator import (
        generate_pyramid_for_content,
        upsert_pyramid,
    )
    from app.astra_memory.retrieval import _fetch_cold_storage
    
    # Fetch content
    content = _fetch_cold_storage(db, artifact_type, artifact_id)
    if not content:
        raise HTTPException(status_code=404, detail=f"Content not found: {artifact_type}/{artifact_id}")
    
    # Generate pyramid
    pyramid_data = generate_pyramid_for_content(content, artifact_type, artifact_id)
    if not pyramid_data:
        raise HTTPException(status_code=422, detail="Failed to generate pyramid (content too short or LLM error)")
    
    # Store
    pyramid = upsert_pyramid(db, artifact_type, artifact_id, pyramid_data)
    
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "l0_sentence": pyramid.l0_sentence,
        "l1_bullets": pyramid.l1_bullets,
        "l2_paragraphs": pyramid.l2_paragraphs,
        "source_hash": pyramid.source_hash,
    }


# =============================================================================
# FEEDBACK / PREFERENCE LEARNING ENDPOINTS
# =============================================================================

class FeedbackRequest(BaseModel):
    message_id: int
    feedback_type: str  # "positive", "negative", "correction"
    comment: Optional[str] = None
    correction_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    status: str
    message_id: int
    feedback_type: str
    preferences_updated: List[str]
    preferences_created: List[str] = []


@router.post("/feedback", response_model=FeedbackResponse)
def record_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
):
    """
    Record user feedback on a message and trigger preference learning.
    
    feedback_type options:
    - "positive": Thumbs up / liked response
    - "negative": Thumbs down / disliked response  
    - "correction": User edited/corrected the response
    
    For corrections, include correction_text with the corrected version.
    """
    from app.astra_memory.learning import record_message_feedback
    
    try:
        result = record_message_feedback(
            db,
            message_id=request.message_id,
            feedback_type=request.feedback_type,
            user_comment=request.comment,
            correction_text=request.correction_text,
            metadata=request.metadata,
        )
        
        return FeedbackResponse(
            status="ok",
            message_id=request.message_id,
            feedback_type=request.feedback_type,
            preferences_updated=result.get("preferences_updated", []),
            preferences_created=result.get("preferences_created", []),
        )
    except Exception as e:
        logger.error(f"[feedback] Recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn/analyze-patterns")
def analyze_patterns(
    project_id: Optional[int] = Query(None, description="Filter by project"),
    limit: int = Query(100, description="Messages to analyze"),
    db: Session = Depends(get_db),
):
    """
    Analyze conversation patterns to discover preferences.
    
    Scans recent messages to find:
    - Provider/model preferences
    - Response length patterns
    - Format preferences
    """
    from app.astra_memory.learning import analyze_conversation_patterns
    
    results = analyze_conversation_patterns(db, project_id=project_id, limit=limit)
    return results


# =============================================================================
# CONFIDENCE DECAY JOB ENDPOINTS
# =============================================================================

class DecayJobResult(BaseModel):
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    preferences_recomputed: int
    preferences_expired: int
    preferences_disputed: int
    hot_index_cleaned: int
    errors: List[str] = []


class DecayMetrics(BaseModel):
    preferences_by_status: Dict[str, int]
    confidence_distribution: Dict[str, int]
    total_evidence_records: int
    hot_index_entries: int


class SchedulerStatus(BaseModel):
    running: bool
    interval_hours: float
    last_run: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None


@router.post("/decay/run", response_model=DecayJobResult)
def run_decay_job(db: Session = Depends(get_db)):
    """
    Run confidence decay job immediately.
    
    This job:
    - Recomputes confidence scores for all preferences (applies time decay)
    - Expires preferences with very low confidence (<0.15)
    - Marks preferences as disputed if confidence drops below 0.3
    - Cleans stale hot index entries (>90 days old)
    """
    from app.astra_memory.decay_job import run_confidence_decay
    
    try:
        result = run_confidence_decay(db)
        return DecayJobResult(**result)
    except Exception as e:
        logger.error(f"[decay] Job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decay/metrics", response_model=DecayMetrics)
def get_metrics(db: Session = Depends(get_db)):
    """
    Get current confidence/decay metrics.
    
    Returns:
    - Preference counts by status (active, expired, disputed)
    - Confidence distribution across buckets
    - Evidence and hot index counts
    """
    from app.astra_memory.decay_job import get_decay_metrics
    
    metrics = get_decay_metrics(db)
    return DecayMetrics(**metrics)


@router.get("/decay/scheduler", response_model=SchedulerStatus)
def scheduler_status():
    """Get decay job scheduler status."""
    from app.astra_memory.decay_job import get_scheduler
    
    scheduler = get_scheduler()
    status = scheduler.get_status()
    return SchedulerStatus(**status)


@router.post("/decay/scheduler/start")
async def start_scheduler(interval_hours: float = Query(24, description="Run interval in hours")):
    """Start the decay job scheduler."""
    from app.astra_memory.decay_job import get_scheduler, DecayJobScheduler
    
    global _scheduler
    from app.astra_memory import decay_job
    
    if decay_job._scheduler is None:
        decay_job._scheduler = DecayJobScheduler(interval_hours=interval_hours)
    
    await decay_job._scheduler.start()
    return {"status": "started", "interval_hours": interval_hours}


@router.post("/decay/scheduler/stop")
async def stop_scheduler():
    """Stop the decay job scheduler."""
    from app.astra_memory.decay_job import get_scheduler
    
    scheduler = get_scheduler()
    await scheduler.stop()
    return {"status": "stopped"}

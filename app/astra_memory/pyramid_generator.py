# FILE: app/astra_memory/pyramid_generator.py
"""
Summary Pyramid Generator for ASTRA Memory System.

Generates L0-L3 summaries for large artifacts:
- L0: 1 sentence summary
- L1: 5 bullet points
- L2: 1-2 paragraphs
- L3: Full text (stored as cold path reference)

Uses LLM to generate summaries, with caching via source_hash.
"""

from __future__ import annotations

import os
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any

from sqlalchemy.orm import Session
from dotenv import load_dotenv

from app.astra_memory.preference_models import SummaryPyramid

logger = logging.getLogger(__name__)

# Load env for API keys
load_dotenv()

# Configuration
PYRAMID_MODEL = os.getenv("ASTRA_PYRAMID_MODEL", "gpt-4.1-mini")
PYRAMID_PROVIDER = os.getenv("ASTRA_PYRAMID_PROVIDER", "openai")
MIN_CONTENT_LENGTH = 200  # Don't generate pyramids for tiny content
MAX_CONTENT_FOR_SUMMARY = 12000  # Truncate very large content


def _get_openai_client():
    """Get OpenAI client with API key from env."""
    from openai import OpenAI
    return OpenAI()


def _compute_content_hash(content: str) -> str:
    """Compute hash of content for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _call_llm(prompt: str, system: str, max_tokens: int = 500) -> Optional[str]:
    """Call LLM for summary generation."""
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=PYRAMID_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,  # Low temperature for consistent summaries
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[pyramid] LLM call failed: {e}")
        return None


def generate_l0_sentence(content: str) -> Optional[str]:
    """Generate L0: Single sentence summary."""
    system = "You are a summarization assistant. Respond with exactly ONE sentence that captures the main point."
    prompt = f"Summarize this in exactly one sentence:\n\n{content[:MAX_CONTENT_FOR_SUMMARY]}"
    return _call_llm(prompt, system, max_tokens=100)


def generate_l1_bullets(content: str) -> Optional[List[str]]:
    """Generate L1: 5 bullet points."""
    system = """You are a summarization assistant. Respond with exactly 5 bullet points.
Format: One bullet per line, starting with "• " (bullet character).
No numbering, no extra text."""
    
    prompt = f"Summarize the key points in exactly 5 bullets:\n\n{content[:MAX_CONTENT_FOR_SUMMARY]}"
    result = _call_llm(prompt, system, max_tokens=300)
    
    if not result:
        return None
    
    # Parse bullets
    bullets = []
    for line in result.split('\n'):
        line = line.strip()
        if line.startswith('•'):
            bullets.append(line[1:].strip())
        elif line.startswith('-'):
            bullets.append(line[1:].strip())
        elif line.startswith('*'):
            bullets.append(line[1:].strip())
        elif line and len(bullets) < 5:
            # Handle lines without bullet prefix
            bullets.append(line)
    
    return bullets[:5] if bullets else None


def generate_l2_paragraphs(content: str) -> Optional[str]:
    """Generate L2: 1-2 paragraph summary."""
    system = """You are a summarization assistant. Write a concise summary in 1-2 paragraphs.
Focus on the main ideas, key decisions, and important details.
Keep it under 200 words."""
    
    prompt = f"Write a 1-2 paragraph summary:\n\n{content[:MAX_CONTENT_FOR_SUMMARY]}"
    return _call_llm(prompt, system, max_tokens=400)


def generate_pyramid_for_content(
    content: str,
    artifact_type: str,
    artifact_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate all pyramid levels for content.
    
    Returns dict with l0_sentence, l1_bullets, l2_paragraphs, source_hash.
    Returns None if content too short or generation fails.
    """
    if not content or len(content) < MIN_CONTENT_LENGTH:
        logger.debug(f"[pyramid] Content too short for {artifact_type}/{artifact_id}")
        return None
    
    source_hash = _compute_content_hash(content)
    
    logger.info(f"[pyramid] Generating summaries for {artifact_type}/{artifact_id}")
    
    # Generate each level
    l0 = generate_l0_sentence(content)
    l1 = generate_l1_bullets(content)
    l2 = generate_l2_paragraphs(content)
    
    if not l0 and not l1 and not l2:
        logger.warning(f"[pyramid] All generations failed for {artifact_type}/{artifact_id}")
        return None
    
    return {
        "l0_sentence": l0,
        "l1_bullets": l1,
        "l2_paragraphs": l2,
        "source_hash": source_hash,
        "l3_token_estimate": len(content) // 4,
    }


def upsert_pyramid(
    db: Session,
    artifact_type: str,
    artifact_id: str,
    pyramid_data: Dict[str, Any],
    l3_cold_path: Optional[str] = None,
) -> SummaryPyramid:
    """Create or update summary pyramid in database."""
    existing = db.query(SummaryPyramid).filter(
        SummaryPyramid.artifact_type == artifact_type,
        SummaryPyramid.artifact_id == artifact_id,
    ).first()
    
    if existing:
        existing.l0_sentence = pyramid_data.get("l0_sentence")
        existing.l1_bullets = pyramid_data.get("l1_bullets")
        existing.l2_paragraphs = pyramid_data.get("l2_paragraphs")
        existing.source_hash = pyramid_data.get("source_hash")
        existing.l3_token_estimate = pyramid_data.get("l3_token_estimate")
        existing.l3_cold_path = l3_cold_path
        existing.generated_at = datetime.now(timezone.utc)
        pyramid = existing
    else:
        pyramid = SummaryPyramid(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            l0_sentence=pyramid_data.get("l0_sentence"),
            l1_bullets=pyramid_data.get("l1_bullets"),
            l2_paragraphs=pyramid_data.get("l2_paragraphs"),
            source_hash=pyramid_data.get("source_hash"),
            l3_token_estimate=pyramid_data.get("l3_token_estimate"),
            l3_cold_path=l3_cold_path,
        )
        db.add(pyramid)
    
    db.commit()
    db.refresh(pyramid)
    return pyramid


def needs_regeneration(
    db: Session,
    artifact_type: str,
    artifact_id: str,
    current_content: str,
) -> bool:
    """Check if pyramid needs regeneration based on content hash."""
    existing = db.query(SummaryPyramid).filter(
        SummaryPyramid.artifact_type == artifact_type,
        SummaryPyramid.artifact_id == artifact_id,
    ).first()
    
    if not existing:
        return True
    
    current_hash = _compute_content_hash(current_content)
    return existing.source_hash != current_hash


# =============================================================================
# BATCH GENERATION FOR DIFFERENT ARTIFACT TYPES
# =============================================================================

def generate_pyramids_for_messages(
    db: Session,
    project_id: Optional[int] = None,
    limit: int = 50,
    force: bool = False,
) -> Dict[str, int]:
    """Generate pyramids for message content."""
    from app.memory.models import Message
    from app.crypto import is_encryption_ready
    
    if not is_encryption_ready():
        logger.warning("[pyramid] Encryption not ready, skipping messages")
        return {"skipped": 0, "generated": 0, "failed": 0}
    
    query = db.query(Message).filter(Message.role == "assistant")
    if project_id:
        query = query.filter(Message.project_id == project_id)
    
    messages = query.order_by(Message.id.desc()).limit(limit).all()
    
    stats = {"skipped": 0, "generated": 0, "failed": 0}
    
    for msg in messages:
        try:
            content = msg.content
            if not content or content.startswith("ENC:") or len(content) < MIN_CONTENT_LENGTH:
                stats["skipped"] += 1
                continue
            
            artifact_id = str(msg.id)
            
            if not force and not needs_regeneration(db, "message", artifact_id, content):
                stats["skipped"] += 1
                continue
            
            pyramid_data = generate_pyramid_for_content(content, "message", artifact_id)
            if pyramid_data:
                upsert_pyramid(db, "message", artifact_id, pyramid_data)
                stats["generated"] += 1
            else:
                stats["failed"] += 1
                
        except Exception as e:
            logger.error(f"[pyramid] Message {msg.id} failed: {e}")
            stats["failed"] += 1
    
    return stats


def generate_pyramids_for_notes(
    db: Session,
    project_id: Optional[int] = None,
    limit: int = 50,
    force: bool = False,
) -> Dict[str, int]:
    """Generate pyramids for notes."""
    from app.memory.models import Note
    
    query = db.query(Note)
    if project_id:
        query = query.filter(Note.project_id == project_id)
    
    notes = query.order_by(Note.id.desc()).limit(limit).all()
    
    stats = {"skipped": 0, "generated": 0, "failed": 0}
    
    for note in notes:
        try:
            content = f"# {note.title}\n\n{note.content}" if note.content else ""
            if not content or content.startswith("ENC:") or len(content) < MIN_CONTENT_LENGTH:
                stats["skipped"] += 1
                continue
            
            artifact_id = str(note.id)
            
            if not force and not needs_regeneration(db, "note", artifact_id, content):
                stats["skipped"] += 1
                continue
            
            pyramid_data = generate_pyramid_for_content(content, "note", artifact_id)
            if pyramid_data:
                upsert_pyramid(db, "note", artifact_id, pyramid_data)
                stats["generated"] += 1
            else:
                stats["failed"] += 1
                
        except Exception as e:
            logger.error(f"[pyramid] Note {note.id} failed: {e}")
            stats["failed"] += 1
    
    return stats


def generate_pyramids_for_documents(
    db: Session,
    project_id: Optional[int] = None,
    limit: int = 20,
    force: bool = False,
) -> Dict[str, int]:
    """Generate pyramids for uploaded documents."""
    from app.memory.models import DocumentContent
    
    query = db.query(DocumentContent)
    if project_id:
        query = query.filter(DocumentContent.project_id == project_id)
    
    docs = query.order_by(DocumentContent.id.desc()).limit(limit).all()
    
    stats = {"skipped": 0, "generated": 0, "failed": 0}
    
    for doc in docs:
        try:
            content = doc.raw_text or ""
            if not content or content.startswith("ENC:") or len(content) < MIN_CONTENT_LENGTH:
                stats["skipped"] += 1
                continue
            
            artifact_id = str(doc.id)
            
            if not force and not needs_regeneration(db, "document", artifact_id, content):
                stats["skipped"] += 1
                continue
            
            pyramid_data = generate_pyramid_for_content(content, "document", artifact_id)
            if pyramid_data:
                upsert_pyramid(db, "document", artifact_id, pyramid_data)
                stats["generated"] += 1
            else:
                stats["failed"] += 1
                
        except Exception as e:
            logger.error(f"[pyramid] Document {doc.id} failed: {e}")
            stats["failed"] += 1
    
    return stats


def run_pyramid_generation(
    db: Session,
    project_id: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Run pyramid generation for all artifact types.
    
    Args:
        db: Database session
        project_id: Optional project filter
        force: Regenerate even if content unchanged
        
    Returns:
        Stats per artifact type
    """
    logger.info(f"[pyramid] Starting generation (project={project_id}, force={force})")
    
    results = {
        "messages": generate_pyramids_for_messages(db, project_id, force=force),
        "notes": generate_pyramids_for_notes(db, project_id, force=force),
        "documents": generate_pyramids_for_documents(db, project_id, force=force),
    }
    
    total_generated = sum(r["generated"] for r in results.values())
    total_skipped = sum(r["skipped"] for r in results.values())
    total_failed = sum(r["failed"] for r in results.values())
    
    logger.info(f"[pyramid] Complete: generated={total_generated}, skipped={total_skipped}, failed={total_failed}")
    
    return results


def get_pyramid_stats(db: Session) -> Dict[str, Any]:
    """Get statistics about stored pyramids."""
    from sqlalchemy import func
    
    total = db.query(func.count(SummaryPyramid.id)).scalar() or 0
    
    by_type = dict(
        db.query(
            SummaryPyramid.artifact_type,
            func.count(SummaryPyramid.id)
        ).group_by(SummaryPyramid.artifact_type).all()
    )
    
    with_l0 = db.query(func.count(SummaryPyramid.id)).filter(
        SummaryPyramid.l0_sentence.isnot(None)
    ).scalar() or 0
    
    with_l1 = db.query(func.count(SummaryPyramid.id)).filter(
        SummaryPyramid.l1_bullets.isnot(None)
    ).scalar() or 0
    
    with_l2 = db.query(func.count(SummaryPyramid.id)).filter(
        SummaryPyramid.l2_paragraphs.isnot(None)
    ).scalar() or 0
    
    return {
        "total": total,
        "by_type": by_type,
        "coverage": {
            "l0": with_l0,
            "l1": with_l1,
            "l2": with_l2,
        }
    }


__all__ = [
    "generate_pyramid_for_content",
    "generate_l0_sentence",
    "generate_l1_bullets", 
    "generate_l2_paragraphs",
    "upsert_pyramid",
    "needs_regeneration",
    "generate_pyramids_for_messages",
    "generate_pyramids_for_notes",
    "generate_pyramids_for_documents",
    "run_pyramid_generation",
    "get_pyramid_stats",
]

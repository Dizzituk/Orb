# FILE: app/transparency/corrections.py
# Purpose: CorrectionStore — CRUD operations for user corrections.
# Called-by: app.debug.feedback, app.transparency.matcher, app.transparency.router
# Depends-on: app.db, app.transparency.models, app.transparency.schemas
# Last-renovated: 2026-06-11
"""
CorrectionStore — CRUD operations for user corrections.

Manages storage and retrieval of user feedback pinned to
specific reasoning events. Used by:
- Router (POST new corrections)
- Matcher (query relevant past corrections)
- Frontend (display corrections alongside reasoning blocks)

v1.0 (2026-02): Initial implementation
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.transparency.schemas import UserCorrection

logger = logging.getLogger(__name__)


# =============================================================================
# KEYWORD EXTRACTION
# =============================================================================

# Common stop words to filter out when extracting context keywords
_STOP_WORDS = {
    "the", "a", "an", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "mine", "yours", "ours",
    "not", "no", "nor", "but", "and", "or", "so", "if", "then",
    "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "up", "out", "off", "over", "into", "through", "about",
    "just", "also", "very", "too", "quite", "rather", "really",
}


def extract_keywords(text: str, max_keywords: int = 15) -> List[str]:
    """
    Extract meaningful keywords from correction text for matching.

    Strips stop words, keeps technical terms, file paths, and
    domain-specific language.
    """
    if not text:
        return []

    # Normalise
    text_lower = text.lower()

    # Extract file paths (e.g. app/endpoints/chat.py)
    file_paths = re.findall(r'[\w/\\]+\.(?:py|ts|tsx|js|jsx|css|json|sql)', text_lower)

    # Extract technical terms (words with underscores or dots)
    tech_terms = re.findall(r'\b\w+[_.]\w+\b', text_lower)

    # Extract regular words (3+ chars, not stop words)
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    meaningful = [w for w in words if w not in _STOP_WORDS]

    # Combine, deduplicate, and limit
    all_keywords = []
    seen = set()
    for kw in file_paths + tech_terms + meaningful:
        if kw not in seen:
            seen.add(kw)
            all_keywords.append(kw)
        if len(all_keywords) >= max_keywords:
            break

    return all_keywords


# =============================================================================
# CORRECTION STORE
# =============================================================================

class CorrectionStore:
    """CRUD operations for user corrections."""

    @staticmethod
    def add_correction(correction: UserCorrection) -> UserCorrection:
        """Store a new user correction."""
        # Auto-extract keywords if not provided
        if not correction.context_keywords:
            correction.context_keywords = extract_keywords(correction.user_comment)

        try:
            from app.db import get_db_session
            from app.transparency.models import UserCorrectionModel

            db = get_db_session()
            try:
                row = UserCorrectionModel(
                    correction_id=correction.correction_id,
                    reasoning_event_id=correction.reasoning_event_id,
                    job_id=correction.job_id,
                    run_id=correction.run_id,
                    build_project_id=correction.build_project_id,
                    stage_name=correction.stage_name,
                    stage_index=correction.stage_index,
                    decision_index=correction.decision_index,
                    user_comment=correction.user_comment,
                    severity=correction.severity,
                    correction_type=correction.correction_type,
                    context_keywords=correction.context_keywords,
                )
                db.add(row)
                db.commit()

                logger.info(
                    "[corrections] Added correction '%s' for stage '%s' (severity=%s)",
                    correction.correction_id, correction.stage_name, correction.severity,
                )
                return correction
            finally:
                db.close()

        except Exception as e:
            logger.error("[corrections] Failed to add correction: %s", e)
            raise

    @staticmethod
    def get_corrections_for_event(reasoning_event_id: str) -> List[UserCorrection]:
        """Get all corrections for a specific reasoning event."""
        try:
            from app.db import get_db_session
            from app.transparency.models import UserCorrectionModel

            db = get_db_session()
            try:
                rows = (
                    db.query(UserCorrectionModel)
                    .filter_by(reasoning_event_id=reasoning_event_id)
                    .order_by(UserCorrectionModel.created_at)
                    .all()
                )
                return [_row_to_correction(r) for r in rows]
            finally:
                db.close()
        except Exception as e:
            logger.warning("[corrections] Failed to get corrections for event: %s", e)
            return []

    @staticmethod
    def get_corrections_for_project(build_project_id: str) -> List[UserCorrection]:
        """Get all corrections for a build project."""
        try:
            from app.db import get_db_session
            from app.transparency.models import UserCorrectionModel

            db = get_db_session()
            try:
                rows = (
                    db.query(UserCorrectionModel)
                    .filter_by(build_project_id=build_project_id)
                    .order_by(UserCorrectionModel.created_at.desc())
                    .all()
                )
                return [_row_to_correction(r) for r in rows]
            finally:
                db.close()
        except Exception as e:
            logger.warning("[corrections] Failed to get corrections for project: %s", e)
            return []

    @staticmethod
    def get_corrections_by_stage(
        stage_name: str,
        limit: int = 50,
    ) -> List[UserCorrection]:
        """Get recent corrections for a specific pipeline stage."""
        try:
            from app.db import get_db_session
            from app.transparency.models import UserCorrectionModel

            db = get_db_session()
            try:
                rows = (
                    db.query(UserCorrectionModel)
                    .filter_by(stage_name=stage_name)
                    .order_by(UserCorrectionModel.created_at.desc())
                    .limit(limit)
                    .all()
                )
                return [_row_to_correction(r) for r in rows]
            finally:
                db.close()
        except Exception as e:
            logger.warning("[corrections] Failed to get corrections by stage: %s", e)
            return []


    @staticmethod
    def get_corrections_by_stage_and_project(
        stage_name: str,
        build_project_id: str = "",
        limit: int = 50,
    ) -> List[UserCorrection]:
        """Get corrections filtered by stage AND project.
        
        Returns project-specific corrections first, then universal ones
        (where build_project_id is empty), up to the limit.
        """
        try:
            from app.db import get_db_session
            from app.transparency.models import UserCorrectionModel

            db = get_db_session()
            try:
                # Project-specific corrections
                project_rows = []
                if build_project_id:
                    project_rows = (
                        db.query(UserCorrectionModel)
                        .filter_by(stage_name=stage_name, build_project_id=build_project_id)
                        .order_by(UserCorrectionModel.created_at.desc())
                        .limit(limit)
                        .all()
                    )

                # Universal corrections (no project specified, or severity=broke_things)
                remaining = limit - len(project_rows)
                universal_rows = []
                if remaining > 0:
                    universal_query = (
                        db.query(UserCorrectionModel)
                        .filter_by(stage_name=stage_name)
                        .filter(
                            (UserCorrectionModel.build_project_id == "")
                            | (UserCorrectionModel.severity == "broke_things")
                        )
                    )
                    # Exclude ones we already got from project query
                    if project_rows:
                        project_ids = [r.correction_id for r in project_rows]
                        universal_query = universal_query.filter(
                            ~UserCorrectionModel.correction_id.in_(project_ids)
                        )
                    universal_rows = (
                        universal_query
                        .order_by(UserCorrectionModel.created_at.desc())
                        .limit(remaining)
                        .all()
                    )

                all_rows = project_rows + universal_rows
                return [_row_to_correction(r) for r in all_rows]
            finally:
                db.close()
        except Exception as e:
            logger.warning("[corrections] Failed to get corrections by stage+project: %s", e)
            return []
    @staticmethod
    def delete_correction(correction_id: str) -> bool:
        """Delete a correction by ID."""
        try:
            from app.db import get_db_session
            from app.transparency.models import UserCorrectionModel

            db = get_db_session()
            try:
                row = db.query(UserCorrectionModel).filter_by(
                    correction_id=correction_id
                ).first()
                if row:
                    db.delete(row)
                    db.commit()
                    return True
                return False
            finally:
                db.close()
        except Exception as e:
            logger.error("[corrections] Failed to delete correction: %s", e)
            return False


def _row_to_correction(row) -> UserCorrection:
    """Convert a DB row to a UserCorrection dataclass."""
    return UserCorrection(
        correction_id=row.correction_id,
        reasoning_event_id=row.reasoning_event_id,
        job_id=row.job_id or "",
        run_id=row.run_id or "",
        build_project_id=row.build_project_id or "",
        stage_name=row.stage_name,
        stage_index=row.stage_index or 0,
        decision_index=row.decision_index,
        user_comment=row.user_comment,
        severity=row.severity,
        correction_type=row.correction_type,
        context_keywords=row.context_keywords or [],
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


__all__ = [
    "CorrectionStore",
    "extract_keywords",
]

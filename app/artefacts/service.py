# FILE: app/artefacts/service.py
"""
Phase 4 Artefact Store Service

Manages project artefacts with:
- Versioning (v1, v2, v3, ...)
- Optimistic concurrency control (etag)
- Status tracking (current, superseded, draft)
- Structured metadata

PHASE 4 FIXES:
- Fixed etag generation to include version number
- Two versions with identical content now get different etags
- Etag format: sha256(content + ":v" + version)[:32]

Artefacts include:
- Research dossiers
- Architecture documents
- Code patch proposals
- Script proposals
- Canonical prompts and configs
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4
from sqlalchemy.orm import Session

from app.jobs.models import Artefact
from app.jobs.schemas import ErrorType

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ArtefactError(Exception):
    """Base exception for artefact operations."""
    pass


class ArtefactNotFoundError(ArtefactError):
    """Artefact does not exist."""
    pass


class ArtefactConflictError(ArtefactError):
    """Concurrent modification detected (etag mismatch)."""
    pass


# =============================================================================
# ARTEFACT SERVICE
# =============================================================================

class ArtefactService:
    """
    Service for managing project artefacts.
    
    Features:
    - Write artefacts with automatic versioning
    - Read artefacts by ID or filters
    - Update artefacts with optimistic locking (etag)
    - Supersede old versions
    - List artefacts with filters
    """
    
    @staticmethod
    def _generate_etag(content: str, version: int = 1) -> str:
        """
        Generate etag from content hash AND version.
        
        FIXED: Now includes version in hash to ensure different versions
        with identical content get different etags.
        
        Format: sha256(content + ":v" + str(version))[:32]
        """
        versioned_content = f"{content}:v{version}"
        return hashlib.sha256(versioned_content.encode('utf-8')).hexdigest()[:32]
    
    @staticmethod
    def write_artefact(
        db: Session,
        project_id: int,
        artefact_type: str,
        name: str,
        content: str,
        metadata: Optional[dict] = None,
        created_by_job_id: Optional[str] = None,
        etag: Optional[str] = None,
        artefact_id: Optional[str] = None,
    ) -> Artefact:
        """
        Write or update an artefact.
        
        If artefact_id is provided and etag matches:
            - Create new version
            - Mark old version as superseded
        
        If artefact_id is provided but etag doesn't match:
            - Raise ArtefactConflictError
        
        If artefact_id is not provided:
            - Create new artefact (version 1)
        
        Args:
            db: Database session
            project_id: Project identifier
            artefact_type: Type of artefact (research_dossier, architecture_doc, etc.)
            name: Human-readable name
            content: Artefact content (will be encrypted)
            metadata: Optional metadata (will be encrypted)
            created_by_job_id: Job that created this artefact
            etag: Expected etag for concurrency control (required for updates)
            artefact_id: Existing artefact ID (for updates)
        
        Returns:
            Created/updated Artefact instance
        
        Raises:
            ArtefactNotFoundError: If artefact_id provided but not found
            ArtefactConflictError: If etag doesn't match current version
        """
        if artefact_id:
            # Update existing artefact
            existing = db.query(Artefact).filter(
                Artefact.id == artefact_id,
                Artefact.project_id == project_id
            ).first()
            
            if not existing:
                raise ArtefactNotFoundError(f"Artefact not found: {artefact_id}")
            
            # Check etag for concurrency control
            if etag != existing.etag:
                raise ArtefactConflictError(
                    f"Concurrent modification detected. "
                    f"Expected etag={etag}, but current etag={existing.etag}"
                )
            
            # Mark existing version as superseded
            existing.status = "superseded"
            db.add(existing)
            
            # Calculate new version number
            new_version = existing.version + 1
            
            # FIXED: Generate etag with version included
            new_etag = ArtefactService._generate_etag(content, new_version)
            
            # Create new version
            new_artefact = Artefact(
                id=str(uuid4()),
                project_id=project_id,
                artefact_type=artefact_type,
                name=name,
                version=new_version,
                etag=new_etag,
                status="current",
                content=content,
                metadata_json=metadata,
                created_by_job_id=created_by_job_id,
                supersedes_id=existing.id,
            )
            
            logger.info(
                f"[artefact] Updated: {artefact_id} -> {new_artefact.id} "
                f"v{existing.version} -> v{new_artefact.version}"
            )
        else:
            # Create new artefact (version 1)
            new_version = 1
            new_etag = ArtefactService._generate_etag(content, new_version)
            
            new_artefact = Artefact(
                id=str(uuid4()),
                project_id=project_id,
                artefact_type=artefact_type,
                name=name,
                version=new_version,
                etag=new_etag,
                status="current",
                content=content,
                metadata_json=metadata,
                created_by_job_id=created_by_job_id,
            )
            
            logger.info(
                f"[artefact] Created: {new_artefact.id} "
                f"type={artefact_type} name={name}"
            )
        
        db.add(new_artefact)
        db.commit()
        db.refresh(new_artefact)
        
        return new_artefact
    
    @staticmethod
    def read_artefact(
        db: Session,
        artefact_id: str,
        project_id: int,
    ) -> Optional[Artefact]:
        """
        Read an artefact by ID.
        
        Args:
            db: Database session
            artefact_id: Artefact identifier
            project_id: Project identifier (for security)
        
        Returns:
            Artefact instance or None if not found
        """
        return db.query(Artefact).filter(
            Artefact.id == artefact_id,
            Artefact.project_id == project_id
        ).first()
    
    @staticmethod
    def list_artefacts(
        db: Session,
        project_id: int,
        artefact_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Artefact], int]:
        """
        List artefacts with filters.
        
        Args:
            db: Database session
            project_id: Project identifier
            artefact_type: Optional type filter
            status: Optional status filter (current, superseded, draft)
            limit: Maximum results
            offset: Pagination offset
        
        Returns:
            Tuple of (list of Artefact instances, total count)
        """
        query = db.query(Artefact).filter(Artefact.project_id == project_id)
        
        if artefact_type:
            query = query.filter(Artefact.artefact_type == artefact_type)
        
        if status:
            query = query.filter(Artefact.status == status)
        
        # Get total count before pagination
        total = query.count()
        
        query = query.order_by(Artefact.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        return query.all(), total
    
    @staticmethod
    def get_latest_version(
        db: Session,
        project_id: int,
        artefact_type: str,
        name: str,
    ) -> Optional[Artefact]:
        """
        Get the latest version of an artefact by type and name.
        
        Args:
            db: Database session
            project_id: Project identifier
            artefact_type: Artefact type
            name: Artefact name
        
        Returns:
            Latest Artefact instance or None
        """
        return db.query(Artefact).filter(
            Artefact.project_id == project_id,
            Artefact.artefact_type == artefact_type,
            Artefact.name == name,
            Artefact.status == "current"
        ).order_by(Artefact.version.desc()).first()
    
    @staticmethod
    def get_version_history(
        db: Session,
        project_id: int,
        artefact_type: str,
        name: str,
    ) -> list[Artefact]:
        """
        Get all versions of an artefact.
        
        Args:
            db: Database session
            project_id: Project identifier
            artefact_type: Artefact type
            name: Artefact name
        
        Returns:
            List of all versions, newest first
        """
        return db.query(Artefact).filter(
            Artefact.project_id == project_id,
            Artefact.artefact_type == artefact_type,
            Artefact.name == name,
        ).order_by(Artefact.version.desc()).all()
    
    @staticmethod
    def mark_as_draft(
        db: Session,
        artefact_id: str,
        project_id: int,
    ) -> bool:
        """
        Mark an artefact as draft.
        
        Args:
            db: Database session
            artefact_id: Artefact identifier
            project_id: Project identifier (for security)
        
        Returns:
            True if successful, False if not found
        """
        artefact = db.query(Artefact).filter(
            Artefact.id == artefact_id,
            Artefact.project_id == project_id
        ).first()
        
        if not artefact:
            return False
        
        artefact.status = "draft"
        db.commit()
        
        logger.info(f"[artefact] Marked as draft: {artefact_id}")
        return True
    
    @staticmethod
    def delete_artefact(
        db: Session,
        artefact_id: str,
        project_id: int,
    ) -> bool:
        """
        Delete an artefact.
        
        Note: This is a hard delete. Consider marking as superseded instead.
        
        Args:
            db: Database session
            artefact_id: Artefact identifier
            project_id: Project identifier (for security)
        
        Returns:
            True if deleted, False if not found
        """
        artefact = db.query(Artefact).filter(
            Artefact.id == artefact_id,
            Artefact.project_id == project_id
        ).first()
        
        if not artefact:
            return False
        
        db.delete(artefact)
        db.commit()
        
        logger.info(f"[artefact] Deleted: {artefact_id}")
        return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def write_research_dossier(
    db: Session,
    project_id: int,
    name: str,
    content: str,
    job_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Artefact:
    """
    Convenience function to write a research dossier.
    
    Research dossiers are structured documents with:
    - Problem/question
    - Key concepts
    - Summary of findings
    - Contradictions/uncertainties
    - Gaps/unknowns
    - Recommendations
    - Sources
    """
    return ArtefactService.write_artefact(
        db=db,
        project_id=project_id,
        artefact_type="research_dossier",
        name=name,
        content=content,
        metadata=metadata,
        created_by_job_id=job_id,
    )


def write_architecture_doc(
    db: Session,
    project_id: int,
    name: str,
    content: str,
    job_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Artefact:
    """Convenience function to write an architecture document."""
    return ArtefactService.write_artefact(
        db=db,
        project_id=project_id,
        artefact_type="architecture_doc",
        name=name,
        content=content,
        metadata=metadata,
        created_by_job_id=job_id,
    )


def write_code_patch_proposal(
    db: Session,
    project_id: int,
    name: str,
    content: str,
    job_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Artefact:
    """Convenience function to write a code patch proposal."""
    return ArtefactService.write_artefact(
        db=db,
        project_id=project_id,
        artefact_type="code_patch_proposal",
        name=name,
        content=content,
        metadata=metadata,
        created_by_job_id=job_id,
    )


__all__ = [
    "ArtefactError",
    "ArtefactNotFoundError",
    "ArtefactConflictError",
    "ArtefactService",
    "write_research_dossier",
    "write_architecture_doc",
    "write_code_patch_proposal",
]
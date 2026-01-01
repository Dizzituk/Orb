# FILE: app/helpers/context.py
"""
Context building helpers for LLM endpoints.

Extracted from main.py for better organization.
"""

from sqlalchemy.orm import Session
from app.memory import service as memory_service


def build_context_block(db: Session, project_id: int) -> str:
    """Build context from notes + tasks."""
    sections = []

    notes = memory_service.list_notes(db, project_id)[:10]
    if notes:
        notes_text = "\n".join(
            f"- [{n.id}] {n.title}: {n.content[:200]}..."
            for n in notes
        )
        sections.append(f"PROJECT NOTES:\n{notes_text}")

    tasks = memory_service.list_tasks(db, project_id, status="pending")[:10]
    if tasks:
        tasks_text = "\n".join(f"- {t.title}" for t in tasks)
        sections.append(f"PENDING TASKS:\n{tasks_text}")

    return "\n\n".join(sections) if sections else ""


def build_document_context(db: Session, project_id: int, user_message: str = "") -> str:
    """
    Build context from previously uploaded documents (from database).
    
    NOTE: This returns SUMMARIES of OLD documents, not the current upload.
    For current uploads, use document_content_parts directly.
    """
    try:
        from app.memory.models import DocumentContent
        
        recent_docs = (db.query(DocumentContent)
                      .filter(DocumentContent.project_id == project_id)
                      .order_by(DocumentContent.created_at.desc())
                      .limit(5)
                      .all())
        
        if not recent_docs:
            return ""
        
        context_parts = []
        for doc in recent_docs:
            summary = doc.summary[:300] if doc.summary else "(no summary)"
            context_parts.append(f"[{doc.filename}]: {summary}")
        
        return "\n".join(context_parts)
    except Exception as e:
        print(f"[build_document_context] Error: {e}")
        return ""

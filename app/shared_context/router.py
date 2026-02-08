from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.shared_context.builder import build_shared_context_package
from app.shared_context.schemas import SharedContextPackage

router = APIRouter(prefix="/shared-context", tags=["Shared Context"])


class PreviewRequest(BaseModel):
    project_id: int
    query: str
    top_k: int = 8
    source_types: Optional[List[str]] = None
    budget_chars: int = 6000
    max_items: int = 4


@router.post("/preview", response_model=SharedContextPackage)
def preview_shared_context(
    req: PreviewRequest,
    auth: AuthResult = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """
    Preview what shared context would be injected for a given query.
    
    This endpoint is essential for:
    - Debugging retrieval quality
    - Tuning top_k and budget_chars
    - Validating provenance format
    """
    return build_shared_context_package(
        db=db,
        project_id=req.project_id,
        query=req.query,
        top_k=req.top_k,
        source_types=req.source_types,
        budget_chars=req.budget_chars,
        max_items=req.max_items
    )
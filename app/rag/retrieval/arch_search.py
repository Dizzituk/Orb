"""
Architecture search.

REUSES existing embedding search and intent classification.
"""

from dataclasses import dataclass
from typing import List, Optional
from sqlalchemy.orm import Session

from app.rag.models import (
    ArchDirectoryIndex,
    ArchCodeChunk,
    SourceType,
)

# REUSE EXISTING - do not reimplement!
from app.embeddings.service import search_embeddings
from app.astra_memory.retrieval import classify_intent_depth
from app.astra_memory.preference_models import IntentDepth

ARCH_PROJECT_ID = 0


@dataclass
class ArchSearchResult:
    """Single search result."""
    source_type: str
    source_id: int
    score: float
    content: str
    
    # Resolved metadata
    canonical_path: str = ""
    name: str = ""
    chunk_type: str = ""
    start_line: int = 0
    end_line: int = 0
    signature: str = ""
    docstring: str = ""


@dataclass
class ArchSearchResponse:
    """Full search response."""
    query: str
    intent_depth: IntentDepth
    results: List[ArchSearchResult]
    total_searched: int
    directories_found: int
    chunks_found: int


class ArchitectureSearch:
    """Search architecture using existing infrastructure."""
    
    def __init__(self, db: Session, project_id: int = ARCH_PROJECT_ID):
        self.db = db
        self.project_id = project_id
    
    def search(
        self,
        query: str,
        top_k_dirs: int = 3,
        top_k_chunks: int = 10,
        source_types: List[str] = None,
    ) -> ArchSearchResponse:
        """
        Search architecture.
        
        Args:
            query: User query
            top_k_dirs: Max directory results
            top_k_chunks: Max chunk results
            source_types: Filter by type
            
        Returns:
            ArchSearchResponse
        """
        # REUSE existing intent classification
        intent = classify_intent_depth(query)
        
        if source_types is None:
            source_types = [SourceType.ARCH_DIRECTORY, SourceType.ARCH_CHUNK]
        
        # Adjust based on intent
        if intent in (IntentDepth.D0, IntentDepth.D1):
            top_k_chunks = min(top_k_chunks, 3)
        
        total_k = top_k_dirs + top_k_chunks
        
        # REUSE existing search
        raw_results, total_searched = search_embeddings(
            db=self.db,
            project_id=self.project_id,
            query=query,
            top_k=total_k,
            source_types=source_types,
        )
        
        # Convert and resolve metadata
        results = []
        dirs_found = 0
        chunks_found = 0
        
        for raw in raw_results:
            result = ArchSearchResult(
                source_type=raw.source_type,
                source_id=raw.source_id,
                score=raw.similarity,
                content=raw.content,
            )
            
            self._resolve_metadata(result)
            
            # Apply limits
            if result.source_type == SourceType.ARCH_DIRECTORY:
                if dirs_found < top_k_dirs:
                    results.append(result)
                    dirs_found += 1
            else:
                if chunks_found < top_k_chunks:
                    results.append(result)
                    chunks_found += 1
        
        return ArchSearchResponse(
            query=query,
            intent_depth=intent,
            results=results,
            total_searched=total_searched,
            directories_found=dirs_found,
            chunks_found=chunks_found,
        )
    
    def _resolve_metadata(self, result: ArchSearchResult):
        """Fill in metadata from database."""
        if result.source_type == SourceType.ARCH_DIRECTORY:
            directory = self.db.query(ArchDirectoryIndex).get(result.source_id)
            if directory:
                result.canonical_path = directory.canonical_path
                result.name = directory.name
                
        elif result.source_type == SourceType.ARCH_CHUNK:
            chunk = self.db.query(ArchCodeChunk).get(result.source_id)
            if chunk:
                result.canonical_path = chunk.file_path
                result.name = chunk.chunk_name
                result.chunk_type = chunk.chunk_type
                result.start_line = chunk.start_line or 0
                result.end_line = chunk.end_line or 0
                result.signature = chunk.signature or ""
                result.docstring = chunk.docstring or ""


def search_architecture(
    db: Session,
    query: str,
    top_k: int = 10,
) -> ArchSearchResponse:
    """Convenience function."""
    return ArchitectureSearch(db).search(
        query,
        top_k_dirs=3,
        top_k_chunks=top_k - 3,
    )

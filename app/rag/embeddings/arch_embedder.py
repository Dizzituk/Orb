"""
Architecture embedder.

Thin wrapper around existing app/embeddings/service.py.
DO NOT REIMPLEMENT - just call existing functions.
"""

import logging
from typing import Dict
from sqlalchemy.orm import Session

from app.rag.models import (
    ArchDirectoryIndex,
    ArchCodeChunk,
    SourceType,
    ChunkType,
)

# REUSE EXISTING - do not reimplement!
from app.embeddings.service import generate_embedding, store_embedding

logger = logging.getLogger(__name__)

# Project ID for architecture embeddings
ARCH_PROJECT_ID = 0


class ArchitectureEmbedder:
    """Embed architecture data using existing system."""
    
    def __init__(
        self,
        db: Session,
        scan_id: int,
        project_id: int = ARCH_PROJECT_ID,
        batch_size: int = 50,
    ):
        self.db = db
        self.scan_id = scan_id
        self.project_id = project_id
        self.batch_size = batch_size
    
    def embed_all(self) -> Dict[str, int]:
        """Embed directories and chunks."""
        stats = {
            "directories": 0,
            "chunks": 0,
            "errors": 0,
        }
        
        # Embed directories
        dir_stats = self._embed_directories()
        stats["directories"] = dir_stats["embedded"]
        stats["errors"] += dir_stats["errors"]
        
        # Embed chunks
        chunk_stats = self._embed_chunks()
        stats["chunks"] = chunk_stats["embedded"]
        stats["errors"] += chunk_stats["errors"]
        
        logger.info(f"Embedding complete: {stats}")
        return stats
    
    def _embed_directories(self) -> Dict[str, int]:
        """Embed directory summaries."""
        stats = {"embedded": 0, "errors": 0}
        
        directories = self.db.query(ArchDirectoryIndex).filter(
            ArchDirectoryIndex.scan_id == self.scan_id,
            ArchDirectoryIndex.summary.isnot(None),
            ArchDirectoryIndex.summary != "",
        ).all()
        
        logger.info(f"Embedding {len(directories)} directories")
        
        for i, directory in enumerate(directories):
            try:
                # REUSE existing function
                embedding = generate_embedding(directory.summary)
                
                if embedding is None:
                    stats["errors"] += 1
                    continue
                
                # REUSE existing function
                store_embedding(
                    db=self.db,
                    project_id=self.project_id,
                    source_type=SourceType.ARCH_DIRECTORY,
                    source_id=directory.id,
                    content=directory.summary,
                    embedding=embedding,
                    chunk_index=0,
                )
                
                stats["embedded"] += 1
                
                if (i + 1) % self.batch_size == 0:
                    self.db.commit()
                    logger.info(f"Embedded {i + 1} directories...")
                    
            except Exception as e:
                logger.error(f"Error embedding directory {directory.id}: {e}")
                stats["errors"] += 1
        
        self.db.commit()
        return stats
    
    def _embed_chunks(self) -> Dict[str, int]:
        """Embed chunk descriptors."""
        stats = {"embedded": 0, "errors": 0}
        
        chunks = self.db.query(ArchCodeChunk).filter(
            ArchCodeChunk.scan_id == self.scan_id,
            ArchCodeChunk.chunk_type.in_(ChunkType.EMBEDDABLE),
            ArchCodeChunk.descriptor.isnot(None),
            ArchCodeChunk.descriptor != "",
        ).all()
        
        logger.info(f"Embedding {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            try:
                # REUSE existing function
                embedding = generate_embedding(chunk.descriptor)
                
                if embedding is None:
                    stats["errors"] += 1
                    continue
                
                # REUSE existing function
                store_embedding(
                    db=self.db,
                    project_id=self.project_id,
                    source_type=SourceType.ARCH_CHUNK,
                    source_id=chunk.id,
                    content=chunk.descriptor,
                    embedding=embedding,
                    chunk_index=0,
                )
                
                stats["embedded"] += 1
                
                if (i + 1) % self.batch_size == 0:
                    self.db.commit()
                    logger.info(f"Embedded {i + 1} chunks...")
                    
            except Exception as e:
                logger.error(f"Error embedding chunk {chunk.id}: {e}")
                stats["errors"] += 1
        
        self.db.commit()
        return stats


def embed_architecture_scan(
    db: Session,
    scan_id: int,
    project_id: int = ARCH_PROJECT_ID,
) -> Dict[str, int]:
    """Convenience function."""
    return ArchitectureEmbedder(db, scan_id, project_id).embed_all()

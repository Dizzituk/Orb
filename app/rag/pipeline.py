"""
RAG pipeline orchestrator.

Runs all stages in order:
1. Build directory index from INDEX_*.json
2. Generate directory summaries
3. Load signatures from SIGNATURES_*.json (glob pattern)
4. Generate chunk descriptors
5. Embed everything
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.rag.models import ArchScanRun
from app.rag.indexing.directory_indexer import DirectoryIndexBuilder
from app.rag.indexing.directory_summary import generate_summaries_for_scan
from app.rag.chunking.signature_loader import (
    SignatureLoader,
    find_latest_signatures_file,
)
from app.rag.descriptors.descriptor_gen import generate_descriptors_for_scan
from app.rag.embeddings.arch_embedder import embed_architecture_scan

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Full RAG indexing pipeline."""
    
    def __init__(
        self,
        db: Session,
        scan_dir: str,
        project_id: int = 0,
    ):
        self.db = db
        self.scan_dir = scan_dir
        self.project_id = project_id
        self.scan_run: Optional[ArchScanRun] = None
    
    def run(self) -> Dict[str, Any]:
        """
        Run full pipeline.
        
        Returns:
            Stats dictionary
        """
        stats = {
            "scan_id": 0,
            "directories": 0,
            "chunks": 0,
            "descriptors": 0,
            "embeddings": 0,
            "errors": [],
        }
        
        try:
            # Create scan run
            self.scan_run = self._create_scan_run()
            stats["scan_id"] = self.scan_run.id
            
            # Find files
            sig_path = find_latest_signatures_file(self.scan_dir)
            if not sig_path:
                raise FileNotFoundError(
                    f"No SIGNATURES file in {self.scan_dir}"
                )
            
            idx_path = sig_path.replace("SIGNATURES_", "INDEX_")
            
            logger.info(f"Using signatures: {sig_path}")
            logger.info(f"Using index: {idx_path}")
            
            # Get repo root
            with open(sig_path, "r", encoding="utf-8") as f:
                sig_data = json.load(f)
            repo_root = sig_data.get("scan_repo_root", "")
            repo_root = repo_root.replace("\\\\", "\\")
            
            # Step 1: Directory index
            logger.info("Step 1: Building directory index...")
            if os.path.exists(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                builder = DirectoryIndexBuilder(self.db, self.scan_run.id)
                builder.load_from_index_json(idx_data, repo_root)
                dir_stats = builder.save_to_db()
                stats["directories"] = dir_stats.get("created", 0)
            
            # Step 2: Directory summaries
            logger.info("Step 2: Generating directory summaries...")
            generate_summaries_for_scan(self.db, self.scan_run.id)
            
            # Step 3: Load signatures
            logger.info("Step 3: Loading signatures...")
            loader = SignatureLoader(self.db, self.scan_run.id, repo_root)
            chunk_stats = loader.load_from_file(sig_path)
            stats["chunks"] = chunk_stats.get("chunks_created", 0)
            
            # Step 4: Generate descriptors
            logger.info("Step 4: Generating descriptors...")
            desc_count = generate_descriptors_for_scan(self.db, self.scan_run.id)
            stats["descriptors"] = desc_count
            
            # Step 5: Embed
            logger.info("Step 5: Generating embeddings...")
            embed_stats = embed_architecture_scan(
                self.db,
                self.scan_run.id,
                self.project_id,
            )
            stats["embeddings"] = (
                embed_stats.get("directories", 0) +
                embed_stats.get("chunks", 0)
            )
            
            # Complete
            self._complete_scan_run(stats)
            logger.info(f"Pipeline complete: {stats}")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            stats["errors"].append(str(e))
            if self.scan_run:
                self._fail_scan_run(str(e))
            raise
        
        return stats
    
    def _create_scan_run(self) -> ArchScanRun:
        """Create scan run record."""
        scan_run = ArchScanRun(
            status="running",
            started_at=datetime.utcnow(),
        )
        self.db.add(scan_run)
        self.db.commit()
        return scan_run
    
    def _complete_scan_run(self, stats: Dict[str, Any]):
        """Mark complete."""
        self.scan_run.status = "complete"
        self.scan_run.completed_at = datetime.utcnow()
        self.scan_run.directories_indexed = stats.get("directories", 0)
        self.scan_run.chunks_extracted = stats.get("chunks", 0)
        self.scan_run.embeddings_created = stats.get("embeddings", 0)
        self.db.commit()
    
    def _fail_scan_run(self, error: str):
        """Mark failed."""
        self.scan_run.status = "failed"
        self.scan_run.completed_at = datetime.utcnow()
        self.scan_run.error_message = error
        self.db.commit()


def run_rag_pipeline(
    db: Session,
    scan_dir: str,
    project_id: int = 0,
) -> Dict[str, Any]:
    """Convenience function."""
    return RAGPipeline(db, scan_dir, project_id).run()


def get_latest_scan_id(db: Session) -> Optional[int]:
    """Get most recent complete scan ID."""
    scan = db.query(ArchScanRun).filter(
        ArchScanRun.status == "complete"
    ).order_by(ArchScanRun.id.desc()).first()
    
    return scan.id if scan else None

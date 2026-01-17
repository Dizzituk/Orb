"""
Signature loader.

Loads code chunks from zobie SIGNATURES_*.json (timestamped).
No AST parsing - zobie already did that.
"""

import os
import json
import glob
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session

from app.rag.models import ArchCodeChunk, ChunkType
from app.rag.utils.canonical_paths import canonicalize_path


def find_latest_signatures_file(scan_dir: str) -> Optional[str]:
    """Find most recent SIGNATURES_*.json file."""
    pattern = os.path.join(scan_dir, "SIGNATURES_*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def map_kind_to_chunk_type(kind: str) -> str:
    """Map zobie 'kind' to ChunkType."""
    mapping = {
        "function": ChunkType.FUNCTION,
        "async_function": ChunkType.ASYNC_FUNCTION,
        "class": ChunkType.CLASS,
        "method": ChunkType.METHOD,
        "async_method": ChunkType.ASYNC_METHOD,
    }
    return mapping.get(kind, kind)


class SignatureLoader:
    """Load signatures from zobie output."""
    
    def __init__(
        self,
        db: Session,
        scan_id: int,
        repo_root: str = None,
    ):
        self.db = db
        self.scan_id = scan_id
        self.repo_root = repo_root
    
    def load_from_file(
        self,
        signatures_path: str,
        index_path: str = None,
    ) -> Dict[str, int]:
        """
        Load signatures from SIGNATURES_*.json.
        
        Returns:
            {chunks_created, files_processed, skipped}
        """
        stats = {
            "chunks_created": 0,
            "files_processed": 0,
            "skipped": 0,
        }
        
        # Load data
        with open(signatures_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Get repo root
        if not self.repo_root:
            self.repo_root = data.get("scan_repo_root", "")
            self.repo_root = self.repo_root.replace("\\\\", "\\")
        
        # Process files
        by_file = data.get("by_file", {})
        
        for file_path, signatures in by_file.items():
            file_stats = self._process_file(file_path, signatures)
            stats["chunks_created"] += file_stats["created"]
            stats["skipped"] += file_stats["skipped"]
            stats["files_processed"] += 1
        
        self.db.commit()
        return stats
    
    def _process_file(
        self,
        file_path: str,
        signatures: List[Dict],
    ) -> Dict[str, int]:
        """Process signatures for one file."""
        stats = {"created": 0, "skipped": 0}
        
        # Build absolute path
        if self.repo_root and not os.path.isabs(file_path):
            abs_path = os.path.join(self.repo_root, file_path)
        else:
            abs_path = file_path
        
        # Canonicalize
        try:
            canonical, alias, kind, zone = canonicalize_path(abs_path)
        except ValueError:
            stats["skipped"] = len(signatures)
            return stats
        
        # Process each signature
        for sig in signatures:
            chunk = self._create_chunk(canonical, abs_path, sig)
            if chunk:
                self.db.add(chunk)
                stats["created"] += 1
            else:
                stats["skipped"] += 1
        
        return stats
    
    def _create_chunk(
        self,
        file_canonical: str,
        file_abs: str,
        sig: Dict,
    ) -> Optional[ArchCodeChunk]:
        """Create chunk from signature data."""
        name = sig.get("name", "")
        if not name:
            return None
        
        kind = sig.get("kind", "function")
        chunk_type = map_kind_to_chunk_type(kind)
        
        # Build full signature string
        signature = sig.get("signature", "")
        if signature:
            prefix = "async def" if "async" in kind else "def"
            if chunk_type == ChunkType.CLASS:
                prefix = "class"
            signature = f"{prefix} {name}{signature}"
        
        return ArchCodeChunk(
            scan_id=self.scan_id,
            file_path=file_canonical,
            file_abs_path=file_abs,
            chunk_type=chunk_type,
            chunk_name=name,
            qualified_name=name,
            start_line=sig.get("line"),
            end_line=sig.get("end_line"),
            signature=signature,
            docstring=sig.get("docstring"),
            decorators_json=json.dumps(sig.get("decorators") or []),
            parameters_json=json.dumps(sig.get("parameters") or []),
            returns=sig.get("returns"),
            bases_json=json.dumps(sig.get("bases") or []),
        )

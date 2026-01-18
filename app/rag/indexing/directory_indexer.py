"""
Directory index builder.

Builds directory hierarchy from INDEX_*.json (timestamped).
"""

import os
import json
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session

from app.rag.models import ArchDirectoryIndex
from app.rag.utils.canonical_paths import (
    canonicalize_path,
    get_canonical_directory,
)


class DirectoryIndexBuilder:
    """Build directory index from file list."""
    
    def __init__(self, db: Session, scan_id: int):
        self.db = db
        self.scan_id = scan_id
        self.directories: Dict[str, Dict] = {}
        self.id_map: Dict[str, int] = {}
    
    def load_from_index_json(
        self,
        index_data: Dict[str, Any],
        repo_root: str = None,
    ) -> int:
        """
        Load directory structure from INDEX_*.json.
        
        Returns number of directories created.
        """
        for file_info in index_data.get("scanned_files", []):
            self._add_file(file_info, repo_root)
        
        return len(self.directories)
    
    def _add_file(self, file_info: Dict, repo_root: str):
        """Process one file entry."""
        file_path = file_info.get("path", "")
        if not file_path:
            return
        
        # Build absolute path
        if repo_root and not os.path.isabs(file_path):
            abs_path = os.path.join(repo_root, file_path)
        else:
            abs_path = file_path
        
        # Canonicalize
        try:
            canonical, alias, kind, zone = canonicalize_path(abs_path)
        except ValueError:
            return  # Skip non-scannable
        
        # Get directory
        dir_canonical = get_canonical_directory(canonical)
        
        # Ensure chain exists
        self._ensure_directory_chain(dir_canonical, alias, kind, zone)
        
        # Update aggregates
        self.directories[dir_canonical]["file_count"] += 1
        self.directories[dir_canonical]["total_lines"] += file_info.get("lines", 0)
        self.directories[dir_canonical]["total_bytes"] += file_info.get("bytes", 0)
        
        # Track extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext:
            exts = self.directories[dir_canonical].get("extensions", {})
            exts[ext] = exts.get(ext, 0) + 1
            self.directories[dir_canonical]["extensions"] = exts
    
    def _ensure_directory_chain(
        self,
        canonical: str,
        alias: str,
        kind: str,
        zone: str,
    ):
        """Ensure directory and parents exist."""
        if canonical in self.directories:
            return
        
        # Get parent
        if "/" in canonical.split(":", 1)[1]:
            parent = get_canonical_directory(canonical)
            self._ensure_directory_chain(parent, alias, kind, zone)
        else:
            parent = None
        
        # Get name
        parts = canonical.split("/")
        name = parts[-1] if len(parts) > 1 else alias
        
        self.directories[canonical] = {
            "canonical_path": canonical,
            "name": name,
            "root_alias": alias,
            "root_kind": kind,
            "zone": zone,
            "parent_canonical": parent,
            "depth": canonical.count("/"),
            "file_count": 0,
            "total_lines": 0,
            "total_bytes": 0,
            "extensions": {},
        }
    
    def save_to_db(self) -> Dict[str, int]:
        """
        Save directories to database.
        
        Uses upsert pattern for idempotency.
        """
        stats = {"created": 0, "updated": 0}
        
        # Sort by depth (parents first)
        sorted_dirs = sorted(
            self.directories.items(),
            key=lambda x: x[1].get("depth", 0)
        )
        
        for canonical, info in sorted_dirs:
            # Get parent ID
            parent_id = self.id_map.get(info.get("parent_canonical"))
            
            # Upsert
            existing = self.db.query(ArchDirectoryIndex).filter_by(
                scan_id=self.scan_id,
                canonical_path=canonical,
            ).first()
            
            if existing:
                existing.file_count = info["file_count"]
                existing.total_lines = info["total_lines"]
                existing.total_bytes = info["total_bytes"]
                existing.extensions_json = json.dumps(info.get("extensions", {}))
                existing.parent_id = parent_id
                self.id_map[canonical] = existing.id
                stats["updated"] += 1
            else:
                record = ArchDirectoryIndex(
                    scan_id=self.scan_id,
                    canonical_path=canonical,
                    name=info["name"],
                    root_alias=info["root_alias"],
                    root_kind=info["root_kind"],
                    zone=info["zone"],
                    parent_id=parent_id,
                    depth=info["depth"],
                    file_count=info["file_count"],
                    total_lines=info["total_lines"],
                    total_bytes=info["total_bytes"],
                    extensions_json=json.dumps(info.get("extensions", {})),
                )
                self.db.add(record)
                self.db.flush()
                self.id_map[canonical] = record.id
                stats["created"] += 1
        
        self.db.commit()
        return stats
    
    def get_directory_id(self, canonical: str) -> Optional[int]:
        """Get database ID for canonical path."""
        return self.id_map.get(canonical)

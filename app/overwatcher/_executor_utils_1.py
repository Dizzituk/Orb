from __future__ import annotations
import logging
import os
import subprocess
from app.overwatcher.schemas import Chunk
from pathlib import Path
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


IMPLEMENTER_PROVIDER = os.getenv("ORB_IMPLEMENTER_PROVIDER", "anthropic")

IMPLEMENTER_MODEL = os.getenv("ORB_IMPLEMENTER_MODEL", "claude-sonnet-4-5-20250514")

IMPLEMENTER_FALLBACK_PROVIDER = os.getenv("ORB_IMPLEMENTER_FALLBACK_PROVIDER", "openai")

IMPLEMENTER_FALLBACK_MODEL = os.getenv("ORB_IMPLEMENTER_FALLBACK_MODEL", "gpt-5.2-thinking")

def get_git_changes(repo_path: str, base_ref: str = "HEAD") -> Tuple[List[str], List[str], List[str]]:
    """Get file changes from git.
    
    Returns (added, modified, deleted)
    """
    from .executor import parse_git_diff_stat
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", base_ref],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            logger.warning(f"[executor] git diff failed: {result.stderr}")
            return [], [], []
        
        return parse_git_diff_stat(result.stdout)
        
    except Exception as e:
        logger.warning(f"[executor] Failed to get git changes: {e}")
        return [], [], []

def get_working_tree_changes(repo_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Get uncommitted changes from working tree.
    
    Returns (added, modified, deleted)
    """
    from .executor import parse_git_diff_stat
    try:
        # Get staged changes
        staged = subprocess.run(
            ["git", "diff", "--name-status", "--cached"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Get unstaged changes
        unstaged = subprocess.run(
            ["git", "diff", "--name-status"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Get untracked files
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        added = []
        modified = []
        deleted = []
        
        # Parse staged
        a, m, d = parse_git_diff_stat(staged.stdout)
        added.extend(a)
        modified.extend(m)
        deleted.extend(d)
        
        # Parse unstaged
        a, m, d = parse_git_diff_stat(unstaged.stdout)
        added.extend(a)
        modified.extend(m)
        deleted.extend(d)
        
        # Add untracked as added
        for line in untracked.stdout.strip().split("\n"):
            if line.strip():
                added.append(line.strip())
        
        return added, modified, deleted
        
    except Exception as e:
        logger.warning(f"[executor] Failed to get working tree changes: {e}")
        return [], [], []

def create_backup(
    chunk: Chunk,
    repo_path: str,
    backup_dir: str,
) -> Dict[str, str]:
    """Create backup of files that will be modified.
    
    Returns dict of path -> original content
    """
    backups = {}
    
    for path in chunk.allowed_files.get("modify", []):
        full_path = Path(repo_path) / path
        if full_path.exists():
            try:
                backups[path] = full_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"[executor] Failed to backup {path}: {e}")
    
    # Also store in backup directory
    if backup_dir:
        backup_path = Path(backup_dir) / chunk.chunk_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for path, content in backups.items():
            bp = backup_path / path.replace("/", "_").replace("\\", "_")
            bp.write_text(content, encoding="utf-8")
    
    return backups

def rollback_chunk(
    chunk: Chunk,
    repo_path: str,
    backups: Dict[str, str],
    files_added: List[str],
) -> bool:
    """Rollback a chunk implementation.
    
    Args:
        chunk: The chunk that was implemented
        repo_path: Path to repository
        backups: Original content of modified files
        files_added: Files that were added (to delete)
    
    Returns:
        True if rollback succeeded
    """
    success = True
    
    # Restore modified files
    for path, content in backups.items():
        try:
            full_path = Path(repo_path) / path
            full_path.write_text(content, encoding="utf-8")
            logger.info(f"[executor] Restored: {path}")
        except Exception as e:
            logger.error(f"[executor] Failed to restore {path}: {e}")
            success = False
    
    # Delete added files
    for path in files_added:
        try:
            full_path = Path(repo_path) / path
            if full_path.exists():
                full_path.unlink()
                logger.info(f"[executor] Deleted: {path}")
        except Exception as e:
            logger.error(f"[executor] Failed to delete {path}: {e}")
            success = False
    
    return success

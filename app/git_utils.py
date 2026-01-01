# FILE: app/git_utils.py
"""
Git utilities for ASTRA.

Provides read-only access to git repository state.
Used for spec provenance, architecture tracking, and index updates.

INVARIANT: All operations are READ-ONLY. No pushes, pulls, resets, or mutations.
"""
from __future__ import annotations
import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GitError(Enum):
    """Git operation error types."""
    NO_GIT_REPO = "NO_GIT_REPO"
    UNRESOLVED_COMMIT = "UNRESOLVED_COMMIT"
    INVALID_BRANCH = "INVALID_BRANCH"
    GIT_NOT_INSTALLED = "GIT_NOT_INSTALLED"


@dataclass
class GitResult:
    """Result of a git operation."""
    success: bool
    value: Optional[str] = None
    error: Optional[GitError] = None
    error_message: Optional[str] = None


def get_current_commit(
    repo_path: Optional[str] = None,
    branch: Optional[str] = None,
) -> GitResult:
    """
    Get the current commit hash for a repository.
    
    Args:
        repo_path: Path to the git repository. If None, uses current working directory.
        branch: Optional branch name to get HEAD of. If None, uses current HEAD.
    
    Returns:
        GitResult with commit hash on success, or error details on failure.
    
    Behavior:
        - Default: Returns HEAD of the currently checked-out branch
        - With branch: Returns HEAD of specified branch (does NOT switch branches)
        - Detached HEAD: Returns the commit hash
        - No .git: Returns NO_GIT_REPO error
    
    Usage:
        result = get_current_commit()
        if result.success:
            commit_hash = result.value  # e.g., "abc123def456..."
        else:
            handle_error(result.error)
    """
    # Determine repo path
    if repo_path:
        work_dir = Path(repo_path)
    else:
        work_dir = Path.cwd()
    
    # Check if .git exists
    git_dir = work_dir / ".git"
    if not git_dir.exists():
        # Check if we're in a subdirectory of a git repo
        current = work_dir
        while current != current.parent:
            if (current / ".git").exists():
                work_dir = current
                break
            current = current.parent
        else:
            return GitResult(
                success=False,
                error=GitError.NO_GIT_REPO,
                error_message=f"No .git directory found in {work_dir} or its parents",
            )
    
    # Build git command
    if branch:
        cmd = ["git", "rev-parse", branch]
    else:
        cmd = ["git", "rev-parse", "HEAD"]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            if commit_hash:
                return GitResult(success=True, value=commit_hash)
            else:
                return GitResult(
                    success=False,
                    error=GitError.UNRESOLVED_COMMIT,
                    error_message="git rev-parse returned empty output",
                )
        else:
            stderr = result.stderr.strip()
            if branch and "unknown revision" in stderr.lower():
                return GitResult(
                    success=False,
                    error=GitError.INVALID_BRANCH,
                    error_message=f"Branch '{branch}' not found: {stderr}",
                )
            return GitResult(
                success=False,
                error=GitError.UNRESOLVED_COMMIT,
                error_message=f"git rev-parse failed: {stderr}",
            )
    
    except FileNotFoundError:
        return GitResult(
            success=False,
            error=GitError.GIT_NOT_INSTALLED,
            error_message="git command not found - is git installed?",
        )
    except subprocess.TimeoutExpired:
        return GitResult(
            success=False,
            error=GitError.UNRESOLVED_COMMIT,
            error_message="git rev-parse timed out after 10 seconds",
        )
    except Exception as e:
        return GitResult(
            success=False,
            error=GitError.UNRESOLVED_COMMIT,
            error_message=f"Unexpected error: {str(e)}",
        )


def get_current_branch(repo_path: Optional[str] = None) -> GitResult:
    """
    Get the current branch name.
    
    Returns "(detached)" if in detached HEAD state.
    """
    if repo_path:
        work_dir = Path(repo_path)
    else:
        work_dir = Path.cwd()
    
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            branch = result.stdout.strip()
            if not branch:
                # Detached HEAD state
                return GitResult(success=True, value="(detached)")
            return GitResult(success=True, value=branch)
        else:
            return GitResult(
                success=False,
                error=GitError.UNRESOLVED_COMMIT,
                error_message=f"git branch --show-current failed: {result.stderr.strip()}",
            )
    
    except FileNotFoundError:
        return GitResult(
            success=False,
            error=GitError.GIT_NOT_INSTALLED,
            error_message="git command not found",
        )
    except Exception as e:
        return GitResult(
            success=False,
            error=GitError.UNRESOLVED_COMMIT,
            error_message=f"Unexpected error: {str(e)}",
        )


def get_commit_short(commit_hash: str, length: int = 7) -> str:
    """Get short form of commit hash."""
    return commit_hash[:length] if commit_hash else ""


def is_git_repo(path: Optional[str] = None) -> bool:
    """Check if path is inside a git repository."""
    result = get_current_commit(repo_path=path)
    return result.success

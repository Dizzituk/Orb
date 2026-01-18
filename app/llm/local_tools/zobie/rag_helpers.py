# FILE: app/llm/local_tools/zobie/rag_helpers.py
"""RAG-related helpers for generating INDEX and SIGNATURES JSON, and CODEBASE.md.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same generation behavior.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from .signature_extract import (
    extract_python_signatures,
    extract_js_signatures,
    strip_line_numbers,
)


def generate_signatures_json(
    contents_data: List[Dict[str, Any]],
    repo_root: str,
) -> Dict[str, Any]:
    """
    Generate SIGNATURES JSON in format expected by RAG pipeline.
    
    Expected format:
    {
        "scan_repo_root": "D:\\Orb",
        "by_file": {
            "path/to/file.py": [
                {"name": ..., "kind": ..., "signature": ..., ...}
            ]
        }
    }
    """
    by_file: Dict[str, List[Dict]] = {}
    
    for content_info in contents_data:
        path = content_info.get("path", "")
        content = content_info.get("content", "")
        language = content_info.get("language", "")
        
        if not path or not content:
            continue
        
        if content_info.get("error"):
            continue
        
        # Strip line numbers from content before signature extraction
        # (sandbox_controller returns content with line numbers when include_line_numbers=True)
        raw_content = strip_line_numbers(content)
        
        # Extract signatures based on language
        signatures = []
        ext = os.path.splitext(path)[1].lower()
        
        if ext in (".py", ".pyw", ".pyi") or language == "python":
            signatures = extract_python_signatures(raw_content, path)
        elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs") or language in ("javascript", "typescript"):
            signatures = extract_js_signatures(raw_content, path)
        
        if signatures:
            by_file[path] = signatures
    
    return {
        "scan_repo_root": repo_root,
        "by_file": by_file,
        "total_files": len(by_file),
        "total_signatures": sum(len(sigs) for sigs in by_file.values()),
    }


def generate_index_for_rag(
    files_data: List[Dict[str, Any]],
    contents_data: List[Dict[str, Any]],
    repo_root: str,
) -> Dict[str, Any]:
    """
    Generate INDEX JSON in format expected by RAG pipeline.
    
    Expected format:
    {
        "scanned_files": [
            {"path": "...", "lines": N, "bytes": M}
        ]
    }
    """
    # Build content lookup
    content_by_path = {c.get("path", ""): c for c in contents_data if c.get("path")}
    
    scanned_files = []
    for f in files_data:
        path = f.get("path", "")
        content_info = content_by_path.get(path, {})
        
        scanned_files.append({
            "path": path,
            "lines": content_info.get("line_count", 0),
            "bytes": content_info.get("size_bytes") or f.get("size_bytes", 0),
            "language": content_info.get("language", ""),
        })
    
    return {
        "scan_repo_root": repo_root,
        "scanned_files": scanned_files,
        "total_files": len(scanned_files),
    }


def generate_codebase_md(
    files_data: List[Dict[str, Any]],
    contents_data: List[Dict[str, Any]],
) -> str:
    """
    Generate CODEBASE.md with all source code and line numbers.
    
    Output format:
    # CODEBASE SNAPSHOT
    Generated: 2026-01-07T22:00:00Z
    Files: 420 | Lines: 50000 | Size: 2.5MB
    
    ---
    
    ## D:\Orb\main.py
    **Language:** python | **Lines:** 245 | **Size:** 7.5KB
    ```python
      1: from fastapi import FastAPI
      2: import logging
      ...
    ```
    
    ---
    """
    # Build content lookup
    content_by_path: Dict[str, Dict[str, Any]] = {}
    for c in contents_data:
        path = c.get("path", "")
        if path:
            content_by_path[path] = c
    
    # Stats
    total_files = len(files_data)
    total_lines = sum(c.get("line_count", 0) for c in contents_data if not c.get("error"))
    total_bytes = sum(c.get("size_bytes", 0) for c in contents_data if not c.get("error"))
    files_with_content = sum(1 for c in contents_data if c.get("content") and not c.get("error"))
    
    lines = [
        "# CODEBASE SNAPSHOT",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Files:** {total_files} | **With Content:** {files_with_content} | **Lines:** {total_lines:,} | **Size:** {total_bytes / 1024 / 1024:.2f}MB",
        "",
        "---",
        "",
    ]
    
    # Group by root then sort by path
    by_root: Dict[str, List[Dict]] = {}
    for f in files_data:
        root = f.get("root", "")
        if root not in by_root:
            by_root[root] = []
        by_root[root].append(f)
    
    for root in sorted(by_root.keys()):
        root_name = os.path.basename(root) or root
        lines.append(f"# {root_name}")
        lines.append("")
        
        # Sort files by path
        files_in_root = sorted(by_root[root], key=lambda x: x.get("path", ""))
        
        for f in files_in_root:
            path = f.get("path", "")
            name = f.get("name", "")
            
            # Get content
            content_info = content_by_path.get(path, {})
            content = content_info.get("content", "")
            language = content_info.get("language", "text")
            line_count = content_info.get("line_count", 0)
            size_bytes = content_info.get("size_bytes", 0) or f.get("size_bytes", 0)
            error = content_info.get("error")
            
            # Calculate relative path
            try:
                rel_path = path.replace(root, "").lstrip("\\/").replace("\\", "/")
            except:
                rel_path = name
            
            lines.append(f"## {rel_path}")
            
            if error:
                lines.append(f"**Error:** {error}")
                lines.append("")
            elif content:
                size_kb = size_bytes / 1024 if size_bytes else 0
                lines.append(f"**Language:** {language} | **Lines:** {line_count} | **Size:** {size_kb:.1f}KB")
                lines.append(f"```{language}")
                lines.append(content)
                lines.append("```")
                lines.append("")
            else:
                lines.append("*No content captured*")
                lines.append("")
            
            lines.append("---")
            lines.append("")
    
    return "\n".join(lines)

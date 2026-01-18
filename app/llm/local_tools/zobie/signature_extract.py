# FILE: app/llm/local_tools/zobie/signature_extract.py
"""Signature extraction for RAG pipeline.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same extraction behavior.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


def extract_python_signatures(content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Extract function and class signatures from Python source code.
    
    Returns list of signature dicts matching RAG pipeline expected format:
    {
        "name": "function_name",
        "kind": "function|async_function|class|method|async_method",
        "signature": "(arg1, arg2)",
        "line": 10,
        "end_line": 25,
        "docstring": "...",
        "decorators": ["@decorator"],
        "parameters": [],
        "returns": "str",
        "bases": ["BaseClass"]  # for classes
    }
    """
    signatures = []
    lines = content.splitlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty and comment lines
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        
        # Detect decorators
        decorators = []
        while stripped.startswith("@"):
            decorators.append(stripped)
            i += 1
            if i >= len(lines):
                break
            line = lines[i]
            stripped = line.strip()
        
        # Detect class definition
        class_match = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*:", stripped)
        if class_match:
            name = class_match.group(1)
            bases_str = class_match.group(2) or ""
            bases = []
            if bases_str:
                # Extract base class names
                bases_str = bases_str.strip("()")
                bases = [b.strip() for b in bases_str.split(",") if b.strip()]
            
            # Get docstring
            docstring = _extract_docstring(lines, i + 1)
            
            # Find end of class (crude: next dedented non-empty line or EOF)
            end_line = _find_block_end(lines, i)
            
            signatures.append({
                "name": name,
                "kind": "class",
                "signature": bases_str,
                "line": i + 1,
                "end_line": end_line,
                "docstring": docstring,
                "decorators": decorators,
                "parameters": [],
                "returns": None,
                "bases": bases,
            })
            i += 1
            continue
        
        # Detect function/method definition
        func_match = re.match(r"^(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(->\s*([^:]+))?\s*:", stripped)
        if func_match:
            is_async = bool(func_match.group(1))
            name = func_match.group(2)
            params_str = func_match.group(3) or ""
            returns = (func_match.group(5) or "").strip() if func_match.group(4) else None
            
            # Determine if method (has self/cls as first param)
            is_method = params_str.strip().startswith(("self", "cls"))
            
            if is_async:
                kind = "async_method" if is_method else "async_function"
            else:
                kind = "method" if is_method else "function"
            
            # Get docstring
            docstring = _extract_docstring(lines, i + 1)
            
            # Find end of function
            end_line = _find_block_end(lines, i)
            
            signatures.append({
                "name": name,
                "kind": kind,
                "signature": f"({params_str})",
                "line": i + 1,
                "end_line": end_line,
                "docstring": docstring,
                "decorators": decorators,
                "parameters": _parse_params(params_str),
                "returns": returns,
                "bases": [],
            })
        
        i += 1
    
    return signatures


def _extract_docstring(lines: List[str], start_idx: int) -> Optional[str]:
    """Extract docstring starting from given line index."""
    if start_idx >= len(lines):
        return None
    
    line = lines[start_idx].strip()
    
    # Single-line docstring
    if line.startswith(('"""', "'''")) and line.count('"""') >= 2:
        return line.strip('"""').strip("'''").strip()
    if line.startswith(('"""', "'''")):
        # Multi-line docstring
        quote = line[:3]
        doc_lines = [line[3:]]
        for i in range(start_idx + 1, min(start_idx + 50, len(lines))):
            doc_lines.append(lines[i])
            if quote in lines[i]:
                break
        full = "\n".join(doc_lines)
        return full.replace(quote, "").strip()[:500]  # Limit length
    
    return None


def _find_block_end(lines: List[str], start_idx: int) -> int:
    """Find end line of a Python block (class or function)."""
    if start_idx >= len(lines):
        return start_idx + 1
    
    # Get indentation of the def/class line
    start_line = lines[start_idx]
    base_indent = len(start_line) - len(start_line.lstrip())
    
    end_line = start_idx + 1
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip():  # Empty line
            end_line = i + 1
            continue
        
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= base_indent and line.strip():
            # Found a line at same or lower indentation - block ended
            break
        end_line = i + 1
    
    return end_line


def _parse_params(params_str: str) -> List[Dict[str, Any]]:
    """Parse parameter string into list of param dicts."""
    if not params_str.strip():
        return []
    
    params = []
    # Simple split - won't handle complex type annotations perfectly
    for p in params_str.split(","):
        p = p.strip()
        if not p:
            continue
        
        # Extract name and type annotation
        if ":" in p:
            name_part, type_part = p.split(":", 1)
            name = name_part.strip()
            if "=" in type_part:
                type_ann, default = type_part.split("=", 1)
                params.append({"name": name, "type": type_ann.strip(), "default": default.strip()})
            else:
                params.append({"name": name, "type": type_part.strip()})
        elif "=" in p:
            name, default = p.split("=", 1)
            params.append({"name": name.strip(), "default": default.strip()})
        else:
            params.append({"name": p})
    
    return params


def extract_js_signatures(content: str, file_path: str) -> List[Dict[str, Any]]:
    """Extract function/class signatures from JavaScript/TypeScript."""
    signatures = []
    lines = content.splitlines()
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect class
        class_match = re.match(r"^(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+extends\s+([A-Za-z_][A-Za-z0-9_]*))?\s*\{", stripped)
        if class_match:
            name = class_match.group(1)
            base = class_match.group(2)
            signatures.append({
                "name": name,
                "kind": "class",
                "signature": "",
                "line": i + 1,
                "end_line": i + 1,
                "docstring": None,
                "decorators": [],
                "parameters": [],
                "returns": None,
                "bases": [base] if base else [],
            })
            continue
        
        # Detect function
        func_match = re.match(r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", stripped)
        if func_match:
            name = func_match.group(1)
            params = func_match.group(2)
            is_async = "async" in stripped.split("function")[0]
            signatures.append({
                "name": name,
                "kind": "async_function" if is_async else "function",
                "signature": f"({params})",
                "line": i + 1,
                "end_line": i + 1,
                "docstring": None,
                "decorators": [],
                "parameters": [],
                "returns": None,
                "bases": [],
            })
            continue
        
        # Detect const arrow function
        arrow_match = re.match(r"^(?:export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*[^=]+)?\s*=>", stripped)
        if arrow_match:
            name = arrow_match.group(1)
            params = arrow_match.group(2)
            is_async = "async" in stripped.split("=")[0] or "async" in stripped.split("const")[1].split("=")[0]
            signatures.append({
                "name": name,
                "kind": "async_function" if is_async else "function",
                "signature": f"({params})",
                "line": i + 1,
                "end_line": i + 1,
                "docstring": None,
                "decorators": [],
                "parameters": [],
                "returns": None,
                "bases": [],
            })
    
    return signatures


def strip_line_numbers(content: str) -> str:
    """
    Strip line number prefixes from content.
    
    Handles formats like:
    - "1: code"
    - "  1: code" 
    - " 10: code"
    - "100: code"
    """
    lines = content.splitlines()
    stripped_lines = []
    
    for line in lines:
        # Match optional whitespace, digits, colon, optional single space
        match = re.match(r'^\s*\d+:\s?', line)
        if match:
            stripped_lines.append(line[match.end():])
        else:
            stripped_lines.append(line)
    
    return "\n".join(stripped_lines)


def map_kind_to_chunk_type(kind: str) -> str:
    """
    Map signature 'kind' to ArchCodeChunk chunk_type.
    
    Expected kinds from extract_python_signatures / extract_js_signatures:
    - function, async_function, class, method, async_method
    """
    return kind  # Direct mapping - kinds match ChunkType values

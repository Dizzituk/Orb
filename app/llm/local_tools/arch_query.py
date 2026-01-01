# FILE: D:\Orb\app\llm\local_tools\arch_query.py
"""
Architecture Query Tool - Reads signature files directly (no service).

Reads SIGNATURES_*.json from zobie_mapper output directory.
"""

import os
import re
import json
import glob
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Where zobie_mapper writes output
SIGNATURES_DIR = Path(os.getenv("ORB_SIGNATURES_DIR", "D:/tools/zobie_mapper/out"))


def _find_latest_signatures() -> Optional[Path]:
    """Find most recent SIGNATURES_*.json file."""
    pattern = str(SIGNATURES_DIR / "SIGNATURES_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return Path(max(files, key=os.path.getmtime))


def _load_signatures() -> Optional[Dict[str, Any]]:
    """Load latest signatures file."""
    path = _find_latest_signatures()
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load signatures: {e}")
        return None


def is_service_available() -> bool:
    """Check if signature data exists (kept for compatibility)."""
    return _find_latest_signatures() is not None


def get_file_signatures(file_path: str) -> Dict[str, Any]:
    """Get signatures for a specific file."""
    file_path = file_path.replace("\\", "/")
    
    data = _load_signatures()
    if not data:
        return {"error": "No signature data found. Run 'ZOBIE MAP' first."}
    
    by_file = data.get("by_file", {})
    
    # Try exact match first
    if file_path in by_file:
        return {"file": file_path, "signatures": by_file[file_path]}
    
    # Try partial match
    for path, sigs in by_file.items():
        if path.endswith(file_path) or file_path in path:
            return {"file": path, "signatures": sigs}
    
    return {"error": f"File '{file_path}' not found in signature data"}


def search_symbols(query: str, kind: str = None, limit: int = 20) -> List[Dict]:
    """Search for symbols by name."""
    data = _load_signatures()
    if not data:
        return [{"error": "No signature data found"}]
    
    query_lower = query.lower()
    results = []
    
    for file_path, sigs in data.get("by_file", {}).items():
        for sig in sigs:
            name = sig.get("name", "")
            sig_kind = sig.get("kind", "")
            
            if query_lower in name.lower():
                if kind and sig_kind != kind:
                    continue
                    
                score = 1.0 if name.lower() == query_lower else 0.5
                results.append({
                    "file": file_path,
                    "name": name,
                    "kind": sig_kind,
                    "line": sig.get("line"),
                    "signature": sig.get("signature", ""),
                    "score": score,
                })
                
                if len(results) >= limit:
                    break
        if len(results) >= limit:
            break
    
    return sorted(results, key=lambda x: -x["score"])[:limit]


def get_routes() -> Dict[str, Any]:
    """Get all API routes."""
    pattern = str(SIGNATURES_DIR / "ROUTE_MAP_*.json")
    files = glob.glob(pattern)
    if files:
        latest = max(files, key=os.path.getmtime)
        try:
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"routes": data.get("routes", [])}
        except Exception:
            pass
    return {"error": "No route data found"}


def get_enums() -> Dict[str, Any]:
    """Get all enum definitions."""
    pattern = str(SIGNATURES_DIR / "ENUM_INDEX_*.json")
    files = glob.glob(pattern)
    if files:
        latest = max(files, key=os.path.getmtime)
        try:
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"enums": data.get("enums", [])}
        except Exception:
            pass
    return {"error": "No enum data found"}


def get_structure() -> Dict[str, Any]:
    """Get project structure stats."""
    data = _load_signatures()
    if not data:
        return {"error": "No signature data found"}
    
    by_file = data.get("by_file", {})
    total_sigs = sum(len(sigs) for sigs in by_file.values())
    
    lang_counts = {}
    for path in by_file.keys():
        ext = Path(path).suffix.lower()
        lang = {".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "react", ".tsx": "react"}.get(ext, ext)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    return {
        "scanned_files_count": len(by_file),
        "signature_total_count": total_sigs,
        "language_counts": lang_counts,
    }


def format_signatures(data: Dict[str, Any]) -> str:
    """Format signature data as readable text."""
    if "error" in data:
        return f"Error: {data['error']}"
    
    file_path = data.get("file", "unknown")
    signatures = data.get("signatures", [])
    
    if not signatures:
        return f"No signatures found for {file_path}"
    
    lines = [f"**{file_path}** ({len(signatures)} signatures)\n"]
    
    for sig in signatures:
        kind = sig.get("kind", "?")
        name = sig.get("name", "?")
        line = sig.get("line", "?")
        signature = sig.get("signature", "")
        docstring = sig.get("docstring", "")
        
        if kind == "class":
            bases = sig.get("bases") or []
            bases_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"L{line}: `class {name}{bases_str}`")
            if docstring:
                lines.append(f"  → {docstring[:80]}...")
            methods = sig.get("methods") or []
            if methods:
                lines.append("  Methods:")
                for m in methods[:10]:
                    lines.append(f"    - `{m.get('name', '?')}{m.get('signature', '()')}`")
                if len(methods) > 10:
                    lines.append(f"    ... ({len(methods)} total)")
        else:
            async_prefix = "async " if kind == "async_function" else ""
            lines.append(f"L{line}: `{async_prefix}def {name}{signature}`")
            if docstring:
                lines.append(f"  → {docstring[:80]}...")
        lines.append("")
    
    return "\n".join(lines)


def format_search_results(results: List[Dict]) -> str:
    """Format search results."""
    if not results:
        return "No matches found."
    if "error" in results[0]:
        return f"Error: {results[0]['error']}"
    lines = [f"**Found {len(results)} matches:**\n"]
    for r in results:
        lines.append(f"- `{r.get('file', '?')}:{r.get('line', '?')}` **{r.get('name', '?')}**{r.get('signature', '')} ({r.get('kind', '?')})")
    return "\n".join(lines)


def query_architecture(question: str) -> str:
    """Answer architecture questions from local signature files."""
    q_lower = question.lower()
    
    # File path query
    file_match = re.search(r'([a-zA-Z0-9_/\\]+\.py)', question)
    if file_match:
        file_path = file_match.group(1)
        data = get_file_signatures(file_path)
        return format_signatures(data)
    
    # Symbol search
    search_match = re.search(r'(?:find|search|where is)\s+(?:function|class|method)?\s*["\']?(\w+)["\']?', question, re.I)
    if search_match:
        results = search_symbols(search_match.group(1))
        return format_search_results(results)
    
    # Routes
    if any(w in q_lower for w in ["route", "endpoint", "api"]):
        data = get_routes()
        if "error" in data:
            return f"Error: {data['error']}"
        routes = data.get("routes", [])
        lines = [f"**{len(routes)} API routes:**\n"]
        for r in routes[:30]:
            lines.append(f"- `{r.get('method', '?')} {r.get('path', '?')}` → {r.get('handler', '?')}")
        return "\n".join(lines)
    
    # Enums
    if "enum" in q_lower:
        data = get_enums()
        if "error" in data:
            return f"Error: {data['error']}"
        enums = data.get("enums", [])
        lines = [f"**{len(enums)} enums:**\n"]
        for e in enums[:20]:
            members = ", ".join(e.get("members", [])[:5])
            lines.append(f"- `{e.get('name', '?')}` [{members}]")
        return "\n".join(lines)
    
    # Default: structure
    data = get_structure()
    if "error" in data:
        return f"Error: {data['error']}"
    lines = ["**Project Structure:**\n"]
    lines.append(f"- Scanned files: {data.get('scanned_files_count', '?')}")
    lines.append(f"- Total signatures: {data.get('signature_total_count', '?')}")
    return "\n".join(lines)
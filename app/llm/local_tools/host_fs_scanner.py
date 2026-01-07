# FILE: D:\tools\zobie_mapper\host_fs_scanner.py
"""
Host Filesystem Scanner - Scans local filesystem directly (no sandbox controller).

Supports:
- Multiple scan roots (C:\Users\*, D:\*)
- Configurable exclusion patterns
- Same output format as zobie_map.py for compatibility

Usage:
    python host_fs_scanner.py [out_dir] [max_files] [max_scan_files]

Environment variables:
    HOST_SCAN_ROOTS: Comma-separated list of roots (default: C:\Users,D:\)
    HOST_SCAN_MAX_FILE_BYTES: Max file size to read (default: 512000)

v1.0 (2026-01): Initial implementation for Plateau 1 coverage
"""

import ast
import json
import os
import re
import sys
import shutil
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default scan roots
DEFAULT_SCAN_ROOTS = [
    r"C:\Users",
    r"D:\\",
]

# Hard excludes - never scan these
HARD_EXCLUDE_ROOTS = [
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
    r"C:\$Recycle.Bin",
    r"C:\System Volume Information",
]

# Directory patterns to skip (anywhere in path)
EXCLUDE_DIR_PATTERNS = [
    r"\.git$",
    r"\.git[/\\]",
    r"node_modules$",
    r"node_modules[/\\]",
    r"dist$",
    r"build$",
    r"\.venv$",
    r"venv$",
    r"__pycache__$",
    r"__pycache__[/\\]",
    r"\.idea$",
    r"\.vscode$",
    r"logs$",
    r"cache$",
    r"temp$",
    r"tmp$",
    r"\.pytest_cache$",
    r"\.mypy_cache$",
    r"\.ruff_cache$",
    r"\.tox$",
    r"\.nox$",
    r"\.eggs$",
    r"\.egg-info$",
    r"htmlcov$",
    r"\.coverage$",
    r"orb-electron-data$",
    r"Code Cache$",
    r"GPUCache$",
    r"Cache$",
    r"CachedData$",
    r"CachedExtensions$",
    r"CachedExtensionVSIXs$",
    # Windows system folders
    r"AppData[/\\]Local[/\\]Temp",
    r"AppData[/\\]Local[/\\]Microsoft",
    r"AppData[/\\]Local[/\\]Google[/\\]Chrome",
    r"AppData[/\\]Local[/\\]Mozilla",
    r"AppData[/\\]LocalLow",
    r"NTUSER\.DAT",
]

# File patterns to skip
EXCLUDE_FILE_PATTERNS = [
    r"\.log$",
    r"\.iso$",
    r"\.vhd$",
    r"\.vhdx$",
    r"\.qcow2$",
    r"\.img$",
    r"\.zip$",
    r"\.7z$",
    r"\.rar$",
    r"\.tar$",
    r"\.gz$",
    r"\.bz2$",
    r"\.xz$",
    r"\.sqlite\d*$",
    r"\.db$",
    r"\.wal$",
    r"\.shm$",
    r"\.dll$",
    r"\.exe$",
    r"\.msi$",
    r"\.sys$",
    r"\.bin$",
    r"\.dat$",
    r"\.pdb$",
    r"\.obj$",
    r"\.o$",
    r"\.a$",
    r"\.so$",
    r"\.dylib$",
    r"\.pyc$",
    r"\.pyo$",
    r"\.class$",
    r"\.jar$",
    r"\.war$",
    # Large media
    r"\.mp4$",
    r"\.mkv$",
    r"\.avi$",
    r"\.mov$",
    r"\.wmv$",
    r"\.mp3$",
    r"\.wav$",
    r"\.flac$",
    # Images (keep small ones but skip large)
    r"\.psd$",
    r"\.xcf$",
    r"\.raw$",
    r"\.cr2$",
    r"\.nef$",
]

# Sensitive file patterns - never read content
DENY_FILE_PATTERNS = [
    r"(^|[/\\])\.env($|[/\\])",
    r"\.pem$",
    r"\.key$",
    r"\.pfx$",
    r"\.p12$",
    r"\.p8$",
    r"(^|[/\\])id_rsa($|[/\\])",
    r"(^|[/\\])id_ed25519($|[/\\])",
    r"(^|[/\\])known_hosts($|[/\\])",
    r"(^|[/\\])secrets?[/\\]",
    r"(^|[/\\])credentials?[/\\]",
    r"(^|[/\\])\.ssh[/\\]",
    r"(^|[/\\])\.gnupg[/\\]",
    r"(^|[/\\])\.aws[/\\]",
]

# Extensions to scan for code structure
CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".json", ".md", ".markdown",
    ".toml", ".yml", ".yaml",
    ".txt", ".ini", ".cfg", ".conf",
    ".html", ".htm", ".css", ".scss", ".less",
    ".sql", ".sh", ".bash", ".ps1", ".bat", ".cmd",
    ".xml", ".xsl", ".xslt",
    ".rs", ".go", ".java", ".kt", ".scala",
    ".c", ".cpp", ".h", ".hpp",
    ".cs", ".fs", ".vb",
    ".rb", ".php", ".pl", ".pm",
    ".swift", ".m", ".mm",
    ".r", ".R", ".jl",
    ".lua", ".vim", ".el",
    ".dockerfile", ".containerfile",
    ".makefile", ".cmake",
    ".gitignore", ".gitattributes",
    ".env.example", ".env.sample", ".env.template",
}

# Max file size to read content (512KB default)
MAX_FILE_BYTES = int(os.getenv("HOST_SCAN_MAX_FILE_BYTES", "512000"))

# Limits
DEFAULT_MAX_FILES = 500000  # Max files in tree
DEFAULT_MAX_SCAN_FILES = 500  # Max files to extract content from


# =============================================================================
# PATTERN COMPILATION
# =============================================================================

def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            pass
    return compiled

_EXCLUDE_DIR_RX = _compile_patterns(EXCLUDE_DIR_PATTERNS)
_EXCLUDE_FILE_RX = _compile_patterns(EXCLUDE_FILE_PATTERNS)
_DENY_FILE_RX = _compile_patterns(DENY_FILE_PATTERNS)


def _is_hard_excluded(path: str) -> bool:
    """Check if path starts with a hard-excluded root."""
    path_lower = path.lower().replace("/", "\\")
    for excl in HARD_EXCLUDE_ROOTS:
        excl_lower = excl.lower().replace("/", "\\")
        if path_lower.startswith(excl_lower):
            return True
    return False


def _is_excluded_dir(path: str) -> bool:
    """Check if any directory in path matches exclusion patterns."""
    path_norm = path.replace("\\", "/")
    for rx in _EXCLUDE_DIR_RX:
        if rx.search(path_norm):
            return True
    return False


def _is_excluded_file(path: str) -> bool:
    """Check if file matches exclusion patterns."""
    path_norm = path.replace("\\", "/")
    for rx in _EXCLUDE_FILE_RX:
        if rx.search(path_norm):
            return True
    return False


def _is_denied_path(path: str) -> bool:
    """Check if path is sensitive (never read content)."""
    path_norm = path.replace("\\", "/")
    for rx in _DENY_FILE_RX:
        if rx.search(path_norm):
            return True
    return False


def _is_scannable_file(path: str) -> bool:
    """Check if file should have content extracted."""
    ext = os.path.splitext(path.lower())[1]
    if ext in CODE_EXTENSIONS:
        return True
    # Also scan files with no extension if they look like config/scripts
    basename = os.path.basename(path).lower()
    if basename in {"dockerfile", "makefile", "gemfile", "rakefile", "procfile",
                    "vagrantfile", "brewfile", "justfile", "taskfile"}:
        return True
    return False


# =============================================================================
# EXTRACTION HELPERS (from zobie_map.py)
# =============================================================================

def _detect_language(path: str) -> str:
    p = path.lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith(".ts") or p.endswith(".tsx"):
        return "typescript"
    if p.endswith(".js") or p.endswith(".jsx"):
        return "javascript"
    if p.endswith(".json"):
        return "json"
    if p.endswith(".md") or p.endswith(".markdown"):
        return "markdown"
    if p.endswith(".yml") or p.endswith(".yaml"):
        return "yaml"
    if p.endswith(".txt"):
        return "text"
    if p.endswith(".html") or p.endswith(".htm"):
        return "html"
    if p.endswith(".css") or p.endswith(".scss"):
        return "css"
    if p.endswith(".sql"):
        return "sql"
    if p.endswith(".sh") or p.endswith(".bash"):
        return "shell"
    if p.endswith(".ps1"):
        return "powershell"
    if p.endswith(".bat") or p.endswith(".cmd"):
        return "batch"
    return "other"


def _redact_secrets(text: str) -> str:
    """Redact obvious API keys and secrets."""
    if not text:
        return text
    patterns = [
        (r"(sk-[A-Za-z0-9]{10,})", "sk-REDACTED"),
        (r"(AIza[0-9A-Za-z\-_]{10,})", "AIzaREDACTED"),
        (r"(?i)(OPENAI_API_KEY\s*=\s*)(\S+)", r"\1REDACTED"),
        (r"(?i)(ANTHROPIC_API_KEY\s*=\s*)(\S+)", r"\1REDACTED"),
        (r"(?i)(GOOGLE_API_KEY\s*=\s*)(\S+)", r"\1REDACTED"),
        (r"(?i)(API_KEY\s*=\s*)(\S+)", r"\1REDACTED"),
        (r"(?i)(SECRET\s*=\s*)(\S+)", r"\1REDACTED"),
        (r"(?i)(PASSWORD\s*=\s*)(\S+)", r"\1REDACTED"),
    ]
    out = text
    for pat, rep in patterns:
        out = re.sub(pat, rep, out)
    return out


def _extract_python_imports(content: str) -> List[str]:
    imports = set()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m1 = re.match(r"import\s+([A-Za-z0-9_\.]+)", line)
        if m1:
            imports.add(m1.group(1))
            continue
        m2 = re.match(r"from\s+([A-Za-z0-9_\.]+)\s+import", line)
        if m2:
            imports.add(m2.group(1))
    return sorted(imports)


def _extract_js_imports(content: str) -> List[str]:
    imports = set()
    for line in content.splitlines():
        line = line.strip()
        m1 = re.search(r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]", line)
        if m1:
            imports.add(m1.group(1))
        m2 = re.search(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", line)
        if m2:
            imports.add(m2.group(1))
    return sorted(imports)


def _extract_symbols(lang: str, content: str, max_symbols: int = 200) -> List[dict]:
    symbols = []
    lines = content.splitlines()
    
    if lang == "python":
        for i, line in enumerate(lines, start=1):
            m_class = re.match(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if m_class:
                symbols.append({"kind": "class", "name": m_class.group(1), "line": i})
                if len(symbols) >= max_symbols:
                    break
            m_def = re.match(r"^\s*(async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if m_def:
                symbols.append({"kind": "function", "name": m_def.group(2), "line": i})
                if len(symbols) >= max_symbols:
                    break
                    
    elif lang in ("javascript", "typescript"):
        for i, line in enumerate(lines, start=1):
            m_class = re.match(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if m_class:
                symbols.append({"kind": "class", "name": m_class.group(1), "line": i})
                if len(symbols) >= max_symbols:
                    break
            m_fn = re.match(r"^\s*(export\s+)?(async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if m_fn:
                symbols.append({"kind": "function", "name": m_fn.group(3), "line": i})
                if len(symbols) >= max_symbols:
                    break
            m_const = re.match(r"^\s*(export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(async\s*)?\(", line)
            if m_const:
                symbols.append({"kind": "function_like", "name": m_const.group(2), "line": i})
                if len(symbols) >= max_symbols:
                    break
    
    return symbols


def _extract_routes(content: str, max_routes: int = 200) -> List[dict]:
    """Extract FastAPI/Flask routes."""
    routes = []
    lines = content.splitlines()
    
    for i, line in enumerate(lines, start=1):
        # FastAPI style
        m = re.match(r'^\s*@(?:app|router)\.(\w+)\s*\(\s*["\']([^"\']+)["\']', line)
        if m:
            routes.append({
                "method": m.group(1).upper(),
                "path": m.group(2),
                "line": i,
            })
            if len(routes) >= max_routes:
                break
        # Flask style
        m2 = re.match(r'^\s*@(?:app|bp)\.route\s*\(\s*["\']([^"\']+)["\']', line)
        if m2:
            routes.append({
                "method": "GET",
                "path": m2.group(1),
                "line": i,
            })
            if len(routes) >= max_routes:
                break
    
    return routes


def _extract_enums(content: str) -> List[dict]:
    """Extract Python enum definitions."""
    enums = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [
                    (b.id if isinstance(b, ast.Name) else 
                     b.attr if isinstance(b, ast.Attribute) else str(b))
                    for b in node.bases
                ]
                if any("Enum" in str(b) for b in bases):
                    members = []
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    members.append(target.id)
                    enums.append({
                        "name": node.name,
                        "line": node.lineno,
                        "base": bases[0] if bases else "Enum",
                        "members": members,
                        "member_count": len(members),
                    })
    except Exception:
        pass
    return enums


# =============================================================================
# FILESYSTEM WALKER
# =============================================================================

def scan_filesystem(
    roots: List[str],
    max_files: int = DEFAULT_MAX_FILES,
    max_scan_files: int = DEFAULT_MAX_SCAN_FILES,
    progress_callback=None,
) -> Tuple[List[dict], List[dict]]:
    """
    Walk filesystem from given roots, return (tree_entries, scanned_files).
    
    tree_entries: All discovered files with basic metadata
    scanned_files: Files with extracted content/structure
    """
    tree_entries = []
    seen_paths = set()
    
    # Phase 1: Build file tree
    file_count = 0
    
    for root in roots:
        if not os.path.exists(root):
            continue
            
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            # Check hard excludes
            if _is_hard_excluded(dirpath):
                dirnames[:] = []
                continue
                
            # Filter out excluded directories (modifies in place for efficiency)
            dirnames[:] = [d for d in dirnames 
                          if not _is_excluded_dir(os.path.join(dirpath, d))
                          and not d.startswith(".")]
            
            for fname in filenames:
                if file_count >= max_files:
                    break
                    
                fpath = os.path.join(dirpath, fname)
                
                # Skip if already seen (symlinks can cause duplicates)
                fpath_lower = fpath.lower()
                if fpath_lower in seen_paths:
                    continue
                seen_paths.add(fpath_lower)
                
                # Skip excluded files
                if _is_excluded_file(fpath):
                    continue
                if fname.startswith(".") and fname not in {".env.example", ".gitignore", ".gitattributes"}:
                    continue
                    
                try:
                    stat = os.stat(fpath)
                    size = stat.st_size
                    mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
                except Exception:
                    continue
                    
                # Normalize path for consistent output
                rel_path = fpath.replace("\\", "/")
                
                tree_entries.append({
                    "path": rel_path,
                    "size": size,
                    "mtime": mtime,
                })
                file_count += 1
                
                if progress_callback and file_count % 1000 == 0:
                    progress_callback(f"Discovered {file_count} files...")
            
            if file_count >= max_files:
                break
        
        if file_count >= max_files:
            break
    
    if progress_callback:
        progress_callback(f"Tree complete: {len(tree_entries)} files")
    
    # Phase 2: Score and select files for content extraction
    scored = []
    for entry in tree_entries:
        path = entry["path"]
        size = entry["size"]
        
        if not _is_scannable_file(path):
            continue
        if _is_denied_path(path):
            continue
        if size > MAX_FILE_BYTES:
            continue
        if size == 0:
            continue
            
        # Score: prefer smaller files, code files, config files
        score = 100
        ext = os.path.splitext(path.lower())[1]
        
        # Boost important file types
        if ext == ".py":
            score += 50
        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            score += 40
        elif ext in {".json", ".toml", ".yaml", ".yml"}:
            score += 30
        elif ext == ".md":
            score += 20
            
        # Boost important paths
        path_lower = path.lower()
        if "/app/" in path_lower or "/src/" in path_lower:
            score += 30
        if "main" in path_lower or "router" in path_lower:
            score += 20
        if "config" in path_lower or "settings" in path_lower:
            score += 15
        if "test" in path_lower:
            score += 10
            
        # Penalize large files
        if size > 100000:
            score -= 20
        elif size > 50000:
            score -= 10
            
        scored.append({
            "path": path,
            "size": size,
            "score": score,
        })
    
    # Sort by score (highest first) and take top N
    scored.sort(key=lambda x: -x["score"])
    scan_list = scored[:max_scan_files]
    
    if progress_callback:
        progress_callback(f"Selected {len(scan_list)} files for content extraction")
    
    # Phase 3: Extract content from selected files
    scanned_files = []
    
    for i, item in enumerate(scan_list):
        path = item["path"]
        
        if progress_callback and i % 50 == 0:
            progress_callback(f"Scanning {i}/{len(scan_list)}: {os.path.basename(path)}")
        
        try:
            # Convert back to OS path for reading
            os_path = path.replace("/", os.sep)
            with open(os_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_FILE_BYTES)
        except Exception:
            continue
        
        content = _redact_secrets(content)
        lang = _detect_language(path)
        
        # Extract structure
        if lang == "python":
            imports = _extract_python_imports(content)
        elif lang in ("javascript", "typescript"):
            imports = _extract_js_imports(content)
        else:
            imports = []
        
        symbols = _extract_symbols(lang, content)
        routes = _extract_routes(content) if lang == "python" else []
        enums = _extract_enums(content) if lang == "python" else []
        
        scanned_files.append({
            "path": path,
            "bytes": item["size"],
            "language": lang,
            "imports": imports,
            "symbols": symbols,
            "routes": routes,
            "enums": enums,
        })
    
    if progress_callback:
        progress_callback(f"Extraction complete: {len(scanned_files)} files scanned")
    
    return tree_entries, scanned_files


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_output(
    tree_entries: List[dict],
    scanned_files: List[dict],
    out_dir: str,
    roots: List[str],
) -> Dict[str, str]:
    """Generate output files compatible with zobie_map format."""
    
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Build aggregates
    top_level = defaultdict(int)
    for entry in tree_entries:
        path = entry["path"]
        # Get drive letter or first directory
        parts = path.replace("\\", "/").split("/")
        if parts:
            root_key = parts[0]
            if len(root_key) == 2 and root_key[1] == ":":
                root_key = root_key.upper()
            top_level[root_key] += 1
    
    # Count by language
    lang_counts = defaultdict(int)
    total_symbols = 0
    total_routes = 0
    total_enums = 0
    
    for sf in scanned_files:
        lang_counts[sf.get("language", "other")] += 1
        total_symbols += len(sf.get("symbols", []))
        total_routes += len(sf.get("routes", []))
        total_enums += len(sf.get("enums", []))
    
    # Output files
    outputs = {}
    
    # INDEX (main output - compatible with zobie_tools.py)
    index_path = os.path.join(out_dir, f"INDEX_{stamp}.json")
    index_data = {
        "scanner": "host_fs_scanner",
        "version": "1.0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "roots": roots,
        "repo_root": roots[0] if roots else "",
        "scan_repo_root": ", ".join(roots),
        "tree_files_count": len(tree_entries),
        "scanned_files_count": len(scanned_files),
        "top_level_counts": dict(top_level),
        "language_counts": dict(lang_counts),
        "total_symbols": total_symbols,
        "total_routes": total_routes,
        "total_enums": total_enums,
        "scanned_files": scanned_files,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
    outputs["index"] = index_path
    
    # TREE (file listing)
    tree_path = os.path.join(out_dir, f"TREE_{stamp}.json")
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "roots": roots,
            "entries": tree_entries,
        }, f, indent=2)
    outputs["tree"] = tree_path
    
    # REPO_TREE.txt (human readable)
    tree_txt_path = os.path.join(out_dir, f"REPO_TREE_{stamp}.txt")
    with open(tree_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Host Filesystem Scan\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Roots: {', '.join(roots)}\n\n")
        f.write(f"Top-level breakdown:\n")
        for k in sorted(top_level.keys()):
            f.write(f"  {k}: {top_level[k]} files\n")
        f.write(f"\nLanguage breakdown:\n")
        for k in sorted(lang_counts.keys()):
            f.write(f"  {k}: {lang_counts[k]} files\n")
        f.write(f"\nTotals:\n")
        f.write(f"  Files in tree: {len(tree_entries)}\n")
        f.write(f"  Files scanned: {len(scanned_files)}\n")
        f.write(f"  Symbols: {total_symbols}\n")
        f.write(f"  Routes: {total_routes}\n")
        f.write(f"  Enums: {total_enums}\n")
    outputs["tree_txt"] = tree_txt_path
    
    # SYMBOL_INDEX
    symbol_index_path = os.path.join(out_dir, f"SYMBOL_INDEX_{stamp}.json")
    with open(symbol_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "by_file": {sf["path"]: sf.get("symbols", []) for sf in scanned_files},
        }, f, indent=2)
    outputs["symbol_index"] = symbol_index_path
    
    # ENUM_INDEX
    enum_index_path = os.path.join(out_dir, f"ENUM_INDEX_{stamp}.json")
    enum_data = {}
    for sf in scanned_files:
        for e in sf.get("enums", []):
            e["file"] = sf["path"]
            enum_data[e["name"]] = e
    with open(enum_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "enums": enum_data,
            "total_enums": len(enum_data),
        }, f, indent=2)
    outputs["enum_index"] = enum_index_path
    
    # ROUTE_MAP
    route_map_path = os.path.join(out_dir, f"ROUTE_MAP_{stamp}.json")
    all_routes = []
    for sf in scanned_files:
        for r in sf.get("routes", []):
            r["file"] = sf["path"]
            all_routes.append(r)
    with open(route_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "routes": all_routes,
            "total_routes": len(all_routes),
        }, f, indent=2)
    outputs["route_map"] = route_map_path
    
    print(f"\nOutputs written to: {out_dir}")
    for name, path in outputs.items():
        print(f"  {name}: {os.path.basename(path)}")
    
    return outputs


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Usage:
        python host_fs_scanner.py [out_dir] [max_files] [max_scan_files]
    
    Environment:
        HOST_SCAN_ROOTS: Comma-separated roots (default: C:\\Users,D:\\)
    """
    # Parse arguments
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out"
    )
    
    max_files = DEFAULT_MAX_FILES
    if len(sys.argv) > 2:
        try:
            max_files = int(sys.argv[2])
        except ValueError:
            pass
    
    max_scan_files = DEFAULT_MAX_SCAN_FILES
    if len(sys.argv) > 3:
        try:
            max_scan_files = int(sys.argv[3])
        except ValueError:
            pass
    
    # Get roots from environment or use defaults
    roots_env = os.getenv("HOST_SCAN_ROOTS", "").strip()
    if roots_env:
        roots = [r.strip() for r in roots_env.split(",") if r.strip()]
    else:
        roots = DEFAULT_SCAN_ROOTS
    
    print(f"Host Filesystem Scanner")
    print(f"  Roots: {roots}")
    print(f"  Max files in tree: {max_files}")
    print(f"  Max files to scan: {max_scan_files}")
    print(f"  Output: {out_dir}")
    print()
    
    # Clean output directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    def progress(msg):
        print(f"  {msg}")
    
    # Run scan
    tree_entries, scanned_files = scan_filesystem(
        roots=roots,
        max_files=max_files,
        max_scan_files=max_scan_files,
        progress_callback=progress,
    )
    
    # Generate output
    outputs = generate_output(
        tree_entries=tree_entries,
        scanned_files=scanned_files,
        out_dir=out_dir,
        roots=roots,
    )
    
    print(f"\nScan complete!")
    print(f"  Total files discovered: {len(tree_entries)}")
    print(f"  Total files scanned: {len(scanned_files)}")


if __name__ == "__main__":
    main()
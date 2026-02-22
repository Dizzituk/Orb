from pathlib import Path


EVIDENCE_PRIORITY = [
    "architecture_map",      # Highest priority
    "codebase_report",
    "file_read",
    "repo_search",
    "arch_query_fallback",  # Lowest priority - fallback only
]

DEFAULT_MAX_BYTES = 100_000

def _to_absolute_path(path: str) -> str:
    """Convert relative path to absolute path in repo."""
    if Path(path).is_absolute():
        return path
    
    # Assume relative to D:\Orb
    return str(Path(r"D:\Orb") / path)

"""
Directory summary generator.

Creates embeddable text summaries for directories.
"""

import json
from sqlalchemy.orm import Session
from app.rag.models import ArchDirectoryIndex


def extension_to_language(ext: str) -> str:
    """Map extension to language name."""
    mapping = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".jsx": "React",
        ".tsx": "React/TS",
        ".java": "Java",
        ".go": "Go",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".php": "PHP",
        ".cs": "C#",
        ".cpp": "C++",
        ".vue": "Vue",
        ".svelte": "Svelte",
    }
    return mapping.get(ext.lower(), ext.upper().lstrip("."))


def format_lines(count: int) -> str:
    """Format line count."""
    if count < 1000:
        return str(count)
    elif count < 1000000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1000000:.1f}M"


def generate_directory_summary(directory: ArchDirectoryIndex) -> str:
    """
    Generate embeddable summary.
    
    Format: "app/llm/ | 28 files, 18K lines | Primary: Python"
    """
    parts = []
    
    # Path
    parts.append(f"{directory.canonical_path}/")
    
    # Stats
    stats = f"{directory.file_count} files"
    if directory.total_lines:
        stats += f", {format_lines(directory.total_lines)} lines"
    parts.append(stats)
    
    # Primary language
    if directory.extensions_json:
        try:
            extensions = json.loads(directory.extensions_json)
            if extensions:
                primary_ext = max(extensions, key=extensions.get)
                primary_lang = extension_to_language(primary_ext)
                parts.append(f"Primary: {primary_lang}")
        except (json.JSONDecodeError, ValueError):
            pass
    
    return " | ".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def generate_summaries_for_scan(db: Session, scan_id: int) -> int:
    """
    Generate summaries for all directories in scan.
    
    Returns number of summaries generated.
    """
    directories = db.query(ArchDirectoryIndex).filter_by(
        scan_id=scan_id
    ).all()
    
    count = 0
    for directory in directories:
        summary = generate_directory_summary(directory)
        directory.summary = summary
        directory.summary_tokens = estimate_tokens(summary)
        count += 1
    
    db.commit()
    return count

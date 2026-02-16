"""
Source file extraction utilities for architecture executor.

Detects source files referenced in architecture sections that indicate
code is being extracted/moved/split from existing files.
"""

import re
from typing import List


def _detect_source_files_from_architecture(
    file_section: str,
    architecture_content: str,
    rel_path: str
) -> List[str]:
    """
    Parse architecture section to detect source files from which code is being extracted.
    
    Uses three strategies:
    1. Regex patterns for explicit extraction language ("extract from X.py", etc.)
    2. Backtick file references with context-window check for extraction keywords
    3. Parent-file inference for *_parts/*_modules/*_components/*_lib/*_utils directories
    
    Args:
        file_section: Architecture section text for the target file
        architecture_content: Full architecture document (not used in current implementation)
        rel_path: Target file path being created (relative, may have backslashes)
        
    Returns:
        List of relative source file paths, normalized to forward slashes, excluding target file
    """
    sources = set()
    normalized_target = rel_path.replace("\\", "/")

    # Strategy 1: Explicit extraction patterns
    extraction_patterns = [
        r'extract(?:ing|ed)?\s+(?:from|out of)\s+[`"]?([^`"\s]+\.py)[`"]?',
        r'mov(?:e|ing|ed)\s+(?:from|out of)\s+[`"]?([^`"\s]+\.py)[`"]?',
        r'split(?:ting)?\s+(?:from|out of)\s+[`"]?([^`"\s]+\.py)[`"]?',
        r'currently\s+in\s+[`"]?([^`"\s]+\.py)[`"]?',
        r'originally\s+(?:from|in)\s+[`"]?([^`"\s]+\.py)[`"]?',
        r'refactor(?:ing|ed)?\s+(?:from|out of)\s+[`"]?([^`"\s]+\.py)[`"]?'
    ]
    
    for pattern in extraction_patterns:
        for match in re.finditer(pattern, file_section, re.IGNORECASE):
            source_path = match.group(1).replace("\\", "/")
            if source_path != normalized_target:
                sources.add(source_path)

    # Strategy 2: Backtick file references with context window
    backtick_pattern = r'`([^`]+\.py)`'
    extraction_keywords = [
        'extract', 'move', 'split', 'refactor', 'currently', 'originally',
        'from', 'out of', 'migrat'
    ]
    
    for match in re.finditer(backtick_pattern, file_section):
        candidate = match.group(1).replace("\\", "/")
        if candidate == normalized_target:
            continue
            
        # Check 200-char window around the match
        start = max(0, match.start() - 200)
        end = min(len(file_section), match.end() + 200)
        context = file_section[start:end].lower()
        
        if any(kw in context for kw in extraction_keywords):
            sources.add(candidate)

    # Strategy 3: Parent-file inference for split directories
    split_suffixes = ['_parts', '_modules', '_components', '_lib', '_utils']
    normalized_lower = normalized_target.lower()
    
    for suffix in split_suffixes:
        if f'/{suffix}/' in normalized_lower:
            # Extract parent file path
            parts = normalized_target.split('/')
            for i, part in enumerate(parts):
                if part.lower().endswith(suffix):
                    # Construct parent file: remove suffix directory and child file
                    parent_parts = parts[:i]
                    if parent_parts:
                        # Add .py if the parent directory name suggests a module
                        parent_name = parts[i].replace(suffix, '')
                        if parent_name:
                            parent_parts.append(f"{parent_name}.py")
                            parent_path = '/'.join(parent_parts)
                            if parent_path != normalized_target:
                                sources.add(parent_path)
                    break

    return sorted(sources)
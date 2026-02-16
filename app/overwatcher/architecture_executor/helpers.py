"""
Helper utilities for architecture executor.

Provides low-level text processing functions used across the executor package.
These functions are pure utilities with no external dependencies (stdlib only).
"""

from typing import Any

__all__ = [
    "_extract_llm_content",
    "_strip_markdown_fences",
]


def _extract_llm_content(llm_result: Any) -> str:
    """
    Extract string content from an LLM result object.
    
    Handles both string responses and structured objects with a 'content' attribute.
    
    Args:
        llm_result: The LLM response object (str or object with .content)
        
    Returns:
        The extracted string content
        
    Raises:
        ValueError: If content cannot be extracted from the result
    """
    if isinstance(llm_result, str):
        return llm_result
    if hasattr(llm_result, "content"):
        content = llm_result.content
        if isinstance(content, str):
            return content
    raise ValueError(f"Cannot extract content from LLM result of type {type(llm_result)}")


def _strip_markdown_fences(content: str) -> str:
    """
    Remove markdown code fences from content if present.
    
    Strips leading ```[language] and trailing ``` markers, preserving the code inside.
    If no fences are found, returns the content unchanged.
    
    Args:
        content: The content string, possibly wrapped in markdown fences
        
    Returns:
        The content with fences removed (if present)
    """
    lines = content.strip().split("\n")
    if not lines:
        return content
    
    # Check for opening fence
    if lines[0].startswith("```"):
        lines = lines[1:]
    
    # Check for closing fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    
    return "\n".join(lines)
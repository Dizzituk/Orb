"""
Context assembler.

Assembles search results into LLM context with token budget.
"""

from dataclasses import dataclass
from typing import List

from app.rag.retrieval.arch_search import (
    ArchSearchResult,
    ArchSearchResponse,
)
from app.rag.models import SourceType


@dataclass
class AssembledContext:
    """Assembled context for LLM."""
    text: str
    total_tokens: int
    directories_included: int
    chunks_included: int
    truncated: bool


class ContextAssembler:
    """Assemble search results into context."""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
    
    def assemble(
        self,
        response: ArchSearchResponse,
        include_headers: bool = True,
    ) -> AssembledContext:
        """
        Assemble results into formatted context.
        
        Args:
            response: Search response
            include_headers: Include section headers
            
        Returns:
            AssembledContext
        """
        sections = []
        total_tokens = 0
        dirs_included = 0
        chunks_included = 0
        truncated = False
        
        # Separate by type
        directories = [
            r for r in response.results
            if r.source_type == SourceType.ARCH_DIRECTORY
        ]
        chunks = [
            r for r in response.results
            if r.source_type == SourceType.ARCH_CHUNK
        ]
        
        # Add directories first (orientation)
        if directories:
            if include_headers:
                sections.append("## Relevant Directories\n")
                total_tokens += 5
            
            for d in directories:
                entry = self._format_directory(d)
                entry_tokens = self._estimate_tokens(entry)
                
                if total_tokens + entry_tokens > self.max_tokens:
                    truncated = True
                    break
                
                sections.append(entry)
                total_tokens += entry_tokens
                dirs_included += 1
        
        # Add chunks (detail)
        if chunks and not truncated:
            if include_headers:
                sections.append("\n## Relevant Code\n")
                total_tokens += 5
            
            for c in chunks:
                entry = self._format_chunk(c)
                entry_tokens = self._estimate_tokens(entry)
                
                if total_tokens + entry_tokens > self.max_tokens:
                    truncated = True
                    break
                
                sections.append(entry)
                total_tokens += entry_tokens
                chunks_included += 1
        
        return AssembledContext(
            text="".join(sections),
            total_tokens=total_tokens,
            directories_included=dirs_included,
            chunks_included=chunks_included,
            truncated=truncated,
        )
    
    def _format_directory(self, result: ArchSearchResult) -> str:
        """Format directory result."""
        return f"**{result.canonical_path}**: {result.content}\n\n"
    
    def _format_chunk(self, result: ArchSearchResult) -> str:
        """Format chunk result."""
        parts = []
        
        # Header with location
        location = result.canonical_path
        if result.start_line:
            location += f":{result.start_line}"
            if result.end_line and result.end_line != result.start_line:
                location += f"-{result.end_line}"
        
        parts.append(f"### {result.name} ({location})\n")
        
        # Signature
        if result.signature:
            parts.append(f"```python\n{result.signature}\n```\n")
        
        # Docstring (truncated)
        if result.docstring:
            doc = result.docstring
            if len(doc) > 300:
                doc = doc[:300].rsplit(" ", 1)[0] + "..."
            parts.append(f"{doc}\n")
        
        parts.append("\n")
        return "".join(parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate."""
        return int(len(text.split()) * 1.3)


def assemble_context(
    response: ArchSearchResponse,
    max_tokens: int = 4000,
) -> AssembledContext:
    """Convenience function."""
    return ContextAssembler(max_tokens).assemble(response)


def retrieve_architecture_context(
    db,
    query: str,
    max_tokens: int = 4000,
) -> str:
    """
    One-shot: search + assemble.
    
    Args:
        db: Database session
        query: User query
        max_tokens: Token budget
        
    Returns:
        Formatted context string
    """
    from app.rag.retrieval.arch_search import search_architecture
    
    response = search_architecture(db, query)
    context = assemble_context(response, max_tokens)
    return context.text

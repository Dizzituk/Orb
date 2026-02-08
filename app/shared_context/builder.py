from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from app.shared_context.schemas import SharedContextItem, SharedContextPackage
from app.embeddings.service import search_embeddings


def build_shared_context_package(
    db: Session,
    project_id: int,
    query: str,
    top_k: int = 8,
    source_types: Optional[List[str]] = None,
    budget_chars: int = 6000,
    max_items: int = 4
) -> SharedContextPackage:
    """
    Build a shared context package from semantic search results.
    
    Args:
        db: Database session
        project_id: Project scope
        query: Search query (typically user message)
        top_k: How many results to retrieve from embeddings
        source_types: Filter by ["note", "message", "file"] or None for all
        budget_chars: Max characters in formatted_block
        max_items: Max items to include (best by similarity)
    
    Returns:
        SharedContextPackage with formatted_block ready for injection
    """
    # Step 1: Retrieval
    results, total_searched = search_embeddings(
        db, project_id, query, top_k, source_types
    )
    
    # Step 2: Selection - sort by similarity descending and take top max_items
    sorted_results = sorted(results, key=lambda r: r.similarity, reverse=True)
    selected_results = sorted_results[:max_items]
    
    # Step 3: Format and budget
    items: List[SharedContextItem] = []
    formatted_lines: List[str] = ["=== SHARED CONTEXT (retrieved) ===", ""]
    current_chars = len("\n".join(formatted_lines))
    truncated = False
    
    for result in selected_results:
        # Build the item
        provenance = f"{result.source_type}:{result.source_id} chunk={result.chunk_index} sim={result.similarity:.2f}"
        pointer = f"/memory/{result.source_type}/{result.source_id}"
        
        item = SharedContextItem(
            summary=result.content,
            provenance=provenance,
            pointer=pointer,
            similarity=result.similarity
        )
        
        # Format the item block
        item_block = [
            f"• {result.content}",
            f"  Source: {provenance}",
            f"  Link: {pointer}",
            ""
        ]
        item_text = "\n".join(item_block)
        
        # Check budget
        if current_chars + len(item_text) > budget_chars:
            truncated = True
            break
        
        # Add to output
        items.append(item)
        formatted_lines.extend(item_block)
        current_chars += len(item_text)
    
    formatted_block = "\n".join(formatted_lines)
    
    # Step 4: Return package
    return SharedContextPackage(
        items=items,
        formatted_block=formatted_block,
        total_retrieved=len(results),
        total_searched=total_searched,
        truncated=truncated
    )
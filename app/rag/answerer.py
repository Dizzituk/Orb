"""
Grounded Q&A engine for architecture queries.

Takes a question, searches RAG index, builds context, and answers with LLM.

v1.0 (2026-01): Initial implementation
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from app.rag.models import ArchCodeChunk, ArchDirectoryIndex


@dataclass
class ArchAnswer:
    """Answer from architecture RAG."""
    question: str
    answer: str
    sources: List[dict]
    chunks_searched: int
    model_used: str


def _build_context_from_chunks(chunks: List[ArchCodeChunk], max_tokens: int = 8000) -> Tuple[str, List[dict]]:
    """
    Build context string from chunks for LLM.
    
    Returns: (context_string, sources_list)
    """
    context_parts = []
    sources = []
    approx_tokens = 0
    
    for chunk in chunks:
        # Build chunk representation
        chunk_text = f"## {chunk.file_path}\n"
        if chunk.chunk_name:
            chunk_text += f"**{chunk.chunk_type}**: `{chunk.chunk_name}`\n"
        if chunk.signature:
            chunk_text += f"**Signature**: `{chunk.signature}`\n"
        if chunk.docstring:
            chunk_text += f"**Docstring**: {chunk.docstring[:200]}\n"
        if chunk.start_line:
            chunk_text += f"**Lines**: {chunk.start_line}-{chunk.end_line or '?'}\n"
        chunk_text += "\n"
        
        # Approximate token count (rough: 4 chars = 1 token)
        chunk_tokens = len(chunk_text) // 4
        if approx_tokens + chunk_tokens > max_tokens:
            break
            
        context_parts.append(chunk_text)
        approx_tokens += chunk_tokens
        
        sources.append({
            "file": chunk.file_path,
            "name": chunk.chunk_name,
            "type": chunk.chunk_type,
            "line": chunk.start_line,
        })
    
    return "\n".join(context_parts), sources


def _search_chunks_keyword(db: Session, query: str, limit: int = 20) -> List[ArchCodeChunk]:
    """
    Keyword-based search when embeddings aren't available.
    
    Searches chunk_name, file_path, signature, docstring.
    """
    # Split query into keywords
    keywords = [k.strip().lower() for k in query.split() if len(k.strip()) > 2]
    
    if not keywords:
        return []
    
    # Build filter - match any keyword in name, path, signature, or docstring
    from sqlalchemy import or_, func
    
    filters = []
    for kw in keywords[:5]:  # Limit keywords
        filters.append(func.lower(ArchCodeChunk.chunk_name).contains(kw))
        filters.append(func.lower(ArchCodeChunk.file_path).contains(kw))
        filters.append(func.lower(ArchCodeChunk.signature).contains(kw))
        filters.append(func.lower(ArchCodeChunk.docstring).contains(kw))
    
    chunks = db.query(ArchCodeChunk).filter(
        or_(*filters)
    ).limit(limit).all()
    
    return chunks


def _check_embedding_availability(db: Session) -> Tuple[int, int]:
    """
    Check how many chunks have embeddings.
    
    Returns: (embedded_count, total_count)
    """
    from sqlalchemy import func
    
    total = db.query(func.count(ArchCodeChunk.id)).scalar() or 0
    embedded = db.query(func.count(ArchCodeChunk.id)).filter(
        ArchCodeChunk.embedded == True
    ).scalar() or 0
    
    return embedded, total


def _search_chunks_embedding(db: Session, query: str, limit: int = 20) -> Tuple[List[ArchCodeChunk], str]:
    """
    Embedding-based semantic search.
    
    Returns: (chunks, search_mode_used)
    """
    # First check if we have embeddings
    embedded_count, total_count = _check_embedding_availability(db)
    
    if embedded_count == 0:
        print(f"[rag:answerer] No embeddings available ({total_count} chunks), using keyword search")
        return [], "keyword_only"
    
    if embedded_count < total_count:
        print(f"[rag:answerer] Partial embeddings ({embedded_count}/{total_count}), semantic search may be incomplete")
    
    try:
        from app.embeddings.service import search_embeddings
        
        results, _ = search_embeddings(
            db=db,
            project_id=0,
            query=query,
            top_k=limit,
            source_types=["arch_code_chunk"],
        )
        
        if not results:
            return [], "no_results"
        
        # Fetch full chunk records
        chunk_ids = [r.source_id for r in results]
        chunks = db.query(ArchCodeChunk).filter(
            ArchCodeChunk.id.in_(chunk_ids)
        ).all()
        
        # Preserve ranking order
        chunk_map = {c.id: c for c in chunks}
        ordered_chunks = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
        
        mode = "semantic" if embedded_count == total_count else "semantic_partial"
        return ordered_chunks, mode
        
    except Exception as e:
        print(f"[rag:answerer] Embedding search failed: {e}, falling back to keyword")
        return [], "embedding_error"


def ask_architecture(
    db: Session,
    question: str,
    use_embeddings: bool = True,
) -> ArchAnswer:
    """
    Answer a question about the codebase using RAG.
    
    Uses hybrid search strategy:
    1. Try semantic (embedding) search first if available
    2. Fall back to keyword search if semantic fails/empty
    
    Args:
        db: Database session
        question: User's question
        use_embeddings: Try embedding search first (falls back to keyword)
        
    Returns:
        ArchAnswer with response and sources
    """
    # Search for relevant chunks using hybrid strategy
    chunks: List[ArchCodeChunk] = []
    search_mode = "none"
    
    # Step 1: Try semantic search if enabled
    if use_embeddings:
        semantic_chunks, semantic_mode = _search_chunks_embedding(db, question, limit=15)
        
        if semantic_chunks:
            chunks = semantic_chunks
            search_mode = semantic_mode
            print(f"[rag:answerer] Semantic search returned {len(chunks)} chunks (mode={search_mode})")
        else:
            print(f"[rag:answerer] Semantic search returned empty (mode={semantic_mode}), trying keyword fallback")
    
    # Step 2: Fall back to keyword search if semantic didn't return results
    if not chunks:
        chunks = _search_chunks_keyword(db, question, limit=15)
        if chunks:
            search_mode = "keyword"
            print(f"[rag:answerer] Keyword search returned {len(chunks)} chunks")
        else:
            print(f"[rag:answerer] Keyword search also returned empty")
    
    if not chunks:
        return ArchAnswer(
            question=question,
            answer="I couldn't find any relevant code in the architecture index. Try running `Astra, command: CREATE ARCHITECTURE MAP` first, then `/rag/index` to create embeddings.",
            sources=[],
            chunks_searched=0,
            model_used="none",
        )
    
    # Build context
    context, sources = _build_context_from_chunks(chunks)
    
    # Build prompt
    system_prompt = """You are an expert code assistant answering questions about the Orb/ASTRA codebase.

Use the provided code context to answer the user's question accurately. 
- Reference specific files and functions when relevant
- If the context doesn't contain enough information, say so
- Be concise but thorough"""

    user_prompt = f"""## Code Context

{context}

## Question

{question}

## Instructions

Answer based on the code context above. Reference specific files and functions."""

    # Call LLM (use cheap model for speed)
    try:
        from app.llm.clients import call_openai
        
        # call_openai returns (content, usage_dict) and takes:
        # system_prompt, messages, temperature, model
        content, usage = call_openai(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3,
            model=os.getenv("ORB_RAG_MODEL", "gpt-4o-mini"),
        )
        
        answer_text = content if content else "Failed to generate answer"
        model_used = os.getenv("ORB_RAG_MODEL", "gpt-4o-mini")
        
    except Exception as e:
        answer_text = f"LLM call failed: {e}"
        model_used = "error"
    
    return ArchAnswer(
        question=question,
        answer=answer_text,
        sources=sources,
        chunks_searched=len(chunks),
        model_used=model_used,
    )


async def ask_architecture_async(
    db: Session,
    question: str,
    use_embeddings: bool = True,
) -> ArchAnswer:
    """
    Async version of ask_architecture for use in async contexts (FastAPI streams).
    
    Uses await async_call_openai() instead of sync call_openai().
    Logic is identical to ask_architecture() - keep them in sync.
    
    Args:
        db: Database session
        question: User's question
        use_embeddings: Try embedding search first (falls back to keyword)
        
    Returns:
        ArchAnswer with response and sources
    """
    # Search for relevant chunks using hybrid strategy
    chunks: List[ArchCodeChunk] = []
    search_mode = "none"
    
    # Step 1: Try semantic search if enabled
    if use_embeddings:
        semantic_chunks, semantic_mode = _search_chunks_embedding(db, question, limit=15)
        
        if semantic_chunks:
            chunks = semantic_chunks
            search_mode = semantic_mode
            print(f"[rag:answerer] Semantic search returned {len(chunks)} chunks (mode={search_mode})")
        else:
            print(f"[rag:answerer] Semantic search returned empty (mode={semantic_mode}), trying keyword fallback")
    
    # Step 2: Fall back to keyword search if semantic didn't return results
    if not chunks:
        chunks = _search_chunks_keyword(db, question, limit=15)
        if chunks:
            search_mode = "keyword"
            print(f"[rag:answerer] Keyword search returned {len(chunks)} chunks")
        else:
            print(f"[rag:answerer] Keyword search also returned empty")
    
    if not chunks:
        return ArchAnswer(
            question=question,
            answer="I couldn't find any relevant code in the architecture index. Try running `Astra, command: CREATE ARCHITECTURE MAP` first, then `/rag/index` to create embeddings.",
            sources=[],
            chunks_searched=0,
            model_used="none",
        )
    
    # Build context
    context, sources = _build_context_from_chunks(chunks)
    
    # Build prompt
    system_prompt = """You are an expert code assistant answering questions about the Orb/ASTRA codebase.

Use the provided code context to answer the user's question accurately. 
- Reference specific files and functions when relevant
- If the context doesn't contain enough information, say so
- Be concise but thorough"""

    user_prompt = f"""## Code Context

{context}

## Question

{question}

## Instructions

Answer based on the code context above. Reference specific files and functions."""

    # Call LLM using async variant (FastAPI-safe)
    try:
        from app.llm.clients import async_call_openai
        
        content, usage = await async_call_openai(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3,
            model=os.getenv("ORB_RAG_MODEL", "gpt-4o-mini"),
        )
        
        answer_text = content if content else "Failed to generate answer"
        model_used = os.getenv("ORB_RAG_MODEL", "gpt-4o-mini")
        
    except Exception as e:
        answer_text = f"LLM call failed: {e}"
        model_used = "error"
    
    return ArchAnswer(
        question=question,
        answer=answer_text,
        sources=sources,
        chunks_searched=len(chunks),
        model_used=model_used,
    )

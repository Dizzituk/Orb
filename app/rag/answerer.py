# FILE: app/rag/answerer.py
"""
Grounded Q&A engine for architecture queries.

Takes a question, searches RAG index, builds context from both
architecture chunks and memory stores, and answers with a
complexity-appropriate LLM.

v10.0 (2026-02-23): Job 10B/10C/10D — complexity routing, memory
    grounding, upgraded prompts, provider-agnostic LLM call.
v1.0 (2026-01): Initial implementation
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.rag.models import ArchCodeChunk, ArchDirectoryIndex
from app.rag._answerer_model_selection import select_rag_model
from app.rag._answerer_memory_context import build_memory_context
from app.rag._answerer_structured_context import build_structured_context
from app.rag._answerer_text_search import search_codebase_text
from app.rag._answerer_prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt

logger = logging.getLogger(__name__)


@dataclass
class ArchAnswer:
    """Answer from architecture RAG."""
    question: str
    answer: str
    sources: List[dict]
    chunks_searched: int
    model_used: str


# =========================================================================
# Chunk context building (unchanged from v1.0)
# =========================================================================

def _build_context_from_chunks(
    chunks: List[ArchCodeChunk],
    max_tokens: int = 8000,
) -> Tuple[str, List[dict]]:
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


# =========================================================================
# Chunk search — keyword
# =========================================================================

def _search_chunks_keyword(
    db: Session,
    query: str,
    limit: int = 20,
) -> List[ArchCodeChunk]:
    """
    Keyword-based search when embeddings aren't available.

    Searches chunk_name, file_path, signature, docstring.
    """
    keywords = [k.strip().lower() for k in query.split() if len(k.strip()) > 2]
    if not keywords:
        return []

    from sqlalchemy import or_, func

    filters = []
    for kw in keywords[:5]:
        filters.append(func.lower(ArchCodeChunk.chunk_name).contains(kw))
        filters.append(func.lower(ArchCodeChunk.file_path).contains(kw))
        filters.append(func.lower(ArchCodeChunk.signature).contains(kw))
        filters.append(func.lower(ArchCodeChunk.docstring).contains(kw))

    chunks = db.query(ArchCodeChunk).filter(
        or_(*filters)
    ).limit(limit).all()

    return chunks


# =========================================================================
# Chunk search — embedding
# =========================================================================

def _check_embedding_availability(db: Session) -> Tuple[int, int]:
    """Returns: (embedded_count, total_count)"""
    from sqlalchemy import func

    total = db.query(func.count(ArchCodeChunk.id)).scalar() or 0
    embedded = db.query(func.count(ArchCodeChunk.id)).filter(
        ArchCodeChunk.embedded == True  # noqa: E712
    ).scalar() or 0

    return embedded, total


def _search_chunks_embedding(
    db: Session,
    query: str,
    limit: int = 20,
) -> Tuple[List[ArchCodeChunk], str]:
    """
    Embedding-based semantic search.

    Returns: (chunks, search_mode_used)
    """
    embedded_count, total_count = _check_embedding_availability(db)

    if embedded_count == 0:
        logger.info("[rag:answerer] No embeddings available (%d chunks), using keyword search", total_count)
        return [], "keyword_only"

    if embedded_count < total_count:
        logger.info("[rag:answerer] Partial embeddings (%d/%d), semantic search may be incomplete", embedded_count, total_count)

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
        logger.warning("[rag:answerer] Embedding search failed: %s, falling back to keyword", e)
        return [], "embedding_error"


# =========================================================================
# Hybrid chunk search
# =========================================================================

def _search_chunks_hybrid(
    db: Session,
    question: str,
    use_embeddings: bool = True,
) -> Tuple[List[ArchCodeChunk], str]:
    """
    Hybrid search: tries semantic first, falls back to keyword.

    Returns: (chunks, search_mode)
    """
    chunks: List[ArchCodeChunk] = []
    search_mode = "none"

    if use_embeddings:
        semantic_chunks, semantic_mode = _search_chunks_embedding(db, question, limit=15)
        if semantic_chunks:
            return semantic_chunks, semantic_mode
        logger.info("[rag:answerer] Semantic search empty (mode=%s), trying keyword", semantic_mode)

    chunks = _search_chunks_keyword(db, question, limit=15)
    if chunks:
        return chunks, "keyword"

    logger.info("[rag:answerer] Keyword search also returned empty")
    return [], "none"


# =========================================================================
# Main async entry point (primary)
# =========================================================================

async def ask_architecture_async(
    db: Session,
    question: str,
    use_embeddings: bool = True,
    project_id: str = "astra-core",
) -> ArchAnswer:
    """
    Answer a question about the codebase using grounded RAG.

    Pipeline:
        1. Select model via complexity classification (Job 10B)
        2. Search architecture chunks (hybrid: semantic → keyword)
        3. Query memory stores for additional context (Job 10C)
        4. Assemble prompt with memory + code context (Job 10D)
        5. Call LLM via provider registry (provider-agnostic)

    Args:
        db:             Database session
        question:       User's codebase question
        use_embeddings: Try embedding search first
        project_id:     Project scope for memory queries

    Returns:
        ArchAnswer with response, sources, and model info
    """
    # Step 1: Select model based on question complexity
    provider, model, tier = select_rag_model(question)

    # Step 2: Search for relevant chunks
    chunks, search_mode = _search_chunks_hybrid(db, question, use_embeddings)

    if not chunks:
        return ArchAnswer(
            question=question,
            answer=(
                "I couldn't find any relevant code in the architecture index. "
                "Try running `Astra, command: CREATE ARCHITECTURE MAP` first, "
                "then `index the architecture` to create embeddings."
            ),
            sources=[],
            chunks_searched=0,
            model_used="none",
        )

    logger.info(
        "[rag:answerer] %d chunks found (mode=%s), complexity=%s → %s/%s",
        len(chunks), search_mode, tier, provider, model,
    )

    # Step 3: Build code context from chunks
    code_context, sources = _build_context_from_chunks(chunks)

    # Step 4: Build memory context from all domain stores
    memory_context = build_memory_context(
        question=question,
        project_id=project_id,
    )

    # Step 4b: Build structured codebase overview (Job 11A)
    structured_context = build_structured_context(token_budget=2000)

    # Step 4c: On-demand text search if question looks like a count/find query (Job 11B)
    text_search_context = search_codebase_text(question, token_budget=1500)

    # Step 5: Assemble prompt — combine all context sources
    combined_code_context = code_context
    if structured_context:
        combined_code_context = structured_context + "\n\n" + combined_code_context
    if text_search_context:
        combined_code_context = combined_code_context + "\n\n" + text_search_context

    user_prompt = build_rag_user_prompt(
        memory_context=memory_context,
        code_context=combined_code_context,
        question=question,
    )

    # Step 6: Call LLM via provider registry
    try:
        from app.providers._registry_utils_3 import llm_call as registry_llm_call

        result = await registry_llm_call(
            provider_id=provider,
            model_id=model,
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=4096,
        )

        if result.is_success():
            answer_text = result.content or "No response generated."
            model_used = f"{result.provider_id}/{result.model_id}"
        else:
            error_msg = getattr(result, "error_message", "Unknown error")
            answer_text = f"LLM call failed ({result.status}): {error_msg}"
            model_used = f"{provider}/{model} (error)"
            logger.error("[rag:answerer] LLM call failed: %s — %s", result.status, error_msg)

    except Exception as e:
        answer_text = f"LLM call failed: {e}"
        model_used = f"{provider}/{model} (exception)"
        logger.error("[rag:answerer] LLM call exception: %s", e)

    return ArchAnswer(
        question=question,
        answer=answer_text,
        sources=sources,
        chunks_searched=len(chunks),
        model_used=model_used,
    )


# =========================================================================
# Sync wrapper (deprecated — use ask_architecture_async instead)
# =========================================================================

def ask_architecture(
    db: Session,
    question: str,
    use_embeddings: bool = True,
    project_id: str = "astra-core",
) -> ArchAnswer:
    """
    Sync wrapper around ask_architecture_async().

    Deprecated: prefer the async version directly.
    Kept for backward compatibility with any sync callers.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop — can't use asyncio.run().
        # Create a new thread to run the async version.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                ask_architecture_async(db, question, use_embeddings, project_id),
            )
            return future.result(timeout=120)
    else:
        return asyncio.run(
            ask_architecture_async(db, question, use_embeddings, project_id)
        )

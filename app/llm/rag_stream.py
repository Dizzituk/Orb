# Purpose: RAG Query stream handler.
# Called-by: app.llm.routing.handler_registry
# Depends-on: app.rag.answerer, app.rag.models, app.rag.pipeline
# Last-renovated: 2026-06-11
"""
RAG Query stream handler.

Searches indexed codebase and answers questions with LLM.

v1.0 (2026-01): Initial implementation
"""

import json
import logging
import os
from typing import AsyncGenerator

from sqlalchemy.orm import Session

from app.rag.answerer import ask_architecture_async
from app.rag.models import ArchCodeChunk
from app.rag.pipeline import run_rag_pipeline

logger = logging.getLogger(__name__)

# Default scan directory
DEFAULT_SCAN_DIR = os.getenv("ZOBIE_OUTPUT_DIR", r"D:\Orb\.architecture")


def _persist_rag_turn(
    db: Session,
    project_id: int,
    message: str,
    answer: str,
    model: str = "rag_answerer",
    persist_user: bool = False,
) -> None:
    """Persist the RAG turn to chat history (assistant always; user when the
    caller hasn't already saved it). Without this, RAG answers were absent from
    history so follow-up questions lost context. Non-fatal."""
    try:
        from app.memory import service as memory_service
        from app.memory import schemas as memory_schemas
        if persist_user:
            memory_service.create_message(db, memory_schemas.MessageCreate(
                project_id=project_id, role="user", content=message, provider="local",
            ))
        if answer:
            memory_service.create_message(db, memory_schemas.MessageCreate(
                project_id=project_id, role="assistant", content=answer,
                provider="local", model=model,
            ))
    except Exception as e:
        logger.warning("[rag_stream] Failed to persist RAG turn (non-fatal): %s", e)


async def generate_rag_query_stream(
    project_id: int,
    message: str,
    db: Session,
    trace=None,
    persist_user: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Stream RAG codebase query response.

    Handles two cases:
    1. "index the architecture" - triggers indexing
    2. Question - searches and answers

    persist_user: when True, also save the user turn to history (callers on the
    normal-routing path don't persist it themselves; the chat-mode and command
    paths already do, so they leave this False).
    """
    # Persist the user turn up-front when the caller hasn't already (normal-
    # routing path). Done here so EVERY branch — including the index-command
    # early returns — records it; assistant turns are saved per-branch below.
    if persist_user:
        _persist_rag_turn(db, project_id, message, answer="", persist_user=True)

    msg_lower = message.lower().strip()
    
    # Check for index command
    if "index" in msg_lower and ("architecture" in msg_lower or "rag" in msg_lower or "codebase" in msg_lower):
        yield "data: " + json.dumps({
            'type': 'token',
            'content': "🔄 **Indexing architecture for RAG...**\n\n"
        }) + "\n\n"
        
        yield "data: " + json.dumps({
            'type': 'token',
            'content': f"📂 Scan directory: `{DEFAULT_SCAN_DIR}`\n"
        }) + "\n\n"
        
        # Check if directory exists
        if not os.path.isdir(DEFAULT_SCAN_DIR):
            yield "data: " + json.dumps({
                'type': 'token',
                'content': f"❌ **Directory not found:** `{DEFAULT_SCAN_DIR}`\n\nRun `Astra, command: CREATE ARCHITECTURE MAP` first."
            }) + "\n\n"
            yield "data: " + json.dumps({
                'type': 'done',
                'provider': 'local',
                'model': 'rag_indexer',
            }) + "\n\n"
            return
        
        # Check for SIGNATURES file
        import glob
        sig_files = glob.glob(os.path.join(DEFAULT_SCAN_DIR, "SIGNATURES_*.json"))
        if not sig_files:
            yield "data: " + json.dumps({
                'type': 'token',
                'content': "❌ **No SIGNATURES file found.**\n\nRun `Astra, command: CREATE ARCHITECTURE MAP` first."
            }) + "\n\n"
            yield "data: " + json.dumps({
                'type': 'done',
                'provider': 'local',
                'model': 'rag_indexer',
            }) + "\n\n"
            return
        
        yield "data: " + json.dumps({
            'type': 'token',
            'content': f"📝 Found {len(sig_files)} signature file(s)\n"
        }) + "\n\n"
        
        yield "data: " + json.dumps({
            'type': 'token',
            'content': "⏳ Running pipeline (this may take a minute)...\n\n"
        }) + "\n\n"
        
        try:
            result = run_rag_pipeline(
                db=db,
                scan_dir=DEFAULT_SCAN_DIR,
                project_id=0,
            )
            
            chunks = result.get("chunks", 0)
            dirs = result.get("directories", 0)
            embeddings = result.get("embeddings", 0)
            
            yield "data: " + json.dumps({
                'type': 'token',
                'content': (
                    f"✅ **RAG Indexing Complete**\n\n"
                    f"- **Code chunks:** {chunks:,}\n"
                    f"- **Directories:** {dirs:,}\n"
                    f"- **Embeddings:** {embeddings:,}\n\n"
                    f"You can now ask questions about the codebase!"
                )
            }) + "\n\n"
            
        except Exception as e:
            import traceback
            logger.error(f"RAG indexing failed: {e}")
            logger.error(traceback.format_exc())
            yield "data: " + json.dumps({
                'type': 'token',
                'content': f"❌ **Indexing failed:** {str(e)}\n\nMake sure you've run `CREATE ARCHITECTURE MAP` first."
            }) + "\n\n"
        
        yield "data: " + json.dumps({
            'type': 'done',
            'provider': 'local',
            'model': 'rag_indexer',
        }) + "\n\n"
        return
    
    # Check if we have any indexed chunks
    chunk_count = db.query(ArchCodeChunk).count()
    if chunk_count == 0:
        _no_index_msg = (
            "⚠️ **No codebase index found**\n\n"
            "To use RAG search, first run:\n"
            "1. `Astra, command: CREATE ARCHITECTURE MAP` - scans and extracts signatures\n"
            "2. `index the architecture` - creates embeddings for search\n\n"
            "Then you can ask questions like:\n"
            "- `search codebase: what functions handle streaming?`\n"
            "- `In the codebase, where is job routing implemented?`"
        )
        yield "data: " + json.dumps({
            'type': 'token',
            'content': _no_index_msg,
        }) + "\n\n"

        # User turn already saved at function top (when persist_user); assistant only.
        _persist_rag_turn(db, project_id, message, _no_index_msg,
                          model="rag_query", persist_user=False)

        yield "data: " + json.dumps({
            'type': 'done',
            'provider': 'local',
            'model': 'rag_query',
        }) + "\n\n"
        return
    
    # Extract the actual question
    question = _extract_question(message)
    
    yield "data: " + json.dumps({
        'type': 'token',
        'content': f"🔍 **Searching codebase for:** {question}\n\n"
    }) + "\n\n"
    
    # Search and answer
    full_answer = ""      # accumulated for history persistence
    answer_model = "rag_answerer"
    try:
        result = await ask_architecture_async(db=db, question=question, project_id=str(project_id))
        answer_model = result.model_used or "rag_answerer"

        # Stream the answer
        full_answer += result.answer or ""
        yield "data: " + json.dumps({
            'type': 'token',
            'content': result.answer
        }) + "\n\n"

        # Add sources if any (deduplicated)
        if result.sources:
            # Deduplicate by (file, name, line) keeping first occurrence
            seen = set()
            unique_sources = []
            for src in result.sources:
                key = (src.get('file'), src.get('name'), src.get('line'))
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(src)

            sources_text = "\n\n---\n**Sources** (searched " + str(result.chunks_searched) + " chunks):\n"
            for src in unique_sources[:5]:  # Limit to 5 after dedup
                sources_text += f"- `{src.get('file', '?')}` → `{src.get('name', '?')}` ({src.get('type', '?')}, line {src.get('line', '?')})\n"

            full_answer += sources_text
            yield "data: " + json.dumps({
                'type': 'token',
                'content': sources_text
            }) + "\n\n"

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        full_answer = f"❌ **Query failed:** {str(e)}"
        yield "data: " + json.dumps({
            'type': 'token',
            'content': full_answer
        }) + "\n\n"

    # Persist the assistant answer so RAG answers join chat history. The user
    # turn was already saved at function top (when persist_user) or by the caller.
    _persist_rag_turn(db, project_id, message, full_answer,
                      model=answer_model, persist_user=False)

    yield "data: " + json.dumps({
        'type': 'done',
        'provider': answer_model,
        'model': 'rag_query',
    }) + "\n\n"


def _extract_question(message: str) -> str:
    """Extract the actual question from trigger phrases."""
    import re
    
    # Remove common prefixes
    prefixes = [
        r"^[Ss]earch\s+(?:the\s+)?codebase:\s*",
        r"^[Aa]sk\s+about\s+(?:the\s+)?codebase:\s*",
        r"^[Cc]odebase\s+(?:search|query):\s*",
        r"^[Ii]n\s+(?:the|this)\s+codebase,?\s*",
    ]
    
    question = message
    for pattern in prefixes:
        question = re.sub(pattern, "", question, count=1)
    
    return question.strip()

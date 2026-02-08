"""
Command Output Persistence Module

This module provides a reusable SSE stream wrapper that accumulates command output
and persists it to the database after streaming completes.

Key features:
- Transparent SSE proxy (zero latency impact)
- Accumulates content from token events
- Persists after stream ends (success, error, or disconnect)
- Updates provider/model from done events
- Robust SSE chunk parsing

Design decisions documented in CRITICAL_CLAIMS register (see architecture spec).
"""

import json
import logging
from typing import AsyncIterator, Optional
from sqlalchemy.orm import Session

from app.memory import schemas
from app.memory.service import create_message

logger = logging.getLogger(__name__)


async def wrap_sse_and_persist(
    *,
    gen: AsyncIterator[str],
    db: Session,
    project_id: int,
    provider: Optional[str],
    model: Optional[str]
) -> AsyncIterator[str]:
    """
    Transparent SSE proxy that persists command output after streaming completes.
    
    This wrapper:
    1. Forwards all SSE chunks to the client unchanged (zero latency impact)
    2. Accumulates content from {"type": "token", "text": "..."} events
    3. Updates provider/model from {"type": "done", "provider": "...", "model": "..."} if present
    4. Persists a single assistant message in finally: block after stream ends
    
    The finally: block runs regardless of how the stream terminates:
    - Normal completion (generator exhausted)
    - Error raised during streaming
    - Client disconnect (GeneratorExit)
    
    This ensures partial outputs are persisted even on failures, which is
    desirable for continuity (user can see what was generated before the error).
    
    Args:
        gen: Async generator yielding SSE chunks
        db: Database session for persistence
        project_id: Project ID for the message
        provider: Initial provider value (can be updated by done event)
        model: Initial model value (can be updated by done event)
    
    Yields:
        SSE chunks unchanged (transparent proxy)
    
    Side Effects:
        Creates assistant message in database after stream completes
    
    Decision References:
        - D-001: SSE chunk parsing strategy (only parse 'data:' lines)
        - D-002: Persist partial output on error/disconnect
        - D-003: Provider/model fallback values
    """
    accumulator = []
    final_provider = provider or "local"
    final_model = model or "command_router"
    
    try:
        async for chunk in gen:
            # Forward chunk to client immediately (zero latency)
            yield chunk
            
            # Parse SSE chunk for accumulation
            # SSE format: lines starting with "data: " contain JSON events
            # Other lines (comments, metadata) are ignored
            for line in chunk.split('\n'):
                if line.startswith('data: '):
                    try:
                        # Extract JSON payload (skip "data: " prefix)
                        json_str = line[6:].strip()
                        if not json_str:
                            continue
                        
                        event = json.loads(json_str)
                        
                        # Accumulate text from token events
                        if event.get("type") == "token":
                            text = event.get("text", "")
                            if text:
                                accumulator.append(text)
                        
                        # Update provider/model from done event
                        elif event.get("type") == "done":
                            if "provider" in event:
                                final_provider = event["provider"]
                            if "model" in event:
                                final_model = event["model"]
                    
                    except json.JSONDecodeError as e:
                        # Skip malformed JSON (robust parsing)
                        logger.debug(f"Failed to parse SSE JSON: {e}")
                        continue
                    except Exception as e:
                        # Catch-all for unexpected parsing errors
                        logger.warning(f"Unexpected error parsing SSE chunk: {e}")
                        continue
    
    finally:
        # Persist after stream ends (success, error, or disconnect)
        # This runs even if generator raises exception or client disconnects
        if accumulator:
            content = "".join(accumulator)
            
            try:
                create_message(
                    db,
                    schemas.MessageCreate(
                        project_id=project_id,
                        role="assistant",
                        content=content,
                        provider=final_provider,
                        model=final_model
                    )
                )
                logger.info(
                    f"Persisted command output: project_id={project_id}, "
                    f"length={len(content)}, provider={final_provider}, model={final_model}"
                )
            except Exception as e:
                # Log persistence failure but don't crash stream
                # Stream has already completed from client perspective
                logger.error(
                    f"Failed to persist command output: project_id={project_id}, "
                    f"error={e}",
                    exc_info=True
                )
        else:
            logger.debug(
                f"No content to persist: project_id={project_id} "
                "(empty accumulator)"
            )
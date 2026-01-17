"""
Descriptor generator.

Creates embeddable text from code chunks.
"""

import json
from sqlalchemy.orm import Session
from app.rag.models import ArchCodeChunk, ChunkType


def generate_chunk_descriptor(chunk: ArchCodeChunk) -> str:
    """
    Generate embeddable descriptor for chunk.
    
    Format: "async def stream_chat(...) | Streams responses... | @router.post"
    """
    parts = []
    
    # Signature (most important)
    if chunk.signature:
        parts.append(chunk.signature)
    else:
        prefix = _type_to_prefix(chunk.chunk_type)
        parts.append(f"{prefix} {chunk.chunk_name}")
    
    # Return type if not in signature
    if chunk.returns and "->" not in (chunk.signature or ""):
        parts.append(f"-> {chunk.returns}")
    
    # Docstring (truncated)
    if chunk.docstring:
        doc = _truncate_docstring(chunk.docstring)
        if doc:
            parts.append(doc)
    
    # Decorators
    if chunk.decorators_json:
        try:
            decorators = json.loads(chunk.decorators_json)
            if decorators:
                dec_str = ", ".join(decorators[:3])
                parts.append(f"Decorators: {dec_str}")
        except json.JSONDecodeError:
            pass
    
    # Base classes (for classes)
    if chunk.chunk_type == ChunkType.CLASS and chunk.bases_json:
        try:
            bases = json.loads(chunk.bases_json)
            if bases:
                parts.append(f"Bases: {', '.join(bases[:3])}")
        except json.JSONDecodeError:
            pass
    
    return " | ".join(parts)


def _type_to_prefix(chunk_type: str) -> str:
    """Get code prefix for chunk type."""
    mapping = {
        ChunkType.FUNCTION: "def",
        ChunkType.ASYNC_FUNCTION: "async def",
        ChunkType.CLASS: "class",
        ChunkType.METHOD: "def",
        ChunkType.ASYNC_METHOD: "async def",
    }
    return mapping.get(chunk_type, "def")


def _truncate_docstring(docstring: str, max_len: int = 150) -> str:
    """Truncate docstring to first sentence or max length."""
    if not docstring:
        return ""
    
    doc = docstring.strip()
    
    # Take first sentence
    for end in [". ", ".\n", ".\t"]:
        idx = doc.find(end)
        if 0 < idx < max_len:
            return doc[:idx + 1]
    
    # Or truncate
    if len(doc) <= max_len:
        return doc
    
    return doc[:max_len].rsplit(" ", 1)[0] + "..."


def estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def generate_descriptors_for_scan(db: Session, scan_id: int) -> int:
    """
    Generate descriptors for all chunks in scan.
    
    Returns number generated.
    """
    chunks = db.query(ArchCodeChunk).filter(
        ArchCodeChunk.scan_id == scan_id,
        ArchCodeChunk.chunk_type.in_(ChunkType.EMBEDDABLE),
    ).all()
    
    count = 0
    for chunk in chunks:
        descriptor = generate_chunk_descriptor(chunk)
        chunk.descriptor = descriptor
        chunk.descriptor_tokens = estimate_tokens(descriptor)
        count += 1
    
    db.commit()
    return count

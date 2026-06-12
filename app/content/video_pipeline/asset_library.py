# FILE: app/content/video_pipeline/asset_library.py
# Purpose: Video Asset Library — semantic search over previously downloaded clips.
# Called-by: app.content.video_pipeline.asset_resolver, app.content.video_pipeline.bake_segment, app.content.video_pipeline.orchestrator
# Depends-on: app.db, app.embeddings.service, app.memory.rag_entries_model
# Last-renovated: 2026-06-11
"""
Video Asset Library — semantic search over previously downloaded clips.

Every clip the pipeline downloads (Pexels, fal.ai, HeyGen) gets indexed
here with its visual description, keywords, and a Gemini embedding.
Future pipeline runs search the library FIRST before calling any API.

Uses the existing rag_entries table with domain='video_asset'.
Embeddings are Gemini Embedding 2 (1536d), same as the rest of RAG.

Enhanced with rich clip analysis: when clips are indexed, Gemini
analyzes the actual visual content and stores a detailed description,
quality score, content tags, and camera motion data.
"""
import json
import logging
import os
import struct
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

VIDEO_ASSET_DOMAIN = "video_asset"
VIDEO_ASSET_PROJECT = "astra-core"

# Clip cooldown: clips used in a video cannot be reused for this
# many days. Prevents the same b-roll appearing across videos.
CLIP_COOLDOWN_DAYS = 90
CLIP_USAGE_FILE = Path("data/content/video_pipeline/clip_usage.json")


def _load_clip_usage() -> Dict[str, str]:
    """Load {file_path: last_used_iso} from disk."""
    if CLIP_USAGE_FILE.exists():
        try:
            return json.loads(CLIP_USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_clip_usage(usage: Dict[str, str]) -> None:
    """Persist clip usage to disk."""
    CLIP_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLIP_USAGE_FILE.write_text(
        json.dumps(usage, indent=2), encoding="utf-8",
    )


def mark_clip_used(file_path: str) -> None:
    """Record that a clip was used in a video today."""
    from datetime import datetime, timezone
    usage = _load_clip_usage()
    key = os.path.abspath(file_path)
    usage[key] = datetime.now(timezone.utc).isoformat()
    _save_clip_usage(usage)
    logger.debug(f"[asset_library] Marked used: {os.path.basename(file_path)}")


def is_clip_on_cooldown(file_path: str) -> bool:
    """Check if a clip was used within the cooldown period."""
    from datetime import datetime, timezone, timedelta
    usage = _load_clip_usage()
    key = os.path.abspath(file_path)
    last_used = usage.get(key)
    if not last_used:
        return False
    try:
        used_dt = datetime.fromisoformat(last_used)
        cutoff = datetime.now(timezone.utc) - timedelta(days=CLIP_COOLDOWN_DAYS)
        return used_dt > cutoff
    except Exception:
        return False


def index_asset(
    file_path: str,
    source: str,
    segment_id: str,
    search_keywords: List[str],
    visual_description: str = "",
    duration_s: float = 0.0,
    cost_usd: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
    clip_analysis: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Index a downloaded video asset for future semantic search.

    Generates a Gemini embedding from the visual description + keywords,
    then stores it in rag_entries with domain='video_asset'.

    If clip_analysis is provided (from clip_analyzer.py), the rich
    description replaces the search-query-based description for
    much more accurate future matching.

    Args:
        file_path: Path to the downloaded video clip
        source: Where it came from (pexels, fal_ai, heygen)
        segment_id: The segment it was resolved for
        search_keywords: Keywords used to find this clip
        visual_description: What the clip shows visually
        duration_s: Clip duration in seconds
        cost_usd: How much it cost
        metadata: Any extra metadata
        clip_analysis: Rich analysis from clip_analyzer (optional)

    Returns:
        True if indexed successfully
    """
    if not file_path or not os.path.exists(file_path):
        return False

    # If we have rich analysis from Gemini, use it for the embedding
    # text instead of the search keywords. This means future searches
    # match against what is ACTUALLY in the clip, not what we hoped
    # to find when we searched Pexels.
    if clip_analysis:
        rich_desc = clip_analysis.get("description", "")
        tags = clip_analysis.get("content_tags", [])
        motion = clip_analysis.get("camera_motion", "")
        embed_text = (
            f"{rich_desc} "
            f"{' '.join(tags)} "
            f"camera: {motion}"
        ).strip()
        # Update the visual description with the rich one
        if rich_desc:
            visual_description = rich_desc
    else:
        # Fallback: combine keywords + description (original behaviour)
        embed_text = f"{' '.join(search_keywords)}. {visual_description}".strip()

    if not embed_text:
        embed_text = f"video clip from {source}"

    # Generate embedding
    try:
        from app.embeddings.service import generate_embedding
        embedding = generate_embedding(embed_text, task_type="RETRIEVAL_DOCUMENT")
        if not embedding:
            logger.warning(
                f"[asset_library] Failed to generate embedding for "
                f"{segment_id}"
            )
            return False
    except Exception as e:
        logger.warning(f"[asset_library] Embedding generation failed: {e}")
        return False

    # Build chunk text with all metadata for keyword fallback
    chunk_data = {
        "file_path": file_path,
        "source": source,
        "segment_id": segment_id,
        "keywords": search_keywords,
        "visual_description": visual_description,
        "duration_s": duration_s,
        "cost_usd": cost_usd,
        "metadata": metadata or {},
    }

    # Merge rich analysis into the stored data
    if clip_analysis:
        chunk_data["analysis"] = {
            "description": clip_analysis.get("description", ""),
            "content_tags": clip_analysis.get("content_tags", []),
            "quality_score": clip_analysis.get("quality_score", 0),
            "dominant_colours": clip_analysis.get("dominant_colours", []),
            "camera_motion": clip_analysis.get("camera_motion", ""),
        }

    chunk_text = json.dumps(chunk_data)

    # Store in rag_entries
    try:
        from app.db import SessionLocal
        from app.memory.rag_entries_model import RAGEntry

        # Pack embedding as binary (same format as other RAG entries)
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        with SessionLocal() as db:
            # Check for duplicate (same file_path)
            existing = db.query(RAGEntry).filter(
                RAGEntry.domain == VIDEO_ASSET_DOMAIN,
                RAGEntry.file_path == file_path,
                RAGEntry.status == "ACTIVE",
            ).first()

            if existing:
                # If we have new analysis data, update the existing entry
                if clip_analysis:
                    existing.chunk_text = chunk_text
                    existing.embedding = embedding_bytes
                    db.commit()
                    logger.info(
                        f"[asset_library] Updated with rich analysis: "
                        f"{os.path.basename(file_path)}"
                    )
                else:
                    logger.debug(
                        f"[asset_library] Already indexed: {file_path}"
                    )
                return True

            entry = RAGEntry(
                project_id=VIDEO_ASSET_PROJECT,
                domain=VIDEO_ASSET_DOMAIN,
                file_path=file_path,
                chunk_text=chunk_text,
                embedding=embedding_bytes,
                status="ACTIVE",
                ingest_source="video_pipeline",
            )
            db.add(entry)
            db.commit()

        quality = ""
        if clip_analysis:
            quality = f", quality={clip_analysis.get('quality_score', '?')}/10"
        logger.info(
            f"[asset_library] Indexed {source} asset: "
            f"{os.path.basename(file_path)} "
            f"({len(search_keywords)} keywords{quality})"
        )
        return True

    except Exception as e:
        logger.warning(f"[asset_library] Failed to store entry: {e}")
        return False


def search_library(
    query: str,
    max_results: int = 5,
    min_similarity: float = 0.5,
    min_quality: int = 0,
) -> List[Dict[str, Any]]:
    """
    Search the asset library for clips matching a visual description.

    Uses Gemini embedding similarity to find the best matching clips
    from previous pipeline runs.

    Args:
        query: Visual description or keywords to search for
        max_results: Maximum clips to return
        min_similarity: Minimum cosine similarity threshold
        min_quality: Minimum quality score (0-10) to include

    Returns:
        List of dicts with file_path, similarity, and metadata
    """
    # Generate query embedding
    try:
        from app.embeddings.service import generate_embedding
        query_embedding = generate_embedding(
            query, task_type="RETRIEVAL_QUERY",
        )
        if not query_embedding:
            return []
    except Exception as e:
        logger.warning(f"[asset_library] Query embedding failed: {e}")
        return []

    # Search rag_entries
    try:
        from app.db import SessionLocal
        from app.memory.rag_entries_model import RAGEntry

        with SessionLocal() as db:
            entries = db.query(RAGEntry).filter(
                RAGEntry.domain == VIDEO_ASSET_DOMAIN,
                RAGEntry.status == "ACTIVE",
                RAGEntry.embedding.isnot(None),
            ).all()

            if not entries:
                return []

            # Score each entry by cosine similarity
            results = []
            for entry in entries:
                try:
                    # Unpack embedding
                    n_floats = len(entry.embedding) // 4
                    stored_emb = list(
                        struct.unpack(f"{n_floats}f", entry.embedding)
                    )

                    sim = _cosine_similarity(query_embedding, stored_emb)
                    if sim >= min_similarity:
                        # Parse the metadata from chunk_text
                        data = json.loads(entry.chunk_text)

                        # Filter by quality score if analysis exists
                        if min_quality > 0:
                            analysis = data.get("analysis", {})
                            quality = analysis.get("quality_score", 10)
                            if quality < min_quality:
                                continue

                        data["similarity"] = round(sim, 4)
                        data["rag_entry_id"] = entry.id
                        results.append(data)
                except Exception:
                    continue

            # Filter out clips on cooldown (used in last 90 days)
            fresh_results = []
            for r in results:
                fp = r.get("file_path", "")
                if fp and is_clip_on_cooldown(fp):
                    logger.debug(
                        f"[asset_library] Skipping {os.path.basename(fp)} "
                        f"(on {CLIP_COOLDOWN_DAYS}-day cooldown)"
                    )
                    continue
                fresh_results.append(r)

            # Sort by similarity descending
            fresh_results.sort(key=lambda x: x["similarity"], reverse=True)
            return fresh_results[:max_results]

    except Exception as e:
        logger.warning(f"[asset_library] Search failed: {e}")
        return []


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_library_stats() -> Dict[str, Any]:
    """Get stats about the asset library."""
    try:
        from app.db import SessionLocal
        from app.memory.rag_entries_model import RAGEntry
        from sqlalchemy import func

        with SessionLocal() as db:
            total = db.query(func.count(RAGEntry.id)).filter(
                RAGEntry.domain == VIDEO_ASSET_DOMAIN,
                RAGEntry.status == "ACTIVE",
            ).scalar() or 0

            return {"total_assets": total}
    except Exception:
        return {"total_assets": 0}

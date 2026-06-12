# FILE: app/memory/ingest/__init__.py
# Purpose: Document ingestion pipeline package (Spec Section 9).
# Called-by: app.memory.ingest.pipeline
# Depends-on: app.memory.ingest.classifier, app.memory.ingest.parsers, app.memory.ingest.pipeline
# Last-renovated: 2026-06-11
"""
Document ingestion pipeline package (Spec Section 9).

5-stage pipeline: Parse → Extract → Classify → Deduplicate → Store

Public API:
    IngestPipeline  — Main pipeline class
    parse_file      — Direct access to format parsers
    classify_item   — Direct access to classifier
    check_duplicate — Direct access to deduplicator

Usage:
    from app.memory.ingest import IngestPipeline

    pipeline = IngestPipeline()
    result = pipeline.ingest_file("conversations.json")
    result = pipeline.ingest_gpt_export("chatgpt_export.json")

    # Review low-confidence items
    for i, item in enumerate(pipeline.review_queue):
        print(f"{i}: [{item.confidence}] {item.text[:60]}")
    pipeline.approve_review(0)
    pipeline.reject_review(1)
"""

from app.memory.ingest.pipeline import (
    IngestPipeline,
    IngestResult,
    ReviewItem,
)
from app.memory.ingest.parsers import (
    parse_file,
    ParsedChunk,
)
from app.memory.ingest.classifier import (
    classify_item,
    ClassifiedItem,
    REVIEW_THRESHOLD,
)
from app.memory.ingest.deduplicator import (
    check_duplicate,
    check_batch,
    DedupeResult,
)

__all__ = [
    # Pipeline
    "IngestPipeline",
    "IngestResult",
    "ReviewItem",
    # Parsers
    "parse_file",
    "ParsedChunk",
    # Classifier
    "classify_item",
    "ClassifiedItem",
    "REVIEW_THRESHOLD",
    # Deduplicator
    "check_duplicate",
    "check_batch",
    "DedupeResult",
]

# FILE: app/memory/enrichment/__init__.py
# Purpose: Media & document enrichment layer (descriptions + metadata tags).
# Called-by: app.drive.content_indexer, app.memory.upload_knowledge_hook, backfill CLI
# Depends-on: app.memory.enrichment.media_enricher, app.memory.enrichment.backfill_job
# Last-renovated: 2026-06-12
"""
Media & document enrichment (memory architecture Job 3, 2026-06-12).

Public surface:
  enrich_image / enrich_music / enrich_document_abstract — produce descriptions
  store_media_document — persist via the existing document pipeline
  hot_index_document — make a document conversationally retrievable
  run_backfill — resumable backfill over the existing corpus
"""
from app.memory.enrichment.media_enricher import (
    enrich_image,
    enrich_music,
    enrich_document_abstract,
    store_media_document,
    hot_index_document,
    llm_keys_available,
)
from app.memory.enrichment.backfill_job import run_backfill, reset_state

__all__ = [
    "enrich_image",
    "enrich_music",
    "enrich_document_abstract",
    "store_media_document",
    "hot_index_document",
    "llm_keys_available",
    "run_backfill",
    "reset_state",
]

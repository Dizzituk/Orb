# FILE: app/memory/ingest/pipeline.py
# Purpose: Document ingestion pipeline (Spec Section 9.2, Job 7A).
# Called-by: app.memory.ingest
# Depends-on: app.db, app.memory.ingest, app.memory.ingest.classifier, app.memory.ingest.parsers (+1 more)
# Last-renovated: 2026-06-11
"""
Document ingestion pipeline (Spec Section 9.2, Job 7A).

Orchestrates the 5-stage pipeline:
    1. PARSE    — Format-specific text extraction (parsers.py)
    2. EXTRACT  — Identify discrete knowledge items from chunks
    3. CLASSIFY — Tag domain, memory layer, confidence (classifier.py)
    4. DEDUP    — Check for overlapping or conflicting entries (deduplicator.py)
    5. STORE    — Write to appropriate memory table

Safety rails (7D):
    - Every stored item is source-tagged with ingest_source
    - No item stored without domain + project_id
    - Low-confidence items go to review queue (7C)
    - Unclassifiable items go to pending
    - Each run is logged with counts per domain/layer

Usage:
    from app.memory.ingest.pipeline import IngestPipeline

    pipeline = IngestPipeline()
    result = pipeline.ingest_file("conversations.json")
    # result.stored = 42
    # result.review_queue = 7
    # result.duplicates = 13
    # result.errors = 0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry
from app.memory.ingest.parsers import parse_file, ParsedChunk
from app.memory.ingest.classifier import classify_item, ClassifiedItem
from app.memory.ingest.deduplicator import check_duplicate, DedupeResult

logger = logging.getLogger(__name__)


# =========================================================================
# Ingest result
# =========================================================================

@dataclass
class IngestResult:
    """Summary of an ingest run."""
    source_file: str
    total_chunks: int = 0
    extracted: int = 0
    classified: int = 0
    stored: int = 0
    duplicates: int = 0
    conflicts: int = 0
    review_queue: int = 0
    errors: int = 0
    started_at: str = ""
    completed_at: str = ""
    domain_counts: dict = field(default_factory=dict)
    layer_counts: dict = field(default_factory=dict)
    error_details: list = field(default_factory=list)


# =========================================================================
# Review queue entry (in-memory for now)
# =========================================================================

@dataclass
class ReviewItem:
    """An item pending human review."""
    text: str
    domain: str
    memory_layer: str
    confidence: float
    reason: str                 # Why it needs review
    source_file: str
    metadata: dict = field(default_factory=dict)
    conflict_id: Optional[int] = None


# =========================================================================
# Pipeline
# =========================================================================

class IngestPipeline:
    """
    5-stage document ingestion pipeline.

    Processes files through parse→extract→classify→dedup→store,
    routing low-confidence items to a review queue.
    """

    def __init__(self, project_id: str = "astra-core"):
        self.project_id = project_id
        self._review_queue: list[ReviewItem] = []

    @property
    def review_queue(self) -> list[ReviewItem]:
        """Items awaiting human review."""
        return list(self._review_queue)

    @property
    def review_count(self) -> int:
        return len(self._review_queue)

    # -----------------------------------------------------------------
    # Main entry points
    # -----------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str,
        source: str = "ingest_pipeline",
    ) -> IngestResult:
        """
        Ingest a single file through the full pipeline.

        Args:
            file_path: Path to the file to ingest.
            source: Source tag for audit trail.

        Returns:
            IngestResult with counts and error details.
        """
        result = IngestResult(
            source_file=file_path,
            started_at=datetime.utcnow().isoformat(),
        )

        # Stage 1: PARSE
        try:
            chunks = parse_file(file_path)
            result.total_chunks = len(chunks)
        except (FileNotFoundError, ValueError) as e:
            result.errors = 1
            result.error_details.append(f"Parse error: {e}")
            result.completed_at = datetime.utcnow().isoformat()
            logger.error("[ingest] Parse failed: %s — %s", file_path, e)
            return result

        # Stage 2: EXTRACT (filter to meaningful items)
        items = self._extract(chunks)
        result.extracted = len(items)

        # Stages 3–5: Classify → Dedup → Store
        for item in items:
            try:
                self._process_item(item, source, result)
            except Exception as e:
                result.errors += 1
                result.error_details.append(
                    f"Item error (chunk {item.chunk_index}): {e}"
                )
                logger.error(
                    "[ingest] Item processing error: %s", e, exc_info=True,
                )

        result.completed_at = datetime.utcnow().isoformat()

        logger.info(
            "[ingest] %s complete: %d parsed, %d extracted, "
            "%d stored, %d dupes, %d review, %d errors",
            file_path, result.total_chunks, result.extracted,
            result.stored, result.duplicates,
            result.review_queue, result.errors,
        )

        return result

    def ingest_gpt_export(self, file_path: str) -> IngestResult:
        """
        Ingest an OpenAI GPT conversation export (Job 7B).

        Convenience wrapper that sets appropriate source tag.
        The parser auto-detects GPT export format from JSON structure.
        """
        return self.ingest_file(file_path, source="gpt_export")

    # -----------------------------------------------------------------
    # Review queue management (7C)
    # -----------------------------------------------------------------

    def approve_review(self, index: int) -> Optional[int]:
        """
        Approve a review queue item — store it into memory.

        Args:
            index: Index into the review queue.

        Returns:
            The stored entry ID, or None if index is invalid.
        """
        if index < 0 or index >= len(self._review_queue):
            return None

        item = self._review_queue.pop(index)
        entry_id = self._store_item(
            text=item.text,
            domain=item.domain,
            memory_layer=item.memory_layer,
            source=item.source_file,
            ingest_source="review_approved",
            metadata=item.metadata,
        )
        logger.info(
            "[ingest] Review approved: entry %d, domain=%s",
            entry_id, item.domain,
        )
        return entry_id

    def reject_review(self, index: int) -> bool:
        """
        Reject a review queue item — discard it.

        Returns True if successfully removed.
        """
        if index < 0 or index >= len(self._review_queue):
            return False

        item = self._review_queue.pop(index)
        logger.info(
            "[ingest] Review rejected: '%s' (domain=%s)",
            item.text[:50], item.domain,
        )
        return True

    def clear_review_queue(self) -> int:
        """Clear all pending review items. Returns count cleared."""
        count = len(self._review_queue)
        self._review_queue.clear()
        return count

    # -----------------------------------------------------------------
    # Stage 2: EXTRACT
    # -----------------------------------------------------------------

    def _extract(self, chunks: list[ParsedChunk]) -> list[ParsedChunk]:
        """
        Filter parsed chunks to meaningful knowledge items.

        Removes:
            - Very short chunks (under 20 chars)
            - Pure code blocks without explanatory context
            - System/tool messages from GPT exports
            - Timestamps and metadata-only chunks
        """
        meaningful = []
        for chunk in chunks:
            text = chunk.text.strip()

            # Skip very short content
            if len(text) < 20:
                continue

            # Skip system messages in GPT exports
            role = chunk.metadata.get("role", "")
            if role in ("system", "tool"):
                continue

            # Skip pure code without natural language
            if self._is_pure_code(text):
                continue

            meaningful.append(chunk)

        return meaningful

    @staticmethod
    def _is_pure_code(text: str) -> bool:
        """Check if text is purely code with no explanation."""
        lines = text.split("\n")
        if len(lines) < 3:
            return False

        code_indicators = 0
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith(("def ", "class ", "import ", "from "))
                or stripped.startswith(("#!", "//", "/*"))
                or stripped.endswith(("{", "}", ";"))
                or stripped.startswith("```")
            ):
                code_indicators += 1

        return code_indicators / len(lines) > 0.7

    # -----------------------------------------------------------------
    # Stage 3+4+5: Classify → Dedup → Store
    # -----------------------------------------------------------------

    def _process_item(
        self,
        chunk: ParsedChunk,
        source: str,
        result: IngestResult,
    ) -> None:
        """Process a single extracted item through stages 3-5."""

        # Stage 3: CLASSIFY
        classified = classify_item(
            text=chunk.text,
            source=source,
            role=chunk.metadata.get("role"),
            source_file=chunk.source_file,
            metadata=chunk.metadata,
            project_id=self.project_id,
        )
        result.classified += 1
        _incr(result.domain_counts, classified.domain)
        _incr(result.layer_counts, classified.memory_layer)

        # Safety rail: no unclassifiable items bypass
        if classified.domain == "general" and classified.confidence < 0.3:
            self._review_queue.append(ReviewItem(
                text=classified.text,
                domain=classified.domain,
                memory_layer=classified.memory_layer,
                confidence=classified.confidence,
                reason="Unclassifiable (very low confidence)",
                source_file=classified.source_file,
                metadata=classified.metadata,
            ))
            result.review_queue += 1
            return

        # Low confidence → review queue (7C)
        if classified.needs_review:
            self._review_queue.append(ReviewItem(
                text=classified.text,
                domain=classified.domain,
                memory_layer=classified.memory_layer,
                confidence=classified.confidence,
                reason=f"Low confidence ({classified.confidence:.2f})",
                source_file=classified.source_file,
                metadata=classified.metadata,
            ))
            result.review_queue += 1
            return

        # Stage 4: DEDUP
        dedup = check_duplicate(
            text=classified.text,
            domain=classified.domain,
            project_id=self.project_id,
            memory_layer=classified.memory_layer,
        )

        if dedup.status == "DUPLICATE":
            result.duplicates += 1
            return

        if dedup.status == "CONFLICT":
            self._review_queue.append(ReviewItem(
                text=classified.text,
                domain=classified.domain,
                memory_layer=classified.memory_layer,
                confidence=classified.confidence,
                reason=f"Conflict: {dedup.reason}",
                source_file=classified.source_file,
                metadata=classified.metadata,
                conflict_id=dedup.existing_id,
            ))
            result.conflicts += 1
            result.review_queue += 1
            return

        # Stage 5: STORE
        self._store_item(
            text=classified.text,
            domain=classified.domain,
            memory_layer=classified.memory_layer,
            source=classified.source,
            ingest_source=source,
            metadata=classified.metadata,
        )
        result.stored += 1

    def _store_item(
        self,
        text: str,
        domain: str,
        memory_layer: str,
        source: str,
        ingest_source: str,
        metadata: dict,
    ) -> int:
        """Write an item to the rag_entries table."""
        db = get_db_session()
        try:
            entry = RAGEntry(
                project_id=self.project_id,
                domain=domain,
                chunk_text=text,
                status="ACTIVE",
                package_role=memory_layer,
                ingest_source=ingest_source,
                indexed_at=datetime.utcnow(),
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            return entry.id
        finally:
            db.close()


# =========================================================================
# Helpers
# =========================================================================

def _incr(counts: dict, key: str) -> None:
    """Increment a counter in a dict."""
    counts[key] = counts.get(key, 0) + 1

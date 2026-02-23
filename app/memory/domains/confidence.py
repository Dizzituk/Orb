# FILE: app/memory/domains/confidence.py
"""
Confidence Score domain store (Spec Section 4.5).

Stores phrase-to-intent confidence mappings for the translation layer.
Fundamentally different from other domains:
  - Retrieval: fuzzy string matching, NOT semantic search
  - Lifecycle: reinforce or decay, NOT replace or accumulate
  - No embeddings needed

The confidence_scores table (app/memory/confidence_model.py) is the
backing store. This module provides the ConfidenceStore interface
plus the scoring formula from the spec:

    confidence = (confirmations / (confirmations + corrections + 2)) * recency_factor
    recency_factor = 1.0 if used within 30 days, else 0.95^(days_since_30)

Risk-adjusted thresholds (Spec Section 5):
    Destructive (delete, push): always confirm — threshold locked at 1.0
    State-changing (pipeline, refactor): standard — 0.9
    Read-only (show, status): reduced — 0.7
    Chat: no threshold

Usage:
    store = ConfidenceStore()
    result = store.match("run the scan", context_tags=["idle"])
    store.reinforce("run the scan", "SCAN_CODEBASE")
    store.correct("run the scan", "SCAN_CODEBASE", "CHAT_ONLY")
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.confidence_model import ConfidenceScore
from app.memory.schemas_unified import MemoryResult, StoreRequest, DomainStats

logger = logging.getLogger(__name__)

DOMAIN = "confidence"

# Decay parameters
RECENCY_DAYS = 30
DECAY_RATE = 0.95

# Risk-adjusted thresholds
THRESHOLDS = {
    "destructive": 1.0,     # delete, push, wipe — always confirm
    "state_changing": 0.9,  # pipeline, refactor, build
    "read_only": 0.7,       # show, status, list
    "chat": 0.0,            # no threshold
}

# Intent risk classification
DESTRUCTIVE_INTENTS = {
    "DELETE_FILE", "PURGE_QUARANTINE", "GIT_PUSH",
    "WIPE_DOMAIN", "RESET_STATE",
}
STATE_CHANGING_INTENTS = {
    "RUN_PIPELINE", "REFACTOR", "SCAN_CODEBASE",
    "CREATE_ARCHITECTURE_MAP", "BUILD_SPEC",
    "EXECUTE_SANDBOX", "START_JOB",
}
READ_ONLY_INTENTS = {
    "SHOW_STATUS", "LIST_FILES", "SHOW_ARCHITECTURE",
    "SHOW_LOGS", "SHOW_PREFERENCES", "DESCRIBE_FILE",
}


class ConfidenceStore:
    """
    DomainStore for phrase-to-intent confidence scores.

    Uses fuzzy string matching for retrieval and reinforcement/decay
    for lifecycle management.
    """

    @property
    def domain_name(self) -> str:
        return DOMAIN

    # -----------------------------------------------------------------
    # Core operations
    # -----------------------------------------------------------------

    def match(
        self,
        phrase: str,
        context_tags: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """
        Find the best matching intent for a phrase.

        Uses normalised substring matching against stored patterns.
        Returns None if no match found or confidence below threshold.

        Args:
            phrase: The user's input phrase.
            context_tags: Optional context for filtering.

        Returns:
            Dict with intent, confidence, threshold, auto_execute,
            or None if no match.
        """
        db = get_db_session()
        try:
            normalised = _normalise(phrase)
            candidates = self._find_candidates(db, normalised, context_tags)

            if not candidates:
                return None

            # Pick the best match
            best = max(candidates, key=lambda c: c["score"])

            # Apply recency decay to stored confidence
            raw_conf = best["record"].confidence
            recency = _recency_factor(best["record"].last_used)
            effective_conf = raw_conf * recency

            # Determine threshold for this intent
            threshold = _get_threshold(best["record"].intent)
            auto_execute = effective_conf >= threshold

            return {
                "intent": best["record"].intent,
                "confidence": round(effective_conf, 4),
                "raw_confidence": round(raw_conf, 4),
                "recency_factor": round(recency, 4),
                "threshold": threshold,
                "auto_execute": auto_execute,
                "phrase_pattern": best["record"].phrase_pattern,
                "match_score": round(best["score"], 4),
                "confirmations": best["record"].confirmations,
                "corrections": best["record"].corrections,
            }
        finally:
            db.close()

    def reinforce(
        self,
        phrase: str,
        intent: str,
        context_tags: Optional[list[str]] = None,
    ) -> float:
        """
        Reinforce a phrase-to-intent mapping.

        If the mapping exists, increment confirmations and recalculate.
        If not, create it with initial confidence.

        Returns the new confidence score.
        """
        db = get_db_session()
        try:
            normalised = _normalise(phrase)
            record = self._get_exact(db, normalised, intent)

            if record:
                record.confirmations += 1
                record.last_used = datetime.utcnow()
                record.confidence = _calculate_confidence(
                    record.confirmations, record.corrections
                )
                db.commit()
                logger.debug(
                    "[confidence] Reinforced: '%s' → %s (conf=%.3f)",
                    normalised[:40], intent, record.confidence,
                )
                return record.confidence
            else:
                # Create new mapping
                tags_json = json.dumps(context_tags) if context_tags else None
                record = ConfidenceScore(
                    phrase_pattern=normalised,
                    intent=intent,
                    confidence=_calculate_confidence(1, 0),
                    confirmations=1,
                    corrections=0,
                    last_used=datetime.utcnow(),
                    context_tags=tags_json,
                )
                db.add(record)
                db.commit()
                logger.debug(
                    "[confidence] Created: '%s' → %s (conf=%.3f)",
                    normalised[:40], intent, record.confidence,
                )
                return record.confidence
        finally:
            db.close()

    def correct(
        self,
        phrase: str,
        wrong_intent: str,
        right_intent: str,
        context_tags: Optional[list[str]] = None,
    ) -> tuple[float, float]:
        """
        Correct a phrase-to-intent mapping.

        Increments corrections on the wrong mapping and reinforces
        the correct one.

        Returns (wrong_new_confidence, right_new_confidence).
        """
        # Step 1: Penalise the wrong mapping
        wrong_conf = 0.0
        db = get_db_session()
        try:
            normalised = _normalise(phrase)
            wrong_record = self._get_exact(db, normalised, wrong_intent)
            if wrong_record:
                wrong_record.corrections += 1
                wrong_record.confidence = _calculate_confidence(
                    wrong_record.confirmations, wrong_record.corrections
                )
                wrong_conf = wrong_record.confidence
                db.commit()
        finally:
            db.close()

        # Step 2: Reinforce the correct mapping (opens its own session)
        right_conf = self.reinforce(phrase, right_intent, context_tags)

        logger.info(
            "[confidence] Corrected: '%s' %s(%.3f) → %s(%.3f)",
            normalised[:40], wrong_intent, wrong_conf,
            right_intent, right_conf,
        )
        return (wrong_conf, right_conf)

    def decay_stale(self, days_threshold: int = 30) -> int:
        """
        Decay confidence on mappings not used within threshold.

        Recalculates confidence with recency factor applied.
        Entries with confidence below 0.1 after decay are purged.

        Returns count of entries decayed.
        """
        db = get_db_session()
        try:
            cutoff = datetime.utcnow()
            entries = db.query(ConfidenceScore).all()

            decayed = 0
            purged = 0
            for entry in entries:
                recency = _recency_factor(entry.last_used, cutoff)
                if recency < 1.0:
                    new_conf = entry.confidence * recency
                    if new_conf < 0.1:
                        db.delete(entry)
                        purged += 1
                    else:
                        entry.confidence = round(new_conf, 4)
                        decayed += 1

            if decayed or purged:
                db.commit()
                logger.info(
                    "[confidence] Decay: %d decayed, %d purged",
                    decayed, purged,
                )

            return decayed + purged
        finally:
            db.close()

    # -----------------------------------------------------------------
    # DomainStore Protocol: query
    # -----------------------------------------------------------------

    def query(
        self,
        text: str,
        project_id: str = "astra-core",
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[MemoryResult]:
        """
        Search confidence entries by phrase matching.

        Note: confidence entries are not project-scoped (they apply
        globally to the translation layer).
        """
        db = get_db_session()
        try:
            normalised = _normalise(text)
            candidates = self._find_candidates(db, normalised)

            results = []
            for c in sorted(candidates, key=lambda x: x["score"], reverse=True)[:limit]:
                rec = c["record"]
                if c["score"] >= min_relevance:
                    results.append(MemoryResult(
                        id=rec.id,
                        domain=DOMAIN,
                        content=f"{rec.phrase_pattern} → {rec.intent} (conf={rec.confidence:.3f})",
                        project_id=project_id,
                        relevance=c["score"],
                        source_table="confidence_scores",
                        status="active",
                        metadata={
                            "phrase_pattern": rec.phrase_pattern,
                            "intent": rec.intent,
                            "confidence": rec.confidence,
                            "confirmations": rec.confirmations,
                            "corrections": rec.corrections,
                        },
                    ))
            return results
        finally:
            db.close()

    def store(self, request: StoreRequest) -> int:
        """Store via reinforce — confidence entries are created by learning."""
        intent = request.metadata.get("intent", "UNKNOWN")
        self.reinforce(request.content, intent)
        return 0

    def quarantine(self, file_path: str, project_id: str = "astra-core") -> int:
        """Confidence entries don't quarantine."""
        return 0

    def count(self, project_id: str = "astra-core") -> int:
        db = get_db_session()
        try:
            return db.query(func.count(ConfidenceScore.id)).scalar() or 0
        finally:
            db.close()

    def get_stats(self, project_id: str = "astra-core") -> DomainStats:
        db = get_db_session()
        try:
            total = db.query(func.count(ConfidenceScore.id)).scalar() or 0
            high_conf = (
                db.query(func.count(ConfidenceScore.id))
                .filter(ConfidenceScore.confidence >= 0.7)
                .scalar() or 0
            )
            return DomainStats(
                domain=DOMAIN,
                total_entries=total,
                active_entries=total,
                embedded_entries=high_conf,  # Reuse field for high-confidence count
            )
        finally:
            db.close()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _find_candidates(
        self,
        db: Session,
        normalised: str,
        context_tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """Find candidate matches using fuzzy substring matching."""
        if not normalised:
            return []

        # Get all entries and score by similarity
        entries = db.query(ConfidenceScore).all()

        candidates = []
        for entry in entries:
            score = _fuzzy_score(normalised, entry.phrase_pattern)
            if score > 0.3:
                # Context tag filtering
                if context_tags and entry.context_tags:
                    try:
                        stored_tags = json.loads(entry.context_tags)
                        tag_overlap = len(set(context_tags) & set(stored_tags))
                        if tag_overlap > 0:
                            score += 0.1 * tag_overlap
                    except json.JSONDecodeError:
                        pass

                candidates.append({"record": entry, "score": score})

        return candidates

    def _get_exact(
        self,
        db: Session,
        normalised: str,
        intent: str,
    ) -> Optional[ConfidenceScore]:
        """Get exact phrase+intent match."""
        return (
            db.query(ConfidenceScore)
            .filter(
                ConfidenceScore.phrase_pattern == normalised,
                ConfidenceScore.intent == intent,
            )
            .first()
        )


# =========================================================================
# Scoring functions
# =========================================================================

def _calculate_confidence(confirmations: int, corrections: int) -> float:
    """
    Spec formula:
    confidence = confirmations / (confirmations + corrections + 2)

    The +2 is Bayesian smoothing — prevents 1/1 = 100% confidence
    from a single confirmation.
    """
    return confirmations / (confirmations + corrections + 2)


def _recency_factor(
    last_used: Optional[datetime],
    now: Optional[datetime] = None,
) -> float:
    """
    Spec formula:
    1.0 if used within 30 days, else 0.95^(days_since_30)
    """
    if not last_used:
        return 0.5  # Unknown recency — neutral

    now = now or datetime.utcnow()
    days_since = (now - last_used).total_seconds() / 86400.0

    if days_since <= RECENCY_DAYS:
        return 1.0

    days_over = days_since - RECENCY_DAYS
    return DECAY_RATE ** days_over


def _get_threshold(intent: str) -> float:
    """Get the risk-adjusted confidence threshold for an intent."""
    upper = intent.upper()
    if upper in DESTRUCTIVE_INTENTS:
        return THRESHOLDS["destructive"]
    if upper in STATE_CHANGING_INTENTS:
        return THRESHOLDS["state_changing"]
    if upper in READ_ONLY_INTENTS:
        return THRESHOLDS["read_only"]
    return THRESHOLDS["chat"]


def _normalise(text: str) -> str:
    """Normalise a phrase for matching."""
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def _fuzzy_score(query: str, pattern: str) -> float:
    """
    Score similarity between query and stored pattern.

    Uses token overlap ratio — simple but effective for
    short command-like phrases.
    """
    if not query or not pattern:
        return 0.0

    # Exact match
    if query == pattern:
        return 1.0

    # Substring match
    if query in pattern or pattern in query:
        shorter = min(len(query), len(pattern))
        longer = max(len(query), len(pattern))
        return 0.7 + (0.3 * shorter / longer)

    # Token overlap
    q_tokens = set(query.split())
    p_tokens = set(pattern.split())

    if not q_tokens or not p_tokens:
        return 0.0

    overlap = q_tokens & p_tokens
    union = q_tokens | p_tokens

    return len(overlap) / len(union) if union else 0.0

# FILE: app/translation/phrase_cache.py
"""
Phrase → Intent cache for ASTRA Translation Layer.
Supports learning shortcuts from usage so more messages stay on Tier 0.

The cache is updated when:
1. A Tier 1 classification succeeds with high confidence
2. User gives explicit feedback

Future messages matching cached patterns:
- Use cached mapping directly (Tier 0)
- Skip the classifier
- Still pass through context + confirmation gates for safety
"""
from __future__ import annotations
import re
import json
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
from .schemas import CanonicalIntent, PhraseCacheEntry, LatencyTier

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Confidence threshold for auto-caching Tier 1 results
AUTO_CACHE_CONFIDENCE_THRESHOLD = 0.85

# Max entries in cache before cleanup
MAX_CACHE_SIZE = 1000

# Minimum hits before promoting to Tier 0 rules
MIN_HITS_FOR_PROMOTION = 5


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_phrase(text: str) -> str:
    """
    Normalize a phrase for cache matching.
    
    Normalization:
    - Lowercase
    - Strip whitespace
    - Collapse multiple spaces
    - Remove punctuation (except for special command patterns)
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Keep colons for wake phrases like "astra, feedback:"
    text = re.sub(r'[^\w\s:,]', '', text)
    return text


def extract_pattern(text: str) -> str:
    """
    Extract a generalizable pattern from text.
    
    E.g., "Create architecture map for my-project" 
       -> "create architecture map for {target}"
    """
    normalized = normalize_phrase(text)
    
    # Replace specific identifiers with placeholders
    patterns_to_generalize = [
        (r'\b[a-f0-9\-]{36}\b', '{uuid}'),           # UUIDs
        (r'\b[a-f0-9]{8,}\b', '{id}'),               # Hex IDs
        (r'\bjob[_\-]?\d+\b', '{job}'),              # job-1, job_2, etc.
        (r'\bfor\s+[\w\-]+$', 'for {target}'),       # "for my-project"
        (r'\bsandbox[_\-]?\d*\b', '{sandbox}'),      # sandbox, sandbox-1
    ]
    
    result = normalized
    for pattern, replacement in patterns_to_generalize:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


# =============================================================================
# PHRASE CACHE
# =============================================================================

class PhraseCache:
    """
    Per-user phrase → intent cache.
    Enables learning shortcuts from usage patterns.
    """
    
    def __init__(self, user_id: str, cache_dir: Optional[Path] = None):
        """
        Initialize cache for a user.
        
        Args:
            user_id: Unique user identifier
            cache_dir: Directory to persist cache (optional)
        """
        self.user_id = user_id
        self.cache_dir = cache_dir
        self._cache: Dict[str, PhraseCacheEntry] = {}
        self._load_cache()
    
    def lookup(self, text: str) -> Optional[Tuple[CanonicalIntent, PhraseCacheEntry]]:
        """
        Look up a phrase in the cache.
        
        Returns:
            (intent, entry) if found, None otherwise
        """
        # Try exact normalized match first
        normalized = normalize_phrase(text)
        if normalized in self._cache:
            entry = self._cache[normalized]
            self._record_hit(normalized)
            return entry.intent, entry
        
        # Try pattern match
        pattern = extract_pattern(text)
        if pattern != normalized and pattern in self._cache:
            entry = self._cache[pattern]
            self._record_hit(pattern)
            return entry.intent, entry
        
        return None
    
    def add(
        self,
        text: str,
        intent: CanonicalIntent,
        confidence: float = 1.0,
        source: str = "feedback",
    ) -> PhraseCacheEntry:
        """
        Add a phrase → intent mapping to the cache.
        
        Args:
            text: The original phrase
            intent: The canonical intent
            confidence: Classification confidence
            source: How this entry was added
            
        Returns:
            The created cache entry
        """
        normalized = normalize_phrase(text)
        
        entry = PhraseCacheEntry(
            pattern=normalized,
            intent=intent,
            confidence=confidence,
            source=source,
        )
        
        self._cache[normalized] = entry
        
        # Also add pattern version for generalization
        pattern = extract_pattern(text)
        if pattern != normalized:
            pattern_entry = PhraseCacheEntry(
                pattern=pattern,
                intent=intent,
                confidence=confidence * 0.95,  # Slightly lower for patterns
                source=source,
            )
            self._cache[pattern] = pattern_entry
        
        self._save_cache()
        logger.info(f"Added to phrase cache: '{normalized}' -> {intent.value}")
        
        return entry
    
    def add_from_tier1(
        self,
        text: str,
        intent: CanonicalIntent,
        confidence: float,
    ) -> Optional[PhraseCacheEntry]:
        """
        Conditionally add an entry from a high-confidence Tier 1 classification.
        
        Only adds if confidence exceeds AUTO_CACHE_CONFIDENCE_THRESHOLD.
        """
        if confidence < AUTO_CACHE_CONFIDENCE_THRESHOLD:
            return None
        
        return self.add(
            text=text,
            intent=intent,
            confidence=confidence,
            source="high_confidence_classification",
        )
    
    def remove(self, text: str) -> bool:
        """
        Remove a phrase from the cache.
        
        Returns:
            True if removed, False if not found
        """
        normalized = normalize_phrase(text)
        if normalized in self._cache:
            del self._cache[normalized]
            self._save_cache()
            return True
        return False
    
    def update_from_feedback(
        self,
        original_text: str,
        expected_intent: CanonicalIntent,
        was_misclassified: bool,
    ) -> None:
        """
        Update cache based on user feedback.
        
        Args:
            original_text: The original message
            expected_intent: What the intent should have been
            was_misclassified: True if it was a misfire
        """
        normalized = normalize_phrase(original_text)
        
        if was_misclassified:
            # Add the correct mapping
            self.add(
                text=original_text,
                intent=expected_intent,
                confidence=1.0,
                source="feedback",
            )
        else:
            # Reinforce existing mapping if present
            if normalized in self._cache:
                entry = self._cache[normalized]
                entry.confidence = min(entry.confidence + 0.1, 1.0)
                entry.hit_count += 1
                self._save_cache()
    
    def get_promotion_candidates(self) -> List[PhraseCacheEntry]:
        """
        Get entries that have enough hits to be promoted to Tier 0 rules.
        """
        return [
            entry for entry in self._cache.values()
            if entry.hit_count >= MIN_HITS_FOR_PROMOTION
            and not entry.promoted_to_tier0
        ]
    
    def mark_promoted(self, pattern: str) -> None:
        """Mark an entry as promoted to Tier 0."""
        if pattern in self._cache:
            self._cache[pattern].promoted_to_tier0 = True
            self._save_cache()
    
    def _record_hit(self, key: str) -> None:
        """Record a cache hit."""
        if key in self._cache:
            entry = self._cache[key]
            entry.hit_count += 1
            entry.last_hit = datetime.utcnow()
    
    def _load_cache(self) -> None:
        """Load cache from disk if available."""
        if self.cache_dir is None:
            return
        
        cache_file = self.cache_dir / f"phrase_cache_{self.user_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self._cache = {
                        k: PhraseCacheEntry(**v) 
                        for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._cache)} cache entries for user {self.user_id}")
            except Exception as e:
                logger.warning(f"Failed to load phrase cache: {e}")
                self._cache = {}
    
    def _save_cache(self) -> None:
        """Persist cache to disk."""
        if self.cache_dir is None:
            return
        
        # Cleanup if too large
        if len(self._cache) > MAX_CACHE_SIZE:
            self._cleanup_cache()
        
        cache_file = self.cache_dir / f"phrase_cache_{self.user_id}.json"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                data = {
                    k: v.model_dump(mode='json') 
                    for k, v in self._cache.items()
                }
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save phrase cache: {e}")
    
    def _cleanup_cache(self) -> None:
        """Remove least-used entries when cache is too large."""
        # Sort by hit count and last hit
        entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].hit_count, x[1].last_hit or datetime.min),
        )
        
        # Keep top MAX_CACHE_SIZE / 2 entries
        keep_count = MAX_CACHE_SIZE // 2
        self._cache = dict(entries[-keep_count:])
        logger.info(f"Cleaned up phrase cache, kept {len(self._cache)} entries")


# =============================================================================
# GLOBAL CACHE MANAGER
# =============================================================================

_cache_instances: Dict[str, PhraseCache] = {}


def get_phrase_cache(user_id: str, cache_dir: Optional[Path] = None) -> PhraseCache:
    """
    Get or create a phrase cache for a user.
    
    Args:
        user_id: User identifier
        cache_dir: Optional directory for persistence
        
    Returns:
        PhraseCache instance for the user
    """
    if user_id not in _cache_instances:
        _cache_instances[user_id] = PhraseCache(user_id, cache_dir)
    return _cache_instances[user_id]


def clear_all_caches() -> None:
    """Clear all cached instances (mainly for testing)."""
    _cache_instances.clear()

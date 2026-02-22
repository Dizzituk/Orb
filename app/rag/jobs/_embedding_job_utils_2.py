from __future__ import annotations
from enum import Enum
from .embedding_job import EmbeddingJobStatus
_current_status = EmbeddingJobStatus()


class EmbeddingPriority(Enum):
    """Embedding priority tiers."""
    TIER1_CRITICAL = 1   # Entry points, routers, dispatch - semantic search useful fast
    TIER2_HIGH = 2       # Pipeline core - spec gate, overwatcher, weaver
    TIER3_MEDIUM = 3     # Services, models, DB - infrastructure queries
    TIER4_LOW = 4        # Handlers, utils, clients
    TIER5_NORMAL = 5     # Everything else

def get_embedding_status() -> EmbeddingJobStatus:
    """Get current embedding job status."""
    # Try to load from file if not running (for persistence across restarts)
    if not _current_status.running:
        loaded = EmbeddingJobStatus.load_from_file()
        if loaded:
            return loaded
    return _current_status

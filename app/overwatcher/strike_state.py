# FILE: app/overwatcher/strike_state.py
"""Strike State Management (Spec §9.4).

Tracks strikes by ErrorSignature, not attempt count.
- Same signature: increment strike
- Different signature: reset to 1
- Strike 3: HARD_STOP

Key principle: The strike counter follows the ERROR, not the attempt.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Strike Outcome
# =============================================================================

class StrikeOutcome(str, Enum):
    """Outcome after recording a strike."""
    CONTINUE = "continue"          # Strike 1 or 2 - keep trying
    DEEP_RESEARCH = "deep_research"  # Strike 2 - deeper investigation allowed
    HARD_STOP = "hard_stop"        # Strike 3 - no more retries


# =============================================================================
# Strike Record
# =============================================================================

@dataclass
class StrikeRecord:
    """Record of a single strike event."""
    job_id: str
    stage: str
    signature_hash: str
    strike_count: int
    diagnosis: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evidence: Dict[str, Any] = field(default_factory=dict)
    outcome: StrikeOutcome = StrikeOutcome.CONTINUE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "signature_hash": self.signature_hash,
            "strike_count": self.strike_count,
            "diagnosis": self.diagnosis,
            "timestamp": self.timestamp,
            "evidence": self.evidence,
            "outcome": self.outcome.value,
        }


# =============================================================================
# Strike Manager
# =============================================================================

class StrikeManager:
    """Manages strike state across a job.
    
    Spec §9.4:
    - Strike counters keyed by signature, not attempt
    - Same signature: increment
    - Different signature: reset to 1
    - Strike 2: deep research allowed
    - Strike 3: HARD_STOP
    """
    
    def __init__(self):
        # signature_hash -> current strike count
        self._strikes: Dict[str, int] = {}
        # All strike records for audit
        self._history: List[StrikeRecord] = []
        # Current signature per stage (for reset detection)
        self._current_signature: Dict[str, str] = {}
    
    def record_strike(
        self,
        *,
        job_id: str,
        stage: str,
        error_signature: Any,  # ErrorSignature object
        diagnosis: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> StrikeRecord:
        """Record a strike and return the record with outcome.
        
        Args:
            job_id: Job identifier
            stage: Pipeline stage (e.g., "verification", "spec_gate")
            error_signature: ErrorSignature object with signature_hash
            diagnosis: Human-readable diagnosis
            evidence: Optional evidence dict
        
        Returns:
            StrikeRecord with outcome (CONTINUE, DEEP_RESEARCH, or HARD_STOP)
        """
        sig_hash = error_signature.signature_hash if error_signature else "unknown"
        prev_sig = self._current_signature.get(stage)
        
        # Check if same signature
        if prev_sig and prev_sig == sig_hash:
            # Same error - increment
            self._strikes[sig_hash] = self._strikes.get(sig_hash, 0) + 1
        else:
            # Different error - reset to 1
            self._strikes[sig_hash] = 1
            self._current_signature[stage] = sig_hash
        
        strike_count = self._strikes[sig_hash]
        
        # Determine outcome
        if strike_count >= 3:
            outcome = StrikeOutcome.HARD_STOP
        elif strike_count == 2:
            outcome = StrikeOutcome.DEEP_RESEARCH
        else:
            outcome = StrikeOutcome.CONTINUE
        
        # Create record
        record = StrikeRecord(
            job_id=job_id,
            stage=stage,
            signature_hash=sig_hash,
            strike_count=strike_count,
            diagnosis=diagnosis,
            evidence=evidence or {},
            outcome=outcome,
        )
        
        self._history.append(record)
        
        logger.info(
            f"[strike_state] Strike {strike_count}/3 for {sig_hash[:16]}... "
            f"outcome={outcome.value}"
        )
        
        return record
    
    def get_strike_count(self, signature_hash: str) -> int:
        """Get current strike count for a signature."""
        return self._strikes.get(signature_hash, 0)
    
    def get_current_signature(self, stage: str) -> Optional[str]:
        """Get current signature for a stage."""
        return self._current_signature.get(stage)
    
    def get_history(self) -> List[StrikeRecord]:
        """Get all strike records."""
        return self._history.copy()
    
    def get_history_for_signature(self, signature_hash: str) -> List[StrikeRecord]:
        """Get strike records for a specific signature."""
        return [r for r in self._history if r.signature_hash == signature_hash]
    
    def is_exhausted(self, signature_hash: str) -> bool:
        """Check if signature has exhausted all strikes."""
        return self._strikes.get(signature_hash, 0) >= 3
    
    def reset(self) -> None:
        """Reset all strike state (for new job)."""
        self._strikes.clear()
        self._history.clear()
        self._current_signature.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "strikes": self._strikes.copy(),
            "history": [r.to_dict() for r in self._history],
            "current_signature": self._current_signature.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrikeManager":
        """Restore state from persistence."""
        manager = cls()
        manager._strikes = data.get("strikes", {})
        manager._current_signature = data.get("current_signature", {})
        
        for record_data in data.get("history", []):
            record = StrikeRecord(
                job_id=record_data["job_id"],
                stage=record_data["stage"],
                signature_hash=record_data["signature_hash"],
                strike_count=record_data["strike_count"],
                diagnosis=record_data["diagnosis"],
                timestamp=record_data.get("timestamp", ""),
                evidence=record_data.get("evidence", {}),
                outcome=StrikeOutcome(record_data.get("outcome", "continue")),
            )
            manager._history.append(record)
        
        return manager


# =============================================================================
# Convenience Functions
# =============================================================================

def signatures_match(sig1: Any, sig2: Any) -> bool:
    """Check if two ErrorSignature objects match."""
    if sig1 is None or sig2 is None:
        return False
    
    hash1 = sig1.signature_hash if hasattr(sig1, "signature_hash") else str(sig1)
    hash2 = sig2.signature_hash if hasattr(sig2, "signature_hash") else str(sig2)
    
    return hash1 == hash2


def compute_strike_outcome(strike_count: int) -> StrikeOutcome:
    """Compute outcome from strike count."""
    if strike_count >= 3:
        return StrikeOutcome.HARD_STOP
    elif strike_count == 2:
        return StrikeOutcome.DEEP_RESEARCH
    else:
        return StrikeOutcome.CONTINUE


__all__ = [
    "StrikeOutcome",
    "StrikeRecord",
    "StrikeManager",
    "signatures_match",
    "compute_strike_outcome",
]

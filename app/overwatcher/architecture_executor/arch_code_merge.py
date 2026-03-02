"""
Architecture Code Merge.

Takes extracted code blocks from arch_code_extractor and merges them
into the implementation pipeline. Works in two modes:

1. Direct mode (high confidence): Extracted code IS the file content.
   Bypasses LLM entirely, like the existing verbatim path but with
   better coverage.

2. Pre-fill mode (moderate confidence): Extracted code is injected
   into the prompt as pre-filled content. The LLM's job becomes
   verification and gap-filling rather than generation from scratch.

v1.0 (2026-03-02): Initial implementation for code block merge system.
v1.1 (2026-03-02): Removed scaffold engine integration (scaffold engine removed from pipeline).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .arch_code_extractor import (
    ExtractionResult,
    get_extraction_for_task,
    DIRECT_USE_THRESHOLD,
    PREFILL_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MergeDecision:
    """The merge strategy decision for a single file task."""
    strategy: str  # "direct", "prefill", "llm_generate"
    content: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""

    @property
    def skip_llm(self) -> bool:
        """Whether this decision means we skip the LLM call entirely."""
        return self.strategy == "direct"

    @property
    def use_verify_prompt(self) -> bool:
        """Whether the LLM should use the verification prompt."""
        return self.strategy == "prefill"


# ---------------------------------------------------------------------------
# Merge decision logic
# ---------------------------------------------------------------------------

def decide_merge_strategy(
    extraction_result: Optional[ExtractionResult],
    file_path: str,
    action: str = "create",
) -> MergeDecision:
    """Decide how to merge extracted code into the implementation pipeline.

    Decision tree:
    1. No extracted code -> "llm_generate" (standard path)
    2. High confidence extracted code + CREATE -> "direct" (skip LLM)
    4. Moderate confidence extracted code -> "prefill" (LLM verifies)
    5. Low confidence or MODIFY action -> "llm_generate" with context

    Args:
        extraction_result: Result from extract_code_for_files.
        file_path: Target file path.
        action: "create" or "modify".

    Returns:
        MergeDecision with strategy and optional content.
    """
    extracted_code, confidence = get_extraction_for_task(
        extraction_result, file_path,
    )

    if not extracted_code:
        return MergeDecision(
            strategy="llm_generate",
            reason="No extracted code found in architecture document",
        )

    # MODIFY actions always go through LLM (need to merge with existing)
    if action == "modify":
        return MergeDecision(
            strategy="prefill",
            content=extracted_code,
            confidence=confidence,
            reason=f"MODIFY action: extracted code used as reference (confidence={confidence:.2f})",
        )


    # High confidence CREATE: use directly
    if confidence >= DIRECT_USE_THRESHOLD:
        return MergeDecision(
            strategy="direct",
            content=extracted_code,
            confidence=confidence,
            reason=f"High confidence extraction (confidence={confidence:.2f})",
        )

    # Moderate confidence: pre-fill for LLM verification
    if confidence >= PREFILL_THRESHOLD:
        return MergeDecision(
            strategy="prefill",
            content=extracted_code,
            confidence=confidence,
            reason=f"Moderate confidence: LLM will verify (confidence={confidence:.2f})",
        )

    # Low confidence: standard generation with extracted code as context
    return MergeDecision(
        strategy="llm_generate",
        content=extracted_code,
        confidence=confidence,
        reason=f"Low confidence extraction (confidence={confidence:.2f}) — used as context only",
    )
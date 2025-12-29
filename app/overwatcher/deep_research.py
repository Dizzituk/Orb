# FILE: app/overwatcher/deep_research.py
"""Deep Research for Strike 2 error recovery.

Spec v2.3 §9.5: Deep Research is allowed ONLY on Strike 2 for the same ErrorSignature.

Inputs:
- Exact error signature + stack trace
- Minimal code context needed to interpret the failure

Outputs:
- Short, cited explanation
- FIX_ACTIONS (still no code)

Tooling:
- OpenAI Web Search tool via Responses API
- Cap search calls and retrieved tokens to control spend
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.overwatcher.error_signature import ErrorSignature

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Deep Research uses GPT-5.2 Pro with web search tool
DEEP_RESEARCH_PROVIDER = os.getenv("ORB_DEEP_RESEARCH_PROVIDER", "openai")
DEEP_RESEARCH_MODEL = os.getenv("ORB_DEEP_RESEARCH_MODEL", "gpt-5.2-pro")

# Cost controls
MAX_SEARCH_CALLS = 3
MAX_SEARCH_TOKENS = 4000
MAX_OUTPUT_TOKENS = 1500


# =============================================================================
# Output Schema
# =============================================================================

@dataclass
class ResearchSource:
    """A source from web search."""
    
    title: str
    url: str
    excerpt: str
    relevance: float = 0.0  # 0-1 relevance score
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "excerpt": self.excerpt,
            "relevance": self.relevance,
        }


@dataclass
class DeepResearchResult:
    """Result from Deep Research (Strike 2).
    
    Still follows Overwatcher contract: no code.
    """
    
    explanation: str  # Short explanation of the error
    likely_cause: str  # Most likely root cause
    suggested_fix: str  # High-level fix suggestion (no code)
    sources: List[ResearchSource] = field(default_factory=list)
    confidence: float = 0.0
    search_calls_used: int = 0
    tokens_retrieved: int = 0
    
    def to_dict(self) -> dict:
        return {
            "explanation": self.explanation,
            "likely_cause": self.likely_cause,
            "suggested_fix": self.suggested_fix,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "search_calls_used": self.search_calls_used,
            "tokens_retrieved": self.tokens_retrieved,
        }
    
    def to_context_string(self) -> str:
        """Convert to string for Overwatcher context injection."""
        lines = [
            "### Deep Research Findings",
            "",
            f"**Explanation:** {self.explanation}",
            "",
            f"**Likely Cause:** {self.likely_cause}",
            "",
            f"**Suggested Fix:** {self.suggested_fix}",
            "",
        ]
        
        if self.sources:
            lines.append("**Sources:**")
            for i, src in enumerate(self.sources[:3], 1):
                lines.append(f"{i}. [{src.title}]({src.url})")
                lines.append(f"   {src.excerpt[:200]}...")
        
        lines.append(f"\n_Confidence: {self.confidence:.0%}, Searches: {self.search_calls_used}_")
        
        return "\n".join(lines)


# =============================================================================
# Prompts
# =============================================================================

DEEP_RESEARCH_SYSTEM = """You are a software debugging expert with access to web search.

YOUR TASK:
1. Analyze the error signature and stack trace
2. Search the web for relevant solutions, documentation, or similar issues
3. Provide a short, cited explanation and fix suggestion

CRITICAL RULES:
- NO CODE in your response
- Keep explanation concise (2-3 sentences)
- Cite sources with URLs
- Focus on the most likely root cause

Respond with JSON:
{
  "explanation": "Brief explanation of the error",
  "likely_cause": "Most likely root cause",
  "suggested_fix": "High-level fix suggestion (NO CODE)",
  "sources": [
    {"title": "Source title", "url": "https://...", "excerpt": "Relevant excerpt", "relevance": 0.9}
  ],
  "confidence": 0.0-1.0
}"""

DEEP_RESEARCH_USER = """Analyze this error and search for solutions.

## Error Signature
- Type: {exception_type}
- Test: {failing_test}
- Module: {module_path}
- Hash: {signature_hash}

## Stack Trace
```
{stack_trace}
```

## Context
{context}

Search for:
1. Similar errors in {exception_type} with this pattern
2. Common causes in the {module_path} module type
3. Known issues with the failing test pattern

Provide your analysis with sources."""


def build_research_prompt(
    error_signature: ErrorSignature,
    stack_trace: str,
    context: str = "",
) -> tuple[str, str]:
    """Build Deep Research prompts.
    
    Returns (system_prompt, user_prompt)
    """
    user = DEEP_RESEARCH_USER.format(
        exception_type=error_signature.exception_type,
        failing_test=error_signature.failing_test_name or "N/A",
        module_path=error_signature.module_path or "unknown",
        signature_hash=error_signature.signature_hash,
        stack_trace=stack_trace[:2000],  # Limit stack trace
        context=context[:1000],  # Limit context
    )
    
    return DEEP_RESEARCH_SYSTEM, user


# =============================================================================
# Search Query Generation
# =============================================================================

def generate_search_queries(error_signature: ErrorSignature) -> List[str]:
    """Generate search queries from error signature.
    
    Returns up to MAX_SEARCH_CALLS queries.
    """
    queries = []
    
    # Query 1: Exception type + Python
    queries.append(f"Python {error_signature.exception_type} fix solution")
    
    # Query 2: Exception + test pattern (if available)
    if error_signature.failing_test_name:
        test_pattern = error_signature.failing_test_name.split("::")[-1]
        queries.append(f"{error_signature.exception_type} pytest {test_pattern}")
    
    # Query 3: Module-specific search
    if error_signature.module_path:
        module_name = error_signature.module_path.split("/")[-1].replace(".py", "")
        queries.append(f"Python {module_name} {error_signature.exception_type}")
    
    return queries[:MAX_SEARCH_CALLS]


# =============================================================================
# Result Parsing
# =============================================================================

def parse_research_result(raw_output: str) -> DeepResearchResult:
    """Parse LLM output to DeepResearchResult."""
    import json
    import re
    
    if not raw_output:
        return DeepResearchResult(
            explanation="Deep research returned empty result",
            likely_cause="Unknown",
            suggested_fix="Unable to determine fix from research",
        )
    
    text = raw_output.strip()
    
    # Extract JSON from code fence if present
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    
    try:
        data = json.loads(text)
        
        sources = []
        for src_data in data.get("sources", []):
            sources.append(ResearchSource(
                title=src_data.get("title", ""),
                url=src_data.get("url", ""),
                excerpt=src_data.get("excerpt", ""),
                relevance=src_data.get("relevance", 0.0),
            ))
        
        return DeepResearchResult(
            explanation=data.get("explanation", ""),
            likely_cause=data.get("likely_cause", ""),
            suggested_fix=data.get("suggested_fix", ""),
            sources=sources,
            confidence=data.get("confidence", 0.0),
        )
        
    except json.JSONDecodeError:
        # Try to extract key information from text
        return DeepResearchResult(
            explanation=text[:500] if text else "Failed to parse research result",
            likely_cause="Parse error",
            suggested_fix="Review raw research output",
        )


# =============================================================================
# Main API
# =============================================================================

async def run_deep_research(
    *,
    error_signature: ErrorSignature,
    stack_trace: str,
    context: str,
    job_id: str,
    chunk_id: str,
    job_artifact_root: str,
    llm_call_fn: Callable,
    provider_id: str = None,
    model_id: str = None,
) -> DeepResearchResult:
    """Run Deep Research for Strike 2 error recovery.
    
    Spec v2.3 §9.5:
    - Only allowed on Strike 2 (same ErrorSignature)
    - Uses web search tool
    - Outputs short explanation + fix suggestion (no code)
    
    Args:
        error_signature: The error signature to research
        stack_trace: Full stack trace
        context: Additional context (code excerpts, etc.)
        job_id: Job UUID
        chunk_id: Chunk ID
        job_artifact_root: Root for artifacts
        llm_call_fn: Async function to call LLM
        provider_id: LLM provider (default: openai)
        model_id: LLM model (default: gpt-5.2-pro)
    
    Returns:
        DeepResearchResult with explanation and sources
    """
    from app.pot_spec.ledger import emit_stage_started
    from app.pot_spec.ledger_overwatcher import (
        emit_deep_research_triggered,
        emit_deep_research_completed,
    )
    from uuid import uuid4
    
    stage_run_id = str(uuid4())
    
    provider_id = provider_id or DEEP_RESEARCH_PROVIDER
    model_id = model_id or DEEP_RESEARCH_MODEL
    
    logger.info(f"[deep_research] Starting research for chunk {chunk_id}")
    
    # Emit events
    try:
        emit_stage_started(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            stage_id="deep_research",
            stage_run_id=stage_run_id,
        )
        
        # Generate search query for logging
        queries = generate_search_queries(error_signature)
        emit_deep_research_triggered(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            chunk_id=chunk_id,
            strike_number=2,
            research_query=queries[0] if queries else "N/A",
        )
    except Exception as e:
        logger.warning(f"[deep_research] Failed to emit events: {e}")
    
    # Build prompt
    system_prompt, user_prompt = build_research_prompt(
        error_signature=error_signature,
        stack_trace=stack_trace,
        context=context,
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Call LLM with web search tool
    # Note: The actual web search tool integration depends on your LLM client
    # This is a simplified version that assumes the LLM can do web search
    try:
        result = await llm_call_fn(
            provider_id=provider_id,
            model_id=model_id,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
            }],
        )
        
        raw_output = result.content if hasattr(result, "content") else str(result)
        research_result = parse_research_result(raw_output)
        
        # Estimate tokens retrieved (rough)
        research_result.search_calls_used = min(len(research_result.sources), MAX_SEARCH_CALLS)
        research_result.tokens_retrieved = len(raw_output) // 4
        
    except Exception as e:
        logger.error(f"[deep_research] LLM call failed: {e}")
        research_result = DeepResearchResult(
            explanation=f"Deep research failed: {e}",
            likely_cause="Research error",
            suggested_fix="Retry or proceed without research",
        )
    
    # Emit completion
    try:
        emit_deep_research_completed(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            chunk_id=chunk_id,
            guidance_summary=research_result.explanation[:200],
            sources_count=len(research_result.sources),
        )
    except Exception as e:
        logger.warning(f"[deep_research] Failed to emit completion: {e}")
    
    logger.info(f"[deep_research] Completed with {len(research_result.sources)} sources, confidence={research_result.confidence}")
    return research_result


__all__ = [
    # Data classes
    "ResearchSource",
    "DeepResearchResult",
    # Functions
    "build_research_prompt",
    "generate_search_queries",
    "parse_research_result",
    "run_deep_research",
    # Config
    "DEEP_RESEARCH_PROVIDER",
    "DEEP_RESEARCH_MODEL",
    "MAX_SEARCH_CALLS",
]

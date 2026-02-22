from __future__ import annotations
import json
import logging
import re
from .refactor_schemas import ChangeDecision, ClassifiedMatch, MatchBucket, RawMatch, RefactorFlag, RiskLevel
from typing import Any, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


FILE_PATTERN_HINTS = {
    # Data/artifacts (likely SKIP)
    r"\.db$": MatchBucket.DATABASE_ARTIFACT,
    r"\.sqlite$": MatchBucket.DATABASE_ARTIFACT,
    r"_backup": MatchBucket.HISTORICAL_DATA,
    r"\\jobs\\": MatchBucket.HISTORICAL_DATA,
    r"\\data\\": MatchBucket.HISTORICAL_DATA,
    r"\\logs\\": MatchBucket.HISTORICAL_DATA,
    r"\\cache\\": MatchBucket.HISTORICAL_DATA,
    # Config (likely FLAG)
    r"\.env": MatchBucket.ENV_VAR_KEY,
    r"config\.": MatchBucket.CONFIG_KEY,
    r"settings\.": MatchBucket.CONFIG_KEY,
    # Docs (likely CHANGE)
    r"README": MatchBucket.DOCUMENTATION,
    r"\.md$": MatchBucket.DOCUMENTATION,
    r"CHANGELOG": MatchBucket.DOCUMENTATION,
    # Tests (likely CHANGE)
    r"test_": MatchBucket.TEST_ASSERTION,
    r"_test\.py": MatchBucket.TEST_ASSERTION,
    r"\\tests\\": MatchBucket.TEST_ASSERTION,
}

LINE_PATTERN_HINTS = {
    # Imports
    r"^\s*(?:from|import)\s+": MatchBucket.IMPORT_PATH,
    # Env vars (all caps with underscores)
    r"[A-Z][A-Z0-9_]{3,}\s*=": MatchBucket.ENV_VAR_KEY,
    r"os\.(?:getenv|environ)": MatchBucket.ENV_VAR_KEY,
    # Comments/docs
    r"^\s*#": MatchBucket.DOCUMENTATION,
    r'"""': MatchBucket.DOCUMENTATION,
    r"'''": MatchBucket.DOCUMENTATION,
    # UI strings
    r"(?:title|label|message|text)\s*[=:]": MatchBucket.UI_LABEL,
    r"(?:print|logger\.)": MatchBucket.UI_LABEL,
    # Test assertions
    r"(?:assert|expect|assertEqual)": MatchBucket.TEST_ASSERTION,
    # API routes
    r"@(?:app|router)\.(?:get|post|put|delete)": MatchBucket.API_ROUTE,
    r"(?:path|route)\s*=": MatchBucket.API_ROUTE,
}

def _build_classification_prompt(
    matches: List[RawMatch],
    search_term: str,
    replace_term: str,
    context: str,
    vision_context: str = "",
) -> str:
    """
    Build the classification prompt for the LLM.
    
    The prompt asks the LLM to classify each match and make a decision.
    
    Args:
        matches: List of raw matches to classify
        search_term: The term being searched for
        replace_term: The replacement term
        context: Additional context about the refactor job
        vision_context: v2.1 - Screenshot/UI analysis from Gemini Vision
    
    v2.1: Added vision_context parameter for intelligent UI classification.
    If vision context is present, the LLM can distinguish between:
    - USER-VISIBLE UI elements (title bars, headings, buttons) → likely CHANGE
    - Internal code paths/identifiers → assess based on risk
    """
    from .refactor_classifier import _get_heuristic_hint
    # Build match list with heuristic hints
    match_entries = []
    for i, m in enumerate(matches):
        hint_bucket, hint_conf = _get_heuristic_hint(m)
        hint_str = f" [hint: {hint_bucket.value}]" if hint_bucket else ""
        
        entry = f"""[{i}] {m.file_path}:{m.line_number}{hint_str}
    Line: {m.line_content[:150]}{'...' if len(m.line_content) > 150 else ''}
    Match: "{m.match_text}"
"""
        match_entries.append(entry)
    
    matches_text = "\n".join(match_entries)
    
    # v2.1: Build vision context section if available
    vision_section = ""
    if vision_context:
        vision_section = f"""
## VISION CONTEXT (Screenshot Analysis)
The user uploaded a screenshot. Here's what was visible in the UI:
{vision_context}

Use this to determine which matches are USER-VISIBLE UI elements:
- Text visible in title bars, headings, buttons, labels → bucket: ui_label, decision: CHANGE
- Text in window titles, status bars, menu text → bucket: ui_label, decision: CHANGE  
- Internal code identifiers not shown to user → assess risk as normal
- File paths, storage keys, env vars → appropriate bucket, likely SKIP

IMPORTANT: If vision context mentions specific UI text like "title bar shows '{search_term}'" or "heading says '{search_term}'", 
matches in code that render that UI text should be classified as ui_label with decision CHANGE.
"""
    
    prompt = f"""You are an expert code refactoring analyst. Classify each match and decide whether to CHANGE, SKIP, or FLAG it.
{vision_section}
## Context
We are renaming "{search_term}" to "{replace_term}" across a codebase.
{context}

## Your Task
For each match below, provide:
1. bucket: Category (code_identifier, import_path, module_package, env_var_key, config_key, api_route, file_folder_name, database_artifact, historical_data, documentation, ui_label, test_assertion, unknown)
2. decision: CHANGE (safe to modify), SKIP (leave alone), or FLAG (inform user but proceed if approved)
3. risk: low, medium, high, or critical
4. reasoning: Brief explanation (1 sentence)
5. impact: What happens if changed (optional, for FLAG/SKIP)

## Decision Guidelines
- CHANGE: UI labels, documentation, comments, internal variable names, test assertions
- SKIP: Env var keys (recommend alias), database files (migration needed), historical logs/artifacts (no value)
- FLAG: Import paths (cascade updates needed), package names (breaking change), API routes (external consumers)

## Matches to Classify
{matches_text}

## Response Format
Return ONLY valid JSON array. No markdown, no explanation. Each object must have:
{{"index": 0, "bucket": "...", "decision": "change|skip|flag", "risk": "low|medium|high|critical", "reasoning": "...", "impact": "..."}}

Example:
[
  {{"index": 0, "bucket": "ui_label", "decision": "change", "risk": "low", "reasoning": "User-facing string with no code dependencies"}},
  {{"index": 1, "bucket": "env_var_key", "decision": "skip", "risk": "high", "reasoning": "External systems may read this key", "impact": "Breaking change for .env files"}}
]

Classify all {len(matches)} matches:"""
    
    return prompt

def _parse_classification_response(
    response_text: str,
    original_matches: List[RawMatch],
) -> List[ClassifiedMatch]:
    """
    Parse the LLM's JSON response into ClassifiedMatch objects.
    """
    from .refactor_classifier import _fallback_heuristic_classification
    classified = []
    
    # Try to extract JSON array from response
    try:
        # Clean response - remove markdown code blocks if present
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            # Remove markdown code blocks
            clean_text = re.sub(r"```(?:json)?\n?", "", clean_text)
            clean_text = clean_text.strip()
        
        # Parse JSON
        classifications = json.loads(clean_text)
        
        if not isinstance(classifications, list):
            raise ValueError("Response is not a JSON array")
        
        # Map classifications back to matches
        for item in classifications:
            idx = item.get("index", -1)
            if idx < 0 or idx >= len(original_matches):
                continue
            
            match = original_matches[idx]
            
            # Parse bucket
            bucket_str = item.get("bucket", "unknown").lower()
            try:
                bucket = MatchBucket(bucket_str)
            except ValueError:
                bucket = MatchBucket.UNKNOWN
            
            # Parse decision
            decision_str = item.get("decision", "flag").lower()
            try:
                decision = ChangeDecision(decision_str)
            except ValueError:
                decision = ChangeDecision.FLAG
            
            # Parse risk
            risk_str = item.get("risk", "medium").lower()
            try:
                risk = RiskLevel(risk_str)
            except ValueError:
                risk = RiskLevel.MEDIUM
            
            classified.append(ClassifiedMatch(
                file_path=match.file_path,
                line_number=match.line_number,
                line_content=match.line_content,
                match_text=match.match_text,
                bucket=bucket,
                confidence=0.85,  # LLM classification confidence
                change_decision=decision,
                reasoning=item.get("reasoning", ""),
                risk_level=risk,
                impact_note=item.get("impact"),
            ))
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("[refactor_classifier] Failed to parse LLM response: %s", e)
        logger.debug("[refactor_classifier] Raw response: %s", response_text[:500])
        # Fallback to heuristics
        return _fallback_heuristic_classification(original_matches, "", "")
    
    return classified

def _generate_flags(
    matches: List[ClassifiedMatch],
    search_term: str,
    replace_term: str,
) -> List[RefactorFlag]:
    """Generate informational flags for the user."""
    flags = []
    
    # Check for env var changes
    env_matches = [m for m in matches if m.bucket == MatchBucket.ENV_VAR_KEY]
    if env_matches:
        skip_count = sum(1 for m in env_matches if m.change_decision == ChangeDecision.SKIP)
        if skip_count > 0:
            flags.append(RefactorFlag(
                flag_type="ENV_VAR_ALIAS",
                message=f"Found {len(env_matches)} environment variable keys. Recommend alias approach: introduce {replace_term}_* while keeping {search_term}_* as fallback.",
                severity="WARNING",
                affected_files=list(set(m.file_path for m in env_matches))[:5],
                affected_count=len(env_matches),
                recommendation=f"Add fallback logic: read {replace_term}_* first, fall back to {search_term}_*",
            ))
    
    # Check for import path changes
    import_matches = [m for m in matches if m.bucket == MatchBucket.IMPORT_PATH]
    if len(import_matches) > 10:
        flags.append(RefactorFlag(
            flag_type="IMPORT_CASCADE",
            message=f"Found {len(import_matches)} import statements. These will be updated as part of the refactor.",
            severity="INFO",
            affected_files=list(set(m.file_path for m in import_matches))[:10],
            affected_count=len(import_matches),
        ))
    
    # Check for database/historical data
    data_matches = [m for m in matches if m.bucket in (MatchBucket.DATABASE_ARTIFACT, MatchBucket.HISTORICAL_DATA)]
    if data_matches:
        skip_count = sum(1 for m in data_matches if m.change_decision == ChangeDecision.SKIP)
        if skip_count > 0:
            flags.append(RefactorFlag(
                flag_type="DATA_EXCLUDED",
                message=f"Excluding {skip_count} matches in database files, backups, and job artifacts. These contain historical data and won't be modified.",
                severity="INFO",
                affected_files=list(set(m.file_path for m in data_matches if m.change_decision == ChangeDecision.SKIP))[:5],
                affected_count=skip_count,
            ))
    
    # Check for API route changes
    route_matches = [m for m in matches if m.bucket == MatchBucket.API_ROUTE]
    if route_matches:
        flags.append(RefactorFlag(
            flag_type="API_ROUTES",
            message=f"Found {len(route_matches)} API route definitions. External consumers may need to be updated.",
            severity="CAUTION",
            affected_files=list(set(m.file_path for m in route_matches))[:5],
            affected_count=len(route_matches),
            recommendation="Consider adding route aliases for backward compatibility",
        ))
    
    return flags

EXPANSION_NEEDED_PATTERNS = {
    # Token/key/storage patterns
    r'[a-z]+_[a-z]+_?(?:key|token|prefix|id)': 'potential_storage_key',
    r'[A-Z]+_[A-Z]+_?(?:KEY|TOKEN|PREFIX|ID)': 'potential_env_var',
    r'localStorage|sessionStorage': 'storage_api',
    r'process\.env\.': 'env_access',
    r'getenv|environ': 'env_access',
    # Path patterns
    r'[A-Z]:\\': 'windows_path',
    r'/(?:home|usr|var|etc)/': 'unix_path',
    # Config patterns
    r'(?:config|settings)\[': 'config_access',
    r'\.(?:json|yaml|yml|toml|ini)': 'config_file',
}

def _read_file_lines_from_sandbox(
    file_path: str,
    sandbox_client: Any,
) -> Optional[List[str]]:
    """
    v3.0: Read file contents via sandbox client and return as lines.
    
    Uses existing /fs/contents endpoint.
    """
    if not sandbox_client:
        return None
    
    try:
        # Try to use sandbox client's file read capability
        if hasattr(sandbox_client, 'read_file'):
            content = sandbox_client.read_file(file_path)
            if content:
                return content.splitlines()
        
        # Alternative: try fs_read_file if available
        if hasattr(sandbox_client, 'fs_read_file'):
            result = sandbox_client.fs_read_file(file_path)
            if result and result.get('success') and result.get('content'):
                return result['content'].splitlines()
        
        # Alternative: try get_file_content
        if hasattr(sandbox_client, 'get_file_content'):
            content = sandbox_client.get_file_content(file_path)
            if content:
                return content.splitlines()
        
        return None
        
    except Exception as e:
        logger.warning("[refactor_classifier] v3.0 Failed to read file %s: %s", file_path, e)
        return None

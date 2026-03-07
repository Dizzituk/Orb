from __future__ import annotations
import logging
import re
from .file_discovery import DEFAULT_ROOTS as DISCOVERY_DEFAULT_ROOTS, DiscoveryResult, discover_files
from .refactor_classifier import build_refactor_plan
from .refactor_formatter import format_confirmation_message, format_human_readable, format_machine_readable
from .refactor_schemas import RawMatch, RefactorPlan
from .rename_policy import RenamePlan, build_rename_plan
from .spec_models import MultiFileOperation
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
# v8.0: get_sandbox_client intentionally left as None.
# The multi-file refactor detection path misclassifies architecture specs
# as rename jobs when the sandbox is reachable. The CREATE spec path
# (which fires when sandbox is unavailable) produces correct results.
# TODO: Fix the refactor classifier to not trigger on architecture specs
# before re-enabling sandbox access here.
_SANDBOX_CLIENT_AVAILABLE = False
get_sandbox_client = None
_LLM_CALL_AVAILABLE = True
llm_call = None


MULTI_FILE_SCOPE_INDICATORS = [
    # Explicit scope keywords
    r"\b(?:all|every|everywhere|entire|whole)\b",
    r"\bcodebase\b",
    r"\brepo(?:sitory)?\b",
    r"\bproject\b",
    r"\bacross\s+(?:all\s+)?(?:files?|the\s+codebase)\b",
    r"\bthroughout\b",
    r"\bsystem[- ]?wide\b",
    r"\bproject[- ]?wide\b",
    # v2.3: Drive references - more flexible matching
    r"\b[A-Za-z]\s+drive\b",  # "D drive"
    r"\b[A-Za-z]:\s*(?:\\|/)",  # "D:\" or "D:/"
    r"\b[A-Za-z]:[,\s]",  # v2.3: "D:," or "D: " (Weaver output format)
    r"\bon\s+[A-Za-z]:\b",  # "on D:"
    r"\bover\s+(?:the\s+)?[A-Za-z]\s+drive\b",
    r"\bsearch\s+over\b",
    r"\bscan\s+(?:the\s+)?(?:entire|whole|all)\b",
    # File/folder scope
    r"\bfile\s+(?:names?|structures?)\b",
    r"\bfolder\s+names?\b",
    r"\bwithin\s+(?:the\s+)?(?:code|files?|folders?)\b",
    r"\bin\s+(?:all\s+)?(?:files?|folders?|the\s+code)\b",
    # v2.3: Rename/refactor scope indicators
    r"\brename\s+(?:from|to)\b",
    r"\breplace\s+(?:all|instances|occurrences)\b",
    r"\bchange\s+(?:the\s+)?(?:ui|front-?end|branding)\b",
    r"\borb\s+(?:desktop|system)\b",  # Specific project references
    r"\bfront-?end\s+ui\b",
]

_SCOPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in MULTI_FILE_SCOPE_INDICATORS]

def _has_multi_file_scope(text: str) -> bool:
    """Check if text indicates multi-file scope."""
    if not text:
        return False
    for pattern in _SCOPE_PATTERNS:
        if pattern.search(text):
            return True
    return False

STOPWORDS = frozenset({
    'and', 'or', 'the', 'to', 'a', 'an', 'in', 'on', 'at', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'it', 'be', 'are', 'was', 'were', 'been', 'being',
    'all', 'any', 'every', 'each', 'some', 'that', 'this', 'these', 'those',
    'them', 'they', 'their', 'there', 'then', 'than',
    'find', 'search', 'look', 'replace', 'rename', 'change', 'update',
    'file', 'files', 'folder', 'folders', 'code', 'codebase',
    'extras', 'features', 'scope', 'task', 'micro', 'plan', 'audit',
    'safe', 'reference', 'references', 'occurrence', 'occurrences',
    'case', 'insensitive', 'sensitive',
    # v8.0: Generic English words that appear naturally in architecture specs.
    # These were causing false refactor classification on CREATE jobs.
    'frontend', 'backend', 'placeholder', 'real', 'new', 'old', 'current',
    'existing', 'original', 'updated', 'modified', 'simple', 'complex',
    'default', 'custom', 'standard', 'basic', 'advanced', 'legacy', 'modern',
    'where', 'when', 'how', 'what', 'which', 'who', 'why',
})

UNICODE_QUOTES = '"\'""\'\''

def _detect_multi_file_intent(
    combined_text: str,
    constraints_hint: Optional[Dict] = None,
    project_paths: Optional[List[str]] = None,
    vision_results: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Detect multi-file operation intent from combined text or constraints.
    
    v2.2: Falls back to context-aware inference when direct extraction fails.
    """
    from .multi_file_detection import _extract_replacement_only, _extract_search_and_replace_terms, _infer_search_term_from_context
    # Check if constraints_hint already has multi-file metadata
    if constraints_hint:
        multi_file_meta = constraints_hint.get("multi_file_metadata")
        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            logger.info("[multi_file_detection] v2.1 Multi-file metadata found in constraints_hint: %s",
                        multi_file_meta)
            return multi_file_meta
    
    if not combined_text:
        return None
    
    text = combined_text.strip()
    text_lower = text.lower()
    
    if len(text_lower) < 15:
        return None
    
    # STEP 1: Check for scope indicators
    has_scope = _has_multi_file_scope(text)
    print(f"[multi_file_detection] v2.3 Multi-file scope check: has_scope={has_scope}")
    if not has_scope:
        return None
    
    # STEP 2: Check for refactor/rename intent keywords
    # v2.3: Fixed patterns to match actual Weaver output (lowercase matching)
    refactor_intent_patterns = [
        # Direct verb patterns
        r"\b(?:rename|replace|change|update)\s+(?:all|every|it|them|that|to|with|instances|branding|occurrences|references)",
        r"\b(?:want\s+to|need\s+to|should)\s+(?:rename|replace|change)",
        # Search + action patterns
        r"\bfind\s+(?:all\s+)?(?:references?|occurrences?|instances?)\b",
        r"\bsearch\s+(?:for|over)\b.*?(?:change|rename|replace)",
        # Safety patterns
        r"\bwithout\s+breaking\b",
        r"\brename\s+(?:plan\s+)?to\b",
        r"\bsafe\s+rename\b",
        # v2.3: Weaver output patterns (case-insensitive matching on lowercase text)
        r"\bfront-?end\s+rename\b",  # "Front-end rename"
        r"\brename\s+(?:of\s+)?[a-z]+\s+(?:ui\s+)?to\s+[a-z]",  # "rename of orb ui to astra"
        r"\binstances\s*/\s*branding\b",  # "instances/branding" (the slash!)
        r"\binstances\s+of\s",  # "instances of"
        r"\bbranding\s+of\s",  # "branding of"
        r"\b(?:of|from)\s+['\"]?[a-z]+['\"]?\s+(?:to|with)\s+['\"]?[a-z]+['\"]?",  # "of 'orb' with 'astra'"
        r"\bui\s+to\s+[a-z]",  # "UI to Astra"
        r"\bto\s+astra\b",  # explicit "to Astra"
        r"\borb\s+to\s+astra\b",  # explicit "Orb to Astra"
        r"\breplace\s+['\"][^'\"]+['\"]\s+with\s",  # "replace 'X' with" (quoted terms only)
        r"\bchange\s+.*?\s+to\s+astra\b",  # "change ... to Astra"
        r"\bso\s+it'?s\s+called\b",  # "so it's called"
        r"\bcalled\s+astra\b",  # "called Astra"
    ]
    
    has_refactor_intent = False
    for pattern in refactor_intent_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            has_refactor_intent = True
            break
    
    print(f"[multi_file_detection] v2.3 Refactor intent check: has_intent={has_refactor_intent}")
    
    if not has_refactor_intent:
        return None
    
    # STEP 3: Try direct extraction first
    terms = _extract_search_and_replace_terms(text)
    
    if terms:
        search_pattern = terms.get("search_pattern", "")
        replacement_pattern = terms.get("replacement_pattern", "")
        if search_pattern and replacement_pattern:
            logger.info("[multi_file_detection] v2.2 MULTI-FILE REFACTOR detected: '%s' -> '%s'",
                        search_pattern, replacement_pattern)
            return {
                "is_multi_file": True,
                "operation_type": "refactor",
                "search_pattern": search_pattern,
                "replacement_pattern": replacement_pattern,
            }
    
    # STEP 4: v2.2 FALLBACK - Try context-aware inference
    print("[multi_file_detection] v2.2 Direct extraction failed, trying context-aware inference...")
    
    replacement_pattern = _extract_replacement_only(text)
    if replacement_pattern:
        search_pattern = _infer_search_term_from_context(
            text=text,
            project_paths=project_paths,
            vision_results=vision_results,
            constraints_hint=constraints_hint,
        )
        
        if search_pattern:
            logger.info("[multi_file_detection] v2.2 CONTEXT-INFERRED REFACTOR: '%s' -> '%s'",
                        search_pattern, replacement_pattern)
            return {
                "is_multi_file": True,
                "operation_type": "refactor",
                "search_pattern": search_pattern,
                "replacement_pattern": replacement_pattern,
                "inferred": True,  # Flag that this was context-inferred
            }
    
    logger.warning("[multi_file_detection] v2.2 Has refactor intent + scope but extraction failed")
    return None

def _convert_discovery_to_raw_matches(discovery_result: DiscoveryResult) -> List[RawMatch]:
    """v2.0: Convert file discovery results to RawMatch format for classification."""
    raw_matches = []
    for fm in discovery_result.files:
        for lm in fm.line_matches:
            raw_matches.append(RawMatch(
                file_path=fm.path,
                line_number=lm.line_number,
                line_content=lm.line_content,
                match_text=lm.line_content.strip() if lm.line_content else "",
            ))
    return raw_matches

async def _build_multi_file_operation(
    operation_type: str,
    search_pattern: str,
    replacement_pattern: str = "",
    file_filter: Optional[str] = None,
    sandbox_client: Optional[Any] = None,
    job_description: str = "",
    provider_id: str = "anthropic",
    model_id: str = "claude-sonnet-4-20250514",
    explicit_roots: Optional[List[str]] = None,
    vision_context: str = "",
) -> MultiFileOperation:
    """
    Run file discovery, classify matches, and build MultiFileOperation for spec.
    
    v2.2: Added explicit_roots parameter to scope discovery to specific projects.
    When explicit_roots is provided, ONLY those paths are searched (not DEFAULT_ROOTS).
    
    v2.4: Added vision_context parameter for intelligent UI classification.
    When vision context is present, the classifier can distinguish between
    user-visible UI text (title bars, headings) and internal code paths.
    """
    from .multi_file_detection import _is_valid_term
    logger.info("[multi_file_detection] v2.4 Building multi-file operation: type=%s, pattern=%s, vision_context=%d chars",
                operation_type, search_pattern, len(vision_context))
    
    if vision_context:
        print(f"[multi_file_detection] v2.4 VISION CONTEXT available for refactor ({len(vision_context)} chars)")
    
    if not _is_valid_term(search_pattern):
        error_msg = f"REJECTED: search pattern '{search_pattern}' is a stopword or invalid"
        logger.error("[multi_file_detection] v2.1 %s", error_msg)
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            requires_confirmation=True,
            error_message=error_msg,
        )
    
    if not _SANDBOX_CLIENT_AVAILABLE or not get_sandbox_client:
        logger.warning("[multi_file_detection] v2.1 Sandbox client not available")
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            requires_confirmation=(operation_type == "refactor"),
            error_message="Sandbox client not available for file discovery",
        )
    
    try:
        client = sandbox_client or get_sandbox_client()
        
        # v2.2: Use explicit_roots if provided, otherwise fall back to defaults
        search_roots = explicit_roots if explicit_roots else DISCOVERY_DEFAULT_ROOTS
        logger.info("[multi_file_detection] v2.2 Discovery roots: %s (explicit=%s)",
                    search_roots, explicit_roots is not None)
        
        result = discover_files(
            search_pattern=search_pattern,
            sandbox_client=client,
            file_filter=file_filter,
            roots=search_roots,
        )
        
        if not result.success:
            return MultiFileOperation(
                is_multi_file=True,
                operation_type=operation_type,
                search_pattern=search_pattern,
                replacement_pattern=replacement_pattern,
                requires_confirmation=(operation_type == "refactor"),
                error_message=result.error_message,
            )
        
        raw_matches = _convert_discovery_to_raw_matches(result)
        
        refactor_plan: Optional[RefactorPlan] = None
        rename_plan: Optional[RenamePlan] = None
        classification_markdown = ""
        classification_json: Dict[str, Any] = {}
        confirmation_message = ""
        
        if operation_type == "refactor" and raw_matches:
            try:
                match_dicts = [
                    {'file_path': m.file_path, 'line_number': m.line_number, 'line_content': m.line_content}
                    for m in raw_matches
                ]
                
                rename_plan = build_rename_plan(
                    matches=match_dicts,
                    search_pattern=search_pattern,
                    replace_pattern=replacement_pattern,
                )
                
                classification_markdown = rename_plan.get_report()
                classification_json = {
                    'search_pattern': rename_plan.search_pattern,
                    'replace_pattern': rename_plan.replace_pattern,
                    'safe_count': rename_plan.safe_count,
                    'unsafe_count': rename_plan.unsafe_count,
                    'migration_count': rename_plan.migration_count,
                    'safe_files': list(set(m.file_path for m in rename_plan.safe_to_rename)),
                    'excluded_files': list(set(m.file_path for m in rename_plan.unsafe_excluded)),
                    'migration_files': list(set(m.file_path for m in rename_plan.migration_required)),
                    'required_checks': rename_plan.required_checks,
                }
                
                if rename_plan.unsafe_count > 0:
                    confirmation_message = (
                        f"⚠️ PARTIAL RENAME: {rename_plan.safe_count} items will be renamed, "
                        f"{rename_plan.unsafe_count} EXCLUDED (would break invariants), "
                        f"{rename_plan.migration_count} require migration."
                    )
                else:
                    confirmation_message = f"✅ FULL RENAME: All {rename_plan.safe_count} items can be safely renamed."
                    
            except Exception as policy_err:
                logger.error("[multi_file_detection] v2.1 Rename policy failed: %s", policy_err)
                try:
                    refactor_plan = await build_refactor_plan(
                        raw_matches=raw_matches,
                        search_term=search_pattern,
                        replace_term=replacement_pattern,
                        context=job_description or f"Refactor: rename '{search_pattern}' to '{replacement_pattern}'",
                        provider_id=provider_id,
                        model_id=model_id,
                        llm_call_func=llm_call if _LLM_CALL_AVAILABLE else None,
                        vision_context=vision_context,  # v2.4: Pass vision context
                    )
                    if refactor_plan:
                        classification_markdown = format_human_readable(refactor_plan)
                        classification_json = format_machine_readable(refactor_plan)
                        confirmation_message = format_confirmation_message(refactor_plan)
                except Exception as classify_err:
                    logger.error("[multi_file_detection] v2.1 LLM classification also failed: %s", classify_err)
        
        file_preview = classification_markdown if classification_markdown else result.get_summary_report()
        
        if rename_plan:
            target_files = list(set(m.file_path for m in rename_plan.safe_to_rename))
        elif refactor_plan:
            target_files = list(set(m.file_path for m in refactor_plan.get_change_matches()))
        else:
            target_files = [fm.path for fm in result.files]
        
        # v3.0: Do NOT set requires_confirmation=True just because it's a refactor
        # The confirmation decision is made by spec_runner based on risk class
        # multi_file_detection's job: detect + collect + metadata (NOT gating)
        # v3.1: Also store raw_matches and rename_plan for spec_runner's v3 builder
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            target_files=target_files,
            total_files=result.total_files,
            total_occurrences=result.total_occurrences,
            file_filter=file_filter,
            file_preview=file_preview,
            discovery_truncated=result.truncated,
            discovery_duration_ms=result.duration_ms,
            roots_searched=result.roots_searched,
            requires_confirmation=False,  # v3.0: Gating decision moved to spec_runner
            confirmed=False,  # v3.0: Deprecated, kept for backward compat
            error_message=None,
            refactor_plan=refactor_plan,
            classification_markdown=classification_markdown,
            classification_json=classification_json,
            confirmation_message=confirmation_message,
            raw_matches=raw_matches,  # v3.1: Store for spec_runner's deterministic builder
        )
        
    except Exception as e:
        logger.error("[multi_file_detection] v2.1 Multi-file operation failed: %s", e)
        # v3.0: Even on error, don't hardcode requires_confirmation
        # Error handling and gating is spec_runner's responsibility
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            requires_confirmation=False,  # v3.0: Gating decision moved to spec_runner
            error_message=f"Discovery failed: {str(e)}",
        )

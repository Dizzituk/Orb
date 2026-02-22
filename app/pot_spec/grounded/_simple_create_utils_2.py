import logging
import os
import re
from app.pot_spec.grounded._simple_create_utils import CONCEPT_KEYWORDS, KEYWORD_STOPWORDS, MIN_KEYWORD_LENGTH, NEGATION_PATTERNS, PLACEHOLDER_GOALS
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


_EVIDENCE_MAX_FILE_CHARS = int(os.getenv("ASTRA_EVIDENCE_MAX_FILE_CHARS", "50000"))

_CREATE_ANALYSIS_TIMEOUT = int(os.getenv("ASTRA_CREATE_ANALYSIS_TIMEOUT", "180"))

def _extract_task_keywords(text: str) -> List[str]:
    """
    v2.0: Extract relevant CONCEPT keywords from task description.
    
    Returns concept names (not individual matched words), filtered of stopwords.
    Individual keywords are only used for concept detection, not returned directly.
    """
    text_lower = text.lower()
    concepts_found = set()
    
    for concept, trigger_phrases in CONCEPT_KEYWORDS.items():
        for phrase in trigger_phrases:
            if phrase in text_lower:
                concepts_found.add(concept)
                break  # One match per concept is enough
    
    # Also extract meaningful proper nouns (likely tech names)
    # v2.0: Require 3+ chars, not in stopwords
    proper_nouns = re.findall(r'\b([A-Z][a-z]{2,}(?:[A-Z][a-z]+)*)\b', text)
    for noun in proper_nouns:
        noun_lower = noun.lower()
        if noun_lower not in KEYWORD_STOPWORDS and len(noun_lower) >= MIN_KEYWORD_LENGTH:
            concepts_found.add(noun_lower)
    
    # Extract specific technology names mentioned (case-insensitive)
    tech_names = [
        'fastapi', 'react', 'electron', 'whisper', 'porcupine', 'openwakeword',
        'websocket', 'pytorch', 'onnx', 'silero', 'rnnoise', 'webrtcvad',
    ]
    for tech in tech_names:
        if tech in text_lower:
            concepts_found.add(tech)
    
    return list(concepts_found)

def _extract_constraints(text: str) -> List[str]:
    """
    v2.0: Extract explicit constraints from task description.
    
    Detects negation patterns like "no cloud APIs", "local only", "no mobile".
    Returns human-readable constraint strings.
    """
    text_lower = text.lower()
    constraints = []
    seen_tags = set()
    
    for pattern, tag in NEGATION_PATTERNS:
        if tag not in seen_tags and re.search(pattern, text_lower):
            seen_tags.add(tag)
            # Map tag to human-readable constraint
            constraint_map = {
                'no_cloud_api': 'No cloud APIs — all processing must be local',
                'no_audio_upload': 'No audio data may leave the machine',
                'local_only': 'Local-only processing — no external network calls for core functionality',
                'no_mobile': 'No mobile app in this phase',
                'no_multilingual': 'No multi-language support in this phase',
                'privacy_critical': 'Privacy constraint — no data leaves the machine',
                'phase_1_scope': 'Phase 1 scope limitations apply',
                'desktop_only': 'Desktop-only in this phase',
                'no_tts': 'No text-to-speech',
            }
            constraints.append(constraint_map.get(tag, tag))
    
    return constraints

CONCEPT_DIRECTORY_PATTERNS = {
    "voice": ["voice", "audio", "speech"],
    "transcription": ["voice", "transcription", "stt"],
    "streaming": ["streaming", "stream", "sse"],
    "model_management": ["services", "models"],
    "api_endpoint": ["routers", "routes", "endpoints", "api"],
    "state_management": ["store", "context", "hooks"],
    "wake_word": ["voice", "wakeword", "hotword"],
    # v4.3: Extended concept mappings for common feature domains
    "context": ["stream", "routing", "context", "memory", "llm"],
    "model": ["llm", "routing", "stream", "models"],
    "chat": ["llm", "stream", "chat", "routing", "memory"],
    "memory": ["memory", "db", "store", "persistence"],
    "persistence": ["memory", "db", "store", "models"],
    "routing": ["routing", "llm", "stream", "routers"],
    "llm": ["llm", "stream", "routing", "pipeline"],
    "pipeline": ["pipeline", "llm", "pot_spec"],
    "specgate": ["pot_spec", "llm", "pipeline"],
    "weaver": ["llm", "stream", "pipeline"],
    "upload": ["memory", "routers", "endpoints"],
    "file_upload": ["memory", "routers", "endpoints"],
    "user": ["auth", "memory", "routers"],
    "db": ["db", "models", "memory"],
    "database": ["db", "models", "memory"],
    "retention": ["memory", "db", "jobs"],
    "session": ["llm", "stream", "memory", "auth"],
}

def _suggest_new_files(
    concepts: List[str],
    constraints: List[str],
    tech_stack: TechStack,
    project_paths: List[str],
) -> List[str]:
    """
    v2.0: Suggest new files based on task concepts and constraints.
    
    CRITICAL FIX: Respects constraints (e.g., "no cloud APIs" prevents 
    suggesting openai_client.py). Uses concepts, not raw keywords.
    """
    suggested = []
    constraint_tags = set()
    
    # Parse constraint tags for filtering
    for c in constraints:
        c_lower = c.lower()
        if 'no cloud' in c_lower or 'local' in c_lower:
            constraint_tags.add('local_only')
        if 'no mobile' in c_lower:
            constraint_tags.add('no_mobile')
    
    # Identify which projects are frontend vs backend
    frontend_path = None
    backend_path = None
    
    for path in project_paths:
        path_lower = path.lower()
        if 'desktop' in path_lower or 'frontend' in path_lower or 'ui' in path_lower:
            frontend_path = path
        elif os.path.exists(os.path.join(path, 'requirements.txt')):
            backend_path = path
        elif os.path.exists(os.path.join(path, 'package.json')):
            frontend_path = path
        elif os.path.exists(os.path.join(path, 'app')):
            backend_path = path
    
    # v4.6: REMOVED all hardcoded concept-to-file mappings.
    # Previously this function had static suggestions like:
    #   voice/transcription → app/routers/transcribe.py, app/services/transcription_service.py
    #   streaming → app/routers/audio_stream.py
    # These polluted specs for unrelated jobs (e.g., "streaming" matched SSE chat streaming
    # and suggested audio_stream.py). The LLM analysis step already generates file suggestions
    # from actual codebase evidence — it doesn't need a static lookup table.
    #
    # The pipeline must be domain-agnostic. New file suggestions come from LLM analysis only.
    
    return suggested

def _host_list_directory(dir_path: str, max_entries: int = 200, project_paths: Optional[List[str]] = None) -> Tuple[bool, str]:
    """List directory contents from the host filesystem for evidence fulfilment.

    v4.1: Added project_paths for resolving relative directory paths.
    Returns (success, listing_or_error_message).
    """
    dir_path = dir_path.replace('/', os.sep).replace('\\', os.sep)

    # v4.1: Resolve relative paths against project roots
    if not os.path.exists(dir_path) and project_paths:
        for root in project_paths:
            candidate = os.path.join(root, dir_path)
            candidate = candidate.replace('/', os.sep).replace('\\', os.sep)
            if os.path.exists(candidate):
                logger.info("[SPEC_GATE_EVIDENCE] Resolved relative dir: %s \u2192 %s", dir_path, candidate)
                dir_path = candidate
                break

    if not os.path.exists(dir_path):
        return False, f"Directory not found: {dir_path}"
    if not os.path.isdir(dir_path):
        return False, f"Path is not a directory: {dir_path}"

    try:
        entries = []
        for entry in sorted(os.listdir(dir_path)):
            full = os.path.join(dir_path, entry)
            tag = "[DIR]" if os.path.isdir(full) else "[FILE]"
            entries.append(f"  {tag} {entry}")
            if len(entries) >= max_entries:
                entries.append(f"  ... ({len(os.listdir(dir_path)) - max_entries} more entries)")
                break
        listing = f"{dir_path}/\n" + "\n".join(entries)
        logger.info("[SPEC_GATE_EVIDENCE] Listed %d entries from %s", len(entries), dir_path)
        return True, listing
    except Exception as exc:
        logger.warning("[SPEC_GATE_EVIDENCE] Failed to list %s: %s", dir_path, exc)
        return False, f"List error: {exc}"

def _sanitize_goal(goal: str, what_to_do: str) -> str:
    """
    v1.2: Ensure goal is not a placeholder.
    """
    if not goal:
        goal = ""
    
    goal_clean = goal.split('\n')[0].strip().lower()
    
    is_placeholder = (
        not goal_clean or
        goal_clean in PLACEHOLDER_GOALS or
        goal_clean.startswith("job description") or
        goal_clean.startswith("weaver output")
    )
    
    if is_placeholder and what_to_do:
        intent_match = re.search(r'intent\s*[:\-]\s*(.+?)(?:\n|$)', what_to_do, re.IGNORECASE)
        if intent_match:
            return intent_match.group(1).strip()
        
        built_match = re.search(r'what\s+is\s+being\s+built\s*[:\-]\s*(.+?)(?:\n|$)', what_to_do, re.IGNORECASE)
        if built_match:
            return built_match.group(1).strip()
        
        for line in what_to_do.split('\n'):
            line = line.strip()
            if line and line.lower() not in PLACEHOLDER_GOALS and not line.startswith('#'):
                if not line.lower().startswith('what is being built'):
                    return line[:200]
    
    return goal.split('\n')[0].strip() if goal else "Implement requested feature"

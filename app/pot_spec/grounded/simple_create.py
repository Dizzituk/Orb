# FILE: app/pot_spec/grounded/simple_create.py
"""
SpecGate CREATE Path - LLM-Grounded Feature Spec Builder

Provides grounded specs for CREATE tasks (new features) by combining
filesystem evidence with LLM analysis using the allocated model.

Flow:
1. Extract meaningful keywords from task description (stopword-filtered)
2. Scan codebase for relevant integration points (semantic, not substring)
3. Detect tech stack and patterns
4. Extract constraints from task description (e.g., "no cloud APIs")
5. Use allocated LLM to analyze evidence and generate intelligent spec
6. Build grounded spec with WHERE + HOW + INTEGRATION + CONSTRAINTS

v1.0 (2026-02-02): Initial implementation
v1.2 (2026-02-02): Goal validation
v2.0 (2026-02-04): MAJOR FIX — LLM-grounded analysis
    - Added LLM analysis step using the allocated model (was zero-LLM before)
    - Fixed keyword extraction: added stopwords, min length, no garbage matches
    - Fixed integration point discovery: semantic relevance, not substring matching
    - Fixed file suggestions: constraint-aware, removed hardcoded openai_client.py
    - Fixed Implementation Steps: LLM-generated from evidence, not copy-paste
    - Fixed acceptance criteria: task-specific, extracted from constraints
    - Added constraint extraction from weaver output (negation detection)
    - spec_runner now passes provider_id/model_id through to this module
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SIMPLE_CREATE_BUILD_ID = "2026-02-05-v2.2-section-authority-labels"
print(f"[SIMPLE_CREATE_LOADED] BUILD_ID={SIMPLE_CREATE_BUILD_ID}")

# =============================================================================
# ENV-DRIVEN MODEL OVERRIDE FOR CREATE ANALYSIS
# =============================================================================
# The spec_gate stage allocates a model (often gpt-5.2-pro) for the WHOLE stage,
# but CREATE analysis is structured analysis — doesn't need a pro-tier model.
# A faster model gives better latency without sacrificing quality here.
#
# Set ASTRA_CREATE_ANALYSIS_MODEL to override (e.g., "gpt-5-mini", "gpt-5.2")
# If not set, uses the model allocated by spec_gate_stream.
_CREATE_ANALYSIS_MODEL = os.getenv("ASTRA_CREATE_ANALYSIS_MODEL", "")
_CREATE_ANALYSIS_TIMEOUT = int(os.getenv("ASTRA_CREATE_ANALYSIS_TIMEOUT", "180"))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IntegrationPoint:
    """A file/location where new code should integrate."""
    file_path: str
    file_name: str
    relevance: str  # Why this file matters
    action: str  # "modify" or "reference" or "create_nearby"
    line_hint: Optional[int] = None
    content_preview: Optional[str] = None


@dataclass 
class TechStack:
    """Detected technology stack."""
    frontend_framework: Optional[str] = None  # React, Vue, Angular, vanilla
    frontend_language: Optional[str] = None   # TypeScript, JavaScript
    backend_framework: Optional[str] = None   # FastAPI, Express, Django
    backend_language: Optional[str] = None    # Python, Node, Go
    styling: Optional[str] = None             # CSS, Tailwind, styled-components
    state_management: Optional[str] = None    # Redux, Zustand, Context
    api_pattern: Optional[str] = None         # REST, GraphQL


@dataclass
class CreateEvidence:
    """Evidence gathered for a CREATE task."""
    tech_stack: TechStack
    integration_points: List[IntegrationPoint]
    existing_patterns: Dict[str, str]  # pattern_name -> example code/file
    suggested_files: List[str]  # New files to create
    keywords_found: Dict[str, List[str]]  # keyword -> files containing it
    constraints: List[str]  # Extracted constraints (e.g., "no cloud APIs")
    llm_analysis: Optional[str] = None  # LLM-generated analysis


# =============================================================================
# STOPWORDS & KEYWORD EXTRACTION (v2.0 — FIXED)
# =============================================================================

# Words that should NEVER be used as search keywords for filename matching
KEYWORD_STOPWORDS = {
    # Articles, prepositions, conjunctions
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'of', 'for',
    'by', 'or', 'and', 'but', 'not', 'no', 'nor', 'so', 'as', 'if',
    'do', 'be', 'am', 'are', 'was', 'were', 'has', 'had', 'have',
    'will', 'can', 'may', 'with', 'from', 'that', 'this', 'than',
    'its', 'my', 'any', 'all', 'each', 'every',
    # Common verbs that match too many files
    'get', 'set', 'run', 'use', 'add', 'new', 'put', 'let',
    # Task description noise
    'should', 'must', 'need', 'want', 'like', 'also', 'just',
    'only', 'ever', 'never', 'always', 'still', 'yet',
    # Generic computing terms too broad for filename matching
    'file', 'data', 'type', 'name', 'path', 'list', 'item',
    'mode', 'test', 'main', 'base', 'core', 'util', 'help',
}

# Minimum keyword length for filename matching (prevents "no", "ui", etc.)
MIN_KEYWORD_LENGTH = 3

# Map task concepts to search keywords
# v2.0: Only concepts are returned, individual keywords are used for DETECTION only
CONCEPT_KEYWORDS = {
    "voice": ["voice", "audio", "speech", "microphone", "mic", "record"],
    "text_input": ["input field", "text input", "textarea", "text entry"],
    "button": ["button", "toggle", "click handler"],
    "api_endpoint": ["endpoint", "api route", "http endpoint", "fastapi route"],
    "file_upload": ["upload", "file upload", "blob upload"],
    "transcription": ["transcribe", "transcription", "whisper", "speech-to-text", "stt"],
    "ui_component": ["component", "widget", "ui element"],
    "state_management": ["useState", "context", "store", "redux", "state management"],
    "streaming": ["stream", "sse", "server-sent", "websocket", "real-time"],
    "model_management": ["model download", "model loading", "model manager", "gpu detection"],
    "wake_word": ["wake word", "hotword", "always listening", "voice activation"],
    "noise_suppression": ["noise", "noise suppression", "vad", "voice activity"],
}


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


# =============================================================================
# CONSTRAINT EXTRACTION (v2.0 — NEW)
# =============================================================================

# Patterns that indicate negative constraints
NEGATION_PATTERNS = [
    (r'no\s+(cloud\s+api|cloud\s+service|external\s+api|remote\s+api)', 'no_cloud_api'),
    (r'no\s+audio.*leave', 'no_audio_upload'),
    (r'local[\s-]+only', 'local_only'),
    (r'no\s+mobile', 'no_mobile'),
    (r'no\s+multi[\s-]*language', 'no_multilingual'),
    (r'never\s+leav(?:e|ing)\s+the\s+machine', 'local_only'),
    (r'not\s+(?:a|an)?\s*cloud', 'no_cloud_api'),
    (r'privacy\s+constraint', 'privacy_critical'),
    (r'no\s+network\s+traffic', 'local_only'),
    (r'phase\s+1\s+only', 'phase_1_scope'),
    (r'desktop[\s-]+only', 'desktop_only'),
    (r'no\s+(?:tts|text[\s-]*to[\s-]*speech)', 'no_tts'),
]


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


# =============================================================================
# TECH STACK DETECTION
# =============================================================================

def _read_text_any_encoding(file_path: str) -> str:
    """
    v2.1: Read a text file trying multiple encodings.
    
    Some files (e.g., pip freeze output on Windows) are UTF-16 encoded.
    If default encoding fails to produce readable content, try alternatives.
    """
    # Try encodings in order of likelihood
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            # Sanity check: UTF-16 files read as UTF-8 will have null bytes
            # appearing as spaces between every character
            if enc == 'utf-8' and '\x00' in content:
                continue  # Try next encoding
            # Another sanity check: if every other char is a space and content
            # is very long, it's probably UTF-16 misread as UTF-8
            if enc == 'utf-8' and len(content) > 100:
                sample = content[:200]
                space_ratio = sample.count(' ') / len(sample) if sample else 0
                if space_ratio > 0.35:  # More than 35% spaces is suspicious
                    continue
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue
    
    # Last resort: read as binary and decode with errors='replace'
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        return raw.decode('utf-8', errors='replace')
    except Exception:
        return ""


def _detect_tech_stack(project_path: str, sandbox_client: Any = None) -> TechStack:
    """Detect the technology stack of a project."""
    stack = TechStack()
    
    # Check for common config files
    indicators = {
        "tsconfig.json": ("frontend_language", "TypeScript"),
        "vite.config.ts": ("frontend_framework", "React/Vite"),
        "vite.config.js": ("frontend_framework", "React/Vite"),
        "next.config.js": ("frontend_framework", "Next.js"),
        "angular.json": ("frontend_framework", "Angular"),
        "vue.config.js": ("frontend_framework", "Vue"),
        "requirements.txt": ("backend_language", "Python"),
        "pyproject.toml": ("backend_language", "Python"),
        "tailwind.config.js": ("styling", "Tailwind"),
        "tailwind.config.ts": ("styling", "Tailwind"),
    }
    
    for filename, detection in indicators.items():
        check_path = os.path.join(project_path, filename)
        try:
            if os.path.exists(check_path):
                setattr(stack, detection[0], detection[1])
        except Exception:
            pass
    
    # Check package.json for React/Vue/state management
    try:
        pkg_path = os.path.join(project_path, "package.json")
        if os.path.exists(pkg_path):
            import json
            with open(pkg_path, 'r', encoding='utf-8') as f:
                pkg = json.load(f)
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            
            if "react" in deps:
                stack.frontend_framework = "React"
            if "vue" in deps:
                stack.frontend_framework = "Vue"
            if "typescript" in deps:
                stack.frontend_language = "TypeScript"
            if "@reduxjs/toolkit" in deps or "redux" in deps:
                stack.state_management = "Redux"
            if "zustand" in deps:
                stack.state_management = "Zustand"
    except Exception as e:
        logger.debug("[simple_create] Could not parse package.json: %s", e)
    
    # v2.1: Check for FastAPI/Flask/Django in Python projects
    # Uses _read_text_any_encoding to handle UTF-16 requirements.txt (common on Windows)
    try:
        req_path = os.path.join(project_path, "requirements.txt")
        if os.path.exists(req_path):
            reqs = _read_text_any_encoding(req_path).lower()
            if reqs:  # Only proceed if we actually read content
                if "fastapi" in reqs:
                    stack.backend_framework = "FastAPI"
                    logger.info("[simple_create] v2.1 Detected FastAPI in %s", req_path)
                elif "flask" in reqs:
                    stack.backend_framework = "Flask"
                elif "django" in reqs:
                    stack.backend_framework = "Django"
            else:
                logger.warning("[simple_create] v2.1 requirements.txt was empty or unreadable: %s", req_path)
    except Exception as e:
        logger.debug("[simple_create] Could not parse requirements.txt: %s", e)
    
    # v2.1: Also check main.py imports as a fallback for backend framework detection
    if not stack.backend_framework and stack.backend_language == "Python":
        try:
            main_path = os.path.join(project_path, "main.py")
            if os.path.exists(main_path):
                with open(main_path, 'r', encoding='utf-8') as f:
                    main_content = f.read(2000)  # First 2KB is enough
                if 'fastapi' in main_content.lower() or 'FastAPI' in main_content:
                    stack.backend_framework = "FastAPI"
                    logger.info("[simple_create] v2.1 Detected FastAPI from main.py imports")
                elif 'flask' in main_content.lower():
                    stack.backend_framework = "Flask"
        except Exception as e:
            logger.debug("[simple_create] Could not check main.py: %s", e)
    
    return stack


# =============================================================================
# INTEGRATION POINT DISCOVERY (v2.0 — FIXED)
# =============================================================================

# v2.0: Only match these SPECIFIC architectural files, not keyword substrings
ARCHITECTURAL_FILE_PATTERNS = [
    # Frontend entry points and primary components
    (r'^InputSection\.(tsx|jsx)$', "Primary text input component", "modify"),
    (r'^ChatWindow\.(tsx|jsx)$', "Main chat interface", "modify"),
    (r'^Header\.(tsx|jsx)$', "App header/toolbar", "modify"),
    (r'^App\.(tsx|jsx)$', "Root component", "reference"),
    (r'^main\.(tsx|jsx)$', "Entry point", "reference"),
    (r'^index\.(tsx|jsx)$', "Entry point", "reference"),
    # Frontend API layer
    (r'^api\.(ts|js)$', "API client layer", "modify"),
    (r'^streamingApi\.(ts|js)$', "Streaming API client", "modify"),
    # Backend entry points
    (r'^main\.py$', "Backend entry point", "reference"),
    # Types/interfaces
    (r'^types\.(ts|d\.ts)$', "Type definitions", "reference"),
]

# v2.0: Concept-to-directory patterns — search for files in relevant directories
CONCEPT_DIRECTORY_PATTERNS = {
    "voice": ["voice", "audio", "speech"],
    "transcription": ["voice", "transcription", "stt"],
    "streaming": ["streaming", "stream", "sse"],
    "model_management": ["services", "models"],
    "api_endpoint": ["routers", "routes", "endpoints", "api"],
    "state_management": ["store", "context", "hooks"],
    "wake_word": ["voice", "wakeword", "hotword"],
}


def _find_integration_points(
    project_path: str,
    concepts: List[str],
    sandbox_client: Any = None,
) -> List[IntegrationPoint]:
    """
    v2.0: Find architecturally relevant integration points.
    
    Uses SPECIFIC file patterns (regex on full filename) instead of
    substring matching. Also searches for existing directories that
    match the task's concepts.
    """
    points = []
    
    try:
        for root, dirs, files in os.walk(project_path):
            # Skip non-source directories
            dirs[:] = [d for d in dirs if d not in {
                'node_modules', '.git', '__pycache__', '.venv', 'venv',
                'dist', 'build', '.next', 'coverage', '.architecture',
                '_backup_before_audit', '_patches',
            }]
            
            rel_root = os.path.relpath(root, project_path)
            
            for filename in files:
                # Only check source files
                if not filename.endswith(('.tsx', '.jsx', '.ts', '.js', '.py', '.css')):
                    continue
                
                full_path = os.path.join(root, filename)
                
                # v2.0: Match against SPECIFIC architectural patterns (regex on full filename)
                for pattern, relevance, action in ARCHITECTURAL_FILE_PATTERNS:
                    if re.match(pattern, filename, re.IGNORECASE):
                        points.append(IntegrationPoint(
                            file_path=full_path,
                            file_name=filename,
                            relevance=relevance,
                            action=action,
                        ))
                        break
                
                # v2.0: Match files in concept-relevant directories
                for concept in concepts:
                    dir_patterns = CONCEPT_DIRECTORY_PATTERNS.get(concept, [])
                    for dir_pat in dir_patterns:
                        # Check if file is under a directory matching the concept
                        if dir_pat in rel_root.lower():
                            if not any(p.file_path == full_path for p in points):
                                points.append(IntegrationPoint(
                                    file_path=full_path,
                                    file_name=filename,
                                    relevance=f"In '{dir_pat}/' directory (relevant to {concept})",
                                    action="reference",
                                ))
                            break
    except Exception as e:
        logger.warning("[simple_create] Error scanning project: %s", e)
    
    # Dedupe and prioritize
    seen = set()
    unique = []
    for p in points:
        if p.file_path not in seen:
            seen.add(p.file_path)
            unique.append(p)
    
    # Sort: modify actions first, then by relevance
    unique.sort(key=lambda x: (0 if x.action == "modify" else 1, x.file_name))
    
    return unique[:15]  # Limit to top 15


# =============================================================================
# PATTERN EXTRACTION
# =============================================================================

def _extract_patterns(
    integration_points: List[IntegrationPoint],
    tech_stack: TechStack,
) -> Dict[str, str]:
    """Extract coding patterns from existing files."""
    patterns = {}
    
    for point in integration_points:
        if point.action != "modify":
            continue
        
        try:
            with open(point.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract React component pattern
            if point.file_name.endswith(('.tsx', '.jsx')):
                # Find component definition
                comp_match = re.search(
                    r'((?:export\s+)?(?:const|function)\s+\w+\s*[=:]\s*(?:\([^)]*\)|[^=])*\s*(?:=>|{)[^}]*(?:return\s*\()?[^)]*<)',
                    content[:2000]
                )
                if comp_match:
                    patterns[f"component_pattern:{point.file_name}"] = comp_match.group(0)[:500]
                
                # Find import pattern
                import_match = re.search(r"^(import\s+.+\n)+", content, re.MULTILINE)
                if import_match:
                    patterns[f"import_pattern:{point.file_name}"] = import_match.group(0)[:300]
            
            # Extract API call pattern
            if 'api' in point.file_name.lower():
                fetch_match = re.search(
                    r'((?:export\s+)?(?:async\s+)?(?:function|const)\s+\w+\s*[=:]?\s*(?:async\s*)?\([^)]*\)[^{]*{[^}]*fetch[^}]*})',
                    content,
                    re.DOTALL
                )
                if fetch_match:
                    patterns["api_call_pattern"] = fetch_match.group(0)[:600]
                    
        except Exception as e:
            logger.debug("[simple_create] Could not read %s: %s", point.file_path, e)
    
    return patterns


# =============================================================================
# FILE SUGGESTION (v2.0 — FIXED: constraint-aware)
# =============================================================================

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
    
    ext = '.tsx' if tech_stack.frontend_language == 'TypeScript' else '.jsx'
    hook_ext = ext.replace('x', '')  # .ts or .js
    
    # Voice/Audio feature
    if any(c in concepts for c in ['voice', 'transcription', 'wake_word', 'noise_suppression']):
        if frontend_path or tech_stack.frontend_framework:
            suggested.append(f"src/components/VoiceInput{ext}")
            suggested.append(f"src/hooks/useVoiceRecorder{hook_ext}")
        
        if backend_path or tech_stack.backend_framework:
            if tech_stack.backend_framework == 'FastAPI':
                suggested.append("app/routers/transcribe.py")
                suggested.append("app/services/transcription_service.py")
                suggested.append("app/services/model_manager.py")
    
    # Wake word specific
    if 'wake_word' in concepts:
        if backend_path or tech_stack.backend_framework:
            suggested.append("app/services/wake_word_service.py")
    
    # v2.0: REMOVED the old mapping that suggested openai_client.py
    # The old code had: if any(kw in ['api', 'openai', 'whisper']) → openai_client.py
    # This is wrong — "no cloud APIs" + "whisper" means LOCAL whisper, not OpenAI API
    
    # Streaming feature
    if 'streaming' in concepts:
        if backend_path or tech_stack.backend_framework:
            suggested.append("app/routers/audio_stream.py")
    
    # Dedupe
    seen = set()
    unique = []
    for f in suggested:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    
    return unique


# =============================================================================
# LLM ANALYSIS (v2.0 — NEW)
# =============================================================================

CREATE_ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect analyzing a feature request.

You will receive:
1. A feature description (from the Weaver stage)
2. Detected tech stack
3. Discovered integration points (existing files)
4. Extracted constraints

YOUR TASK:
Produce a structured analysis with these sections:

## Architecture Overview
Brief description of the feature's architecture (3-5 sentences).

## Implementation Steps
Numbered, actionable implementation steps. Each step should reference specific 
files (existing or new) and describe what changes are needed. Order by dependency.

## Files to Modify
For each existing file that needs changes, explain WHAT changes are needed and WHY.

## New Files to Create
For each new file, explain its purpose and key contents.

## Acceptance Criteria
Task-specific, testable acceptance criteria. Include criteria derived from 
explicit constraints (e.g., "no network traffic during transcription").

IMPORTANT:
- Be specific to THIS feature, not generic
- Respect ALL constraints listed
- Reference actual integration points provided
- Keep implementation steps concrete and actionable
- Do NOT suggest cloud services if constraints say local-only
- Do NOT suggest files/features outside the stated phase scope"""


# Default fallback model when the allocated model times out
_FALLBACK_MODELS = [
    ("openai", "gpt-5-mini"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
]


async def _run_llm_analysis(
    goal: str,
    what_to_do: str,
    tech_stack: TechStack,
    integration_points: List[IntegrationPoint],
    constraints: List[str],
    suggested_files: List[str],
    provider_id: str,
    model_id: str,
    llm_call_func: Optional[Callable],
) -> Optional[str]:
    """
    v2.1: Use an LLM to analyze the feature request.
    
    Model selection priority:
    1. ASTRA_CREATE_ANALYSIS_MODEL env var (if set)
    2. Allocated model from spec_gate_stream
    3. On timeout: retry with faster fallback model
    
    Returns LLM-generated analysis or None if all attempts fail.
    """
    if not llm_call_func:
        logger.warning("[simple_create] v2.1 LLM unavailable, falling back to template")
        return None
    
    # v2.1: Model override from env
    use_provider = provider_id
    use_model = model_id
    if _CREATE_ANALYSIS_MODEL:
        use_model = _CREATE_ANALYSIS_MODEL
        logger.info("[simple_create] v2.1 MODEL OVERRIDE: %s (from ASTRA_CREATE_ANALYSIS_MODEL)", use_model)
        print(f"[simple_create] v2.1 MODEL OVERRIDE: {use_model}")
    
    # Build context for LLM
    stack_desc = []
    if tech_stack.frontend_framework:
        stack_desc.append(f"Frontend: {tech_stack.frontend_framework}" +
                         (f" ({tech_stack.frontend_language})" if tech_stack.frontend_language else ""))
    if tech_stack.backend_framework:
        stack_desc.append(f"Backend: {tech_stack.backend_framework}" +
                         (f" ({tech_stack.backend_language})" if tech_stack.backend_language else ""))
    if tech_stack.styling:
        stack_desc.append(f"Styling: {tech_stack.styling}")
    
    integration_desc = []
    for p in integration_points[:10]:
        integration_desc.append(f"- {p.file_name} ({p.action}): {p.relevance}")
    
    constraints_desc = "\n".join(f"- {c}" for c in constraints) if constraints else "None specified"
    
    user_prompt = f"""Feature Request:
{goal}

Full Description:
{what_to_do[:3000]}

Tech Stack:
{chr(10).join(stack_desc) if stack_desc else 'Not detected'}

Existing Integration Points:
{chr(10).join(integration_desc) if integration_desc else 'None found'}

Suggested New Files:
{chr(10).join(f'- {f}' for f in suggested_files) if suggested_files else 'None'}

Constraints:
{constraints_desc}

Please provide your structured analysis."""

    # v2.1: Build attempt list — primary model, then fallbacks on timeout
    attempts = [(use_provider, use_model, _CREATE_ANALYSIS_TIMEOUT)]
    for fb_provider, fb_model in _FALLBACK_MODELS:
        # Don't add fallback if it's the same as primary
        if fb_provider != use_provider or fb_model != use_model:
            attempts.append((fb_provider, fb_model, 90))  # Fallbacks get standard timeout
    
    for attempt_idx, (attempt_provider, attempt_model, attempt_timeout) in enumerate(attempts):
        is_retry = attempt_idx > 0
        
        try:
            if is_retry:
                logger.info("[simple_create] v2.1 RETRY with fallback: %s/%s (timeout=%ds)",
                           attempt_provider, attempt_model, attempt_timeout)
                print(f"[simple_create] v2.1 RETRY: {attempt_provider}/{attempt_model} (timeout={attempt_timeout}s)")
            else:
                logger.info("[simple_create] v2.1 LLM ANALYSIS CALL: provider=%s, model=%s, timeout=%ds",
                           attempt_provider, attempt_model, attempt_timeout)
                print(f"[simple_create] v2.1 LLM ANALYSIS: calling {attempt_provider}/{attempt_model} (timeout={attempt_timeout}s)")
            
            result = await llm_call_func(
                provider_id=attempt_provider,
                model_id=attempt_model,
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=CREATE_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=4096,
                timeout_seconds=attempt_timeout,
            )
            
            if result.is_success() and result.content:
                analysis = result.content.strip()
                model_label = f"{attempt_provider}/{attempt_model}"
                if is_retry:
                    model_label += " (fallback)"
                logger.info("[simple_create] v2.1 LLM ANALYSIS SUCCESS: %d chars via %s", len(analysis), model_label)
                print(f"[simple_create] v2.1 LLM ANALYSIS SUCCESS: {len(analysis)} chars via {model_label}")
                return analysis
            
            error_msg = getattr(result, 'error_message', 'Unknown error')
            logger.warning("[simple_create] v2.1 LLM ANALYSIS FAILED (%s/%s): %s",
                          attempt_provider, attempt_model, error_msg)
            print(f"[simple_create] v2.1 LLM ANALYSIS FAILED: {error_msg}")
            
            # Only retry on timeout-like errors
            is_timeout = 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower()
            if not is_timeout:
                # Non-timeout error — don't bother retrying with a different model
                return None
            
        except Exception as e:
            error_str = str(e)
            logger.warning("[simple_create] v2.1 LLM ANALYSIS EXCEPTION (%s/%s): %s",
                          attempt_provider, attempt_model, error_str)
            print(f"[simple_create] v2.1 LLM ANALYSIS EXCEPTION: {error_str}")
            
            is_timeout = 'timeout' in error_str.lower() or 'timed out' in error_str.lower()
            if not is_timeout:
                return None
    
    # All attempts failed
    logger.warning("[simple_create] v2.1 ALL LLM ATTEMPTS FAILED, falling back to template")
    print("[simple_create] v2.1 ALL LLM ATTEMPTS FAILED — using template fallback")
    return None


# =============================================================================
# SPEC BUILDER (v2.0 — FIXED)
# =============================================================================

# v1.2: Placeholder goals that should never be used
PLACEHOLDER_GOALS = {
    "job description",
    "job description from weaver", 
    "weaver output",
    "task description",
    "implement requested feature",
    "complete the requested task",
}


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


def _extract_acceptance_from_constraints(constraints: List[str]) -> List[str]:
    """
    v2.0: Generate task-specific acceptance criteria from constraints.
    """
    criteria = []
    for constraint in constraints:
        c_lower = constraint.lower()
        if 'no cloud' in c_lower or 'local' in c_lower:
            criteria.append("No network traffic during transcription (verify with network monitor)")
        if 'no audio' in c_lower and 'leave' in c_lower:
            criteria.append("Audio data never leaves the machine — no outbound connections")
        if 'desktop' in c_lower:
            criteria.append("Works on desktop platform as specified")
        if 'phase 1' in c_lower:
            criteria.append("Only Phase 1 features implemented — no scope creep")
    return criteria


def build_create_spec(
    goal: str,
    what_to_do: str,
    evidence: CreateEvidence,
    project_paths: List[str],
) -> str:
    """
    v2.0: Build a grounded CREATE spec with evidence and LLM analysis.
    
    If LLM analysis is available, uses it for implementation steps and
    acceptance criteria. Falls back to weaver output if LLM unavailable.
    """
    lines = []
    
    sanitized_goal = _sanitize_goal(goal, what_to_do)
    
    # Header
    lines.append("# SPoT Spec — New Feature")
    lines.append("")
    
    # Goal
    lines.append("## Goal")
    lines.append("")
    lines.append(sanitized_goal)
    lines.append("")
    
    # Tech Stack Context
    lines.append("## Tech Stack (Detected)")
    lines.append("")
    stack = evidence.tech_stack
    if stack.frontend_framework:
        lines.append(f"- **Frontend**: {stack.frontend_framework}" + 
                    (f" ({stack.frontend_language})" if stack.frontend_language else ""))
    if stack.backend_framework:
        lines.append(f"- **Backend**: {stack.backend_framework}" +
                    (f" ({stack.backend_language})" if stack.backend_language else ""))
    if stack.styling:
        lines.append(f"- **Styling**: {stack.styling}")
    if stack.state_management:
        lines.append(f"- **State**: {stack.state_management}")
    lines.append("")
    
    # Constraints (v2.0 — NEW)
    if evidence.constraints:
        lines.append("## Constraints")
        lines.append("")
        for c in evidence.constraints:
            lines.append(f"- ⛔ {c}")
        lines.append("")
    
    # Integration Points (WHERE)
    lines.append("## Integration Points")
    lines.append("")
    
    modify_points = [p for p in evidence.integration_points if p.action == "modify"]
    reference_points = [p for p in evidence.integration_points if p.action != "modify"]
    
    if modify_points:
        lines.append("### Files to Modify (Suggested — architecture may choose alternatives)")
        lines.append("")
        lines.append("*These are LLM-suggested integration points from codebase analysis.*")
        lines.append("*The architecture may use different files or approaches if they better serve the requirements.*")
        lines.append("")
        for p in modify_points[:5]:
            lines.append(f"- `{p.file_name}` — {p.relevance}")
        lines.append("")
    
    if reference_points:
        lines.append("### Reference Files (patterns to follow — not mandatory)")
        lines.append("")
        for p in reference_points[:5]:
            lines.append(f"- `{p.file_name}` — {p.relevance}")
        lines.append("")
    
    # Suggested New Files
    if evidence.suggested_files:
        lines.append("### New Files to Create (Suggested — architecture determines final structure)")
        lines.append("")
        for f in evidence.suggested_files:
            lines.append(f"- `{f}`")
        lines.append("")
    
    # v2.0: LLM Analysis or Requirements
    if evidence.llm_analysis:
        lines.append("## LLM Architecture Analysis (Suggested — architecture determines final approach)")
        lines.append("")
        lines.append("*This analysis was generated by LLM codebase review. It is guidance, not a binding requirement.*")
        lines.append("")
        lines.append(evidence.llm_analysis)
        lines.append("")
    else:
        # Fallback: include weaver output as requirements (not "implementation steps")
        lines.append("## Requirements (from Weaver)")
        lines.append("")
        if what_to_do:
            for line in what_to_do.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.lower().startswith('what is being built'):
                    lines.append(stripped)
        lines.append("")
    
    # Patterns (HOW)
    if evidence.existing_patterns:
        lines.append("## Existing Patterns to Follow")
        lines.append("")
        for name, pattern in list(evidence.existing_patterns.items())[:3]:
            short_name = name.split(':')[-1] if ':' in name else name
            lines.append(f"### Pattern from `{short_name}`")
            lines.append("```")
            pattern_preview = pattern[:400] + "..." if len(pattern) > 400 else pattern
            lines.append(pattern_preview)
            lines.append("```")
            lines.append("")
    
    # v2.0: Task-specific Acceptance Criteria
    lines.append("## Acceptance")
    lines.append("")
    
    # Always include base criteria
    lines.append("- [ ] Feature works as described in requirements")
    lines.append("- [ ] Integrates with existing UI patterns")
    lines.append("- [ ] No console errors")
    lines.append("- [ ] App boots without issues")
    
    # Add constraint-derived criteria (v2.0)
    constraint_criteria = _extract_acceptance_from_constraints(evidence.constraints)
    for cc in constraint_criteria:
        lines.append(f"- [ ] {cc}")
    
    lines.append("")
    
    # Evidence Summary
    lines.append("## Evidence Summary")
    lines.append("")
    lines.append(f"- Integration points found: {len(evidence.integration_points)}")
    lines.append(f"- Patterns extracted: {len(evidence.existing_patterns)}")
    lines.append(f"- Constraints detected: {len(evidence.constraints)}")
    lines.append(f"- LLM analysis: {'Yes' if evidence.llm_analysis else 'No (fallback mode)'}")
    lines.append(f"- Project paths: {', '.join(project_paths)}")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN ENTRY POINT (v2.0 — UPDATED)
# =============================================================================

async def build_grounded_create_spec(
    goal: str,
    what_to_do: str,
    project_paths: List[str],
    sandbox_client: Any = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    llm_call_func: Optional[Callable] = None,
) -> Tuple[str, CreateEvidence]:
    """
    v2.0: Build a grounded spec for CREATE tasks with LLM analysis.
    
    Now accepts provider_id, model_id, and llm_call_func to enable
    LLM-powered analysis using the model allocated by the spec_gate_stream.
    Falls back to template-only mode if LLM unavailable.
    
    Returns:
        Tuple of (spec_markdown, evidence)
    """
    logger.info("[simple_create] v2.0 Building LLM-grounded CREATE spec")
    print(f"[simple_create] v2.0 GROUNDED CREATE: {goal[:60]}...")
    
    # v2.0: Extract CONCEPTS (not raw keywords)
    combined_text = f"{goal} {what_to_do}"
    concepts = _extract_task_keywords(combined_text)
    print(f"[simple_create] v2.0 Concepts: {concepts[:10]}")
    
    # v2.0: Extract constraints
    constraints = _extract_constraints(combined_text)
    print(f"[simple_create] v2.0 Constraints: {constraints}")
    
    # Detect tech stack for each project path
    tech_stack = TechStack()
    for path in project_paths:
        if os.path.isdir(path):
            detected = _detect_tech_stack(path, sandbox_client)
            for attr in ['frontend_framework', 'frontend_language', 'backend_framework',
                        'backend_language', 'styling', 'state_management', 'api_pattern']:
                if getattr(detected, attr) and not getattr(tech_stack, attr):
                    setattr(tech_stack, attr, getattr(detected, attr))
    
    print(f"[simple_create] v2.0 Tech stack: {tech_stack.frontend_framework}/{tech_stack.backend_framework}")
    
    # v2.0: Find integration points using CONCEPTS (not raw keywords)
    all_points = []
    for path in project_paths:
        if os.path.isdir(path):
            points = _find_integration_points(path, concepts, sandbox_client)
            all_points.extend(points)
    
    print(f"[simple_create] v2.0 Found {len(all_points)} integration points")
    
    # Extract patterns from integration points
    patterns = _extract_patterns(all_points, tech_stack)
    print(f"[simple_create] v2.0 Extracted {len(patterns)} patterns")
    
    # v2.0: Suggest new files with CONSTRAINT awareness
    suggested_files = _suggest_new_files(concepts, constraints, tech_stack, project_paths)
    
    # v2.0: Run LLM analysis if model available
    llm_analysis = None
    if provider_id and model_id:
        # Import llm_call if not provided
        if llm_call_func is None:
            try:
                from app.providers.registry import llm_call as registry_llm_call
                llm_call_func = registry_llm_call
                print(f"[simple_create] v2.0 Loaded llm_call from registry")
            except ImportError:
                print(f"[simple_create] v2.0 WARNING: Could not import llm_call from registry")
        
        if llm_call_func:
            llm_analysis = await _run_llm_analysis(
                goal=goal,
                what_to_do=what_to_do,
                tech_stack=tech_stack,
                integration_points=all_points,
                constraints=constraints,
                suggested_files=suggested_files,
                provider_id=provider_id,
                model_id=model_id,
                llm_call_func=llm_call_func,
            )
    else:
        print(f"[simple_create] v2.0 NO LLM: provider_id={provider_id}, model_id={model_id}")
    
    # Build evidence bundle
    evidence = CreateEvidence(
        tech_stack=tech_stack,
        integration_points=all_points,
        existing_patterns=patterns,
        suggested_files=suggested_files,
        keywords_found={c: [] for c in concepts},
        constraints=constraints,
        llm_analysis=llm_analysis,
    )
    
    # Build spec
    spec = build_create_spec(
        goal=goal,
        what_to_do=what_to_do,
        evidence=evidence,
        project_paths=project_paths,
    )
    
    print(f"[simple_create] v2.0 SPEC READY: {len(spec)} chars (LLM={'yes' if llm_analysis else 'no'})")
    
    return spec, evidence

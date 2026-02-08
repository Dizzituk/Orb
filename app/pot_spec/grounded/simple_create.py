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

SIMPLE_CREATE_BUILD_ID = "2026-02-08-v4.5-specgate-final-loop-and-caps"
print(f"[SIMPLE_CREATE_LOADED] BUILD_ID={SIMPLE_CREATE_BUILD_ID}")

# v4.0: Max evidence fulfilment loops (matches ASTRA_EVIDENCE_MAX_LOOPS convention)
_EVIDENCE_MAX_LOOPS = int(os.getenv("ASTRA_EVIDENCE_MAX_LOOPS", "3"))  # v4.2: Increased from 2 to 3
# v4.0: Max chars to read per file during evidence fulfilment
_EVIDENCE_MAX_FILE_CHARS = int(os.getenv("ASTRA_EVIDENCE_MAX_FILE_CHARS", "50000"))

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


# =============================================================================
# v3.6: CONTENT-SIGNAL SCORING for integration point disambiguation
# =============================================================================
# Problem: filename-only matching surfaces false positives like Orb/static/main.py
# alongside the real FastAPI entrypoint D:/Orb/main.py. Both match the
# architectural pattern r'^main\.py$' but only one is architecturally relevant.
#
# Solution: For ambiguous filenames (main.py, index.py, app.py), read a small
# content sample and score based on signals. Negative scores for paths under
# static/dist/build/public directories.

_CONTENT_SIGNALS = [
    (re.compile(r'FastAPI\s*\('), +10),
    (re.compile(r'from\s+fastapi\s+import\s+FastAPI'), +10),
    (re.compile(r'include_router\s*\('), +5),
    (re.compile(r'app\.mount\s*\('), +3),
    (re.compile(r'@app\.on_event'), +3),
    (re.compile(r'uvicorn\.run'), +3),
]

_NEGATIVE_PATH_SEGMENTS = {'static', 'dist', 'build', 'public', 'assets', 'out', '.next'}


def _score_integration_point(file_path: str, project_path: str) -> int:
    """
    v3.6: Score an integration point based on content signals and path heuristics.
    Only reads the first 3KB of the file for efficiency.
    Positive scores = more likely architecturally relevant.
    Negative scores = likely a false positive.
    """
    score = 0
    rel_path = os.path.relpath(file_path, project_path).lower().replace('\\', '/')
    path_segments = set(rel_path.split('/'))

    if path_segments & _NEGATIVE_PATH_SEGMENTS:
        score -= 10

    filename = os.path.basename(file_path).lower()
    if filename in ('main.py', 'app.py', 'index.py', 'server.py'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                head = f.read(3072)
            for pattern, delta in _CONTENT_SIGNALS:
                if pattern.search(head):
                    score += delta
        except Exception:
            pass

    return score

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
    
    # v3.7: Score each integration point using content signals + path heuristics.
    # Drop negative-scored points (false positives like static/main.py).
    # Sort remainder: modify actions first, then highest score, then filename.
    scored = []
    dropped = []
    for p in unique:
        s = _score_integration_point(p.file_path, project_path)
        if s < 0:
            dropped.append((p.file_name, s))
        else:
            scored.append((p, s))
    
    if dropped:
        print(f"[simple_create] v3.7 DROPPED {len(dropped)} negative-scored integration point(s): "
              f"{[(name, sc) for name, sc in dropped]}")
        logger.info("[simple_create] v3.7 Dropped %d negative-scored points: %s", len(dropped), dropped)
    
    scored.sort(key=lambda x: (0 if x[0].action == "modify" else 1, -x[1], x[0].file_name))
    result = [p for p, _ in scored]
    
    return result[:15]  # Limit to top 15


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
# HOST-DIRECT FILE READER (v4.0 — SpecGate Evidence Fulfilment)
# =============================================================================
# SpecGate runs BEFORE the sandbox is available, so evidence fulfilment must
# use host-direct filesystem access. This is intentionally local to simple_create
# to make the "SpecGate uses host-direct access" boundary explicit.


def _host_read_file(file_path: str, max_chars: int = 0, project_paths: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Read a file from the host filesystem for evidence fulfilment.

    v4.1: Added project_paths parameter for resolving relative paths.
    If file_path is not absolute or doesn't exist, tries resolving against
    each project root (e.g. 'app/llm/stream_router.py' → 'D:\\Orb\\app\\llm\\stream_router.py').

    Returns (success, content_or_error_message).
    Uses _read_text_any_encoding for robust encoding handling.
    """
    if not max_chars:
        max_chars = _EVIDENCE_MAX_FILE_CHARS

    # Normalise path separators for Windows
    file_path = file_path.replace('/', os.sep).replace('\\', os.sep)

    # v4.1: Resolve relative paths against project roots
    if not os.path.exists(file_path) and project_paths:
        for root in project_paths:
            candidate = os.path.join(root, file_path)
            candidate = candidate.replace('/', os.sep).replace('\\', os.sep)
            if os.path.exists(candidate):
                logger.info("[SPEC_GATE_EVIDENCE] Resolved relative path: %s → %s", file_path, candidate)
                file_path = candidate
                break

    if not os.path.exists(file_path):
        logger.info("[SPEC_GATE_EVIDENCE] File not found: %s", file_path)
        return False, f"File not found: {file_path}"

    if not os.path.isfile(file_path):
        logger.info("[SPEC_GATE_EVIDENCE] Not a file: %s", file_path)
        return False, f"Path is not a file: {file_path}"

    try:
        content = _read_text_any_encoding(file_path)
        if not content:
            return False, f"File is empty or unreadable: {file_path}"
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... [truncated at {max_chars} chars, file has {len(content)} total]"
        logger.info("[SPEC_GATE_EVIDENCE] Read %d chars from %s", min(len(content), max_chars), file_path)
        return True, content
    except Exception as exc:
        logger.warning("[SPEC_GATE_EVIDENCE] Failed to read %s: %s", file_path, exc)
        return False, f"Read error: {exc}"


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


# =============================================================================
# EVIDENCE FULFILMENT LOOP (v4.0 — SpecGate Evidence Fulfilment)
# =============================================================================
# Uses parsing/stripping utilities from evidence_loop.py but dispatches file
# reads via host-direct access (not sandbox). This keeps SpecGate independent
# of the sandbox lifecycle while reusing the robust 3-layer YAML parsing.


async def _fulfil_evidence_requests(
    llm_analysis: str,
    provider_id: str,
    model_id: str,
    llm_call_func: Callable,
    project_paths: List[str],
    goal: str = "",
    what_to_do: str = "",
) -> str:
    """Fulfil EVIDENCE_REQUEST blocks in the LLM analysis by reading actual files.

    v4.0: Parses ERs from the analysis, reads requested files from the host
    filesystem, then re-prompts the LLM with real evidence so it can produce
    a grounded spec instead of hallucinating architecture.

    Uses parse_evidence_requests() and strip_fulfilled_requests() from
    evidence_loop.py for robust ER parsing (3-layer YAML defence).

    Max loops: _EVIDENCE_MAX_LOOPS (default 2). After exhaustion, remaining
    ERs are force-resolved with FORCED_RESOLUTION markers.

    Returns updated LLM analysis with ERs replaced by RESOLVED_REQUEST or
    FORCED_RESOLUTION markers and real evidence injected.
    """
    try:
        from app.llm.pipeline.evidence_loop import (
            parse_evidence_requests,
            strip_fulfilled_requests,
            strip_forced_stop_requests,
        )
    except ImportError as exc:
        logger.warning("[SPEC_GATE_EVIDENCE] Cannot import evidence_loop: %s — skipping fulfilment", exc)
        print(f"[SPEC_GATE_EVIDENCE] WARNING: evidence_loop import failed: {exc}")
        return llm_analysis

    current_analysis = llm_analysis

    for loop_idx in range(_EVIDENCE_MAX_LOOPS):
        # Parse outstanding EVIDENCE_REQUESTs
        requests = parse_evidence_requests(current_analysis)
        if not requests:
            logger.info("[SPEC_GATE_EVIDENCE] Loop %d/%d: No EVIDENCE_REQUESTs found — done",
                        loop_idx + 1, _EVIDENCE_MAX_LOOPS)
            print(f"[SPEC_GATE_EVIDENCE] Loop {loop_idx + 1}/{_EVIDENCE_MAX_LOOPS}: No ERs — done")
            break

        logger.info("[SPEC_GATE_EVIDENCE] Loop %d/%d: %d EVIDENCE_REQUEST(s): %s",
                    loop_idx + 1, _EVIDENCE_MAX_LOOPS, len(requests),
                    [r.get('id') for r in requests])
        print(f"[SPEC_GATE_EVIDENCE] Loop {loop_idx + 1}/{_EVIDENCE_MAX_LOOPS}: "
              f"{len(requests)} ER(s): {[r.get('id') for r in requests]}")

        # Dispatch file reads for each ER
        fulfilled_ids = set()
        evidence_bundle_parts = []  # Accumulated evidence text for re-prompt

        for req in requests:
            req_id = req.get("id", "UNKNOWN")
            tool_calls = req.get("tool_calls", [])
            need = req.get("need", "")
            er_results = []

            logger.info("[SPEC_GATE_EVIDENCE] Processing %s: need=%s, tools=%d",
                        req_id, need[:80], len(tool_calls))
            # v4.2: Diagnostic logging for tool dispatch debugging
            for _tc_debug in tool_calls:
                print(f"[SPEC_GATE_EVIDENCE] {req_id}: tool='{_tc_debug.get('tool', 'NONE')}' "
                      f"args={dict(_tc_debug.get('args', {}))}")

            for tc in tool_calls:
                tool_name = tc.get("tool", "")
                args = tc.get("args", {})

                # Dispatch host-direct reads
                # v4.1: Extended tool name matching — LLM may use various tool names
                # from the evidence contract. Map them all to host-direct reads.
                if tool_name in ("sandbox_inspector.read_sandbox_file",
                                 "evidence_collector.add_file_read_to_bundle",
                                 "read_file",
                                 "sandbox_inspector.file_exists_in_sandbox",
                                 "arch_query.get_file_signatures"):
                    file_path = args.get("file_path") or args.get("path", "")
                    if file_path:
                        success, content = _host_read_file(
                            file_path,
                            max_chars=args.get("max_chars", _EVIDENCE_MAX_FILE_CHARS),
                            project_paths=project_paths,
                        )
                        er_results.append({
                            "tool": tool_name,
                            "file_path": file_path,
                            "success": success,
                            "content": content[:4000] if success else None,
                            "error": content if not success else None,
                        })
                        logger.info("[SPEC_GATE_EVIDENCE] %s: read_file %s → %s (%d chars)",
                                    req_id, file_path, "OK" if success else "FAIL",
                                    len(content) if success else 0)

                elif tool_name in ("sandbox_inspector.run_sandbox_discovery_chain",
                                   "list_directory"):
                    dir_path = args.get("anchor") or args.get("path", "")
                    if dir_path:
                        success, listing = _host_list_directory(dir_path, project_paths=project_paths)
                        er_results.append({
                            "tool": tool_name,
                            "path": dir_path,
                            "success": success,
                            "content": listing[:2000] if success else None,
                            "error": listing if not success else None,
                        })
                        logger.info("[SPEC_GATE_EVIDENCE] %s: list_dir %s → %s",
                                    req_id, dir_path, "OK" if success else "FAIL")

                elif tool_name in ("evidence_collector.verify_path_exists",
                                   "evidence_collector.find_in_evidence"):
                    # v4.2: Map verify_path_exists to a simple file existence check
                    file_path = args.get("path") or args.get("file_path", "")
                    if file_path:
                        resolved = file_path.replace('/', os.sep).replace('\\', os.sep)
                        if not os.path.exists(resolved) and project_paths:
                            for root in project_paths:
                                candidate = os.path.join(root, resolved)
                                if os.path.exists(candidate):
                                    resolved = candidate
                                    break
                        exists = os.path.exists(resolved)
                        er_results.append({
                            "tool": tool_name,
                            "file_path": resolved,
                            "success": exists,
                            "content": f"Path {'exists' if exists else 'does NOT exist'}: {resolved}" if exists else None,
                            "error": f"Path does not exist: {resolved}" if not exists else None,
                        })
                        logger.info("[SPEC_GATE_EVIDENCE] %s: verify_path %s → %s",
                                    req_id, resolved, "EXISTS" if exists else "NOT FOUND")

                elif tool_name in ("embeddings_service.search_embeddings",
                                   "evidence_collector.add_search_to_bundle",
                                   "arch_query.search_symbols"):
                    # v4.2: Search tools not available at spec stage — map to directory listing
                    # Use query/anchor to find relevant files via listing
                    query = args.get("query") or args.get("anchor", "")
                    search_path = None
                    if project_paths:
                        search_path = project_paths[0]
                        # If query looks like a path fragment, try listing that dir
                        for root in project_paths:
                            candidate = os.path.join(root, query.replace('.', os.sep))
                            if os.path.isdir(candidate):
                                search_path = candidate
                                break
                    if search_path:
                        success, listing = _host_list_directory(search_path, project_paths=project_paths)
                        er_results.append({
                            "tool": tool_name,
                            "path": search_path,
                            "success": success,
                            "content": f"Directory listing for context (search unavailable at spec stage):\n{listing[:2000]}" if success else None,
                            "error": listing if not success else None,
                        })
                    else:
                        er_results.append({
                            "tool": tool_name,
                            "skipped": True,
                            "reason": f"Search tool '{tool_name}' not available at spec stage; no project path to list",
                        })

                else:
                    # Unsupported tool at spec stage — log and skip
                    logger.info("[SPEC_GATE_EVIDENCE] %s: Skipping unsupported tool '%s' (args=%s)",
                                req_id, tool_name, list(args.keys()))
                    print(f"[SPEC_GATE_EVIDENCE] {req_id}: UNSUPPORTED TOOL '{tool_name}' "
                          f"args={list(args.keys())} — skipping")
                    er_results.append({
                        "tool": tool_name,
                        "skipped": True,
                        "reason": f"Tool '{tool_name}' not available at spec stage",
                    })

            # If we got ANY successful reads, mark this ER as fulfilled
            any_success = any(r.get("success") for r in er_results)
            if any_success:
                fulfilled_ids.add(req_id)
                # Build evidence text block for re-prompt
                evidence_text = f"\n### Evidence for {req_id} (need: {need})\n"
                for r in er_results:
                    if r.get("success") and r.get("content"):
                        label = r.get("file_path") or r.get("path", "unknown")
                        evidence_text += f"\n**{label}:**\n```\n{r['content']}\n```\n"
                    elif r.get("error"):
                        evidence_text += f"\n**Error:** {r['error']}\n"
                evidence_bundle_parts.append(evidence_text)
            else:
                # No successful reads — will be force-resolved after loops
                logger.warning("[SPEC_GATE_EVIDENCE] %s: All tool calls failed", req_id)
                print(f"[SPEC_GATE_EVIDENCE] {req_id}: ALL FAILED — "
                      f"results={[(r.get('tool','?'), r.get('error','?')[:100] if r.get('error') else r.get('reason','?')) for r in er_results]}")
                if er_results:  # Had tool calls but all failed
                    evidence_text = f"\n### Evidence for {req_id} — UNAVAILABLE\n"
                    for r in er_results:
                        if r.get("error"):
                            evidence_text += f"- {r.get('tool', '?')}: {r['error']}\n"
                        elif r.get("skipped"):
                            evidence_text += f"- {r.get('tool', '?')}: {r.get('reason', 'skipped')}\n"
                    evidence_bundle_parts.append(evidence_text)

        if not fulfilled_ids and not evidence_bundle_parts:
            logger.info("[SPEC_GATE_EVIDENCE] No evidence gathered — stopping loop")
            print("[SPEC_GATE_EVIDENCE] No evidence gathered — stopping loop")
            break

        # Strip fulfilled ERs → RESOLVED_REQUEST markers
        current_analysis = strip_fulfilled_requests(current_analysis, fulfilled_ids)

        # Re-prompt LLM with the original analysis + real evidence
        evidence_block = "\n".join(evidence_bundle_parts)

        # v4.4: Determine if this is the final loop
        is_final_loop = (loop_idx >= _EVIDENCE_MAX_LOOPS - 1)

        # v4.4: Cap evidence block to prevent context explosion
        _MAX_EVIDENCE_CHARS = 40000
        if len(evidence_block) > _MAX_EVIDENCE_CHARS:
            evidence_block = evidence_block[:_MAX_EVIDENCE_CHARS] + (
                f"\n\n... [Evidence truncated at {_MAX_EVIDENCE_CHARS} chars. "
                f"Focus on the evidence shown above to produce your analysis.]"
            )
            print(f"[SPEC_GATE_EVIDENCE] Evidence truncated to {_MAX_EVIDENCE_CHARS} chars")

        # v4.4: Cap previous analysis to prevent input explosion
        _MAX_PREV_ANALYSIS_CHARS = 15000
        prev_analysis_text = current_analysis
        if len(prev_analysis_text) > _MAX_PREV_ANALYSIS_CHARS:
            prev_analysis_text = prev_analysis_text[:_MAX_PREV_ANALYSIS_CHARS] + (
                f"\n\n... [Previous analysis truncated. "
                f"Use the evidence below to produce a complete, grounded analysis.]"
            )
            print(f"[SPEC_GATE_EVIDENCE] Previous analysis truncated to {_MAX_PREV_ANALYSIS_CHARS} chars")

        if is_final_loop:
            # v4.4: FINAL LOOP — force spec production, no more ERs allowed
            re_prompt = (
                f"You have completed {loop_idx + 1} rounds of evidence gathering. "
                f"The orchestrator has read real files from the codebase for you across "
                f"all rounds.\n\n"
                f"THIS IS YOUR FINAL ROUND. You MUST now produce your complete, grounded "
                f"analysis. Do NOT emit any more EVIDENCE_REQUEST blocks — they will be "
                f"ignored. Use the evidence you have to produce the best possible analysis.\n\n"
                f"For anything you still don't know, use DECISION_ALLOWED with a sensible "
                f"default, or HUMAN_REQUIRED only if truly high-risk.\n\n"
                f"REQUIRED OUTPUT SECTIONS (all mandatory):\n"
                f"## Architecture Overview\n"
                f"## Implementation Steps (numbered, actionable, referencing real files)\n"
                f"## Files to Modify (with WHAT and WHY for each)\n"
                f"## New Files to Create (or state 'None needed')\n"
                f"## Acceptance Criteria (testable, specific)\n\n"
                f"--- PREVIOUS ANALYSIS ---\n\n"
                f"{prev_analysis_text}\n\n"
                f"--- EVIDENCE FROM CODEBASE ---\n\n"
                f"{evidence_block}\n\n"
                f"--- END EVIDENCE ---\n\n"
                f"Produce your FINAL grounded analysis now. All sections required. "
                f"No EVIDENCE_REQUEST blocks."
            )
            logger.info("[SPEC_GATE_EVIDENCE] FINAL LOOP — forcing spec production")
            print(f"[SPEC_GATE_EVIDENCE] FINAL LOOP — forcing spec production (no more ERs)")
        else:
            re_prompt = (
                f"You previously produced the analysis below, which contained "
                f"EVIDENCE_REQUEST blocks. The orchestrator has now fulfilled "
                f"{len(fulfilled_ids)} of those requests by reading actual files. "
                f"The fulfilled requests have been replaced with RESOLVED_REQUEST markers.\n\n"
                f"Please revise your analysis using the REAL evidence provided below. "
                f"Replace any assumptions or hallucinated architecture with what the "
                f"actual code shows. Keep all other sections intact. "
                f"If you still need more evidence, you may emit new EVIDENCE_REQUEST blocks "
                f"(with new unique IDs, not reusing resolved ones). Focus your new ERs on "
                f"the most critical gaps — prioritise files that directly affect the "
                f"implementation over general exploration.\n\n"
                f"--- PREVIOUS ANALYSIS (with RESOLVED_REQUEST markers) ---\n\n"
                f"{prev_analysis_text}\n\n"
                f"--- FULFILLED EVIDENCE ---\n\n"
                f"{evidence_block}\n\n"
                f"--- END EVIDENCE ---\n\n"
                f"Please provide your revised, grounded analysis."
            )

        logger.info("[SPEC_GATE_EVIDENCE] Re-prompting LLM with %d chars of evidence "
                    "(%d ERs fulfilled, %d unfulfilled)",
                    len(evidence_block), len(fulfilled_ids),
                    len(requests) - len(fulfilled_ids))
        print(f"[SPEC_GATE_EVIDENCE] Re-prompting LLM: {len(fulfilled_ids)} fulfilled, "
              f"{len(requests) - len(fulfilled_ids)} remaining")

        try:
            result = await llm_call_func(
                provider_id=provider_id,
                model_id=model_id,
                messages=[{"role": "user", "content": re_prompt}],
                system_prompt=CREATE_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=8192,
                timeout_seconds=_CREATE_ANALYSIS_TIMEOUT,
            )

            if result.is_success() and result.content:
                current_analysis = result.content.strip()
                logger.info("[SPEC_GATE_EVIDENCE] Re-analysis success: %d chars", len(current_analysis))
                print(f"[SPEC_GATE_EVIDENCE] Re-analysis: {len(current_analysis)} chars")
            else:
                error_msg = getattr(result, 'error_message', 'Unknown error')
                logger.warning("[SPEC_GATE_EVIDENCE] Re-analysis LLM failed: %s — keeping current", error_msg)
                print(f"[SPEC_GATE_EVIDENCE] Re-analysis failed: {error_msg} — keeping current")
                break  # Don't loop further if LLM fails

        except Exception as exc:
            logger.warning("[SPEC_GATE_EVIDENCE] Re-analysis exception: %s — keeping current", exc)
            print(f"[SPEC_GATE_EVIDENCE] Re-analysis exception: {exc} — keeping current")
            break

    # Force-resolve any remaining ERs after all loops
    remaining = parse_evidence_requests(current_analysis)
    if remaining:
        remaining_ids = {r.get("id", "UNKNOWN") for r in remaining}
        logger.warning("[SPEC_GATE_EVIDENCE] Force-resolving %d remaining ER(s) after %d loops: %s",
                       len(remaining), _EVIDENCE_MAX_LOOPS, remaining_ids)
        print(f"[SPEC_GATE_EVIDENCE] Force-resolving {len(remaining)} remaining ER(s): {remaining_ids}")
        current_analysis = strip_forced_stop_requests(current_analysis, remaining_ids)

    return current_analysis


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
- Do NOT suggest files/features outside the stated phase scope

AMBIGUITY HANDLING:
When a design decision has multiple valid approaches (e.g., "should the endpoint
accept multipart/form-data or raw bytes?"), do NOT flag it as HUMAN_REQUIRED if
a safe default exists that covers both options. Instead:
- Pick the most flexible default (e.g., "support both multipart and raw body")
- State it as a DECISION_ALLOWED with the chosen default
- Only flag HUMAN_REQUIRED when there is genuine ambiguity with no safe default,
  or when the choice has significant architectural consequences that cannot be
  reversed without major rework
- Implementation details like exact response field names, optional metadata fields,
  or input format variants are NOT architectural decisions — adopt sensible defaults

ENTRYPOINT IDENTIFICATION (CRITICAL):
When generating EVIDENCE_REQUESTs to locate backend entrypoints (e.g., main.py):
- The goal is to find the file that instantiates FastAPI() (or calls an app factory
  returning FastAPI) and registers routers via include_router().
- Ignore any main.py files under /static/, /dist/, /build/, /public/, or frontend
  project roots. These are NOT the backend entrypoint.
- success_criteria MUST include: "Evidence must include the lines showing
  app = FastAPI(...) (or equivalent factory) and at least one include_router(...)."
- If multiple main.py files exist, the ER must distinguish them by checking content,
  not just path.

CONFIGURATION FILE EVIDENCE:
When the task references external configuration files (e.g., config.ini, .env, YAML)
that are loaded by service wrappers or modules being integrated:
- Include an EVIDENCE_REQUEST to read the configuration file directly.
- Confirm that the sections, keys, and values match what the consuming code expects.
- This prevents runtime mismatches between config parsers and actual config content.
- Fold this into an existing ER if it already reads the consuming module's code,
  or create a dedicated ER if the config file is on a separate path.

ER ID UNIQUENESS:
- Every EVIDENCE_REQUEST must have a unique id (ER-001, ER-002, etc.).
- NEVER emit two EVIDENCE_REQUEST blocks with the same id.
- If you need to request evidence for two different things, use different ids.

EVIDENCE_REQUEST FORMAT (you MUST use this exact YAML format — the parser is strict):
EVIDENCE_REQUEST:
  id: "ER-NNN"
  severity: "CRITICAL" | "NONCRITICAL"
  need: "What you need to know"
  why: "What breaks if you guess wrong"
  scope:
    roots: ["where to look"]
    max_files: 500
  tool_calls:
    - tool: "sandbox_inspector.read_sandbox_file"
      args: {file_path: "full/path/to/file.py"}
      expect: "What you expect to find"
  success_criteria: "What counts as having the answer"
  fallback_if_not_found: "DECISION_ALLOWED" | "HUMAN_REQUIRED"

CRITICAL FORMATTING RULES:
- The block MUST start with 'EVIDENCE_REQUEST:' on its own line (no prefix, no markdown header)
- Fields MUST be indented with 2 spaces under EVIDENCE_REQUEST:
- The id field MUST be quoted: id: "ER-001" (not id: ER-001)
- Do NOT wrap EVIDENCE_REQUESTs in markdown code blocks or headers
- Do NOT use prose-style descriptions — use the YAML structure above

TOOL USAGE:
- Use tool 'sandbox_inspector.read_sandbox_file' with args: {file_path: "FULL_PATH"} for reading files
- Use tool 'sandbox_inspector.run_sandbox_discovery_chain' with args: {anchor: "FULL_PATH"} for listing directories
- ALWAYS use FULL ABSOLUTE PATHS from the Integration Points list provided to you
- Do NOT guess paths. Only request files that appear in the Integration Points list,
  or that you discovered from a previous directory listing or file read.
- If you need to find files not in the Integration Points list, first request a
  directory listing of the parent directory, then request specific files from the results.

When you need to examine files to ground your analysis, emit EVIDENCE_REQUEST blocks
in this exact format. The orchestrator will read the files and re-prompt you with
the actual contents so you can produce a grounded analysis instead of guessing."""


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
    for p in integration_points[:20]:  # v4.2: Increased from 10 to give LLM more valid paths for ERs
        integration_desc.append(f"- {p.file_path} ({p.action}): {p.relevance}")
    
    constraints_desc = "\n".join(f"- {c}" for c in constraints) if constraints else "None specified"
    
    user_prompt = f"""Feature Request:
{goal}

Full Description:
{what_to_do[:3000]}

Tech Stack:
{chr(10).join(stack_desc) if stack_desc else 'Not detected'}

Existing Integration Points (VERIFIED — these files exist and can be read via EVIDENCE_REQUESTs):
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
                max_tokens=8192,  # v4.1: Increased from 4096 — LLM needs room for analysis + YAML ERs
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

            # v4.0: Fulfil EVIDENCE_REQUESTs from the LLM analysis
            # If the LLM produced ERs asking to read specific files, read them
            # and re-prompt with real evidence for a grounded spec.
            if llm_analysis and 'EVIDENCE_REQUEST' in llm_analysis:
                logger.info("[SPEC_GATE_EVIDENCE] LLM analysis contains EVIDENCE_REQUESTs — starting fulfilment")
                print("[SPEC_GATE_EVIDENCE] EVIDENCE_REQUESTs detected — starting fulfilment loop")
                llm_analysis = await _fulfil_evidence_requests(
                    llm_analysis=llm_analysis,
                    provider_id=provider_id,
                    model_id=model_id,
                    llm_call_func=llm_call_func,
                    project_paths=project_paths,
                    goal=goal,
                    what_to_do=what_to_do,
                )
                print(f"[SPEC_GATE_EVIDENCE] Fulfilment complete: {len(llm_analysis)} chars")
            elif llm_analysis:
                logger.info("[SPEC_GATE_EVIDENCE] No EVIDENCE_REQUESTs in LLM analysis — skipping fulfilment")
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

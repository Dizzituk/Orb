# FILE: app/pot_spec/grounded/simple_create.py
"""
SpecGate CREATE Path - Evidence-Based Feature Spec Builder

Provides grounded specs for CREATE tasks (new features) just like
simple_refactor.py does for REFACTOR tasks.

Flow:
1. Extract keywords from task description
2. Scan codebase for relevant integration points
3. Detect tech stack and patterns
4. Build grounded spec with WHERE + HOW + INTEGRATION

v1.0 (2026-02-02): Initial implementation
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SIMPLE_CREATE_BUILD_ID = "2026-02-02-v1.2-goal-validation"
print(f"[SIMPLE_CREATE_LOADED] BUILD_ID={SIMPLE_CREATE_BUILD_ID}")


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


# =============================================================================
# KEYWORD EXTRACTION
# =============================================================================

# Map task concepts to search keywords
CONCEPT_KEYWORDS = {
    "voice": ["voice", "audio", "speech", "microphone", "mic", "record"],
    "text_input": ["input", "text", "textarea", "type", "typing"],
    "button": ["button", "btn", "click", "press", "toggle"],
    "api": ["api", "fetch", "request", "endpoint", "http"],
    "upload": ["upload", "file", "blob", "form"],
    "transcribe": ["transcribe", "transcription", "whisper", "speech-to-text", "stt"],
    "ui_component": ["component", "ui", "interface", "widget"],
    "state": ["state", "useState", "context", "store", "redux"],
}


def _extract_task_keywords(text: str) -> List[str]:
    """Extract relevant keywords from task description."""
    text_lower = text.lower()
    found = set()
    
    for concept, keywords in CONCEPT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.add(concept)
                found.add(kw)
    
    # Also extract capitalized proper nouns (likely tech names)
    proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b', text)
    for noun in proper_nouns:
        if noun.lower() not in {'the', 'a', 'an', 'this', 'that', 'what', 'how'}:
            found.add(noun.lower())
    
    return list(found)


# =============================================================================
# TECH STACK DETECTION
# =============================================================================

def _detect_tech_stack(project_path: str, sandbox_client: Any = None) -> TechStack:
    """Detect the technology stack of a project."""
    stack = TechStack()
    
    # Check for common config files
    indicators = {
        # Frontend
        "package.json": None,
        "tsconfig.json": ("frontend_language", "TypeScript"),
        "vite.config.ts": ("frontend_framework", "React/Vite"),
        "vite.config.js": ("frontend_framework", "React/Vite"),
        "next.config.js": ("frontend_framework", "Next.js"),
        "angular.json": ("frontend_framework", "Angular"),
        "vue.config.js": ("frontend_framework", "Vue"),
        # Backend
        "requirements.txt": ("backend_language", "Python"),
        "pyproject.toml": ("backend_language", "Python"),
        "main.py": ("backend_framework", "FastAPI"),  # Will verify
        "app.py": ("backend_framework", "Flask"),
        "server.js": ("backend_framework", "Express"),
        "server.ts": ("backend_framework", "Express"),
        # Styling
        "tailwind.config.js": ("styling", "Tailwind"),
        "tailwind.config.ts": ("styling", "Tailwind"),
    }
    
    for filename, detection in indicators.items():
        check_path = os.path.join(project_path, filename)
        # Try to check if file exists via sandbox or local
        try:
            if sandbox_client:
                # Use sandbox to check
                pass  # TODO: implement sandbox check
            else:
                # Local check (for when running on host)
                if os.path.exists(check_path):
                    if detection:
                        setattr(stack, detection[0], detection[1])
        except Exception:
            pass
    
    # Check for React specifically
    try:
        pkg_path = os.path.join(project_path, "package.json")
        if os.path.exists(pkg_path):
            import json
            with open(pkg_path, 'r') as f:
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
    
    # Check for FastAPI in Python projects
    try:
        req_path = os.path.join(project_path, "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path, 'r') as f:
                reqs = f.read().lower()
            if "fastapi" in reqs:
                stack.backend_framework = "FastAPI"
            if "flask" in reqs:
                stack.backend_framework = "Flask"
            if "django" in reqs:
                stack.backend_framework = "Django"
    except Exception as e:
        logger.debug("[simple_create] Could not parse requirements.txt: %s", e)
    
    return stack


# =============================================================================
# INTEGRATION POINT DISCOVERY
# =============================================================================

def _find_integration_points(
    project_path: str,
    keywords: List[str],
    sandbox_client: Any = None,
) -> List[IntegrationPoint]:
    """Find files that are likely integration points based on keywords."""
    points = []
    
    # Map keywords to likely file patterns
    keyword_file_patterns = {
        "input": [r"input", r"Input", r"form", r"Form"],
        "button": [r"button", r"Button", r"btn"],
        "text": [r"text", r"Text", r"input", r"Input"],
        "voice": [r"voice", r"Voice", r"audio", r"Audio", r"speech", r"Speech"],
        "api": [r"api\.", r"Api", r"service", r"Service", r"fetch"],
        "header": [r"header", r"Header", r"nav", r"Nav"],
        "chat": [r"chat", r"Chat", r"message", r"Message"],
    }
    
    # Common integration point patterns
    common_patterns = [
        (r"InputSection", "Primary text input component", "modify"),
        (r"ChatWindow", "Main chat interface", "modify"),
        (r"Header", "App header/toolbar", "modify"),
        (r"api\.ts", "API client layer", "modify"),
        (r"api\.js", "API client layer", "modify"),
        (r"App\.tsx", "Root component", "reference"),
        (r"App\.jsx", "Root component", "reference"),
        (r"main\.tsx", "Entry point", "reference"),
        (r"index\.tsx", "Entry point", "reference"),
    ]
    
    try:
        # Walk the project directory
        for root, dirs, files in os.walk(project_path):
            # Skip node_modules, .git, etc.
            dirs[:] = [d for d in dirs if d not in {
                'node_modules', '.git', '__pycache__', '.venv', 'venv',
                'dist', 'build', '.next', 'coverage'
            }]
            
            for filename in files:
                # Only check source files
                if not filename.endswith(('.tsx', '.jsx', '.ts', '.js', '.py', '.css')):
                    continue
                
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, project_path)
                
                # Check against common patterns
                for pattern, relevance, action in common_patterns:
                    if re.search(pattern, filename, re.IGNORECASE):
                        points.append(IntegrationPoint(
                            file_path=full_path,
                            file_name=filename,
                            relevance=relevance,
                            action=action,
                        ))
                        break
                
                # Check keyword matches in filename
                for kw in keywords:
                    if kw.lower() in filename.lower():
                        if not any(p.file_path == full_path for p in points):
                            points.append(IntegrationPoint(
                                file_path=full_path,
                                file_name=filename,
                                relevance=f"Filename contains '{kw}'",
                                action="reference",
                            ))
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
# FILE SUGGESTION
# =============================================================================

def _suggest_new_files(
    keywords: List[str],
    tech_stack: TechStack,
    project_paths: List[str],
) -> List[str]:
    """
    Suggest new files to create based on task keywords and tech stack.
    
    Maps common feature patterns to file structures.
    """
    suggested = []
    
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
    
    # Voice/Audio feature
    if any(kw in keywords for kw in ['voice', 'audio', 'speech', 'microphone', 'record']):
        # Frontend component for voice input
        if frontend_path or tech_stack.frontend_framework:
            ext = '.tsx' if tech_stack.frontend_language == 'TypeScript' else '.jsx'
            suggested.append(f"src/components/VoiceInput{ext}")
            suggested.append(f"src/hooks/useVoiceRecorder{ext.replace('x', '')}")
        
        # Backend endpoint for transcription
        if backend_path or tech_stack.backend_framework:
            if tech_stack.backend_framework == 'FastAPI':
                suggested.append("app/routers/transcribe.py")
            elif tech_stack.backend_framework == 'Express':
                suggested.append("routes/transcribe.js")
    
    # Upload feature
    if any(kw in keywords for kw in ['upload', 'file']):
        if frontend_path or tech_stack.frontend_framework:
            ext = '.tsx' if tech_stack.frontend_language == 'TypeScript' else '.jsx'
            suggested.append(f"src/components/FileUpload{ext}")
        if backend_path or tech_stack.backend_framework:
            if tech_stack.backend_framework == 'FastAPI':
                suggested.append("app/routers/upload.py")
    
    # Button/UI component
    if any(kw in keywords for kw in ['button', 'toggle', 'switch']):
        if frontend_path or tech_stack.frontend_framework:
            ext = '.tsx' if tech_stack.frontend_language == 'TypeScript' else '.jsx'
            # Only suggest if it seems like a new standalone component
            if 'voice' in keywords:
                pass  # Already suggested VoiceInput
            else:
                suggested.append(f"src/components/CustomButton{ext}")
    
    # API integration
    if any(kw in keywords for kw in ['api', 'openai', 'whisper', 'transcribe']):
        if backend_path or tech_stack.backend_framework:
            if tech_stack.backend_framework == 'FastAPI':
                suggested.append("app/services/openai_client.py")
    
    # Dedupe
    seen = set()
    unique = []
    for f in suggested:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    
    return unique


# =============================================================================
# SPEC BUILDER
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
    
    If goal is a placeholder, extract a real goal from what_to_do.
    """
    if not goal:
        goal = ""
    
    goal_clean = goal.split('\n')[0].strip().lower()
    
    # Check if goal is a placeholder
    is_placeholder = (
        not goal_clean or
        goal_clean in PLACEHOLDER_GOALS or
        goal_clean.startswith("job description") or
        goal_clean.startswith("weaver output")
    )
    
    if is_placeholder and what_to_do:
        # Try to extract a real goal from what_to_do
        # Look for "Intent:" line first
        import re
        intent_match = re.search(r'intent\s*[:\-]\s*(.+?)(?:\n|$)', what_to_do, re.IGNORECASE)
        if intent_match:
            return intent_match.group(1).strip()
        
        # Look for "What is being built:" line
        built_match = re.search(r'what\s+is\s+being\s+built\s*[:\-]\s*(.+?)(?:\n|$)', what_to_do, re.IGNORECASE)
        if built_match:
            return built_match.group(1).strip()
        
        # Fall back to first non-empty, non-header line
        for line in what_to_do.split('\n'):
            line = line.strip()
            if line and line.lower() not in PLACEHOLDER_GOALS and not line.startswith('#'):
                if not line.lower().startswith('what is being built'):
                    return line[:200]
    
    return goal.split('\n')[0].strip() if goal else "Implement requested feature"


def build_create_spec(
    goal: str,
    what_to_do: str,
    evidence: CreateEvidence,
    project_paths: List[str],
) -> str:
    """Build a grounded CREATE spec with evidence."""
    lines = []
    
    # v1.2: Sanitize goal to ensure it's not a placeholder
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
    
    # Integration Points (WHERE)
    lines.append("## Integration Points")
    lines.append("")
    
    modify_points = [p for p in evidence.integration_points if p.action == "modify"]
    reference_points = [p for p in evidence.integration_points if p.action != "modify"]
    
    if modify_points:
        lines.append("### Files to Modify")
        lines.append("")
        for p in modify_points[:5]:
            lines.append(f"- `{p.file_name}` — {p.relevance}")
        lines.append("")
    
    if reference_points:
        lines.append("### Reference Files (patterns to follow)")
        lines.append("")
        for p in reference_points[:5]:
            lines.append(f"- `{p.file_name}` — {p.relevance}")
        lines.append("")
    
    # Suggested New Files
    if evidence.suggested_files:
        lines.append("### New Files to Create")
        lines.append("")
        for f in evidence.suggested_files:
            lines.append(f"- `{f}`")
        lines.append("")
    
    # What To Do (from Weaver)
    lines.append("## Implementation Steps")
    lines.append("")
    if what_to_do:
        for line in what_to_do.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('What is being built'):
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
            # Truncate for readability
            pattern_preview = pattern[:400] + "..." if len(pattern) > 400 else pattern
            lines.append(pattern_preview)
            lines.append("```")
            lines.append("")
    
    # Acceptance Criteria
    lines.append("## Acceptance")
    lines.append("")
    lines.append("- [ ] Feature works as described")
    lines.append("- [ ] Integrates with existing UI patterns")
    lines.append("- [ ] No console errors")
    lines.append("- [ ] App boots without issues")
    lines.append("")
    
    # Evidence Summary
    lines.append("## Evidence Summary")
    lines.append("")
    lines.append(f"- Integration points found: {len(evidence.integration_points)}")
    lines.append(f"- Patterns extracted: {len(evidence.existing_patterns)}")
    lines.append(f"- Project paths: {', '.join(project_paths)}")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def build_grounded_create_spec(
    goal: str,
    what_to_do: str,
    project_paths: List[str],
    sandbox_client: Any = None,
) -> Tuple[str, CreateEvidence]:
    """
    Build a grounded spec for CREATE tasks.
    
    Returns:
        Tuple of (spec_markdown, evidence)
    """
    logger.info("[simple_create] v1.0 Building grounded CREATE spec")
    print(f"[simple_create] v1.0 GROUNDED CREATE: {goal[:60]}...")
    
    # Extract keywords from task
    combined_text = f"{goal} {what_to_do}"
    keywords = _extract_task_keywords(combined_text)
    print(f"[simple_create] v1.0 Keywords: {keywords[:10]}")
    
    # Detect tech stack for each project path
    tech_stack = TechStack()
    for path in project_paths:
        if os.path.isdir(path):
            detected = _detect_tech_stack(path, sandbox_client)
            # Merge (prefer non-None values)
            for attr in ['frontend_framework', 'frontend_language', 'backend_framework',
                        'backend_language', 'styling', 'state_management', 'api_pattern']:
                if getattr(detected, attr) and not getattr(tech_stack, attr):
                    setattr(tech_stack, attr, getattr(detected, attr))
    
    print(f"[simple_create] v1.0 Tech stack: {tech_stack.frontend_framework}/{tech_stack.backend_framework}")
    
    # Find integration points
    all_points = []
    for path in project_paths:
        if os.path.isdir(path):
            points = _find_integration_points(path, keywords, sandbox_client)
            all_points.extend(points)
    
    print(f"[simple_create] v1.0 Found {len(all_points)} integration points")
    
    # Extract patterns from integration points
    patterns = _extract_patterns(all_points, tech_stack)
    print(f"[simple_create] v1.0 Extracted {len(patterns)} patterns")
    
    # Suggest new files based on task
    suggested_files = _suggest_new_files(keywords, tech_stack, project_paths)
    
    # Build evidence bundle
    evidence = CreateEvidence(
        tech_stack=tech_stack,
        integration_points=all_points,
        existing_patterns=patterns,
        suggested_files=suggested_files,
        keywords_found={kw: [] for kw in keywords},  # TODO: populate
    )
    
    # Build spec
    spec = build_create_spec(
        goal=goal,
        what_to_do=what_to_do,
        evidence=evidence,
        project_paths=project_paths,
    )
    
    print(f"[simple_create] v1.0 SPEC READY: {len(spec)} chars")
    
    return spec, evidence

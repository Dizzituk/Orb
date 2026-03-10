from __future__ import annotations
import logging
import os
from app.pot_spec.grounded._simple_create_utils_15 import _read_text_any_encoding
from app.pot_spec.grounded._sbx_fs import _sbx_exists, _sbx_read
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.


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
class CreateEvidence:
    """Evidence gathered for a CREATE task."""
    tech_stack: TechStack
    integration_points: List[IntegrationPoint]
    existing_patterns: Dict[str, str]  # pattern_name -> example code/file
    suggested_files: List[str]  # New files to create
    keywords_found: Dict[str, List[str]]  # keyword -> files containing it
    constraints: List[str]  # Extracted constraints (e.g., "no cloud APIs")
    llm_analysis: Optional[str] = None  # LLM-generated analysis
    resolved_target_files: Optional[List[Dict[str, str]]] = None  # v5.2: Pre-resolved file paths from weaver

def _detect_tech_stack(project_path: str, sandbox_client: Any = None) -> TechStack:
    """Detect the technology stack of a project."""
    from ._simple_create_utils_17 import TechStack
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
            if _sbx_exists(check_path):
                setattr(stack, detection[0], detection[1])
        except Exception:
            pass
    
    # Check package.json for React/Vue/state management
    try:
        pkg_path = os.path.join(project_path, "package.json")
        if _sbx_exists(pkg_path):
            import json
            pkg_content = _sbx_read(pkg_path)
            if not pkg_content:
                raise ValueError("Could not read package.json from sandbox")
            pkg = json.loads(pkg_content)
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
        if _sbx_exists(req_path):
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
            if _sbx_exists(main_path):
                main_content = _sbx_read(main_path)
                if main_content:
                    main_content = main_content[:2000]
                    if 'fastapi' in main_content.lower() or 'FastAPI' in main_content:
                        stack.backend_framework = "FastAPI"
                        logger.info("[simple_create] v2.1 Detected FastAPI from main.py imports")
                    elif 'flask' in main_content.lower():
                        stack.backend_framework = "Flask"
        except Exception as e:
            logger.debug("[simple_create] Could not check main.py: %s", e)
    
    return stack

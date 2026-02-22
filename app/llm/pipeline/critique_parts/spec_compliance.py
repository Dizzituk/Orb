# FILE: app/llm/pipeline/critique_parts/spec_compliance.py
"""Block 5: DETERMINISTIC Spec-Compliance Check (v1.3+ / v2.2)

Platform detection, stack mismatch, scope inflation, and all related helpers.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.llm.pipeline.critique_schemas import CritiqueIssue

logger = logging.getLogger(__name__)

# Stack detection keywords
_STACK_KEYWORDS = {
    "python": "Python", "pygame": "Python+Pygame", "tkinter": "Python+Tkinter",
    "pyqt": "Python+PyQt", "flask": "Python+Flask", "fastapi": "Python+FastAPI",
    "django": "Python+Django", "javascript": "JavaScript", "typescript": "TypeScript",
    "react": "TypeScript/React", "electron": "TypeScript/Electron",
    "node.js": "Node.js", "nodejs": "Node.js", "next.js": "TypeScript/Next.js",
    "vue": "TypeScript/Vue", "rust": "Rust", "golang": "Go",
    "c++": "C++", "c#": "C#", "java": "Java",
}

_SCOPE_INFLATION_KEYWORDS = [
    "electron-builder", "packaging", "installer", ".exe", ".msi",
    "telemetry", "analytics", "crash-report", "remote",
    "vite", "webpack", "bundler",
    "playwright", "e2e", "end-to-end",
    "authentication", "auth", "oauth",
    "database", "sqlite", "persistence",
    "%appdata%", "local storage",
    "settings ui", "menus", "overlays",
]

# --- Exclusion context patterns (v1.8 - v1.11) ---
_EXCLUSION_CONTEXT_PATTERNS = [
    r'out\s+of\s+scope', r'not\s+(?:included|supported|targeted|in\s+(?:this|scope|phase))',
    r'excluded', r'will\s+not', r"won't", r"don't", r'no\s+mobile',
    r'phase\s+[2-9]', r'future\s+(?:work|phase|release|version|enhancement|consideration)',
    r'planned\s+for\s+(?:future|later)', r'later\s+(?:phase|version|release)',
    r'beyond\s+(?:scope|phase\s*1|v1|mvp)', r'future\s+\w+',
    r'critique\s+(?:claims?|says?|states?|flagged|reported)',
    r'erroneous\s+assessment', r'factually\s+incorrect', r'rejecting\s+this',
    r'revision\s+notes?', r'review\s*:.*(?:rejected|incorrect|wrong)',
    r'reviewer\s+(?:claim|suggestion|feedback|assertion)', r'false\s+positive',
    r'\bREJECT\b', r'this\s+is\s+(?:a\s+)?(?:false|erroneous|incorrect)',
    r'appears\s+to\s+be\s+an\s+error', r'spec[_-]?compliance', r'platform\s+mismatch\s*:',
    r'explicitly\s+not', r'not\s+(?:yet|now)', r'must\s+not\s+block',
    r'should\s+not\s+block', r'without\s+(?:blocking|preventing)',
    r'allowing\s+(?:future|other|additional|external)',
]

_INLINE_EXCLUSION_PATTERNS = [
    re.compile(r'^\s*[\u274c\u2717\u2718\u2573\u00d7]', re.UNICODE),
    re.compile(r'no\s+mobile', re.IGNORECASE),
    re.compile(r'not\s+(?:in\s+)?(?:this|phase|scope|v1|mvp)', re.IGNORECASE),
    re.compile(r'phase\s+[2-9]', re.IGNORECASE),
    re.compile(r'must\s+not\s+block', re.IGNORECASE),
    re.compile(r'future\s+phase', re.IGNORECASE),
    re.compile(r'explicitly\s+not', re.IGNORECASE),
    re.compile(r'\*\*Reviewer\s+(?:Claim|Suggestion)', re.IGNORECASE),
    re.compile(r'\*\*DECISION\*\*\s*:\s*\*\*REJECT', re.IGNORECASE),
    re.compile(r'false\s+positive', re.IGNORECASE),
    re.compile(r'allowing\s+future\s+\w+', re.IGNORECASE),
]

_EXCLUSION_SECTION_HEADERS = [
    re.compile(r'^#+\s*.*(?:out\s+of\s+scope|future\s+consideration|not\s+in\s+(?:scope|phase)|excluded|deferred|limitation)', re.IGNORECASE),
    re.compile(r'^#+\s*.*(?:revision\s+(?:notes?|log|history))', re.IGNORECASE),
    re.compile(r'^\d+\.\s*(?:future\s+consideration|out\s+of\s+scope|revision\s+(?:notes?|log))', re.IGNORECASE),
    re.compile(r'^\*\*(?:out\s+of\s+scope|future|excluded|not\s+in\s+phase)', re.IGNORECASE),
    re.compile(r'^#+\s*.*(?:reviewer\s+suggestion|revision\s+response|critique\s+rebuttal|spec[_-]?compliance)', re.IGNORECASE),
    re.compile(r'^#+\s*.*(?:platform\s+mismatch\s+(?:claim|analysis))', re.IGNORECASE),
]

_SECTION_HEADER_RE = re.compile(r'^(?:#{1,6}\s|\d+\.\s)')
_EXCLUSION_CONTEXT_RE = [re.compile(p, re.IGNORECASE) for p in _EXCLUSION_CONTEXT_PATTERNS]

# v2.2: Incidental vs structural platform patterns
_INCIDENTAL_PLATFORM_PATTERNS = [
    re.compile(r'mobile[\s-](?:friendly|first|responsive|optimized|compatible|aware|ready)', re.IGNORECASE),
    re.compile(r'mobile\s+(?:browser|browsers|safari|chrome|device|devices|screen|viewport)', re.IGNORECASE),
    re.compile(r'mobile\s+(?:audio|microphone|media|recording|input)', re.IGNORECASE),
    re.compile(r'(?:audio|microphone|media|recording)\s+(?:on|for|from)\s+mobile', re.IGNORECASE),
    re.compile(r'(?:support|handle|detect|check)\s+(?:for\s+)?mobile', re.IGNORECASE),
    re.compile(r'mobile\s+(?:support|compatibility|fallback)', re.IGNORECASE),
    re.compile(r'(?:responsive|breakpoint|media\s+query|@media).*mobile', re.IGNORECASE),
    re.compile(r'mobile.*(?:responsive|breakpoint|media\s+query|@media)', re.IGNORECASE),
    re.compile(r'(?:user[\s-]?agent|navigator|window).*mobile', re.IGNORECASE),
    re.compile(r'mobile.*(?:user[\s-]?agent|detection)', re.IGNORECASE),
    re.compile(r'(?:MediaRecorder|getUserMedia|WebRTC|navigator\.mediaDevices).*mobile', re.IGNORECASE),
    re.compile(r'mobile.*(?:MediaRecorder|getUserMedia|WebRTC)', re.IGNORECASE),
]

_STRUCTURAL_PLATFORM_PATTERNS = [
    re.compile(r'(?:target|platform|deploy|build)\s*(?::|for|to)\s*(?:.*\b)?(?:mobile|android|ios)', re.IGNORECASE),
    re.compile(r'(?:mobile|android|ios)\s+(?:app|application|client|platform|target|deployment)', re.IGNORECASE),
    re.compile(r'(?:app\s+store|play\s+store|google\s+play|apple\s+store|apk|ipa\b)', re.IGNORECASE),
    re.compile(r'(?:react\s+native|flutter|kotlin|swift|xcode|android\s+studio)', re.IGNORECASE),
    re.compile(r'(?:cordova|capacitor|ionic|expo)\b', re.IGNORECASE),
    re.compile(r'\b(?:ios|android)\s+(?:sdk|api|permission|manifest)', re.IGNORECASE),
    re.compile(r'(?:push\s+notification|geofencing|nfc)\s+.*(?:mobile|android|ios)', re.IGNORECASE),
]

_DESKTOP_DEFINITIVE_STACKS = {'TypeScript/Electron', 'Python+PyQt', 'Python+Tkinter', 'C#'}
_MOBILE_DEFINITIVE_STACKS = {'React Native', 'Flutter', 'Kotlin', 'Swift'}

# Keywords needing word-boundary matching
_WORD_BOUNDARY_KEYWORDS = {"rust", "java", "vue"}


def _detect_stack_from_text(text: str) -> List[str]:
    """Detect mentioned technology stacks from text (case-insensitive).
    v1.1 FIX: Word-boundary matching for ambiguous keywords.
    """
    if not text:
        return []
    text_lower = text.lower()
    detected = []
    for keyword, stack_name in _STACK_KEYWORDS.items():
        if keyword.strip() in _WORD_BOUNDARY_KEYWORDS:
            pattern = r'\b' + re.escape(keyword.strip()) + r'\b'
            if re.search(pattern, text_lower):
                detected.append(stack_name)
        else:
            if keyword in text_lower:
                detected.append(stack_name)
    return list(set(detected))


def _extract_spec_constraints(spec_json: Optional[str]) -> Dict[str, Any]:
    """Extract key constraints from SpecGate JSON. v1.4: checks implementation_stack FIRST."""
    if not spec_json:
        return {}
    try:
        spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
    except Exception:
        return {}
    
    constraints = {
        "platform": None, "scope": None, "known_requirements": [],
        "discussed_stack": [], "implementation_stack": None,
        "stack_locked": False, "raw_goal": spec_data.get("goal", ""),
        "raw_summary": spec_data.get("summary", ""),
    }
    
    # v1.4: Check for EXPLICIT implementation_stack field FIRST
    impl_stack = spec_data.get("implementation_stack")
    if impl_stack and isinstance(impl_stack, dict):
        constraints["implementation_stack"] = impl_stack
        constraints["stack_locked"] = impl_stack.get("stack_locked", False)
        explicit_stack = []
        if impl_stack.get("language"):
            explicit_stack.append(impl_stack["language"])
        if impl_stack.get("framework"):
            lang = impl_stack.get("language", "")
            framework = impl_stack["framework"]
            explicit_stack.append(f"{lang}+{framework}" if lang else framework)
        if impl_stack.get("runtime"):
            explicit_stack.append(impl_stack["runtime"])
        if explicit_stack:
            constraints["discussed_stack"] = explicit_stack
            logger.info("[critique] v1.4 Using EXPLICIT implementation_stack: %s (locked=%s)",
                        explicit_stack, constraints["stack_locked"])
            print(f"[DEBUG] [critique] v1.4 EXPLICIT stack from spec: {explicit_stack} (locked={constraints['stack_locked']})")
    
    goal = spec_data.get("goal", "").lower()
    summary = spec_data.get("summary", "").lower()
    all_text = f"{goal} {summary}"
    
    if "desktop" in all_text:
        constraints["platform"] = "Desktop"
    elif "web" in all_text or "browser" in all_text:
        constraints["platform"] = "Web"
    elif "mobile" in all_text or "android" in all_text or "ios" in all_text:
        constraints["platform"] = "Mobile"
    
    scope_keywords = ["minimal", "minimum", "bare", "simple", "basic",
                      "playable", "prototype", "mvp", "first version"]
    for kw in scope_keywords:
        if kw in all_text:
            constraints["scope"] = "minimal"
            break
    
    if not constraints["discussed_stack"]:
        heuristic_stack = _detect_stack_from_text(all_text)
        if heuristic_stack:
            constraints["discussed_stack"] = heuristic_stack
            logger.info("[critique] v1.4 Using HEURISTIC stack detection: %s", heuristic_stack)
            print(f"[DEBUG] [critique] v1.4 HEURISTIC stack from text: {heuristic_stack}")
    
    known_reqs = spec_data.get("known_requirements", []) or spec_data.get("constraints_from_weaver", [])
    if isinstance(known_reqs, list):
        constraints["known_requirements"] = known_reqs
    
    return constraints


def build_platform_exclusion_zones(lines):
    """v1.10: Pre-scan document to identify platform exclusion zones.
    Returns set of line indices inside exclusion zones.
    """
    excluded_lines = set()
    in_exclusion_zone = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_any_header = bool(_SECTION_HEADER_RE.match(stripped))
        if is_any_header:
            is_exclusion_header = any(pat.match(stripped) for pat in _EXCLUSION_SECTION_HEADERS)
            if is_exclusion_header:
                in_exclusion_zone = True
                excluded_lines.add(i)
                continue
            else:
                in_exclusion_zone = False
        if in_exclusion_zone:
            excluded_lines.add(i)
    return excluded_lines


def _check_platform_targeted_not_excluded(
    arch_content: str,
    platform_keywords: List[str],
    context_window_lines: int = 3,
    arch_stack: Optional[List[str]] = None,
) -> bool:
    """v2.2: Structural platform detection with tech stack awareness.
    Five-layer detection: tech stack override, exclusion zones, inline patterns,
    context window, and structural signal validation.
    """
    if not arch_content:
        return False
    
    # Layer 0: Tech stack override
    if arch_stack:
        arch_stack_set = set(arch_stack)
        has_desktop_stack = bool(arch_stack_set & _DESKTOP_DEFINITIVE_STACKS)
        has_mobile_stack = bool(arch_stack_set & _MOBILE_DEFINITIVE_STACKS)
        if has_desktop_stack and not has_mobile_stack:
            print(f"[DEBUG] [critique] v2.2 Tech stack override: {arch_stack_set & _DESKTOP_DEFINITIVE_STACKS} "
                  f"is definitively Desktop — skipping mobile keyword scan")
            logger.info("[critique] v2.2 Tech stack override: %s is Desktop, not scanning for mobile keywords",
                        arch_stack_set & _DESKTOP_DEFINITIVE_STACKS)
            return False
    
    lines = arch_content.splitlines()
    exclusion_zones = build_platform_exclusion_zones(lines)
    
    structural_count = 0
    incidental_count = 0
    excluded_count = 0
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        has_keyword = any(kw in line_lower for kw in platform_keywords)
        if not has_keyword:
            continue
        
        if i in exclusion_zones:
            excluded_count += 1
            continue
        
        is_inline_excluded = any(pat.search(line) for pat in _INLINE_EXCLUSION_PATTERNS)
        if is_inline_excluded:
            excluded_count += 1
            continue
        
        start = max(0, i - context_window_lines)
        end = min(len(lines), i + context_window_lines + 1)
        context_block = " ".join(lines[start:end]).lower()
        is_context_excluded = any(pattern.search(context_block) for pattern in _EXCLUSION_CONTEXT_RE)
        if is_context_excluded:
            excluded_count += 1
            continue
        
        is_incidental = any(pat.search(line) for pat in _INCIDENTAL_PLATFORM_PATTERNS)
        if is_incidental:
            incidental_count += 1
            continue
        
        is_structural = any(pat.search(line) for pat in _STRUCTURAL_PLATFORM_PATTERNS)
        if is_structural:
            structural_count += 1
            logger.info("[critique] v2.2 Line %d STRUCTURAL platform signal: %s", i + 1, line.strip()[:120])
            continue
        
        incidental_count += 1
    
    total = structural_count + incidental_count + excluded_count
    if total == 0:
        return False
    
    if structural_count > 0:
        print(f"[DEBUG] [critique] v2.2 Platform detection: {structural_count} structural, "
              f"{incidental_count} incidental, {excluded_count} excluded mentions")
        return True
    
    print(f"[DEBUG] [critique] v2.2 Platform keywords found but NO structural signals "
          f"({incidental_count} incidental, {excluded_count} excluded) — NOT a platform mismatch")
    return False


def run_deterministic_spec_compliance_check(
    arch_content: str,
    spec_json: Optional[str] = None,
    original_request: str = "",
) -> List[CritiqueIssue]:
    """v1.3 CRITICAL FIX: Deterministic spec-compliance check.
    v1.4: Uses implementation_stack.stack_locked for stricter enforcement.
    
    Catches stack mismatch, scope inflation, platform mismatch BEFORE LLM critique.
    """
    issues: List[CritiqueIssue] = []
    issue_counter = 0
    
    if not arch_content:
        return issues
    
    arch_lower = arch_content.lower()
    constraints = _extract_spec_constraints(spec_json)
    spec_platform = constraints.get("platform")
    spec_scope = constraints.get("scope")
    spec_stack = constraints.get("discussed_stack", [])
    stack_locked = constraints.get("stack_locked", False)
    impl_stack = constraints.get("implementation_stack")
    
    if not stack_locked:
        request_stack = _detect_stack_from_text(original_request)
        all_discussed_stack = list(set(spec_stack + request_stack))
    else:
        all_discussed_stack = spec_stack
    
    arch_stack = _detect_stack_from_text(arch_content)
    
    logger.info("[critique] v1.4 Deterministic check: spec_platform=%s, spec_scope=%s, "
                "discussed_stack=%s, arch_stack=%s, stack_locked=%s",
                spec_platform, spec_scope, all_discussed_stack, arch_stack, stack_locked)
    print(f"[DEBUG] [critique] v1.4 Deterministic spec check:")
    print(f"[DEBUG] [critique]   spec_platform={spec_platform}")
    print(f"[DEBUG] [critique]   spec_scope={spec_scope}")
    print(f"[DEBUG] [critique]   discussed_stack={all_discussed_stack}")
    print(f"[DEBUG] [critique]   arch_stack={arch_stack}")
    print(f"[DEBUG] [critique]   stack_locked={stack_locked}")
    
    # Check 1: STACK MISMATCH
    python_discussed = any("Python" in s for s in all_discussed_stack)
    ts_js_in_arch = any(
        s for s in arch_stack
        if "TypeScript" in s or "JavaScript" in s or "Node" in s or "Electron" in s or "React" in s
    )
    
    if python_discussed and ts_js_in_arch and not any("Python" in s for s in arch_stack):
        issue_counter += 1
        lock_indicator = " [LOCKED]" if stack_locked else ""
        issues.append(CritiqueIssue(
            id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
            spec_ref=f"Discussed stack: Python-based implementation{lock_indicator}",
            arch_ref="Architecture proposes: JavaScript/TypeScript stack",
            category="stack_mismatch", severity="blocking",
            description=(
                f"STACK MISMATCH: The user {'EXPLICITLY CONFIRMED' if stack_locked else 'discussed'} a Python-based implementation "
                f"(detected: {[s for s in all_discussed_stack if 'Python' in s]}), "
                f"but the architecture proposes a JavaScript/TypeScript stack "
                f"(detected: {[s for s in arch_stack if 'TypeScript' in s or 'JavaScript' in s or 'Electron' in s or 'React' in s]}). "
                f"{'This stack choice was LOCKED by user confirmation and CANNOT be changed.' if stack_locked else 'Architecture must use the stack discussed with the user unless explicitly changed.'}"
            ),
            fix_suggestion="Rewrite architecture to use Python + the libraries discussed with the user.",
        ))
    
    pygame_discussed = any("Pygame" in s for s in all_discussed_stack)
    electron_in_arch = "electron" in arch_lower
    if pygame_discussed and electron_in_arch:
        issue_counter += 1
        lock_indicator = " [LOCKED]" if stack_locked else ""
        issues.append(CritiqueIssue(
            id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
            spec_ref=f"Discussed stack: Python + Pygame{lock_indicator}",
            arch_ref="Architecture proposes: Electron framework",
            category="stack_mismatch", severity="blocking",
            description=f"STACK MISMATCH: User {'EXPLICITLY CONFIRMED' if stack_locked else 'discussed'} Python + Pygame but architecture proposes Electron.",
            fix_suggestion="Rewrite architecture to use Python + Pygame as discussed.",
        ))
    
    # v1.4: Explicit stack_locked violation
    if stack_locked and impl_stack:
        locked_language = (impl_stack.get("language") or "").lower()
        locked_framework = (impl_stack.get("framework") or "").lower()
        
        if locked_language:
            arch_uses_locked_language = any(locked_language in s.lower() for s in arch_stack)
            if not arch_uses_locked_language and arch_stack:
                issue_counter += 1
                issues.append(CritiqueIssue(
                    id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
                    spec_ref=f"LOCKED implementation_stack.language: {impl_stack.get('language')}",
                    arch_ref=f"Architecture uses: {arch_stack}",
                    category="stack_mismatch", severity="blocking",
                    description=f"LOCKED STACK VIOLATION: Spec requires '{impl_stack.get('language')}' (stack_locked=True), but architecture proposes: {arch_stack}.",
                    fix_suggestion=f"Rewrite architecture to use {impl_stack.get('language')} as explicitly required.",
                ))
        
        if locked_framework:
            arch_uses_locked_framework = any(locked_framework in s.lower() for s in arch_stack) or locked_framework in arch_lower
            if not arch_uses_locked_framework and arch_stack:
                issue_counter += 1
                issues.append(CritiqueIssue(
                    id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
                    spec_ref=f"LOCKED implementation_stack.framework: {impl_stack.get('framework')}",
                    arch_ref=f"Architecture content does not include: {impl_stack.get('framework')}",
                    category="stack_mismatch", severity="blocking",
                    description=f"LOCKED FRAMEWORK VIOLATION: Spec requires '{impl_stack.get('framework')}' (stack_locked=True) but not found in architecture.",
                    fix_suggestion=f"Rewrite architecture to use {impl_stack.get('framework')} as explicitly required.",
                ))
    
    # Check 2: SCOPE INFLATION
    if spec_scope == "minimal":
        inflation_found = [kw for kw in _SCOPE_INFLATION_KEYWORDS if kw in arch_lower]
        if len(inflation_found) >= 3:
            issue_counter += 1
            issues.append(CritiqueIssue(
                id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
                spec_ref="Scope: minimal / bare minimum playable",
                arch_ref=f"Architecture includes: {inflation_found[:5]}",
                category="scope_inflation", severity="blocking",
                description=f"SCOPE INFLATION: Spec requires 'minimal' but architecture includes: {inflation_found[:5]}.",
                fix_suggestion="Remove scope-inflating features. Focus on core functionality only.",
            ))
    
    # Check 3: PLATFORM CONTEXT
    if spec_platform == "Desktop":
        mobile_is_targeted = _check_platform_targeted_not_excluded(
            arch_content=arch_content,
            platform_keywords=["mobile", "android", "ios"],
            arch_stack=arch_stack,
        )
        if mobile_is_targeted:
            issue_counter += 1
            issues.append(CritiqueIssue(
                id=f"SPEC-COMPLIANCE-{issue_counter:03d}",
                spec_ref=f"Platform: {spec_platform}",
                arch_ref="Architecture targets: Mobile platform",
                category="platform_mismatch", severity="blocking",
                description=f"PLATFORM MISMATCH: Spec requires Desktop but architecture targets Mobile.",
                fix_suggestion="Rewrite architecture to target Desktop platform as specified.",
            ))
    
    if issues:
        print(f"[DEBUG] [critique] v1.3 Deterministic check found {len(issues)} BLOCKING issue(s)")
        logger.warning("[critique] v1.3 Deterministic spec-compliance check: %d blocking issues", len(issues))
    else:
        print(f"[DEBUG] [critique] v1.3 Deterministic check: No obvious spec violations")
        logger.info("[critique] v1.3 Deterministic spec-compliance check: PASSED")
    
    return issues

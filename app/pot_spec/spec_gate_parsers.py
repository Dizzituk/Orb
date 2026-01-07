# FILE: app/pot_spec/spec_gate_parsers.py
"""
Spec Gate v2 - Parsing and Extraction

Contains:
- Text parsing helpers
- Filename extraction
- Output/step/acceptance coercion
- User clarification parsing
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .spec_gate_types import is_placeholder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_COMMAND_RE = re.compile(r"(?im)^\s*astra,\s*command\s*:\s*.*$")
_SECTION_RE = re.compile(
    r"(?im)^\s*(outputs?|steps?|verify|verification|acceptance|evidence)\s*:\s*$"
)
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|(\d+)[.)])\s+(.*\S)\s*$")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def strip_astra_command(text: str) -> str:
    """Remove Astra command prefix from text."""
    if not text:
        return ""
    return _COMMAND_RE.sub("", text).strip()


def split_sections(text: str) -> Dict[str, List[str]]:
    """Split text into outputs/steps/verify sections."""
    out: Dict[str, List[str]] = {"outputs": [], "steps": [], "verify": []}
    if not text:
        return out

    current = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        m = _SECTION_RE.match(line)
        if m:
            head = m.group(1).lower()
            if head.startswith("output"):
                current = "outputs"
            elif head.startswith("step"):
                current = "steps"
            else:
                current = "verify"
            continue

        if current is None:
            continue
        out[current].append(line)

    return out


def extract_bullets(lines: List[str]) -> List[str]:
    """Extract bullet points from lines."""
    items: List[str] = []
    for line in lines or []:
        m = _BULLET_RE.match(line)
        if m:
            items.append(m.group(2).strip())
        else:
            items.append(line.strip())
    seen = set()
    deduped: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


# ---------------------------------------------------------------------------
# Filename extraction
# ---------------------------------------------------------------------------

def extract_filename_from_text(text: str) -> Optional[str]:
    """Extract actual filename from natural language description.
    
    Examples:
        "create message.txt" → "message.txt"
        "file called hello.txt" → "hello.txt"
    """
    patterns = [
        r"['\"]([^'\"]+\.\w{1,5})['\"]",
        r"\((?:like|e\.g\.?|such as)\s*([^\)]+\.\w{1,5})\)",
        r"(?:file|document)\s+(?:called|named)\s+['\"]?([^\s'\"]+\.\w{1,5})['\"]?",
        r"(?:create|write|overwrite)\s+['\"]?([^\s'\"]+\.\w{1,5})['\"]?",
        r"\b([a-zA-Z_][a-zA-Z0-9_-]*\.(?:txt|md|json|py|js|html|css|yaml|yml|xml|csv))\b",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Weaver spec extraction
# ---------------------------------------------------------------------------

def extract_weaver_spec(constraints_hint: Optional[dict]) -> Tuple[Optional[dict], Dict[str, Any]]:
    """Extract Weaver spec and provenance from constraints hint."""
    if not constraints_hint or not isinstance(constraints_hint, dict):
        return None, {}
    weaver_spec = constraints_hint.get("weaver_spec_json")
    prov: Dict[str, Any] = {}
    for k in ("weaver_spec_id", "weaver_spec_hash", "weaver_spec_version", "weaver_model", "weaver_provider"):
        if k in constraints_hint and constraints_hint.get(k) is not None:
            prov[k] = constraints_hint.get(k)
    return (weaver_spec if isinstance(weaver_spec, dict) else None), prov


def best_effort_title_and_objective(weaver_spec: Optional[dict], user_text: str) -> Tuple[str, str]:
    """Extract title and objective from Weaver spec or user text."""
    title = ""
    objective = ""
    if isinstance(weaver_spec, dict):
        title = str(weaver_spec.get("title") or weaver_spec.get("name") or "").strip()
        objective = str(weaver_spec.get("objective") or weaver_spec.get("summary") or "").strip()

    if not title:
        for line in (user_text or "").splitlines():
            t = line.strip()
            if t:
                title = t[:80]
                break
    if not title:
        title = "Untitled Spec"

    if not objective:
        objective = (user_text or "").strip()
        if objective:
            objective = objective[:400]
        else:
            objective = "Define and validate the job spec so the critical pipeline can execute safely."

    return title, objective


# ---------------------------------------------------------------------------
# Coercion functions
# ---------------------------------------------------------------------------

def coerce_output_items(
    x: Any, 
    content_verbatim: Optional[str] = None, 
    location: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Coerce outputs to structured format.
    
    IMPORTANT: Does NOT generate placeholder filenames like output_1.txt.
    Only creates outputs when we can extract a real filename.
    """
    items: List[Dict[str, Any]] = []
    if not x:
        return items

    if isinstance(x, str):
        x = [x]
    if isinstance(x, dict):
        x = [x]
    if not isinstance(x, list):
        return items

    for it in x:
        if isinstance(it, str):
            s = it.strip()
            if not s or is_placeholder(s):
                continue
            
            filename = extract_filename_from_text(s)
            if not filename:
                logger.debug("[spec_gate_parsers] Skipping output with no extractable filename: %s", s[:50])
                continue
            
            s_lower = s.lower()
            action = "modify" if any(k in s_lower for k in ("overwrite", "modify", "update")) else "add"
            
            items.append({
                "type": "file",
                "name": filename,
                "path": location or "",
                "content": content_verbatim or "",
                "action": action,
                "must_exist": "existing" in s_lower or "locate" in s_lower,
                "description": s,
            })
        elif isinstance(it, dict):
            name = str(it.get("name") or it.get("file") or it.get("filename") or it.get("artifact") or "").strip()
            path = str(it.get("path") or it.get("location") or location or "").strip()
            content = str(it.get("content") or content_verbatim or "").strip()
            notes = str(it.get("notes") or it.get("desc") or it.get("description") or "").strip()
            action = str(it.get("action") or "add").strip()
            must_exist = bool(it.get("must_exist", False))
            
            if name and len(name) > 50:
                extracted = extract_filename_from_text(name)
                if extracted:
                    notes = name
                    name = extracted
            
            if not name and notes:
                name = extract_filename_from_text(notes)
            if not name:
                logger.debug("[spec_gate_parsers] Skipping output dict with no extractable name")
                continue
            
            if not is_placeholder(name):
                items.append({
                    "type": "file",
                    "name": name,
                    "path": path,
                    "content": content,
                    "action": action,
                    "must_exist": must_exist,
                    "description": notes,
                })
    return items


def coerce_step_items(x: Any) -> List[str]:
    """Coerce steps to list of strings."""
    if not x:
        return []
    if isinstance(x, str):
        x = [x]
    if isinstance(x, dict):
        if "steps" in x and isinstance(x["steps"], list):
            x = x["steps"]
        else:
            x = [json.dumps(x, ensure_ascii=False)]
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for it in x:
        if isinstance(it, str):
            s = it.strip()
            if s and not is_placeholder(s):
                out.append(s)
        elif isinstance(it, dict):
            s = str(it.get("text") or it.get("step") or it.get("description") or "").strip()
            if not s:
                s = json.dumps(it, ensure_ascii=False)
            if not is_placeholder(s):
                out.append(s)
    seen = set()
    return [s for s in out if not (s in seen or seen.add(s))]


def coerce_acceptance_items(x: Any) -> List[str]:
    """Coerce acceptance criteria to list of strings."""
    if not x:
        return []
    if isinstance(x, str):
        x = [x]
    if isinstance(x, dict):
        if "acceptance_criteria" in x and isinstance(x["acceptance_criteria"], list):
            x = x["acceptance_criteria"]
        else:
            x = [json.dumps(x, ensure_ascii=False)]
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for it in x:
        if isinstance(it, str):
            s = it.strip()
            if s and not is_placeholder(s):
                out.append(s)
        elif isinstance(it, dict):
            s = str(it.get("text") or it.get("criterion") or "").strip()
            if not s:
                s = json.dumps(it, ensure_ascii=False)
            if not is_placeholder(s):
                out.append(s)
    seen = set()
    return [s for s in out if not (s in seen or seen.add(s))]


# ---------------------------------------------------------------------------
# Natural language step extraction (v2.2)
# ---------------------------------------------------------------------------

def _extract_steps_from_natural_language(text: str) -> List[str]:
    """Extract steps from natural language - handles conversational input."""
    steps = []
    text_lower = text.lower()
    
    action_verbs = [
        "find", "locate", "search", "look for", "discover", "work out",
        "go to", "go into", "navigate", "enter", "open", "access", "inside",
        "create", "make", "write", "add", "generate", "overwrite",
        "save", "store", "persist",
        "verify", "check", "confirm", "validate", "ensure",
        "mark", "complete", "finish", "done",
        "read", "load", "get", "fetch",
        "delete", "remove", "clear",
        "update", "modify", "edit", "change",
        "execute", "run", "perform",
    ]
    
    # Skip if not enough action verbs
    verb_count = sum(1 for verb in action_verbs if verb in text_lower)
    if verb_count < 2:
        return []
    
    # Commentary to skip
    skip_phrases = ["i've kind of", "you know", "i mean", "by saying", "but it'll", "anyway", "given it away"]
    
    # Strategy 1: Check for numbered list (1. or 1) or 1 followed by text)
    # Pattern: start of line, optional whitespace, digit(s), optional dot/paren, space, then text
    numbered_pattern = r'(?:^|\n)\s*(\d+)[.\):]?\s+(.+?)(?=\n\s*\d+[.\):]?\s+|\n*$)'
    numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE | re.DOTALL)
    
    if len(numbered_matches) >= 2:
        for num, step_text in numbered_matches:
            step_text = step_text.strip().rstrip('.')
            if step_text and len(step_text) > 3:
                steps.append(step_text[0].upper() + step_text[1:] if len(step_text) > 1 else step_text.upper())
        if steps:
            logger.info(f"[spec_gate_parsers] Extracted {len(steps)} steps from numbered list")
            return _dedupe_steps(steps)
    
    # Strategy 2: Try line-by-line first (for multi-line input)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) >= 3:
        for line in lines:
            line_lower = line.lower()
            if any(s in line_lower for s in skip_phrases):
                continue
            # Strip leading numbers like "1 ", "2.", etc.
            cleaned = re.sub(r'^\d+[.\):]?\s*', '', line).strip()
            if any(v in cleaned.lower() for v in action_verbs) and len(cleaned) > 10:
                cleaned = cleaned.rstrip('.')
                cleaned = re.sub(r'^(the\s+)?(system\s+)?(should\s+|needs?\s+to\s+)?', '', cleaned, flags=re.I).strip()
                if cleaned and len(cleaned) > 5:
                    steps.append(cleaned[0].upper() + cleaned[1:])
        if len(steps) >= 2:
            return _dedupe_steps(steps)
    
    # Strategy 3: Try sentence splitting, handling comma-separated actions
    normalized = text.replace('\n', ' ').strip()
    sentences = re.split(r'(?<=[.!?])\s+', normalized)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        sent_lower = sentence.lower()
        if any(s in sent_lower for s in skip_phrases):
            continue
        if not any(v in sent_lower for v in action_verbs):
            continue
        
        # Check for comma-separated actions (3+ commas with action verbs)
        if sentence.count(',') >= 2:
            parts = re.split(r',\s*', sentence)
            actions_count = sum(1 for p in parts if any(v in p.lower() for v in action_verbs))
            if actions_count >= 3:
                for part in parts:
                    part = part.strip()
                    if any(v in part.lower() for v in action_verbs) and 5 < len(part) < 150:
                        cleaned = part.rstrip('.')
                        cleaned = re.sub(r'^(and\s+|then\s+|also\s+|finally\s+)', '', cleaned, flags=re.I).strip()
                        if cleaned and len(cleaned) > 5:
                            steps.append(cleaned[0].upper() + cleaned[1:])
                continue
        
        # Single action sentence
        cleaned = sentence.rstrip('.')
        cleaned = re.sub(r'^the\s+steps.*?are\s+', '', cleaned, flags=re.I)
        cleaned = re.sub(r'^(the\s+)?(system\s+)?(should\s+|needs?\s+to\s+)?', '', cleaned, flags=re.I).strip()
        if cleaned and len(cleaned) > 5:
            steps.append(cleaned[0].upper() + cleaned[1:])
    
    return _dedupe_steps(steps)


def _dedupe_steps(steps: List[str]) -> List[str]:
    """Deduplicate steps while preserving order."""
    seen = set()
    return [s for s in steps if not (s.lower() in seen or seen.add(s.lower()))]


# ---------------------------------------------------------------------------
# User clarification parsing
# ---------------------------------------------------------------------------

def parse_user_clarification(user_text: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Parse user clarification text for outputs, steps, and acceptance criteria."""
    sections = split_sections(user_text)
    outputs = coerce_output_items(extract_bullets(sections.get("outputs", [])))
    steps = coerce_step_items(extract_bullets(sections.get("steps", [])))
    verify = coerce_acceptance_items(extract_bullets(sections.get("verify", [])))

    # First fallback: process bullet points
    if not (outputs or steps or verify):
        bullets = extract_bullets(user_text.splitlines())
        for b in bullets:
            bl = b.lower()
            if any(k in bl for k in ("verify", "check", "ensure", "must", "should", "expected")):
                verify.append(b)
            elif any(k in bl for k in ("file", "folder", "path", ".py", ".md", ".json", ".txt")):
                filename = extract_filename_from_text(b)
                if filename:
                    outputs.append({
                        "type": "file",
                        "name": filename,
                        "path": "",
                        "content": "",
                        "action": "add",
                        "must_exist": False,
                        "description": b,
                    })
            else:
                steps.append(b)

    # v2.2: Natural language step extraction when bullet parsing didn't find steps
    if not steps:
        steps = _extract_steps_from_natural_language(user_text)
        if steps:
            logger.info(f"[spec_gate_parsers] Extracted {len(steps)} steps from natural language")

    # Second fallback: parse plain paragraph text for file paths
    if not outputs:
        path_patterns = [
            r'(?:Sandbox\s+)?Desktop[\\\/]([^\s,;\.]+[\\\/])?([a-zA-Z0-9_-]+\.(?:txt|md|json|py|js|html|css|yaml|yml))',
            r'(?:file\s+(?:called|named)\s+)([a-zA-Z0-9_-]+\.(?:txt|md|json|py|js|html|css))',
            r'(?:create|write)\s+(?:a\s+)?(?:file\s+)?(?:called\s+)?([a-zA-Z0-9_-]+\.(?:txt|md|json|py|js|html|css))',
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, user_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    filename = match[-1] if match[-1] else match[0]
                    path_part = match[0] if len(match) > 1 and match[0] else ""
                else:
                    filename = match
                    path_part = ""
                
                if filename and not any(o.get("name") == filename for o in outputs):
                    full_path = ""
                    if "sandbox" in user_text.lower() and "desktop" in user_text.lower():
                        full_path = f"Sandbox Desktop\\{path_part}" if path_part else "Sandbox Desktop"
                    
                    outputs.append({
                        "type": "file",
                        "name": filename,
                        "path": full_path.rstrip("\\"),
                        "content": "",
                        "action": "add",
                        "must_exist": "existing" in user_text.lower(),
                        "description": user_text[:200],
                    })
                    break
            if outputs:
                break

    return outputs, steps, verify


# ---------------------------------------------------------------------------
# Output extraction from acceptance
# ---------------------------------------------------------------------------

def extract_outputs_from_acceptance(
    acceptance: List[str],
    content_verbatim: Optional[str] = None,
    location: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract output artifacts from acceptance criteria text."""
    outputs: List[Dict[str, Any]] = []
    
    for criterion in acceptance:
        if not criterion:
            continue
        
        # Windows-style paths
        path_match = re.search(r'([A-Za-z]:\\[^\s,;]+\.(?:txt|md|json|py|js|html|css|yaml|yml))', criterion)
        if path_match:
            full_path = path_match.group(1)
            filename = os.path.basename(full_path)
            outputs.append({
                "type": "file",
                "name": filename,
                "path": os.path.dirname(full_path),
                "content": content_verbatim or "",
                "action": "add",
                "must_exist": False,
                "description": criterion,
            })
            continue
        
        # Sandbox-style paths
        sandbox_match = re.search(
            r'(?:Sandbox\s+)?Desktop[\\\/]([^\s,;]+\.(?:txt|md|json|py|js|html|css))',
            criterion, re.IGNORECASE
        )
        if sandbox_match:
            rel_path = sandbox_match.group(1)
            filename = os.path.basename(rel_path)
            dirname = os.path.dirname(rel_path)
            outputs.append({
                "type": "file",
                "name": filename,
                "path": f"Sandbox Desktop\\{dirname}" if dirname else "Sandbox Desktop",
                "content": content_verbatim or "",
                "action": "add",
                "must_exist": False,
                "description": criterion,
            })
            continue
        
        # Generic filename
        filename = extract_filename_from_text(criterion)
        if filename:
            outputs.append({
                "type": "file",
                "name": filename,
                "path": location or "",
                "content": content_verbatim or "",
                "action": "add",
                "must_exist": "existing" in criterion.lower() or "locate" in criterion.lower(),
                "description": criterion,
            })
    
    return outputs


__all__ = [
    "strip_astra_command",
    "split_sections",
    "extract_bullets",
    "extract_filename_from_text",
    "extract_weaver_spec",
    "best_effort_title_and_objective",
    "coerce_output_items",
    "coerce_step_items",
    "coerce_acceptance_items",
    "parse_user_clarification",
    "extract_outputs_from_acceptance",
]
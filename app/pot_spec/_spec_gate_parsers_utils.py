from __future__ import annotations
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


_COMMAND_RE = re.compile(r"(?im)^\s*astra,\s*command\s*:\s*.*$")

_SECTION_RE = re.compile(
    r"(?im)^\s*(outputs?|steps?|verify|verification|acceptance|evidence)\s*:\s*$"
)

_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|(\d+)[.)])\s+(.*\S)\s*$")

def strip_astra_command(text: str) -> str:
    """Remove Astra command prefix from text."""
    if not text:
        return ""
    return _COMMAND_RE.sub("", text).strip()

def extract_weaver_spec(constraints_hint: Optional[dict]) -> Tuple[Optional[dict], Dict[str, Any]]:
    """Extract Weaver spec and provenance from constraints hint.
    
    v3.0: Now handles both:
      - weaver_job_description_text: Plain text from simple Weaver (v3.0)
      - weaver_spec_json: JSON spec from v2.x Weaver
    
    Returns:
        (weaver_spec_dict, provenance_dict)
    """
    if not constraints_hint or not isinstance(constraints_hint, dict):
        return None, {}
    
    prov: Dict[str, Any] = {}
    
    # Collect provenance keys
    for k in ("weaver_spec_id", "weaver_spec_hash", "weaver_spec_version", 
              "weaver_model", "weaver_provider", "weaver_source"):
        if k in constraints_hint and constraints_hint.get(k) is not None:
            prov[k] = constraints_hint.get(k)
    
    # v3.0: Check for simple Weaver job description text first
    job_desc_text = constraints_hint.get("weaver_job_description_text")
    if job_desc_text and isinstance(job_desc_text, str) and job_desc_text.strip():
        logger.info("[spec_gate_parsers] Using v3.0 weaver_job_description_text (%d chars)", len(job_desc_text))
        prov["weaver_source"] = prov.get("weaver_source") or "weaver_simple"
        # Wrap in dict format for downstream compatibility
        return {
            "job_description": job_desc_text.strip(),
            "source": "weaver_simple",
            "title": "Job Description from Weaver",
        }, prov
    
    # v2.x: Fall back to weaver_spec_json
    weaver_spec = constraints_hint.get("weaver_spec_json")
    if weaver_spec and isinstance(weaver_spec, dict):
        return weaver_spec, prov
    
    return None, prov

def best_effort_title_and_objective(weaver_spec: Optional[dict], user_text: str) -> Tuple[str, str]:
    """Extract title and objective from Weaver spec or user text.
    
    v3.0: Now also checks for 'job_description' field (simple Weaver output).
    """
    title = ""
    objective = ""
    if isinstance(weaver_spec, dict):
        title = str(weaver_spec.get("title") or weaver_spec.get("name") or "").strip()
        # v3.0: Check job_description before falling back to objective/summary
        objective = str(
            weaver_spec.get("job_description") or 
            weaver_spec.get("objective") or 
            weaver_spec.get("summary") or 
            ""
        ).strip()

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

def _dedupe_steps(steps: List[str]) -> List[str]:
    """Deduplicate steps while preserving order."""
    seen = set()
    return [s for s in steps if not (s.lower() in seen or seen.add(s.lower()))]

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

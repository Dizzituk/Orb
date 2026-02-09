# FILE: app/llm/pipeline/critique_parts/grounding_validation.py
"""Block 5b: SPEC-REF GROUNDING VALIDATION (v1.7 - HALLUCINATION DEFENSE)"""

import json
import logging
import re
from typing import List, Optional, Tuple

from app.llm.pipeline.critique_schemas import CritiqueIssue

logger = logging.getLogger(__name__)


def validate_spec_ref_grounding(
    issues: List[CritiqueIssue],
    spec_markdown: Optional[str] = None,
    spec_json: Optional[str] = None,
) -> Tuple[List[CritiqueIssue], List[CritiqueIssue]]:
    """
    v1.7: Validate that blocking issues cite constraints that ACTUALLY EXIST in the spec.
    
    Catches LLM hallucinations like:
    - Citing "cloud_services: false" when spec has no such field
    - Citing "local_only" constraint when spec doesn't mention it
    
    Returns:
        (validated_blocking, downgraded_to_non_blocking)
    """
    if not issues:
        return [], []
    
    spec_corpus = ""
    if spec_markdown:
        spec_corpus += spec_markdown.lower()
    if spec_json:
        try:
            if isinstance(spec_json, str):
                spec_corpus += " " + spec_json.lower()
            else:
                spec_corpus += " " + json.dumps(spec_json).lower()
        except Exception:
            pass
    
    if not spec_corpus.strip():
        print("[DEBUG] [critique] v1.7 No spec corpus available for grounding validation")
        return issues, []
    
    validated: List[CritiqueIssue] = []
    downgraded: List[CritiqueIssue] = []
    
    FABRICATION_PATTERNS = [
        r'cloud_services\s*[:=]\s*(false|true|no|yes)',
        r'local[_\s-]only\s*[:=]\s*(true|yes|required)',
        r'requires?[_\s-]local',
        r'no[_\s-]external[_\s-]apis?',
        r'offline[_\s-]only',
    ]
    
    for issue in issues:
        spec_ref = (getattr(issue, 'spec_ref', '') or '').lower()
        description = (getattr(issue, 'description', '') or '').lower()
        combined_text = f"{spec_ref} {description}"
        
        is_fabricated = False
        fabrication_reason = ""
        
        for pattern in FABRICATION_PATTERNS:
            match = re.search(pattern, combined_text)
            if match:
                cited_text = match.group(0)
                if cited_text not in spec_corpus:
                    is_fabricated = True
                    fabrication_reason = f"Cited '{cited_text}' not found in spec"
                    break
        
        if not is_fabricated and spec_ref:
            field_refs = re.findall(r'constraints?\.([a-z_]+)', spec_ref)
            field_refs += re.findall(r"'([a-z_]+)'", spec_ref)
            field_refs += re.findall(r'"([a-z_]+)"', spec_ref)
            
            for field in field_refs:
                if field in ('integrations', 'platform', 'scope', 'goal', 'summary',
                             'stack', 'language', 'framework', 'requirements'):
                    continue
                if field not in spec_corpus:
                    is_fabricated = True
                    fabrication_reason = f"Referenced field '{field}' not found in spec"
                    break
        
        if is_fabricated:
            issue.severity = "non_blocking"
            downgraded.append(issue)
            print(f"[DEBUG] [critique] v1.7 GROUNDING FAIL: Issue {getattr(issue, 'id', 'N/A')} "
                  f"downgraded - {fabrication_reason}")
            logger.info("[critique] v1.7 Grounding validation failed for %s: %s",
                        getattr(issue, 'id', 'N/A'), fabrication_reason)
        else:
            validated.append(issue)
    
    if downgraded:
        print(f"[DEBUG] [critique] v1.7 Grounding validation: {len(validated)} kept, "
              f"{len(downgraded)} downgraded (hallucinated constraints)")
        logger.info("[critique] v1.7 Grounding validation: %d kept, %d downgraded",
                    len(validated), len(downgraded))
    
    return validated, downgraded

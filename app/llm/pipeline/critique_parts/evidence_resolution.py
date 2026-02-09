# FILE: app/llm/pipeline/critique_parts/evidence_resolution.py
"""Block 5d: EVIDENCE RESOLUTION CHECK (v2.0 - Evidence-or-Request Contract)"""

import logging
from typing import List, Optional

import yaml

from app.llm.pipeline.critique_schemas import CritiqueIssue

logger = logging.getLogger(__name__)

VALID_RESOLUTIONS = {"CITED", "DECISION", "HUMAN_REQUIRED"}


def extract_critical_claims(arch_content: str) -> Optional[list]:
    """Extract CRITICAL_CLAIMS register from architecture output.

    CRITICAL_CLAIMS must be the last structured block.
    Parses from rfind() for robustness against nested YAML,
    blank lines, and field reordering.
    """
    idx = arch_content.rfind("\nCRITICAL_CLAIMS:")
    if idx == -1:
        if arch_content.startswith("CRITICAL_CLAIMS:"):
            idx = 0
        else:
            return None

    yaml_text = arch_content[idx:].strip()

    try:
        parsed = yaml.safe_load(yaml_text)
        if not isinstance(parsed, dict):
            return None
        claims = parsed.get("CRITICAL_CLAIMS")
        if claims is None:
            return None
        if not isinstance(claims, list):
            return None
        return claims
    except yaml.YAMLError:
        return None


def run_evidence_resolution_check(
    arch_content: str,
) -> List[CritiqueIssue]:
    """
    v2.0: Deterministic check — validate CRITICAL_CLAIMS register.

    Rules:
    - Every claim must have resolution in {CITED, DECISION, HUMAN_REQUIRED}
    - CITED claims must have at least one evidence entry
    - DECISION claims must reference a decision_id
    - Missing register entirely = non-blocking warning
    """
    issues: List[CritiqueIssue] = []

    claims = extract_critical_claims(arch_content)

    if claims is None:
        issues.append(CritiqueIssue(
            id="CLAIMS-MISSING",
            spec_ref="Evidence-or-Request Contract",
            arch_ref="Architecture output",
            category="missing_claims_register",
            severity="non_blocking",
            description="No CRITICAL_CLAIMS register found in architecture output",
            fix_suggestion="Add CRITICAL_CLAIMS block as the last section listing all implementation-affecting claims with resolution status",
        ))
        return issues

    for claim in claims:
        claim_id = claim.get("id", "UNKNOWN")
        resolution = claim.get("resolution", "MISSING")
        claim_text = claim.get("claim", "")

        if resolution not in VALID_RESOLUTIONS:
            issues.append(CritiqueIssue(
                id=f"CLAIMS-{claim_id}",
                spec_ref="Evidence-or-Request Contract",
                arch_ref=f"CRITICAL_CLAIMS.{claim_id}: {claim_text[:80]}",
                category="unresolved_critical",
                severity="blocking",
                description=f"Critical claim '{claim_id}' has invalid resolution: '{resolution}'. Must be CITED, DECISION, or HUMAN_REQUIRED.",
                fix_suggestion="Resolve this claim with evidence, an explicit decision, or flag for human input",
            ))
            continue

        if resolution == "CITED":
            evidence = claim.get("evidence", [])
            if not evidence:
                issues.append(CritiqueIssue(
                    id=f"CLAIMS-{claim_id}",
                    spec_ref="Evidence-or-Request Contract",
                    arch_ref=f"CRITICAL_CLAIMS.{claim_id}: {claim_text[:80]}",
                    category="unresolved_critical",
                    severity="blocking",
                    description=f"Critical claim '{claim_id}' marked CITED but has no evidence entries",
                    fix_suggestion="Add evidence entries with file paths and line ranges",
                ))

        elif resolution == "DECISION":
            decision_id = claim.get("decision_id")
            if not decision_id:
                issues.append(CritiqueIssue(
                    id=f"CLAIMS-{claim_id}",
                    spec_ref="Evidence-or-Request Contract",
                    arch_ref=f"CRITICAL_CLAIMS.{claim_id}: {claim_text[:80]}",
                    category="unresolved_critical",
                    severity="blocking",
                    description=f"Critical claim '{claim_id}' marked DECISION but missing decision_id reference",
                    fix_suggestion="Add decision_id referencing a DECISION block with rationale and revisit_if",
                ))

    return issues

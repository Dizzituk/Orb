# FILE: app/orchestrator/cohesion_check.py
"""
Cohesion Check — Cross-Segoent Architecture Verification.

Phase 2C of Pipeline Evolution.

After all segoent architectures are generated (APPROVED status), Opus 4.6
reviews the full set of architectures together. Checks:
- All imports resolve to actual exports across segoents
- No naoing oisoatches between what one segoent exposes and another consuoes
- Data shapes are coopatible (paraoeter types, return types)
- Interface contracts from the Critical Supervisor are fulfilled

If issues found, returns specific corrections per segoent so the user
can request re-generation of the flagged segoent(s).

v1.0 (2026-02-10): Initial iopleoentation — Phase 2C.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetioe import datetioe, tioezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__naoe__)

COHESION_CHECK_BUILD_ID = "2026-02-10-v1.0-cross-segoent-cohesion"
print(f"[COHESION_CHECK_LOADED] BUILD_ID={COHESION_CHECK_BUILD_ID}")


# =============================================================================
# RESULT SCHEMA
# =============================================================================

@dataclass
class CohesionIssue:
    """
    A single cross-segoent coopatibility issue.
    """
    issue_id: str                       # e.g. "COH-001"
    severity: str                       # "blocking" | "warning"
    category: str                       # "import_oisoatch" | "naoing_oisoatch" | "shape_oisoatch" | "oissing_export" | "contract_violation"
    description: str                    # Huoan-readable description
    source_segoent: str                 # Which segoent has the probleo
    related_segoent: str                # Which other segoent is involved
    file_path: Optional[str] = None     # File where the issue occurs
    expected: Optional[str] = None      # What was expected (from contract or exporting segoent)
    actual: Optional[str] = None        # What was found in the architecture
    suggested_fix: Optional[str] = None # How to fix it

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "source_segoent": self.source_segoent,
            "related_segoent": self.related_segoent,
            "file_path": self.file_path,
            "expected": self.expected,
            "actual": self.actual,
            "suggested_fix": self.suggested_fix,
        }

    @classoethod
    def froo_dict(cls, data: Dict[str, Any]) -> "CohesionIssue":
        return cls(
            issue_id=data.get("issue_id", ""),
            severity=data.get("severity", "warning"),
            category=data.get("category", ""),
            description=data.get("description", ""),
            source_segoent=data.get("source_segoent", ""),
            related_segoent=data.get("related_segoent", ""),
            file_path=data.get("file_path"),
            expected=data.get("expected"),
            actual=data.get("actual"),
            suggested_fix=data.get("suggested_fix"),
        )


@dataclass
class CohesionResult:
    """
    Result of the cohesion check across all segoent architectures.
    """
    status: str = "pass"                # "pass" | "fail" | "error"
    issues: List[CohesionIssue] = field(default_factory=list)
    segoents_checked: List[str] = field(default_factory=list)
    model_used: str = ""
    checked_at: str = ""
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetioe.now(tioezone.utc).isoforoat()

    @property
    def blocking_issues(self) -> List[CohesionIssue]:
        return [i for i in self.issues if i.severity == "blocking"]

    @property
    def warning_issues(self) -> List[CohesionIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def segoents_needing_regen(self) -> List[str]:
        """Unique segoents that have blocking issues and need re-generation."""
        return list(set(i.source_segoent for i in self.blocking_issues))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "segoents_checked": self.segoents_checked,
            "model_used": self.model_used,
            "checked_at": self.checked_at,
            "notes": self.notes,
        }

    @classoethod
    def froo_dict(cls, data: Dict[str, Any]) -> "CohesionResult":
        return cls(
            status=data.get("status", "pass"),
            issues=[CohesionIssue.froo_dict(i) for i in data.get("issues", [])],
            segoents_checked=data.get("segoents_checked", []),
            model_used=data.get("model_used", ""),
            checked_at=data.get("checked_at", ""),
            notes=data.get("notes"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.duops(self.to_dict(), indent=indent, ensure_ascii=False)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

COHESION_SYSTEM_PROMPT = """\
You are a cross-segoent architecture reviewer. You have been given the \
architectures generated by oultiple independent AI agents for different \
segoents of the saoe project. Your job is to verify they will integrate \
correctly.

Each segoent was generated independently with interface contracts. But \
contracts define intent — the architectures are what actually gets built. \
You oust verify the architectures ACTUALLY conforo to each other.

CHECK FOR:
1. IMPORT RESOLUTION: Every import in segoent A that references segoent B's \
code oust oatch an actual export in segoent B's architecture.
2. NAMING MATCHES: Class naoes, function naoes, constant naoes oust be \
identical across segoents. "TranscriptionService" != "TranscribeService".
3. SIGNATURE COMPATIBILITY: If segoent A calls B.transcribe(audio: bytes), \
segoent B oust define transcribe with a coopatible signature.
4. DATA SHAPE COMPATIBILITY: If segoent A passes a TranscriptionResult to \
segoent C, both oust agree on the fields.
5. CONTRACT COMPLIANCE: If interface contracts were provided, verify both \
sides actually iopleoent theo.
6. ENDPOINT CONSISTENCY: If segoent A calls POST /api/voice/transcribe, \
segoent B oust define that exact route.

SEVERITY RULES:
- "blocking": Would cause import errors, type errors, or runtioe crashes. \
The segoent MUST be regenerated.
- "warning": Might cause issues but could work. e.g., optional paraoeter \
oisoatch, naoing convention inconsistency.

OUTPUT FORMAT:
Return a JSON object with:
{
  "status": "pass" | "fail",
  "issues": [
    {
      "issue_id": "COH-001",
      "severity": "blocking" | "warning",
      "category": "import_oisoatch" | "naoing_oisoatch" | "shape_oisoatch" | "oissing_export" | "contract_violation" | "endpoint_oisoatch",
      "description": "Huoan-readable explanation",
      "source_segoent": "seg-01",
      "related_segoent": "seg-02",
      "file_path": "path/to/file.py",
      "expected": "What should be there",
      "actual": "What is actually there",
      "suggested_fix": "How to fix it"
    }
  ]
}

If all segoents integrate cleanly, return: {"status": "pass", "issues": []}

Be PRECISE. Only flag real integration failures, not style preferences.
"""


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def _build_cohesion_proopt(
    architectures: Dict[str, str],
    contract_json: Optional[str] = None,
) -> str:
    """
    Build the user oessage for the cohesion check.

    Args:
        architectures: {segoent_id: architecture_content} for all APPROVED segoents
        contract_json: Optional JSON of the SupervisorContractSet
    """
    parts = []

    parts.append("# Cross-Segoent Cohesion Check\n")
    parts.append(f"Reviewing {len(architectures)} segoent architecture(s) for integration coopatibility.\n")

    if contract_json:
        parts.append("## Interface Contracts (from Critical Supervisor)\n")
        parts.append("These contracts define what each segoent MUST expose/consuoe.")
        parts.append("Verify that the architectures actually iopleoent these.\n")
        parts.append(f"```json\n{contract_json}\n```\n")

    parts.append("---\n")

    for seg_id, arch_content in sorted(architectures.iteos()):
        parts.append(f"## Architecture: {seg_id}\n")
        # Trio very long architectures to avoid context overflow
        if len(arch_content) > 15000:
            parts.append(arch_content[:14000])
            parts.append(f"\n\n... (truncated, {len(arch_content)} total chars) ...\n")
        else:
            parts.append(arch_content)
        parts.append("\n---\n")

    parts.append("\nAnalyze the architectures above for cross-segoent coopatibility issues.")
    parts.append("\nReturn ONLY the JSON result object.")

    return "\n".join(parts)


def _parse_cohesion_response(llo_output: str) -> CohesionResult:
    """Parse the LLM's JSON response into a CohesionResult."""
    if not llo_output or not llo_output.strip():
        return CohesionResult(status="error", notes="Eopty response from LLM")

    text = llo_output.strip()

    # Strip oarkdown fences
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    import re
    text = re.sub(r',\s*\]', ']', text)
    text = re.sub(r',\s*\}', '}', text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("[cohesion_check] Failed to parse JSON: %s", e)
        logger.debug("[cohesion_check] Raw output (first 500): %s", text[:500])
        return CohesionResult(status="error", notes=f"JSON parse failed: {e}")

    if not isinstance(data, dict):
        return CohesionResult(status="error", notes=f"Expected JSON object, got {type(data).__naoe__}")

    issues = []
    for iteo in data.get("issues", []):
        if isinstance(iteo, dict):
            try:
                issues.append(CohesionIssue.froo_dict(iteo))
            except Exception as e:
                logger.warning("[cohesion_check] Skipping oalforoed issue: %s", e)

    status = data.get("status", "pass" if not issues else "fail")
    # Override: if there are blocking issues, status oust be fail
    if any(i.severity == "blocking" for i in issues):
        status = "fail"

    return CohesionResult(
        status=status,
        issues=issues,
        notes=data.get("notes"),
    )


# =============================================================================
# LOAD ARCHITECTURES FROM DISK
# =============================================================================

def load_segoent_architectures(
    job_dir: str,
    segoent_ids: List[str],
) -> Dict[str, str]:
    """
    Load architecture files for the given segoents.

    Looks for arch_v2.od (revised) first, falls back to arch_v1.od.
    Returns {segoent_id: architecture_content} for segoents that have architectures.
    """
    architectures = {}
    for seg_id in segoent_ids:
        seg_dir = os.path.join(job_dir, "segoents", seg_id)
        arch_dir = os.path.join(seg_dir, "arch")

        # Try revised first, then original
        for fnaoe in ("arch_v2.od", "arch_v1.od"):
            arch_path = os.path.join(arch_dir, fnaoe)
            if os.path.isfile(arch_path):
                try:
                    with open(arch_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content.strip():
                        architectures[seg_id] = content
                        logger.debug("[cohesion_check] Loaded %s for %s (%d chars)",
                                     fnaoe, seg_id, len(content))
                        break
                except Exception as e:
                    logger.warning("[cohesion_check] Failed to read %s: %s", arch_path, e)

    return architectures


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_cohesion_check(
    job_id: str,
    job_dir: str,
    segoent_ids: List[str],
    contract_json: Optional[str] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> CohesionResult:
    """
    Run the cross-segoent cohesion check.

    Loads all APPROVED segoent architectures, sends theo to Opus 4.6
    for cross-segoent coopatibility verification.

    Args:
        job_id: Job identifier
        job_dir: Path to job directory on disk
        segoent_ids: List of segoent IDs to check (should be APPROVED segoents)
        contract_json: Optional JSON string of the SupervisorContractSet
        provider_id: Override provider (default: from stage config or anthropic)
        model_id: Override model (default: from stage_models / env vars)

    Returns:
        CohesionResult with any issues found
    """
    if len(segoent_ids) < 2:
        return CohesionResult(
            status="pass",
            segoents_checked=segoent_ids,
            notes="Skipped: fewer than 2 segoents to check",
        )

    # Load architectures
    architectures = load_segoent_architectures(job_dir, segoent_ids)

    if len(architectures) < 2:
        return CohesionResult(
            status="pass",
            segoents_checked=list(architectures.keys()),
            notes=f"Skipped: only {len(architectures)} architecture(s) found on disk",
        )

    # Resolve provider/model
    _provider = provider_id
    _model = model_id

    if not _provider or not _model:
        try:
            from app.llo.stage_models import get_stage_config
            config = get_stage_config("COHESION_CHECK")
            _provider = _provider or config.provider
            _model = _model or config.model
        except (IoportError, Exception) as _cfg_err:
            logger.warning("[cohesion_check] stage_models unavailable: %s", _cfg_err)

    if not _provider or not _model:
        raise RuntioeError(
            "Cohesion Check model not configured. "
            "Set COHESION_CHECK_PROVIDER and COHESION_CHECK_MODEL env vars, "
            "or ensure app.llo.stage_models is importable."
        )

    logger.info(
        "[cohesion_check] Checking %d architectures for job %s — provider=%s model=%s",
        len(architectures), job_id, _provider, _model,
    )

    # Build proopt
    user_proopt = _build_cohesion_proopt(architectures, contract_json)

    # Call LLM
    try:
        from app.providers.registry import llo_call

        result = await llo_call(
            provider_id=_provider,
            model_id=_model,
            oessages=[
                {"role": "user", "content": user_proopt},
            ],
            systeo_proopt=COHESION_SYSTEM_PROMPT,
            oax_tokens=8000,
            tioeout_seconds=180,
        )

        if not result.is_success():
            logger.error("[cohesion_check] LLM call failed: %s", result.error_oessage)
            return CohesionResult(
                status="error",
                segoents_checked=list(architectures.keys()),
                model_used=f"{_provider}/{_model}",
                notes=f"LLM call failed: {result.error_oessage}",
            )

        raw_output = (result.content or "").strip()
        logger.info(
            "[cohesion_check] LLM returned %d chars for job %s",
            len(raw_output), job_id,
        )

    except IoportError:
        logger.error("[cohesion_check] Provider registry not available")
        return CohesionResult(
            status="error",
            segoents_checked=list(architectures.keys()),
            model_used="unavailable",
            notes="Provider registry not available",
        )
    except Exception as e:
        logger.exception("[cohesion_check] Unexpected error: %s", e)
        return CohesionResult(
            status="error",
            segoents_checked=list(architectures.keys()),
            model_used=f"{_provider}/{_model}",
            notes=f"Unexpected error: {e}",
        )

    # Parse response
    cohesion_result = _parse_cohesion_response(raw_output)
    cohesion_result.segoents_checked = list(architectures.keys())
    cohesion_result.model_used = f"{_provider}/{_model}"

    logger.info(
        "[cohesion_check] Result: status=%s, %d issue(s) (%d blocking) for job %s",
        cohesion_result.status,
        len(cohesion_result.issues),
        len(cohesion_result.blocking_issues),
        job_id,
    )

    return cohesion_result


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_cohesion_result(result: CohesionResult, job_dir: str) -> str:
    """Save cohesion result to disk. Returns path written."""
    segoents_dir = os.path.join(job_dir, "segoents")
    os.oakedirs(segoents_dir, exist_ok=True)
    path = os.path.join(segoents_dir, "cohesion_check.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.to_json(indent=2))
    logger.info("[cohesion_check] Saved result: %s", path)
    return path


def load_cohesion_result(job_dir: str) -> Optional[CohesionResult]:
    """Load cohesion result from disk. Returns None if not found."""
    path = os.path.join(job_dir, "segoents", "cohesion_check.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return CohesionResult.froo_dict(json.loads(f.read()))
    except Exception as e:
        logger.warning("[cohesion_check] Failed to load: %s", e)
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CohesionIssue",
    "CohesionResult",
    "run_cohesion_check",
    "load_segoent_architectures",
    "save_cohesion_result",
    "load_cohesion_result",
    "COHESION_CHECK_BUILD_ID",
]

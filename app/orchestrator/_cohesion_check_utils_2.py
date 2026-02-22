from __future__ import annotations
import json
import logging
import os
from app.orchestrator._cohesion_check_utils import _build_cohesion_prompt
from app.orchestrator._cohesion_check_utils import _parse_cohesion_response
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


@dataclass
class CohesionIssue:
    """A single cohesion issue found between segments."""
    issue_id: str = ""
    severity: str = "warning"  # "blocking" or "warning"
    category: str = ""         # import_mismatch, naming_mismatch, shape_mismatch,
                               # missing_export, contract_violation, scope_violation,
                               # phantom_segment, endpoint_mismatch
    description: str = ""
    source_segment: str = ""
    related_segment: str = ""
    file_path: str = ""
    expected: str = ""
    actual: str = ""
    suggested_fix: str = ""
    auto_fix_tier: int = 3          # 1=deterministic, 2=micro-LLM, 3=full-regen
    auto_fixed: bool = False        # True if this issue was auto-resolved
    auto_fix_note: str = ""         # What the auto-fixer did

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "source_segment": self.source_segment,
            "related_segment": self.related_segment,
            "file_path": self.file_path,
            "expected": self.expected,
            "actual": self.actual,
            "suggested_fix": self.suggested_fix,
            "auto_fix_tier": self.auto_fix_tier,
            "auto_fixed": self.auto_fixed,
            "auto_fix_note": self.auto_fix_note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohesionIssue":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class CohesionResult:
    """Result of the cohesion check."""
    status: str = "pass"  # "pass", "fail", "error"
    issues: List[CohesionIssue] = field(default_factory=list)
    segments_checked: List[str] = field(default_factory=list)
    notes: str = ""
    layer1_ran: bool = False
    layer2_ran: bool = False

    @property
    def blocking_issues(self) -> List[CohesionIssue]:
        return [i for i in self.issues if i.severity == "blocking"]

    @property
    def warning_issues(self) -> List[CohesionIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def segments_needing_regen(self) -> List[str]:
        segs = set()
        for i in self.blocking_issues:
            if i.source_segment:
                segs.add(i.source_segment)
        return sorted(segs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "segments_checked": self.segments_checked,
            "notes": self.notes,
            "layer1_ran": self.layer1_ran,
            "layer2_ran": self.layer2_ran,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohesionResult":
        result = cls(
            status=data.get("status", "pass"),
            segments_checked=data.get("segments_checked", []),
            notes=data.get("notes", ""),
            layer1_ran=data.get("layer1_ran", False),
            layer2_ran=data.get("layer2_ran", False),
        )
        for issue_data in data.get("issues", []):
            result.issues.append(CohesionIssue.from_dict(issue_data))
        return result

def load_segment_architectures(
    job_dir: str,
    segment_ids: List[str],
) -> Dict[str, str]:
    """
    Load architecture files for the given segments.

    v5.8: Dynamically finds the highest arch_v{N}.md instead of a hardcoded
    fallback list.  This is consistent with segment_loop._find_latest_arch()
    so that cohesion checking and execution always read the same version.

    Returns {segment_id: architecture_content} for segments that have architectures.
    """
    architectures = {}
    for seg_id in segment_ids:
        seg_dir = os.path.join(job_dir, "segments", seg_id)
        arch_dir = os.path.join(seg_dir, "arch")

        if not os.path.isdir(arch_dir):
            continue

        # v5.8: Find the highest version dynamically
        max_version = 0
        best_path = None
        for fname in os.listdir(arch_dir):
            if fname.startswith("arch_v") and fname.endswith(".md"):
                try:
                    v = int(fname.replace("arch_v", "").replace(".md", ""))
                    if v > max_version:
                        max_version = v
                        best_path = os.path.join(arch_dir, fname)
                except ValueError:
                    pass

        if best_path and os.path.isfile(best_path):
            try:
                with open(best_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    architectures[seg_id] = content
                    logger.debug("[cohesion_check] Loaded %s for %s (%d chars)",
                                 os.path.basename(best_path), seg_id, len(content))
            except Exception as e:
                logger.warning("[cohesion_check] Failed to read %s: %s", best_path, e)

    return architectures

async def run_cohesion_check(
    job_id: str,
    job_dir: str,
    segment_ids: List[str],
    contract_json: Optional[str] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    source_file_evidence: Optional[Dict[str, str]] = None,
    skip_llm_layer: bool = False,
) -> CohesionResult:
    """
    Run the cross-segment cohesion check (both layers).

    Layer 1: Deterministic skeleton compliance (always runs, free)
    Layer 2: LLM-based cross-segment analysis (runs if Layer 1 passes)

    v6.1: skip_llm_layer=True for deterministic refactor jobs — the
    architecture was generated from scan data so Layer 2 adds no value.

    Args:
        job_id: Job identifier
        job_dir: Path to job directory on disk
        segment_ids: List of segment IDs to check (APPROVED segments)
        contract_json: Optional JSON string from SkeletonContractSet.to_json()
        provider_id: Override provider for Layer 2 (default: anthropic)
        model_id: Override model for Layer 2 (default: from stage config)

    Returns:
        CohesionResult with any issues found
    """
    if len(segment_ids) < 2:
        return CohesionResult(
            status="pass",
            segments_checked=segment_ids,
            notes="Skipped: fewer than 2 segments to check",
        )

    # Load architectures from disk
    architectures = load_segment_architectures(job_dir, segment_ids)

    if len(architectures) < 2:
        return CohesionResult(
            status="pass",
            segments_checked=list(architectures.keys()),
            notes=f"Skipped: only {len(architectures)} architecture(s) found on disk",
        )

    result = CohesionResult(segments_checked=list(architectures.keys()))

    # Also load manifest for additional context
    manifest_dict = None
    manifest_path = os.path.join(job_dir, "segments", "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_dict = json.load(f)
        except Exception:
            pass

    # =========================================================================
    # LAYER 1: Deterministic skeleton compliance
    # =========================================================================
    logger.info("[cohesion_check] Layer 1: Running skeleton compliance check")
    layer1_issues = run_skeleton_compliance(
        architectures=architectures,
        skeleton_json=contract_json,
        manifest_dict=manifest_dict,
    )
    result.issues.extend(layer1_issues)
    result.layer1_ran = True

    if layer1_issues:
        n_blocking = len([i for i in layer1_issues if i.severity == "blocking"])
        n_warning = len([i for i in layer1_issues if i.severity == "warning"])
        logger.info("[cohesion_check] Layer 1: %d blocking, %d warning", n_blocking, n_warning)
    else:
        logger.info("[cohesion_check] Layer 1: CLEAN — no skeleton violations")

    # If Layer 1 found blocking issues, try auto-fix BEFORE giving up
    if any(i.severity == "blocking" for i in layer1_issues):
        logger.info("[cohesion_check] Layer 1 blocking issues found — attempting auto-fix")
        result.status = "fail"
        result.notes = "Layer 1 (skeleton compliance) found blocking issues"

        # Attempt tiered auto-fix
        result = await attempt_auto_fixes(
            result=result,
            job_dir=job_dir,
            architectures=architectures,
            skeleton_json=contract_json,
            manifest_dict=manifest_dict,
        )

        # If all blocking issues resolved, continue to Layer 2
        if result.blocking_issues:
            result.notes += " — auto-fix could not resolve all blocking issues, Layer 2 skipped"
            return result
        else:
            logger.info("[cohesion_check] Auto-fix resolved all blocking issues — proceeding to Layer 2")
            result.status = "pass"  # Reset for Layer 2 evaluation

    # =========================================================================
    # LAYER 2: LLM-based cross-segment cohesion
    # =========================================================================
    # v6.1: Skip LLM layer for deterministic refactor jobs
    if skip_llm_layer:
        logger.info("[cohesion_check] Layer 2: SKIPPED (deterministic refactor — skip_llm_layer=True)")
        if result.status != "fail":
            result.status = "pass"
            result.notes = (result.notes + " | Layer 2 skipped (deterministic refactor)").strip(" | ")
        return result

    logger.info("[cohesion_check] Layer 2: Running LLM cohesion check")

    # Resolve provider/model
    _provider = provider_id
    _model = model_id

    if not _provider or not _model:
        try:
            from app.llm.stage_models import get_stage_config
            config = get_stage_config("COHESION_CHECK")
            _provider = _provider or config.get("provider", "anthropic")
            _model = _model or config.get("model", "claude-opus-4-6")
        except Exception:
            _provider = _provider or os.getenv("COHERENCE_GUARDIAN_PROVIDER", "anthropic")
            _model = _model or os.getenv("COHERENCE_GUARDIAN_MODEL", "claude-opus-4-6")

    # Build prompt
    prompt = _build_cohesion_prompt(architectures, contract_json, source_file_evidence)

    # Call LLM
    try:
        from app.providers.registry import llm_call

        _messages = [{"role": "user", "content": prompt}]
        _system = (
            "You are a cross-segment architecture reviewer. "
            "Check for interface compatibility issues between segments. "
            "Be precise and only report real issues. "
            "Respond with valid JSON only."
        )

        llm_result_obj = await llm_call(
            provider_id=_provider,
            model_id=_model,
            messages=_messages,
            system_prompt=_system,
            max_tokens=8192,
            timeout_seconds=180,
        )
        llm_response = llm_result_obj.content if llm_result_obj else None

        if llm_response:
            llm_result = _parse_cohesion_response(llm_response)
            result.issues.extend(llm_result.issues)
            if llm_result.notes:
                result.notes = (result.notes + " | " + llm_result.notes).strip(" | ")
            result.layer2_ran = True

            logger.info(
                "[cohesion_check] Layer 2: %d blocking, %d warning",
                len(llm_result.blocking_issues),
                len(llm_result.warning_issues),
            )
        else:
            result.notes = (result.notes + " | Layer 2: empty LLM response").strip(" | ")
            result.layer2_ran = True

    except Exception as llm_err:
        logger.warning("[cohesion_check] Layer 2 LLM call failed: %s", llm_err)
        result.notes = (result.notes + f" | Layer 2 error: {llm_err}").strip(" | ")

    # Determine final status
    if result.blocking_issues:
        # Layer 2 found blocking issues — try auto-fix on those too
        logger.info("[cohesion_check] Layer 2 blocking issues found — attempting auto-fix")

        # Reload architectures in case Layer 1 auto-fix already patched some
        reloaded = load_segment_architectures(job_dir, list(architectures.keys()))

        result = await attempt_auto_fixes(
            result=result,
            job_dir=job_dir,
            architectures=reloaded,
            skeleton_json=contract_json,
            manifest_dict=manifest_dict,
        )

    if result.blocking_issues:
        result.status = "fail"
    else:
        result.status = "pass"

    return result

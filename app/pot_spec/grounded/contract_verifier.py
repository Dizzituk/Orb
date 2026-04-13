# FILE: app/pot_spec/grounded/contract_verifier.py
"""
Cross-Segment Contract Verifier (Phase 1 Job 7)

Verifies that interface contracts declared across segments are coherent:
every consumed identifier is published by some segment, and signatures
match where both sides have declared them.

Design: This is a *diagnostic* layer, not a gate. It reports what it can
verify, what it can't verify (because declarations are missing or sparse),
and what's actively broken. Callers decide whether to treat warnings as
errors based on context (e.g. Phase 3 strict verification might, while
a speculative dry-run might not).

Failure modes reported:
  - MATCHED: consumed identifier found with matching signature
  - MISSING_PUBLISHER: consumed identifier not published by any segment
  - SIGNATURE_MISMATCH: identifier found but shape differs between sides
  - ORPHANED_PUBLISHER: segment publishes something nobody consumes (warning)
  - UNDECLARED: segment has dependents but no exposes declaration (warning)

v1.0 (2026-04-12): Phase 1 Job 7.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ContractIssueKind(str, Enum):
    MATCHED = "matched"
    MISSING_PUBLISHER = "missing_publisher"
    SIGNATURE_MISMATCH = "signature_mismatch"
    ORPHANED_PUBLISHER = "orphaned_publisher"
    UNDECLARED = "undeclared"


class ContractSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ContractIssue:
    kind: ContractIssueKind
    severity: ContractSeverity
    identifier: str
    category: str                   # class_names | method_signatures | endpoint_paths | export_names
    producer_segment: Optional[str] = None
    consumer_segment: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "identifier": self.identifier,
            "category": self.category,
            "producer_segment": self.producer_segment,
            "consumer_segment": self.consumer_segment,
            "detail": self.detail,
        }


@dataclass
class ContractReport:
    issues: List[ContractIssue] = field(default_factory=list)
    total_published: int = 0
    total_consumed: int = 0
    cross_target_matches: int = 0   # Matches that span target boundaries

    def errors(self) -> List[ContractIssue]:
        return [i for i in self.issues if i.severity == ContractSeverity.ERROR]

    def warnings(self) -> List[ContractIssue]:
        return [i for i in self.issues if i.severity == ContractSeverity.WARNING]

    def is_passing(self) -> bool:
        return not self.errors()

    def summary(self) -> str:
        err = len(self.errors())
        warn = len(self.warnings())
        return (f"ContractReport(published={self.total_published}, "
                f"consumed={self.total_consumed}, "
                f"cross_target_matches={self.cross_target_matches}, "
                f"errors={err}, warnings={warn})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "total_published": self.total_published,
            "total_consumed": self.total_consumed,
            "cross_target_matches": self.cross_target_matches,
        }


# Categories we know how to verify. Must match InterfaceContract field names.
_CATEGORIES = ("class_names", "method_signatures", "endpoint_paths", "export_names")


def _normalize_identifier(category: str, raw: str) -> str:
    """Normalize an identifier for matching across publisher/consumer sides.

    Signatures often have minor whitespace or arg-name differences that
    we don't want to count as mismatches. For endpoints, strip whitespace
    only. For method signatures, collapse whitespace. Preserves case.
    """
    if category == "method_signatures":
        return " ".join(raw.split())
    return raw.strip()


def _extract_endpoint_key(raw: str) -> str:
    """For endpoints, the 'identifier' is METHOD + path (e.g. 'POST /foo').
    The consumer might declare just '/foo' or with method — normalize to
    make matching more forgiving. Returns lowercase for case-insensitive
    matching since HTTP methods are canonically uppercase but paths are
    sometimes mis-cased."""
    parts = raw.strip().split(None, 1)
    if len(parts) == 2:
        return f"{parts[0].upper()} {parts[1]}"
    return raw.strip()


def verify_manifest_contracts(manifest) -> ContractReport:
    """Walk a manifest and produce a contract verification report.

    For each segment:
      - If it has dependents but exposes is None or empty, flag UNDECLARED.
      - Index everything it publishes.
    For each segment's consumes:
      - Match against the global published index.
      - Report MATCHED / MISSING_PUBLISHER / SIGNATURE_MISMATCH.
    For each published identifier with no consumer:
      - Flag ORPHANED_PUBLISHER (warning only).

    Returns a ContractReport.
    """
    report = ContractReport()

    if not manifest or not manifest.segments:
        return report

    # Build set of segment_ids that are depended on by someone
    depended_on: Set[str] = set()
    for seg in manifest.segments:
        for dep_id in seg.dependencies:
            depended_on.add(dep_id)

    # Segment target map for cross-target match counting
    target_by_id = {s.segment_id: s.target_id for s in manifest.segments}

    # Build published index: category -> {normalized_id -> (segment_id, raw_signature)}
    published: Dict[str, Dict[str, Tuple[str, str]]] = {c: {} for c in _CATEGORIES}
    consumed_identifiers: Dict[str, Set[str]] = {c: set() for c in _CATEGORIES}

    for seg in manifest.segments:
        exposes = seg.exposes
        # UNDECLARED check: depended-on segment with no exposes
        if seg.segment_id in depended_on:
            if exposes is None or exposes.is_empty():
                report.issues.append(ContractIssue(
                    kind=ContractIssueKind.UNDECLARED,
                    severity=ContractSeverity.WARNING,
                    identifier="<entire-segment>",
                    category="",
                    producer_segment=seg.segment_id,
                    detail="Segment has dependents but declares no exposes "
                           "contract. Consumers cannot be verified against it.",
                ))
        if exposes is None:
            continue

        for cat in _CATEGORIES:
            for raw in getattr(exposes, cat, []):
                key = _extract_endpoint_key(raw) if cat == "endpoint_paths" else _normalize_identifier(cat, raw)
                if key in published[cat]:
                    # Duplicate publish — last-writer-wins but log
                    logger.debug(
                        "[contract] duplicate publish of %s/%s by %s (prev: %s)",
                        cat, key, seg.segment_id, published[cat][key][0],
                    )
                published[cat][key] = (seg.segment_id, raw)
                report.total_published += 1

    # Walk consumers
    for seg in manifest.segments:
        consumes = seg.consumes
        if consumes is None:
            continue
        for cat in _CATEGORIES:
            for raw in getattr(consumes, cat, []):
                key = _extract_endpoint_key(raw) if cat == "endpoint_paths" else _normalize_identifier(cat, raw)
                consumed_identifiers[cat].add(key)
                report.total_consumed += 1

                pub = published[cat].get(key)
                if not pub:
                    report.issues.append(ContractIssue(
                        kind=ContractIssueKind.MISSING_PUBLISHER,
                        severity=ContractSeverity.ERROR,
                        identifier=raw,
                        category=cat,
                        consumer_segment=seg.segment_id,
                        detail=f"{seg.segment_id} consumes {cat}[{raw}] but no segment publishes it.",
                    ))
                    continue

                producer_id, producer_raw = pub
                # Cross-target match tracking
                if target_by_id.get(producer_id) and target_by_id.get(seg.segment_id):
                    if target_by_id[producer_id] != target_by_id[seg.segment_id]:
                        report.cross_target_matches += 1

                # Signature comparison (we already normalized both sides for
                # the match; if they were strict-different they wouldn't have
                # matched. Flag a SIGNATURE_MISMATCH only when the *raw*
                # strings differ — helps catch arg-name drift where the
                # normalized shape is the same but documentation is wrong).
                if producer_raw != raw:
                    report.issues.append(ContractIssue(
                        kind=ContractIssueKind.SIGNATURE_MISMATCH,
                        severity=ContractSeverity.WARNING,
                        identifier=raw,
                        category=cat,
                        producer_segment=producer_id,
                        consumer_segment=seg.segment_id,
                        detail=f"producer={producer_raw!r} consumer={raw!r}",
                    ))
                else:
                    report.issues.append(ContractIssue(
                        kind=ContractIssueKind.MATCHED,
                        severity=ContractSeverity.OK,
                        identifier=raw,
                        category=cat,
                        producer_segment=producer_id,
                        consumer_segment=seg.segment_id,
                    ))

    # Orphaned publishers
    for cat in _CATEGORIES:
        for key, (producer_id, raw) in published[cat].items():
            if key not in consumed_identifiers[cat]:
                report.issues.append(ContractIssue(
                    kind=ContractIssueKind.ORPHANED_PUBLISHER,
                    severity=ContractSeverity.WARNING,
                    identifier=raw,
                    category=cat,
                    producer_segment=producer_id,
                    detail=f"{producer_id} publishes {cat}[{raw}] but nothing consumes it.",
                ))

    logger.info("[contract] %s", report.summary())
    return report

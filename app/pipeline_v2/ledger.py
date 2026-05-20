# FILE: app/pipeline_v2/ledger.py
"""
ASTRA v2.2 Decision Ledger.

Persistent cross-stage record of architectural decisions made during a build.
Every pipeline stage (SpecGate, Scaffold, Agentic Builder, BVL / Verifier)
reads the ledger at start, writes new entries when it makes decisions, and
flags conflicts when a decision contradicts a prior entry.

Ported from app/orchestrator/evidence_ledger.py (2026-02) and extended with:
  - rationale, assumptions, constrains fields on decision entries
  - LedgerConflictError: loud failure when a decision contradicts a prior
    decision (same key, different value) without an explicit `supersedes`
  - Narrowed for v2.2 pipeline usage; old pipeline keeps its own copy

Not to be confused with app/pot_spec/ledger*.py (ndjson event stream for
audit/replay — transactional logging, not decision-tracking).

File layout:
    <job_dir>/decision_ledger.json          # Metadata + entries
    <job_dir>/evidence/                     # Raw file contents (large)

Relationship to the spec's Piece 1 (pipeline_stability_upgrade.md):
  - Every stage reads the ledger at run start (format_ledger_for_prompt).
  - Every stage writes entries when it makes decisions (ledger_append).
  - Conflicts are caught on write, not at integration time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEDGER_VERSION = "2.2.0"
LEDGER_FILENAME = "decision_ledger.json"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LedgerConflictError(Exception):
    """Raised when a decision append would contradict a prior confirmed decision
    without an explicit `supersedes` reference.

    This is the mechanism that stops cross-stage drift: if stage 6 tries to
    overrule a decision stage 2 made, it must do so explicitly.
    """


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LedgerEntry:
    """A single entry in the decision ledger.

    Entry types:
      - decision:      An architectural choice (persistence layer, routing scheme, ...)
      - constraint:    A binding rule downstream must honour
      - codebase_fact: An observed fact about the target repo
      - file_read:     A file was read into evidence/
      - flag:          A concern raised by a stage for downstream attention
      - correction:    An entry that explicitly supersedes a prior one
      - verification:  A verifier recorded that something passed/failed

    Statuses: confirmed, provisional, disputed, corrected, verified.
    """

    entry_id: str           # Auto-generated: "d-001", "d-002", ...
    type: str
    stage: str
    timestamp: str          # ISO format UTC
    status: str
    relevant_to: List[str]  # Segment / unit IDs, or ["all"]
    summary: str            # Human-readable one-liner

    # Decision identity (decisions with a `key` participate in conflict detection)
    key: Optional[str] = None
    value: Optional[str] = None

    # NEW in v2.2 — decision semantics
    rationale: Optional[str] = None          # Why this decision was made
    assumptions: List[str] = field(default_factory=list)  # What it rests on
    constrains: List[str] = field(default_factory=list)   # Downstream areas it binds

    # Optional — varies by type
    segment: Optional[str] = None
    path: Optional[str] = None
    content_ref: Optional[str] = None
    content_hash: Optional[str] = None
    content_size: Optional[int] = None
    category: Optional[str] = None
    source_line: Optional[int] = None
    source_file: Optional[str] = None
    description: Optional[str] = None
    supersedes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize, omitting None values and empty lists for clean JSON."""
        d = asdict(self)
        return {
            k: v for k, v in d.items()
            if v is not None and v != []
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        # Restore list defaults if absent
        filtered.setdefault("assumptions", [])
        filtered.setdefault("constrains", [])
        return cls(**filtered)


@dataclass
class DecisionLedger:
    """The complete decision ledger for a job."""

    job_id: str
    created_at: str
    version: str = LEDGER_VERSION
    entries: List[LedgerEntry] = field(default_factory=list)
    _next_id: int = 1

    def _generate_entry_id(self) -> str:
        entry_id = f"d-{self._next_id:03d}"
        self._next_id += 1
        return entry_id

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def confirmed_entries(self) -> List[LedgerEntry]:
        return [e for e in self.entries if e.status in ("confirmed", "verified")]

    @property
    def decisions(self) -> List[LedgerEntry]:
        return [e for e in self.entries if e.type == "decision"]

    @property
    def corrections(self) -> List[LedgerEntry]:
        return [e for e in self.entries if e.type == "correction"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "version": self.version,
            "entries": [e.to_dict() for e in self.entries],
            "_next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionLedger":
        entries = [LedgerEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            job_id=data["job_id"],
            created_at=data["created_at"],
            version=data.get("version", LEDGER_VERSION),
            entries=entries,
            _next_id=data.get("_next_id", len(entries) + 1),
        )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _find_confirmed_decision(
    ledger: DecisionLedger,
    key: str,
) -> Optional[LedgerEntry]:
    """Return the current confirmed decision for this key, if any.

    Walks newest-first so if multiple exist the latest non-corrected one wins.
    """
    for entry in reversed(ledger.entries):
        if (
            entry.type == "decision"
            and entry.key == key
            and entry.status in ("confirmed", "verified")
        ):
            return entry
    return None


# =============================================================================
# CORE API — lifecycle
# =============================================================================

def create_ledger(job_id: str, job_dir: str) -> DecisionLedger:
    """Create a new empty ledger for a job and persist it."""
    os.makedirs(os.path.join(job_dir, "evidence"), exist_ok=True)
    ledger = DecisionLedger(
        job_id=job_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_ledger(ledger, job_dir)
    logger.info("[ledger] Created new decision ledger for job %s", job_id)
    return ledger


def load_ledger(job_dir: str) -> Optional[DecisionLedger]:
    """Load existing ledger from disk. Returns None if not found."""
    ledger_path = os.path.join(job_dir, LEDGER_FILENAME)
    if not os.path.isfile(ledger_path):
        return None
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ledger = DecisionLedger.from_dict(data)
        logger.info("[ledger] Loaded: %d entries", ledger.entry_count)
        return ledger
    except Exception as exc:
        logger.warning("[ledger] Failed to load: %s", exc)
        return None


def save_ledger(ledger: DecisionLedger, job_dir: str) -> None:
    """Persist ledger to disk (atomic write via temp file + rename)."""
    os.makedirs(job_dir, exist_ok=True)
    ledger_path = os.path.join(job_dir, LEDGER_FILENAME)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=job_dir, prefix=".ledger_tmp_",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, ledger_path)
        logger.debug("[ledger] Saved: %d entries", ledger.entry_count)
    except Exception as exc:
        logger.error("[ledger] Failed to save: %s", exc)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def load_or_create_ledger(job_id: str, job_dir: str) -> DecisionLedger:
    """Idempotent — returns existing ledger or creates a new one."""
    existing = load_ledger(job_dir)
    if existing is not None:
        return existing
    return create_ledger(job_id, job_dir)


# =============================================================================
# CORE API — append with conflict detection
# =============================================================================

def ledger_append(
    ledger: DecisionLedger,
    entry_type: str,
    stage: str,
    relevant_to: List[str],
    summary: str,
    status: str = "confirmed",
    *,
    key: Optional[str] = None,
    value: Optional[str] = None,
    rationale: Optional[str] = None,
    assumptions: Optional[List[str]] = None,
    constrains: Optional[List[str]] = None,
    supersedes: Optional[str] = None,
    **kwargs: Any,
) -> LedgerEntry:
    """Append a new entry. Auto-generates entry_id and timestamp.

    Conflict detection (decision-type entries with a `key`):
      - Same key + same value as an existing confirmed decision → append OK
        (counts as affirmation; ledger records the restatement).
      - Same key + different value + no `supersedes` → LedgerConflictError.
      - Same key + different value + `supersedes` pointing at the wrong
        entry → LedgerConflictError.
      - Same key + different value + `supersedes` correctly targeting the
        existing entry → existing marked 'corrected', new entry appended.

    Non-decision entries and decisions without a `key` skip conflict
    detection entirely.
    """
    if entry_type == "decision" and key is not None:
        existing = _find_confirmed_decision(ledger, key)
        if existing is not None and existing.value != value:
            if supersedes is None:
                raise LedgerConflictError(
                    f"Decision conflict on key='{key}': existing entry "
                    f"{existing.entry_id} has value='{existing.value}' "
                    f"(stage={existing.stage}); new entry proposes "
                    f"value='{value}' (stage={stage}). "
                    f"To overturn, pass supersedes='{existing.entry_id}' "
                    f"with a rationale."
                )
            if supersedes != existing.entry_id:
                raise LedgerConflictError(
                    f"Decision conflict on key='{key}': supersedes="
                    f"'{supersedes}' but current confirmed entry is "
                    f"{existing.entry_id}. Update supersedes."
                )
            # Valid supersede — mark the old one corrected.
            existing.status = "corrected"
            logger.info(
                "[ledger] Decision %s on key='%s' corrected by new entry",
                existing.entry_id, key,
            )

    entry = LedgerEntry(
        entry_id=ledger._generate_entry_id(),
        type=entry_type,
        stage=stage,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=status,
        relevant_to=relevant_to,
        summary=summary,
        key=key,
        value=value,
        rationale=rationale,
        assumptions=list(assumptions or []),
        constrains=list(constrains or []),
        supersedes=supersedes,
        **kwargs,
    )
    ledger.entries.append(entry)
    logger.info(
        "[ledger] %s [%s/%s] %s",
        entry.entry_id, entry_type, stage, summary[:80],
    )
    return entry


def ledger_correct(
    ledger: DecisionLedger,
    supersedes_id: str,
    stage: str,
    description: str,
    relevant_to: List[str],
) -> LedgerEntry:
    """Mark an existing entry as corrected and add a correction entry.

    For decision overrides, prefer calling ledger_append with entry_type=
    'decision', the same key, the new value, and supersedes=<old_id>. This
    function is for non-decision corrections (e.g. correcting a codebase_fact).
    """
    for entry in ledger.entries:
        if entry.entry_id == supersedes_id:
            entry.status = "corrected"
            break
    else:
        logger.warning("[ledger] Entry %s not found to correct", supersedes_id)

    return ledger_append(
        ledger,
        entry_type="correction",
        stage=stage,
        relevant_to=relevant_to,
        summary=f"Corrects {supersedes_id}: {description[:100]}",
        status="correction",
        supersedes=supersedes_id,
        description=description,
    )


# =============================================================================
# CORE API — read slices
# =============================================================================

def ledger_slice(
    ledger: DecisionLedger,
    segment_id: str,
    include_types: Optional[List[str]] = None,
    exclude_corrected: bool = True,
) -> List[LedgerEntry]:
    """Get entries relevant to a specific segment / unit."""
    results = []
    for entry in ledger.entries:
        if segment_id not in entry.relevant_to and "all" not in entry.relevant_to:
            continue
        if include_types and entry.type not in include_types:
            continue
        if exclude_corrected and entry.status == "corrected":
            continue
        results.append(entry)
    return results


def ledger_slice_by_constraint(
    ledger: DecisionLedger,
    stage_concerns: List[str],
    exclude_corrected: bool = True,
) -> List[LedgerEntry]:
    """Get decision entries whose `constrains` field intersects a stage's
    concerns. This is the primary retrieval path for context threading
    (Piece 5 of the stability spec).
    """
    concerns = {c.lower() for c in stage_concerns}
    results = []
    for entry in ledger.entries:
        if entry.type != "decision":
            continue
        if exclude_corrected and entry.status == "corrected":
            continue
        entry_constrains = {c.lower() for c in (entry.constrains or [])}
        if concerns & entry_constrains or "all" in entry_constrains:
            results.append(entry)
    return results


# =============================================================================
# CORE API — file evidence
# =============================================================================

def store_file_evidence(
    ledger: DecisionLedger,
    job_dir: str,
    rel_path: str,
    content: str,
    stage: str,
    relevant_to: List[str],
    summary: Optional[str] = None,
) -> LedgerEntry:
    """Store file content in evidence/ folder and add a ledger entry."""
    evidence_dir = os.path.join(job_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    safe_name = rel_path.replace("\\", "_").replace("/", "_")
    evidence_path = os.path.join(evidence_dir, safe_name)

    with open(evidence_path, "w", encoding="utf-8") as f:
        f.write(content)

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    if summary is None:
        summary = f"File read: {rel_path} ({len(content):,} chars)"

    return ledger_append(
        ledger,
        entry_type="file_read",
        stage=stage,
        relevant_to=relevant_to,
        summary=summary,
        path=rel_path,
        content_ref=f"evidence/{safe_name}",
        content_hash=content_hash,
        content_size=len(content),
    )


def load_file_evidence(
    job_dir: str,
    content_ref: str,
    max_chars: int = 250_000,
) -> Optional[str]:
    """Load file content from the evidence/ folder."""
    evidence_path = os.path.join(job_dir, content_ref)
    if not os.path.isfile(evidence_path):
        return None
    try:
        with open(evidence_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception as exc:
        logger.warning("[ledger] Failed to read %s: %s", evidence_path, exc)
        return None


def seed_ledger_with_source_files(
    ledger: DecisionLedger,
    job_dir: str,
    source_files: Dict[str, str],
) -> int:
    """Seed the ledger with pre-loaded source file evidence.

    Returns number of entries created.
    """
    count = 0
    for rel_path, content in source_files.items():
        line_count = content.count("\n") + 1
        store_file_evidence(
            ledger=ledger,
            job_dir=job_dir,
            rel_path=rel_path,
            content=content,
            stage="seed",
            relevant_to=["all"],
            summary=f"Source file: {rel_path} ({len(content):,} chars, {line_count} lines)",
        )
        count += 1

    if count:
        save_ledger(ledger, job_dir)
        logger.info("[ledger] Seeded with %d source file(s)", count)
    return count


# =============================================================================
# PROMPT FORMATTING
# =============================================================================
# Moved to app/pipeline_v2/ledger_format.py to keep this file focused on
# data + core API. Import from there:
#     from app.pipeline_v2.ledger_format import (
#         format_ledger_for_prompt, format_decisions_only,
#     )

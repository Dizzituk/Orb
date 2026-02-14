# FILE: app/orchestrator/evidence_ledger.py
"""
Evidence Ledger — Living evidence document for ASTRA pipeline jobs.

BUILD_ID: 2026-02-12-v1.0-evidence-ledger-foundation

Every pipeline stage reads from and writes to this ledger:
  - SpecGate seeds it with file reads and codebase facts
  - Skeleton contracts add constraints
  - Critical Pipeline reads evidence and writes architectural decisions
  - Critique reads decisions and flags contradictions
  - Cohesion check reads everything and writes corrections
  - Phase Checkout reads the full ledger for verification

The ledger is persisted to disk after every write and survives restarts.

File layout:
    jobs/<job-id>/evidence_ledger.json    # Metadata + decisions
    jobs/<job-id>/evidence/               # Raw file contents (large)
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

EVIDENCE_LEDGER_BUILD_ID = "2026-02-12-v1.0-evidence-ledger-foundation"
print(f"[EVIDENCE_LEDGER_LOADED] BUILD_ID={EVIDENCE_LEDGER_BUILD_ID}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LedgerEntry:
    """A single entry in the evidence ledger."""

    entry_id: str           # Auto-generated: "ev-001", "ev-002", ...
    type: str               # file_read, codebase_fact, decision, constraint, flag, correction, verification
    stage: str              # specgate, skeleton, critical_pipeline, critique, cohesion_check, checkout
    timestamp: str          # ISO format
    status: str             # confirmed, provisional, disputed, corrected, verified
    relevant_to: List[str]  # Segment IDs or ["all"]
    summary: str            # Human-readable one-liner

    # Optional fields depending on type
    segment: Optional[str] = None
    path: Optional[str] = None
    content_ref: Optional[str] = None
    content_hash: Optional[str] = None
    content_size: Optional[int] = None
    category: Optional[str] = None
    key: Optional[str] = None
    value: Optional[str] = None
    source_line: Optional[int] = None
    source_file: Optional[str] = None
    description: Optional[str] = None
    supersedes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, omitting None values for clean JSON."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        """Deserialize from dict."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class EvidenceLedger:
    """The complete evidence ledger for a job."""

    job_id: str
    created_at: str
    version: str = "1.0"
    entries: List[LedgerEntry] = field(default_factory=list)
    _next_id: int = 1

    def _generate_entry_id(self) -> str:
        entry_id = f"ev-{self._next_id:03d}"
        self._next_id += 1
        return entry_id

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def confirmed_entries(self) -> List[LedgerEntry]:
        return [e for e in self.entries if e.status in ("confirmed", "verified")]

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
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceLedger":
        entries = [LedgerEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            job_id=data["job_id"],
            created_at=data["created_at"],
            version=data.get("version", "1.0"),
            entries=entries,
            _next_id=data.get("_next_id", len(entries) + 1),
        )


# =============================================================================
# CORE API
# =============================================================================

def create_ledger(job_id: str, job_dir: str) -> EvidenceLedger:
    """Create a new empty ledger for a job."""
    evidence_dir = os.path.join(job_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    ledger = EvidenceLedger(
        job_id=job_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_ledger(ledger, job_dir)
    logger.info("[evidence_ledger] Created new ledger for job %s", job_id)
    return ledger


def load_ledger(job_dir: str) -> Optional[EvidenceLedger]:
    """Load existing ledger from disk. Returns None if not found."""
    ledger_path = os.path.join(job_dir, "evidence_ledger.json")
    if not os.path.isfile(ledger_path):
        return None
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ledger = EvidenceLedger.from_dict(data)
        logger.info("[evidence_ledger] Loaded ledger: %d entries", ledger.entry_count)
        return ledger
    except Exception as exc:
        logger.warning("[evidence_ledger] Failed to load ledger: %s", exc)
        return None


def save_ledger(ledger: EvidenceLedger, job_dir: str) -> None:
    """Persist ledger to disk (atomic write via temp file + rename)."""
    ledger_path = os.path.join(job_dir, "evidence_ledger.json")
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=job_dir, prefix=".ledger_tmp_",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, ledger_path)
        logger.debug("[evidence_ledger] Saved ledger: %d entries", ledger.entry_count)
    except Exception as exc:
        logger.error("[evidence_ledger] Failed to save ledger: %s", exc)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def ledger_append(
    ledger: EvidenceLedger,
    entry_type: str,
    stage: str,
    relevant_to: List[str],
    summary: str,
    status: str = "confirmed",
    **kwargs,
) -> LedgerEntry:
    """Append a new entry. Auto-generates entry_id and timestamp."""
    entry = LedgerEntry(
        entry_id=ledger._generate_entry_id(),
        type=entry_type,
        stage=stage,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=status,
        relevant_to=relevant_to,
        summary=summary,
        **kwargs,
    )
    ledger.entries.append(entry)
    logger.info("[evidence_ledger] %s: [%s] %s", entry.entry_id, entry_type, summary[:80])
    return entry


def ledger_correct(
    ledger: EvidenceLedger,
    supersedes_id: str,
    stage: str,
    description: str,
    relevant_to: List[str],
) -> LedgerEntry:
    """Mark an existing entry as corrected and add correction entry."""
    for entry in ledger.entries:
        if entry.entry_id == supersedes_id:
            entry.status = "corrected"
            break
    else:
        logger.warning("[evidence_ledger] Entry %s not found to correct", supersedes_id)

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


def ledger_slice(
    ledger: EvidenceLedger,
    segment_id: str,
    include_types: Optional[List[str]] = None,
    exclude_corrected: bool = True,
) -> List[LedgerEntry]:
    """Get entries relevant to a specific segment."""
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


def store_file_evidence(
    ledger: EvidenceLedger,
    job_dir: str,
    rel_path: str,
    content: str,
    stage: str,
    relevant_to: List[str],
    summary: Optional[str] = None,
) -> LedgerEntry:
    """Store file content in evidence/ folder and add ledger entry."""
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
        logger.warning("[evidence_ledger] Failed to read %s: %s", evidence_path, exc)
        return None


def format_ledger_for_prompt(
    ledger: EvidenceLedger,
    segment_id: str,
    job_dir: str,
    include_file_content: bool = True,
    max_content_chars: int = 120_000,
    include_types: Optional[List[str]] = None,
) -> str:
    """Format ledger slice as markdown for LLM prompt injection."""
    entries = ledger_slice(
        ledger, segment_id,
        include_types=include_types,
        exclude_corrected=True,
    )
    if not entries:
        return ""

    parts = []
    parts.append("### Evidence Ledger (GROUND TRUTH)\n")
    parts.append(
        "The following evidence has been gathered and verified by earlier "
        "pipeline stages. Use this as authoritative reference — do NOT "
        "invent, guess, or approximate values that appear here.\n"
    )

    file_reads = [e for e in entries if e.type == "file_read"]
    decisions = [e for e in entries if e.type == "decision"]
    constraints = [e for e in entries if e.type == "constraint"]
    facts = [e for e in entries if e.type == "codebase_fact"]
    flags = [e for e in entries if e.type == "flag"]

    content_chars_used = 0
    if file_reads:
        parts.append("#### Source Files\n")
        for entry in file_reads:
            if include_file_content and entry.content_ref and content_chars_used < max_content_chars:
                content = load_file_evidence(job_dir, entry.content_ref, max_chars=max_content_chars - content_chars_used)
                if content:
                    parts.append(f"**`{entry.path}`** ({entry.content_size or len(content):,} chars)")
                    parts.append(f"```python\n{content}\n```\n")
                    content_chars_used += len(content)
                else:
                    parts.append(f"- `{entry.path}` ({entry.summary})")
            else:
                parts.append(f"- `{entry.path}` ({entry.summary})")
        parts.append("")

    if facts:
        parts.append("#### Codebase Facts\n")
        for entry in facts:
            parts.append(f"- {entry.summary}")
            if entry.value:
                parts.append(f"  → `{entry.value}`")
        parts.append("")

    if constraints:
        parts.append("#### Constraints\n")
        for entry in constraints:
            parts.append(f"- {entry.summary}")
        parts.append("")

    if decisions:
        parts.append("#### Architectural Decisions (from upstream segments)\n")
        parts.append("These are binding. Match these exactly.\n")
        for entry in decisions:
            src = f" (L{entry.source_line})" if entry.source_line else ""
            parts.append(f"- **{entry.key}**: `{entry.value}`{src}")
            if entry.summary and entry.summary != entry.key:
                parts.append(f"  — {entry.summary}")
        parts.append("")

    if flags:
        parts.append("#### ⚠️ Flags from Review\n")
        for entry in flags:
            parts.append(f"- {entry.summary}")
        parts.append("")

    return "\n".join(parts)


def seed_ledger_with_source_files(
    ledger: EvidenceLedger,
    job_dir: str,
    source_files: Dict[str, str],
) -> int:
    """
    Seed the ledger with pre-loaded source file evidence.

    Called after _load_source_file_evidence() to persist source files
    into the ledger for all downstream stages.

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
            stage="segment_loop",
            relevant_to=["all"],
            summary=f"Source file: {rel_path} ({len(content):,} chars, {line_count} lines)",
        )
        count += 1

    if count:
        save_ledger(ledger, job_dir)
        print(f"[evidence_ledger] Seeded ledger with {count} source file(s)")

    return count

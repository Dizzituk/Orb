# FILE: app/optimize/code_learner.py
# Purpose: Code Learner — captures before/after code snapshots and extracts
# Called-by: app.optimize.orchestrator, app.optimize.proposer, app.optimize.router
# Depends-on: app.providers.registry
# Last-renovated: 2026-06-11
"""
Code Learner — captures before/after code snapshots and extracts
structural lessons from successful optimisations.

When a proposal executes successfully:
  1. Compares the before/after code snapshots
  2. Uses a lightweight LLM call to extract a structured lesson
  3. Stores the lesson with concrete code examples
  4. Feeds lessons back into the proposer for future passes

This gives the recursive loop genuine learning — each pass gets
smarter because it has real examples of what worked before.

v1.0 (2026-03-26): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LESSONS_PATH = os.getenv(
    "ASTRA_LESSONS_PATH",
    os.path.join("D:", os.sep, "Orb", "data", "optimization_lessons.json"),
)

MAX_SNIPPET_LINES = 60  # Cap code snippets to keep storage sane


@dataclass
class CodeLesson:
    """A concrete lesson learned from a successful optimisation."""
    lesson_id: str
    proposal_id: str
    category: str
    title: str

    # What was wrong (human-readable)
    problem_description: str

    # What was done (human-readable)
    fix_description: str

    # The actual code evidence
    file_path: str
    before_snippet: str      # Trimmed code before the change
    after_snippet: str       # Trimmed code after the change

    # Extracted structural lesson (from LLM analysis)
    structural_lesson: str   # Why the new code is better
    reusable_rule: str       # A general rule for future code

    # Metrics
    complexity_before: float = 0.0
    complexity_after: float = 0.0
    size_before: int = 0
    size_after: int = 0

    confidence: float = 0.7
    times_referenced: int = 0
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "proposal_id": self.proposal_id,
            "category": self.category,
            "title": self.title,
            "problem_description": self.problem_description,
            "fix_description": self.fix_description,
            "file_path": self.file_path,
            "before_snippet": self.before_snippet,
            "after_snippet": self.after_snippet,
            "structural_lesson": self.structural_lesson,
            "reusable_rule": self.reusable_rule,
            "complexity_before": self.complexity_before,
            "complexity_after": self.complexity_after,
            "size_before": self.size_before,
            "size_after": self.size_after,
            "confidence": self.confidence,
            "times_referenced": self.times_referenced,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeLesson":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ══════════════════════════════════════════════════════════
# SNAPSHOT CAPTURE
# ══════════════════════════════════════════════════════════

def capture_before_snapshot(
    target_chunks: List[str],
    root_path: str,
) -> Dict[str, str]:
    """Read target files before execution. Returns {path: content}."""
    snapshots = {}
    for chunk_path in target_chunks:
        fpath = Path(root_path) / chunk_path
        try:
            if fpath.exists() and fpath.stat().st_size < 100_000:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                snapshots[chunk_path] = content
        except Exception as e:
            logger.debug("[code_learner] Could not snapshot %s: %s", chunk_path, e)
    return snapshots


def capture_after_snapshot(
    target_chunks: List[str],
    root_path: str,
) -> Dict[str, str]:
    """Read target files after execution. Returns {path: content}."""
    return capture_before_snapshot(target_chunks, root_path)


def _trim_to_changed_region(before: str, after: str) -> tuple[str, str]:
    """Extract the region that actually changed, with context."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()

    # Find first differing line
    start = 0
    for i, (b, a) in enumerate(zip(before_lines, after_lines)):
        if b != a:
            start = max(0, i - 3)  # 3 lines context
            break

    # Find last differing line (from end)
    end_b = len(before_lines)
    end_a = len(after_lines)
    for i in range(1, min(len(before_lines), len(after_lines)) + 1):
        if before_lines[-i] != after_lines[-i]:
            end_b = min(len(before_lines), len(before_lines) - i + 4)
            end_a = min(len(after_lines), len(after_lines) - i + 4)
            break

    trimmed_before = "\n".join(before_lines[start:end_b])
    trimmed_after = "\n".join(after_lines[start:end_a])

    # Cap at MAX_SNIPPET_LINES
    if trimmed_before.count("\n") > MAX_SNIPPET_LINES:
        trimmed_before = "\n".join(trimmed_before.splitlines()[:MAX_SNIPPET_LINES]) + "\n... (trimmed)"
    if trimmed_after.count("\n") > MAX_SNIPPET_LINES:
        trimmed_after = "\n".join(trimmed_after.splitlines()[:MAX_SNIPPET_LINES]) + "\n... (trimmed)"

    return trimmed_before, trimmed_after


# ══════════════════════════════════════════════════════════
# LESSON EXTRACTION (LLM-powered)
# ══════════════════════════════════════════════════════════

LESSON_SYSTEM_PROMPT = """You are a code quality analyst. You will be given a BEFORE and AFTER version of Python code that was optimised.

Your job is to extract:
1. structural_lesson: WHY the new code is better (be specific about the structural improvement — not just "it's cleaner")
2. reusable_rule: A general rule that could be applied to ANY codebase (not specific to this file)

Respond in JSON only:
{"structural_lesson": "...", "reusable_rule": "..."}

No markdown, no backticks, no explanation. Just the JSON object."""


async def extract_lesson(
    before_snippet: str,
    after_snippet: str,
    proposal_title: str,
    category: str,
) -> tuple[str, str]:
    """Use a lightweight LLM call to extract the lesson."""
    try:
        from app.providers.registry import llm_call

        prompt = (
            f"Optimisation: {proposal_title} (category: {category})\n\n"
            f"BEFORE:\n```python\n{before_snippet}\n```\n\n"
            f"AFTER:\n```python\n{after_snippet}\n```"
        )

        result = await llm_call(
            provider="openai",
            model=os.getenv("ASTRA_LESSON_MODEL", "gpt-5-mini"),
            system_prompt=LESSON_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        text = result.content if hasattr(result, "content") else str(result)
        text = text.strip().strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

        parsed = json.loads(text)
        return (
            parsed.get("structural_lesson", "No lesson extracted"),
            parsed.get("reusable_rule", "No rule extracted"),
        )
    except Exception as e:
        logger.warning("[code_learner] LLM lesson extraction failed: %s", e)
        return (
            f"Optimisation in category '{category}' succeeded but lesson extraction failed",
            f"Apply {category} patterns when similar code structures are detected",
        )


# ══════════════════════════════════════════════════════════
# LESSON STORE
# ══════════════════════════════════════════════════════════

class CodeLessonStore:
    """Persists and retrieves code lessons."""

    def __init__(self, path: str = LESSONS_PATH):
        self._path = path
        self._lessons: Dict[str, CodeLesson] = {}
        self._load()

    async def record_lesson(
        self,
        proposal: Any,
        before_snapshots: Dict[str, str],
        after_snapshots: Dict[str, str],
        metrics_before: Dict[str, Any],
        metrics_after: Dict[str, Any],
    ) -> Optional[CodeLesson]:
        """Record a lesson from a successful proposal execution."""
        import hashlib

        # Find the primary file that changed
        primary_file = None
        best_before = ""
        best_after = ""

        for fpath in proposal.target_chunks:
            before = before_snapshots.get(fpath, "")
            after = after_snapshots.get(fpath, "")
            if before and after and before != after:
                primary_file = fpath
                best_before, best_after = _trim_to_changed_region(before, after)
                break

        if not primary_file or not best_before:
            # File was deleted or no meaningful diff
            return None

        # Extract the structural lesson via LLM
        structural_lesson, reusable_rule = await extract_lesson(
            best_before, best_after, proposal.title, proposal.category.value,
        )

        lid = hashlib.sha256(
            f"{proposal.proposal_id}:{primary_file}".encode()
        ).hexdigest()[:12]

        lesson = CodeLesson(
            lesson_id=f"lesson-{lid}",
            proposal_id=proposal.proposal_id,
            category=proposal.category.value,
            title=proposal.title,
            problem_description=proposal.description,
            fix_description=proposal.predicted_improvement,
            file_path=primary_file,
            before_snippet=best_before,
            after_snippet=best_after,
            structural_lesson=structural_lesson,
            reusable_rule=reusable_rule,
            complexity_before=metrics_before.get(primary_file, {}).get("cyclomatic_complexity", 0),
            complexity_after=metrics_after.get(primary_file, {}).get("cyclomatic_complexity", 0),
            size_before=metrics_before.get(primary_file, {}).get("size_bytes", 0),
            size_after=metrics_after.get(primary_file, {}).get("size_bytes", 0),
            confidence=proposal.confidence,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._lessons[lesson.lesson_id] = lesson
        self._persist()

        logger.info("[code_learner] Recorded lesson %s: %s",
                     lesson.lesson_id, structural_lesson[:80])
        return lesson

    def get_lessons_for_category(self, category: str) -> List[CodeLesson]:
        """Get all lessons for a proposal category."""
        return sorted(
            [l for l in self._lessons.values() if l.category == category],
            key=lambda l: l.confidence,
            reverse=True,
        )

    def get_reusable_rules(self, min_confidence: float = 0.6) -> List[str]:
        """Get all reusable rules above confidence threshold."""
        return [
            l.reusable_rule
            for l in self._lessons.values()
            if l.confidence >= min_confidence and l.reusable_rule
        ]

    def get_all_lessons(self) -> List[CodeLesson]:
        return list(self._lessons.values())

    def get_stats(self) -> Dict[str, Any]:
        cats: Dict[str, int] = {}
        for l in self._lessons.values():
            cats[l.category] = cats.get(l.category, 0) + 1
        return {
            "total_lessons": len(self._lessons),
            "categories": cats,
            "path": self._path,
        }

    def _load(self) -> None:
        path = Path(self._path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data.get("lessons", []):
                lesson = CodeLesson.from_dict(item)
                self._lessons[lesson.lesson_id] = lesson
            logger.info("[code_learner] Loaded %d lessons", len(self._lessons))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[code_learner] Load failed: %s", e)

    def _persist(self) -> None:
        try:
            path = Path(self._path)
            path.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "lessons": [l.to_dict() for l in self._lessons.values()],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        except OSError as e:
            logger.error("[code_learner] Persist failed: %s", e)


_store: Optional[CodeLessonStore] = None

def get_lesson_store() -> CodeLessonStore:
    global _store
    if _store is None:
        _store = CodeLessonStore()
    return _store

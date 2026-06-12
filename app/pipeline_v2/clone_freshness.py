# FILE: app/pipeline_v2/clone_freshness.py
# Purpose: Clone freshness gate (2026-06-10, proving-run session).
# Called-by: app.orchestrator.segment_loop, app.pipeline_v2.verifier_agent.agent, tests.test_clone_freshness_smoke
# Depends-on: app.pipeline_v2.sandbox_tools
# Last-renovated: 2026-06-11
"""
Clone freshness gate (2026-06-10, proving-run session).

THE PLACEMENT CONTRACT: ASTRA's own repos (D:/Orb backend, D:/orb-desktop
frontend) are SELF-BUILD targets - the pipeline builds and tests them ONLY
in the sandbox VM's clone (ZOBIE-ORB, 192.168.250.2:8765), never on the
live host. Everything else (astra-bridge etc.) is built directly on the
host and never routes through the sandbox.

This module is the front door of that contract for self-builds: before a
single builder cycle runs, it proves the estate is COHERENT:
  1. HOST CLEAN  - all host work committed (the host is the source of
     truth; uncommitted host work means the clone cannot match it).
  2. HEADS MATCH - clone is on the same commit as the host.
  3. CLONE CLEAN - no leftover debris from prior runs in the clone tree.

Why so strict: building on a stale clone wastes the whole run, and
promoting from one ROLLS LIVE WORK BACKWARDS - promote diffs clone-vs-host
and would "bring across" every file where the host is newer.

Sync mechanism (Taz's design): git via GitHub. Host commits + pushes; the
VM clone resets, cleans and pulls. This gate never syncs by itself - it
reports precisely what is out of step so the human (or a future sanctioned
sync step) can act.

Emergency override: ASTRA_SKIP_CLONE_FRESHNESS=1 (logged loudly).
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

SELF_BUILD_PROJECT_IDS = {"astra-backend", "astra-frontend"}
SELF_BUILD_ROOTS = {"d:/orb", "d:/orb-desktop"}

# Both repos are checked regardless of which one the job targets: a
# self-build of the backend can still touch shared docs/config, and promote
# safety depends on the whole estate being coherent.
REPO_PAIRS = (
    ("Orb backend", "D:/Orb"),
    ("Orb desktop", "D:/orb-desktop"),
)


def is_self_build(profile: Any) -> bool:
    """Single source of truth for 'is this ASTRA building itself?'"""
    if profile is None:
        return False
    pid = str(getattr(profile, "project_id", "") or "").lower()
    if pid in SELF_BUILD_PROJECT_IDS:
        return True
    root = str(getattr(profile, "project_root", "") or "").replace("\\", "/").rstrip("/").lower()
    return root in SELF_BUILD_ROOTS


@dataclass
class RepoFreshness:
    label: str
    root: str
    host_head: str = ""
    host_dirty: int = -1
    clone_head: str = ""
    clone_dirty: int = -1
    ok: bool = False
    reason: str = ""


@dataclass
class FreshnessResult:
    ok: bool
    repos: List[RepoFreshness] = field(default_factory=list)
    reason: str = ""

    def render_lines(self) -> List[str]:
        lines = ["", "=" * 60, "CLONE FRESHNESS GATE (self-build)", "=" * 60]
        for r in self.repos:
            mark = " OK  " if r.ok else "BLOCK"
            lines.append(
                f"   [{mark}] {r.label}: host {r.host_head[:8] or '????????'}"
                f" (dirty {r.host_dirty if r.host_dirty >= 0 else '?'})"
                f"  vs  clone {r.clone_head[:8] or '????????'}"
                f" (dirty {r.clone_dirty if r.clone_dirty >= 0 else '?'})"
            )
            if not r.ok and r.reason:
                lines.append(f"           -> {r.reason}")
        lines.append(f"   VERDICT: {'FRESH - self-build may proceed' if self.ok else 'NOT COHERENT - self-build refused'}")
        if not self.ok:
            lines.append("   Fix: commit + push on the host, then on the VM:")
            lines.append("        git reset --hard ; git clean -fd ; git pull")
        lines.append("=" * 60)
        return lines


def _host_git(root: str) -> Tuple[str, int, str]:
    """(head, dirty_count, error) for a host repo."""
    try:
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if head.returncode != 0:
            return "", -1, (head.stderr or "git rev-parse failed").strip()[:200]
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=20,
        )
        dirty = len([l for l in (status.stdout or "").splitlines() if l.strip()])
        return head.stdout.strip(), dirty, ""
    except Exception as exc:
        return "", -1, str(exc)[:200]


async def _clone_git(root: str) -> Tuple[str, int, str]:
    """(head, dirty_count, error) for the same path inside the sandbox VM."""
    from app.pipeline_v2.sandbox_tools import run_shell
    cmd = (
        f"cd '{root}'; git rev-parse HEAD; "
        f"(git status --porcelain | Measure-Object).Count"
    )
    try:
        res = await run_shell(cmd, timeout_sec=30, profile=None)
    except Exception as exc:
        return "", -1, f"sandbox unreachable: {exc}"[:200]
    if not isinstance(res, dict):
        return "", -1, "sandbox returned no result (VM down?)"
    out = (res.get("stdout") or "").strip().splitlines()
    err = (res.get("stderr") or "").strip()
    lines = [l.strip() for l in out if l.strip()]
    if len(lines) < 2:
        return "", -1, (err or "no git output from clone (VM down or git missing?)")[:200]
    head = lines[0]
    try:
        dirty = int(lines[-1])
    except ValueError:
        dirty = -1
    if len(head) != 40:
        return "", -1, f"unexpected clone git output: {head[:60]}"
    return head, dirty, ""


async def check_clone_freshness(profile: Any = None) -> FreshnessResult:
    """Run the gate. Never raises."""
    if os.getenv("ASTRA_SKIP_CLONE_FRESHNESS", "").strip().lower() in ("1", "true", "yes"):
        logger.warning(
            "[clone_freshness] GATE SKIPPED via ASTRA_SKIP_CLONE_FRESHNESS - "
            "self-build coherence is NOT guaranteed"
        )
        return FreshnessResult(ok=True, reason="skipped via env override")

    result = FreshnessResult(ok=True)
    for label, root in REPO_PAIRS:
        repo = RepoFreshness(label=label, root=root)
        h_head, h_dirty, h_err = _host_git(root)
        c_head, c_dirty, c_err = await _clone_git(root)
        repo.host_head, repo.host_dirty = h_head, h_dirty
        repo.clone_head, repo.clone_dirty = c_head, c_dirty

        if h_err:
            repo.reason = f"host git error: {h_err}"
        elif c_err:
            repo.reason = c_err
        elif not c_head:
            repo.reason = "clone HEAD unreadable"
        elif h_head != c_head:
            repo.reason = "clone is on a different commit to the host (sync needed)"
        elif h_dirty != 0:
            repo.reason = (
                f"host has {h_dirty} uncommitted files - the clone cannot match "
                "the real working state; commit + push first (promote safety)"
            )
        elif c_dirty != 0:
            repo.reason = (
                f"clone working tree has {c_dirty} dirty/untracked files "
                "(old build debris) - reset + clean needed"
            )
        else:
            repo.ok = True

        if not repo.ok:
            result.ok = False
        result.repos.append(repo)

    if not result.ok:
        result.reason = "; ".join(f"{r.label}: {r.reason}" for r in result.repos if not r.ok)
    return result

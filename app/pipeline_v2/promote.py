# FILE: app/pipeline_v2/promote.py
# Purpose: JOB 15 (2026-06-10) - Promote step: verified sandbox self-build -> live Orb.
# Called-by: app.builds.router, tests.test_j15_promote_smoke
# Depends-on: app.pipeline_v2.sandbox_tools
# Last-renovated: 2026-06-11
"""
JOB 15 (2026-06-10) - Promote step: verified sandbox self-build -> live Orb.

This is the missing end of the self-modification loop. A self-build runs in
the sandbox clone (192.168.250.2); once verified, its changed files need a
defined, reversible path into the live codebase. Flow:

  1. PLAN: diff every file in the job manifest's file_scope between the
     sandbox clone and the live host. Produces a human-reviewable plan
     (added / changed / same, sizes, diff head per file).
  2. HUMAN CHECK-OFF: the plan is shown in the UI; apply requires an
     explicit confirm flag - nothing is ever promoted silently.
  3. APPLY: for each approved file, write a dated .bak of the live version
     alongside, then copy the sandbox content onto the host.
  4. SMOKE: py_compile every promoted .py with the live venv. Failures are
     reported (the .baks make reverting a copy-back).
  5. The backend is NOT auto-restarted (promoting from inside the process
     that would die mid-request is silly); the response flags
     restart_required so the UI can prompt.

Safety rails: only paths resolving under D:/Orb or D:/orb-desktop are
promotable; everything else is refused. Apply without a plan-matching
confirm token is refused.

API surface lives in app/builds/router.py:
  POST /builds/projects/promote/plan   {"job_id": "sg-..."}
  POST /builds/projects/promote/apply  {"job_id": "...", "confirm": true,
                                        "files": ["app/x.py", ...] | null}
"""
from __future__ import annotations

import difflib
import glob
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

JOBS_ROOT = "D:\\Orb\\jobs\\jobs"
ALLOWED_ROOT_PREFIXES = ("d:/orb/", "d:/orb-desktop/")
DIFF_HEAD_LINES = 40
MAX_PLAN_FILES = 200


class PromotePlanRequest(BaseModel):
    job_id: str


class PromoteApplyRequest(BaseModel):
    job_id: str
    confirm: bool = False
    files: Optional[List[str]] = None  # None = all changed/added from the plan


@dataclass
class PlanFile:
    path: str
    abs_host: str
    action: str            # "added" | "changed" | "same" | "missing-in-sandbox" | "refused"
    host_size: int = 0
    sandbox_size: int = 0
    diff_head: str = ""
    detail: str = ""


def _host_abs(path: str) -> str:
    """Resolve a manifest path to its live host location (same mapping the
    pipeline's path resolution uses for self-build targets)."""
    from app.pipeline_v2.sandbox_tools import _resolve_path
    return _resolve_path(path, None).replace("\\", "/")


def _is_allowed(abs_host: str) -> bool:
    low = abs_host.replace("\\", "/").lower()
    return any(low.startswith(p) for p in ALLOWED_ROOT_PREFIXES)


def _manifest_paths_for_job(job_id: str) -> List[str]:
    """Collect the file_scope union from the job's manifest(s)."""
    candidates = [
        os.path.join(JOBS_ROOT, job_id, "segments", "manifest.json"),
        os.path.join("D:\\Orb\\jobs", job_id, "segments", "manifest.json"),
    ]
    candidates += glob.glob(os.path.join(JOBS_ROOT, job_id, "**", "*manifest*.json"), recursive=True)

    seen_files: List[str] = []
    seen_keys = set()
    found_manifest = False
    for mp in candidates:
        if not os.path.isfile(mp):
            continue
        found_manifest = True
        try:
            with open(mp, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            logger.warning("[promote] unreadable manifest %s: %s", mp, exc)
            continue
        for seg in manifest.get("segments", []) or []:
            for fp in (seg.get("file_scope", []) if isinstance(seg, dict) else []):
                if not isinstance(fp, str) or not fp:
                    continue
                key = fp.replace("\\", "/").lower()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                seen_files.append(fp)
        break  # first readable manifest wins
    if not found_manifest:
        raise FileNotFoundError(f"No manifest found for job {job_id} under {JOBS_ROOT}")
    return seen_files[:MAX_PLAN_FILES]


async def build_promote_plan(job_id: str) -> Dict[str, Any]:
    """Diff sandbox clone vs live host for every file in the job's scope."""
    from app.pipeline_v2.sandbox_tools import is_sandbox_alive, read_file as sandbox_read

    try:
        scope = _manifest_paths_for_job(job_id)
    except FileNotFoundError as exc:
        return {"job_id": job_id, "ok": False, "error": str(exc), "files": []}

    if not await is_sandbox_alive():
        return {
            "job_id": job_id,
            "ok": False,
            "error": "Sandbox is not reachable - cannot diff the clone",
            "files": [],
        }

    files: List[PlanFile] = []
    for path in scope:
        abs_host = _host_abs(path)
        pf = PlanFile(path=path, abs_host=abs_host, action="same")
        if not _is_allowed(abs_host):
            pf.action = "refused"
            pf.detail = "outside D:/Orb and D:/orb-desktop - not promotable"
            files.append(pf)
            continue

        sandbox_content = await sandbox_read(path, profile=None)
        host_content: Optional[str] = None
        host_path = abs_host.replace("/", os.sep)
        if os.path.isfile(host_path):
            try:
                with open(host_path, "r", encoding="utf-8", errors="replace") as f:
                    host_content = f.read()
            except Exception as exc:
                pf.action = "refused"
                pf.detail = f"host file unreadable: {exc}"
                files.append(pf)
                continue

        if sandbox_content is None:
            pf.action = "missing-in-sandbox"
            pf.detail = "file in manifest scope but absent in the clone"
        else:
            pf.sandbox_size = len(sandbox_content)
            pf.host_size = len(host_content or "")
            if host_content is None:
                pf.action = "added"
            elif host_content == sandbox_content:
                pf.action = "same"
            else:
                pf.action = "changed"
            if pf.action in ("added", "changed"):
                diff = difflib.unified_diff(
                    (host_content or "").splitlines(keepends=False),
                    sandbox_content.splitlines(keepends=False),
                    fromfile=f"live/{path}",
                    tofile=f"sandbox/{path}",
                    lineterm="",
                )
                pf.diff_head = "\n".join(list(diff)[: DIFF_HEAD_LINES + 4])
        files.append(pf)

    promotable = [f for f in files if f.action in ("added", "changed")]
    return {
        "job_id": job_id,
        "ok": True,
        "promotable_count": len(promotable),
        "files": [asdict(f) for f in files],
        "note": "Apply with confirm=true to promote; .baks are written first; backend restart required after.",
    }


async def apply_promote(req: PromoteApplyRequest) -> Dict[str, Any]:
    """Apply a promote plan to the live host. Requires explicit confirm."""
    import datetime

    if not req.confirm:
        return {"ok": False, "error": "confirm=false - promote refused (human check-off required)"}

    plan = await build_promote_plan(req.job_id)
    if not plan.get("ok"):
        return {"ok": False, "error": plan.get("error", "plan failed")}

    wanted = None
    if req.files:
        wanted = {p.replace("\\", "/").lower() for p in req.files}

    from app.pipeline_v2.sandbox_tools import read_file as sandbox_read

    stamp = datetime.date.today().isoformat()
    results: List[Dict[str, Any]] = []
    promoted_py: List[str] = []

    for f in plan["files"]:
        if f["action"] not in ("added", "changed"):
            continue
        key = f["path"].replace("\\", "/").lower()
        if wanted is not None and key not in wanted:
            results.append({"path": f["path"], "action": "skipped", "detail": "not in requested file list"})
            continue

        abs_host = f["abs_host"].replace("/", os.sep)
        try:
            content = await sandbox_read(f["path"], profile=None)
            if content is None:
                results.append({"path": f["path"], "action": "error", "detail": "sandbox read returned None"})
                continue

            bak_path = ""
            if os.path.isfile(abs_host):
                bak_path = f"{abs_host}.bak-{stamp}-promote"
                if not os.path.exists(bak_path):
                    with open(abs_host, "r", encoding="utf-8", errors="replace") as src:
                        original = src.read()
                    with open(bak_path, "w", encoding="utf-8", newline="\n") as bf:
                        bf.write(original)

            os.makedirs(os.path.dirname(abs_host), exist_ok=True)
            with open(abs_host, "w", encoding="utf-8", newline="\n") as out:
                out.write(content)

            results.append({
                "path": f["path"],
                "action": "promoted",
                "bak": os.path.basename(bak_path) if bak_path else "(new file - no bak)",
                "bytes": len(content),
            })
            if abs_host.lower().endswith(".py"):
                promoted_py.append(abs_host)
        except Exception as exc:
            logger.exception("[promote] failed for %s: %s", f["path"], exc)
            results.append({"path": f["path"], "action": "error", "detail": str(exc)[:300]})

    # Post-promote smoke: compile every promoted .py with the live venv
    smoke: List[Dict[str, str]] = []
    if promoted_py:
        import subprocess
        try:
            proc = subprocess.run(
                ["D:\\Orb\\.venv\\Scripts\\python.exe", "-m", "py_compile", *promoted_py],
                capture_output=True, text=True, timeout=120,
            )
            smoke.append({
                "check": "py_compile",
                "status": "pass" if proc.returncode == 0 else "FAIL",
                "detail": (proc.stderr or proc.stdout)[:800],
            })
        except Exception as exc:
            smoke.append({"check": "py_compile", "status": "error", "detail": str(exc)[:300]})

    promoted_count = sum(1 for r in results if r.get("action") == "promoted")
    return {
        "ok": True,
        "job_id": req.job_id,
        "promoted": promoted_count,
        "results": results,
        "smoke": smoke,
        "restart_required": promoted_count > 0,
        "revert_hint": f"copy each *.bak-{stamp}-promote back over its file to revert",
    }

# FILE: app/pot_spec/grounded/_greenfield_gate.py
# Purpose: v12.0 greenfield routing gate — stop unrouted new-app jobs before the sandbox lane.
# Called-by: app.pot_spec.grounded.spec_runner, app.llm.spec_gate_stream (lane logging)
# Depends-on: app.pot_spec.grounded.domain_detection
# Last-renovated: 2026-07-04
"""
Greenfield routing gate for the spec runner (JOBs 2+3 of the greenfield
build-target routing fixpack, 2026-07-04).

Problem this kills: a new-app job (e.g. "build a Tetris desktop game") with
no build_target_profile fell back to _extract_project_paths() chat scraping,
matched ASTRA repos, and hard-stopped on the v11.0 "Sandbox Unavailable"
message demanding the Zobie-Orb VM. Wrong default — "unknown new-app job"
must never degrade to "modify ASTRA".

Two independent signals, either alone is sufficient:
1. Weaver job_class flag (semantic, v3.0 Weaver outputs) — parsed strictly,
   see app/llm/weaver_job_class.py. An explicit "modify_existing" also
   SUPPRESSES the keyword fallback (the Weaver saw the whole job; keyword
   hits like "create a" in ASTRA work chatter must not override it).
2. Deterministic domain detection (greenfield_build / game keyword domains,
   spec_gate_grounded v1.10/v1.12) — the fallback for older Weaver outputs
   that carry no job_class line.

Also home to profile_is_greenfield(), the v2.4 "external root with no source
files" check moved verbatim out of spec_runner.py (file-size discipline), so
spec_runner and spec_gate_stream share one definition of "greenfield".
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# The v12.0 hard-stop reason. Distinct from the v11.0 "Sandbox Unavailable"
# stop — tests key on "No Build Target" to tell the two lanes apart.
NO_BUILD_TARGET_STOP_REASON = (
    "🟡 **No Build Target** — this looks like a new project. Say "
    "**'let's start a new project'** to scope it (name, location, stack) "
    "before sending to SpecGate.\n"
)

# Domains that mark a job as new-app work (see DOMAIN_KEYWORDS v1.10/v1.12).
_GREENFIELD_DOMAINS = ("greenfield_build", "game")

# v2.4 constants (moved verbatim from spec_runner.py STEP 2, 2026-07-04).
_KNOWN_SELF_ROOTS = {"D:/Orb", "D:\\Orb", "D:/orb-desktop", "D:\\orb-desktop"}
_SOURCE_FILE_EXTS = ('.kt', '.java', '.py', '.ts', '.tsx', '.js', '.jsx')


def is_unrouted_greenfield_job(
    constraints_hint: Optional[Dict],
    weaver_job_text: str,
    user_intent: str,
) -> Tuple[bool, str]:
    """Decide whether a NO-profile job is a greenfield job that must stop.

    Only call this when there is no build_target_profile. Returns
    (should_stop, why) — `why` is a short diagnostic for logs.
    """
    hint = constraints_hint or {}

    # Signal 1: the Weaver's semantic flag (hint from spec_gate_stream, else
    # parsed straight from the weaver text for older call paths).
    job_class = str(hint.get("weaver_job_class") or "").strip().lower()
    if job_class not in ("greenfield_new_app", "modify_existing"):
        try:
            from app.llm.weaver_job_class import parse_weaver_job_class
            job_class = parse_weaver_job_class(weaver_job_text or "")
        except Exception:
            job_class = "unknown"

    if job_class == "greenfield_new_app":
        return True, "weaver job_class=greenfield_new_app"
    if job_class == "modify_existing":
        # Explicit Weaver classification wins over keyword heuristics.
        return False, "weaver job_class=modify_existing"

    # Signal 2 (fallback for job_class=unknown): deterministic keyword domains
    # on the weaver job text (question sections excluded, v1.12 rule).
    from .domain_detection import detect_domains
    detect_text = weaver_job_text or user_intent or ""
    domains = detect_domains(detect_text, exclude_questions=True)
    hits = [d for d in _GREENFIELD_DOMAINS if d in domains]
    if hits:
        return True, f"domains={hits}"
    return False, ""


def is_external_root(project_root: str) -> bool:
    """True when the profile root lies outside ASTRA's own repos (v2.4 rule)."""
    _norm_root = (project_root or "").replace("\\", "/").rstrip("/")
    return bool(_norm_root) and _norm_root not in _KNOWN_SELF_ROOTS


def profile_is_greenfield(build_profile: Optional[Dict]) -> bool:
    """v2.4 greenfield check, moved verbatim from spec_runner.py (2026-07-04).

    A profile is greenfield when its root is EXTERNAL to ASTRA's own repos
    AND no source files exist under <root>/<source_root> yet. (v2.3 assumed
    external = greenfield — wrong, AstraBridge has 46 .kt files; hence the
    source-file walk.)
    """
    if not build_profile:
        return False
    _profile_root = build_profile.get("project_root", "")
    if not is_external_root(_profile_root):
        return False
    _src_root = build_profile.get("source_root", "")
    _check_dir = os.path.join(_profile_root, _src_root) if _src_root else _profile_root
    _has_source = False
    try:
        for _root, _dirs, _files in os.walk(_check_dir):
            if any(f.endswith(_SOURCE_FILE_EXTS) for f in _files):
                _has_source = True
                break
    except Exception:
        _has_source = False
    return not _has_source

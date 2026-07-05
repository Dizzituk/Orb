# FILE: app/llm/_sg_target_injection.py
# Purpose: SpecGate build-target hint resolution — session/BuildProject/signal lookup, greenfield override + auto-scope.
# Called-by: app.llm.spec_gate_stream
# Depends-on: app.llm.greenfield_autoscope, app.pipeline_v2.target_registry, app.pot_spec.grounded._greenfield_gate
# Last-renovated: 2026-07-04
"""
Build-target hint resolution for the SpecGate stream. Extracted from
spec_gate_stream.py v2.5–v2.8 (2026-07-04, 30KB cap).

Full wiring chain:
  scoping flow (project_scoping_stream.handle_scoping_confirmation)
    -> register_profile()             [registry + data/build_targets.json, JOB 1]
    -> ProjectSession.finish_scoping  [in-memory, keyed str(chat project_id)]
    -> BuildProject.build_target_id   [DB stamp, JOB 4 — restart-durable]
  resolve_build_target_hint(), every spec gate run:
    Path 1: ProjectSession(str(project_id)) -> get_profile(session.project_id)
    Path 2: BuildProject.build_target_id    -> get_profile(...)  [survives restart]
    Path 3: resolve_project_from_message()  [signal scoring; built-ins only]
    v2.7 GREENFIELD OVERRIDE: job_class=greenfield_new_app discards BUILT-IN
      targets from Paths 2/3 — keyword noise ("desktop app"/"PC" chatter
      scored a Tetris ramble onto astra-frontend). Session + dynamic targets
      always survive.
    v2.8 GREENFIELD AUTO-SCOPE: still no profile? Scope the project straight
      from the weave (greenfield_autoscope leaf) — folder created, target
      registered + persisted, session set, BuildProject stamped. Only when
      the weave names an explicit external location; otherwise fall through.
    -> constraints_hint["build_target_profile"]
  -> run_spec_gate_grounded -> spec_runner STEP 2
     (v2.4 greenfield-vs-existing split when the profile is present,
      v12.0 no-build-target scoping stop when it is absent).

Restart survival needs BOTH halves: the profile must reload (JOB 1 registry
persistence) AND a lookup path must still know the project_id (JOB 4
BuildProject stamp — sessions die with the process).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def resolve_build_target_hint(
    db: Session,
    project_id: int,
    message: str,
    weaver_spec_json: Optional[dict],
) -> Tuple[Optional[Dict], Optional[str]]:
    """Resolve the build_target_profile hint for a spec gate run.

    Returns (profile_dict_or_None, user_note_or_None). The note is a stream
    token for the user (set when auto-scope creates a project). Never raises.
    """
    try:
        from app.pipeline_v2.target_registry import get_profile, BUILTIN_PROFILE_IDS

        _profile = None
        _profile_source = None
        _note: Optional[str] = None
        _weaver_job_class = str((weaver_spec_json or {}).get("job_class") or "unknown")

        # Path 1: ProjectSession (conversation-keyed)
        try:
            from app.shared_context.project_session import get_project_session
            _session = get_project_session(str(project_id))
            if _session.is_set and _session.project_id:
                _profile = get_profile(_session.project_id)
                _profile_source = "session"
        except Exception:
            pass

        # Path 2: BuildProject in database (most reliable — has build_target_id)
        if not _profile:
            try:
                from app.builds.pipeline_bridge import get_or_create_build_project
                _bp = get_or_create_build_project(db, project_id)
                if _bp and _bp.build_target_id:
                    _profile = get_profile(_bp.build_target_id)
                    _profile_source = "build_project"
            except Exception:
                pass

        # Path 3: Auto-detect from the trigger message via target_registry signals
        if not _profile:
            try:
                from app.pipeline_v2.target_registry import resolve_project_from_message
                _detected = resolve_project_from_message(message)
                if _detected:
                    _profile = _detected
                    _profile_source = "message_signals"
            except Exception:
                pass

        # v2.7–v2.9 greenfield handling. The signal is the SAME hierarchy as
        # spec_runner's v12.0 gate (shared helper): explicit Weaver job_class
        # flag, else greenfield/game keyword domains on the weave. Using the
        # shared helper (not the raw flag) closes the flagless-legacy-weave
        # hole where the dispatch-time keyword stamp resurrected the sandbox
        # misroute (review finding, 2026-07-04).
        _job_desc = str((weaver_spec_json or {}).get("job_description") or "")
        _gf_signal = False
        try:
            from app.pot_spec.grounded._greenfield_gate import is_unrouted_greenfield_job
            _gf_signal, _gf_why = is_unrouted_greenfield_job(
                {"weaver_job_class": _weaver_job_class}, _job_desc, "",
            )
        except Exception:
            pass

        if _gf_signal:
            # v2.7 OVERRIDE: built-in targets from Path 2/3 are keyword noise
            # by definition (all four built-ins are existing codebases).
            if (
                _profile is not None
                and _profile_source != "session"
                and _profile.project_id in BUILTIN_PROFILE_IDS
            ):
                logger.info(
                    "[spec_gate_stream] v2.7 GREENFIELD OVERRIDE: discarded "
                    "keyword-scored target %s (source=%s)",
                    _profile.project_id, _profile_source,
                )
                print(
                    f"[spec_gate_stream] v2.7 GREENFIELD OVERRIDE: discarded "
                    f"keyword-scored target {_profile.project_id} (source={_profile_source})"
                )
                _profile = None
                _profile_source = None

            # v2.8/v2.9 AUTO-SCOPE: scope the project straight from the weave.
            # v2.9 (review finding): a greenfield weave naming an explicit
            # location DIFFERENT from the resolved target is a NEW project —
            # a latched session/BuildProject from an earlier project must not
            # swallow it. Same location = idempotent re-run, keep the target.
            try:
                from app.llm.greenfield_autoscope import (
                    autoscope_greenfield_job,
                    extract_greenfield_target,
                )
                _t = extract_greenfield_target(_job_desc)
                _cur_root = (
                    _profile.project_root.replace("\\", "/").rstrip("/").lower()
                    if _profile else None
                )
                _new_root = _t["root"].rstrip("/").lower() if _t else None
                if _profile is None or (_t and _cur_root != _new_root):
                    _auto = autoscope_greenfield_job(
                        db=db, chat_project_id=project_id, weaver_text=_job_desc,
                    )
                    if _auto is not None:
                        _profile = _auto
                        _profile_source = "autoscope"
                        _note = (
                            f"🌱 **New project auto-scoped from the Weaver**: "
                            f"{_auto.project_name} → `{_auto.project_root}`\n"
                            f"   Folder created, build target `{_auto.project_id}` "
                            f"registered and persisted — proceeding in greenfield mode.\n\n"
                        )
                    elif _profile is not None and _t and _cur_root != _new_root:
                        # Weave names a new location we could not scope —
                        # proceeding with the OLD project's root would write
                        # the new app inside it. Stop for manual scoping.
                        logger.warning(
                            "[spec_gate_stream] v2.9 stale target %s dropped: "
                            "weave names different location %s but auto-scope "
                            "failed", _profile.project_id, _t["root"],
                        )
                        print(
                            f"[spec_gate_stream] v2.9 stale target "
                            f"{_profile.project_id} dropped — weave names new "
                            f"location {_t['root']}"
                        )
                        _profile = None
                        _profile_source = None
            except Exception as _as_err:
                logger.warning("[spec_gate_stream] v2.8 auto-scope failed: %s", _as_err)

        if not _profile:
            return None, None

        profile_dict = {
            "project_id": _profile.project_id,
            "project_name": _profile.project_name,
            "project_root": _profile.project_root,
            "language": _profile.language,
            "framework": _profile.framework,
            "build_system": _profile.build_system,
            "source_root": _profile.source_root,
            "package_name": _profile.package_name,
            "architecture_pattern": _profile.architecture_pattern,
        }

        # v2.6 (JOB 4): lane visibility — say which target AND which lane
        # (greenfield vs existing) this job will take.
        _gf = False
        try:
            from app.pot_spec.grounded._greenfield_gate import profile_is_greenfield
            _gf = profile_is_greenfield(profile_dict)
        except Exception:
            pass
        logger.info(
            "[spec_gate_stream] v2.6 BUILD TARGET INJECTED: project_id=%s root=%s "
            "greenfield=%s source=%s",
            _profile.project_id, _profile.project_root,
            "yes" if _gf else "no", _profile_source,
        )
        print(
            f"[spec_gate_stream] v2.6 BUILD TARGET: {_profile.project_id} "
            f"({_profile.project_name}) at {_profile.project_root} "
            f"greenfield={'yes' if _gf else 'no'} source={_profile_source}"
        )
        return profile_dict, _note
    except Exception as e:
        logger.debug("[spec_gate_stream] Profile injection failed: %s", e)
        return None, None

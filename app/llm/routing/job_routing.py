# FILE: app/llm/routing/job_routing.py
# FILE: app/llm/routing/job_routing.py
"""Routing helpers extracted from app.llm.router.

Goal: keep app.llm.router focused on orchestration while preserving behaviour.

This module provides:
- _default_importance_for_job_type
- _default_modalities_for_job_type
- inject_file_map_into_messages
- classify_and_route

Design notes:
- We intentionally lean on app.llm.job_classifier.classify_and_route (the authoritative classifier)
  and keep these wrappers small.
- Modalities/Importance defaults are conservative and defensive; they will not crash if new enum
  members are added/removed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.jobs.schemas import Importance, Modality
from app.llm.schemas import JobType, LLMTask, Provider
from app.llm.job_classifier import classify_and_route as classifier_classify_and_route


def _default_importance_for_job_type(job_type: JobType) -> Importance:
    """Return a sensible default Importance for a given legacy JobType.

    This only affects the Phase-4 envelope defaults; routing decisions are still driven by the classifier.
    """
    # Safe enum fallbacks (in case schema changes)
    normal = getattr(Importance, "NORMAL", None) or getattr(Importance, "MEDIUM", None) or list(Importance)[0]
    low = getattr(Importance, "LOW", None) or normal
    high = getattr(Importance, "HIGH", None) or normal
    critical = getattr(Importance, "CRITICAL", None) or high

    if job_type in (JobType.CHAT_LIGHT,):
        return low

    if job_type in (JobType.APP_ARCHITECTURE, JobType.ORCHESTRATOR):
        return high

    if job_type in (JobType.CRITIQUE_REVIEW,):
        return high

    if job_type in (JobType.VIDEO_CODE_DEBUG, JobType.VIDEO_HEAVY):
        return high

    if job_type in (JobType.CODE_MEDIUM, JobType.TEXT_HEAVY):
        return normal

    return normal


def _default_modalities_for_job_type(job_type: JobType) -> List[Modality]:
    """Return default input modalities for a given legacy JobType.

    Conservative: always includes TEXT. Adds VIDEO/IMAGE when those enum members exist and the job_type implies them.
    """
    mods: List[Modality] = [Modality.TEXT]

    # Add VIDEO if the enum supports it and job type implies it.
    video = getattr(Modality, "VIDEO", None)
    if video is not None and job_type in (JobType.VIDEO_HEAVY, JobType.VIDEO_CODE_DEBUG):
        mods.append(video)

    # Some schemas use IMAGE for screenshots. Add only if available and job type implies mixed media.
    image = getattr(Modality, "IMAGE", None)
    if image is not None and job_type in (JobType.VIDEO_CODE_DEBUG, JobType.VIDEO_HEAVY):
        # Avoid duplicates if VIDEO==IMAGE or weird enum setups
        if image not in mods:
            mods.append(image)

    return mods


def inject_file_map_into_messages(messages: List[Dict[str, Any]], file_map: str) -> List[Dict[str, Any]]:
    """Inject the file map into the prompt messages.

    Strategy:
    - If a system message exists first, append the file map to it.
    - Otherwise, create a new system message at the front.

    Keeps the user content intact and makes the file_map available to all providers.
    """
    if not file_map:
        return messages

    header = "\n\n[FILE MAP]\n"
    payload = header + str(file_map)

    if not messages:
        return [{"role": "system", "content": payload}]

    first = messages[0]
    if isinstance(first, dict) and first.get("role") == "system" and isinstance(first.get("content"), str):
        new_first = dict(first)
        new_first["content"] = (new_first.get("content") or "") + payload
        return [new_first] + messages[1:]

    return [{"role": "system", "content": payload}] + messages


def _flatten_user_text(task: LLMTask) -> str:
    parts: List[str] = []
    for msg in (task.messages or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # multimodal message; take text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = part.get("text") or ""
                    if txt:
                        parts.append(str(txt))
    return "\n".join(parts).strip()


def classify_and_route(task: LLMTask) -> Tuple[Provider, str, JobType, str]:
    """Wrapper around the authoritative classifier.

    Returns: (provider, model_id, job_type, reason)
    """
    message_text = _flatten_user_text(task)
    requested_type: Optional[str] = task.job_type.value if task.job_type else None

    metadata: Dict[str, Any] = {}
    if getattr(task, "project_context", None):
        metadata["project_context"] = task.project_context

    decision = classifier_classify_and_route(
        message=message_text,
        attachments=task.attachments,
        job_type=requested_type,
        metadata=metadata,
    )

    # Be defensive about the decision object shape.
    provider = getattr(decision, "provider", None) or Provider.OPENAI
    model = getattr(decision, "model", None) or ""
    job_type = getattr(decision, "job_type", None) or (task.job_type or JobType.TEXT_HEAVY)
    reason = getattr(decision, "reason", None) or "classified"

    return (provider, model, job_type, reason)

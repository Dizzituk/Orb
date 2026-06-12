# Purpose: core utils
# Called-by: app.llm.routing.core
# Depends-on: app.llm.job_classifier, app.llm.schemas
# Last-renovated: 2026-06-11
import inspect
import os
from app.llm.job_classifier import get_model_config, is_claude_allowed, is_claude_forbidden
from app.llm.schemas import JobType, LLMTask, Provider, RoutingDecision
from typing import List, Tuple


AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"

def _call_classifier(classify_fn, task, message_text: str, requested_type=None):
    """
    Call the job classifier in a backwards/forwards-compatible way.

    Classifiers in this repo historically come in two shapes:
      1) task-first:    classify_and_route(task, ...)
      2) message-first: classify_and_route(message, attachments=None, job_type=None, metadata=None)

    This helper inspects the classifier signature and passes only the
    arguments it is likely to accept, preserving existing behavior while
    avoiding hard failures when signatures evolve.
    """

    def _task_attr(name, default=None):
        try:
            return getattr(task, name)
        except Exception:
            return default

    try:
        sig = inspect.signature(classify_fn)
    except Exception:
        # Conservative fallback: try task-first then message-first.
        try:
            return classify_fn(task, message_text, requested_type=requested_type)
        except TypeError:
            try:
                return classify_fn(task)
            except TypeError:
                return classify_fn(message_text, _task_attr("attachments"))

    params = list(sig.parameters.values())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    names = [p.name for p in positional]

    TASK_NAMES = {"task", "llm_task", "request", "req"}
    MESSAGE_NAMES = {"message", "message_text", "text", "prompt", "input_text"}

    args = []

    # Decide whether this classifier is message-first or task-first.
    # If the first positional param looks like a message, prefer message-first.
    if names and (names[0] in MESSAGE_NAMES) and (names[0] not in TASK_NAMES):
        # message-first: (message, attachments?, job_type?, metadata?)
        args.append(message_text)
        if len(names) >= 2 and names[1] in {"attachments", "files", "file_attachments", "attachment_infos"}:
            args.append(_task_attr("attachments"))
        if len(names) >= 3 and names[2] in {"job_type", "type", "job"}:
            args.append(_task_attr("job_type"))
        if len(names) >= 4 and names[3] in {"metadata", "meta", "context"}:
            args.append(_task_attr("metadata"))
    else:
        # task-first: (task, message_text?)
        if names:
            args.append(task)
        if len(names) >= 2 and names[1] in MESSAGE_NAMES:
            args.append(message_text)

    # Build kwargs only for supported names unless **kwargs exists.
    kwargs = {}
    if has_var_kw:
        kwargs["requested_type"] = requested_type
    else:
        supported = {p.name for p in params}
        if "requested_type" in supported:
            kwargs["requested_type"] = requested_type

    return classify_fn(*args, **kwargs)

def _check_attachment_safety(
    task: LLMTask,
    decision: RoutingDecision,
    has_attachments: bool,
    job_type_specified: bool,
) -> Tuple[str, str, str]:
    """
    Attachment safety logic:
    - If attachments exist, and Claude is forbidden: force OpenAI.
    - If attachments exist, and job type is user-specified: honor requested type but enforce provider safety.
    """
    provider = decision.provider
    model = decision.model
    reason = decision.reason

    if not has_attachments:
        return provider, model, reason

    # If Claude is forbidden for these attachments, force OpenAI
    if is_claude_forbidden(decision.job_type):
        if provider == Provider.ANTHROPIC:
            print("[router] Claude forbidden for attachments; forcing OpenAI")
            provider = Provider.OPENAI
            model = get_model_config(Provider.OPENAI, task.job_type).model
            reason = f"{reason} | Claude forbidden for attachments → OpenAI fallback"
        return provider, model, reason

    # If job type explicitly specified by user, we respect it, but keep provider safe.
    if job_type_specified:
        if provider == Provider.ANTHROPIC and not is_claude_allowed(decision.job_type):
            print("[router] Job type specified but Claude not allowed; forcing OpenAI")
            provider = Provider.OPENAI
            model = get_model_config(Provider.OPENAI, task.job_type).model
            reason = f"{reason} | Claude not allowed for attachments → OpenAI fallback"
        return provider, model, reason

    return provider, model, reason

def list_job_types() -> List[str]:
    return [jt.value for jt in JobType]

def is_policy_routing_enabled() -> bool:
    return True

def enable_policy_routing() -> None:
    return None

def disable_policy_routing() -> None:
    return None

def reload_routing_policy() -> None:
    return None

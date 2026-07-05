# FILE: app/llm/stage_roles.py
# Purpose: Derek stage-role model registry — canonical ASTRA_STAGE_* keys with provenance and fail-loud resolution.
# Called-by: app.settings.stage_roles_api, app.pipeline_v2.config, main (startup validation)
# Depends-on: app.llm.frontier_models
# Last-renovated: 2026-07-04 (Derek phase 2)
"""
The Derek tier map, in env.

Seven canonical stage roles cover the whole build pipeline. Each resolves
provider+model from env with explicit PROVENANCE (which key won) and
FAIL-LOUD semantics (missing/blank model raises StageRoleResolutionError
naming the exact key — never a silent stale default).

    Role            Tier   Env keys (canonical)
    WEAVER_MAIN     HEAVY  ASTRA_STAGE_WEAVER_MAIN_PROVIDER / _MODEL
    SPECGATE_MAIN   APEX   ASTRA_STAGE_SPECGATE_MAIN_PROVIDER / _MODEL
    SPECGATE_AGENT  HANDS  ASTRA_STAGE_SPECGATE_AGENT_PROVIDER / _MODEL
    BUILDER_MAIN    HEAVY  ASTRA_STAGE_BUILDER_MAIN_PROVIDER / _MODEL
    BUILDER_WORKER  HANDS  ASTRA_STAGE_BUILDER_WORKER_PROVIDER / _MODEL
    CHECKOUT_JUDGE  HEAVY  ASTRA_STAGE_CHECKOUT_JUDGE_PROVIDER / _MODEL
    CHECKOUT_EYES   SMALL  ASTRA_STAGE_CHECKOUT_EYES_PROVIDER / _MODEL

Legacy vars (WEAVER_MODEL, SPEC_GATE_MODEL, ASTRA_V2_BUILDER_MODEL, ...)
remain as deprecated aliases: they feed a role when the canonical key is
unset, with a deprecation WARN. Model values may be frontier aliases
(e.g. "anthropic:frontier-opus-thinking") — resolved at call time.

Everything reads env per call (no import-time caching), so Models-tab
saves hot-apply. Under Derek the same keys point at vllm_local endpoints.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StageRoleResolutionError(RuntimeError):
    """A stage role has no usable model/provider. Message names the env key."""


@dataclass(frozen=True)
class RoleSpec:
    """Static definition of one stage role."""
    role: str
    tier: str                      # APEX | HEAVY | HANDS | SMALL
    description: str               # one line, plain English (shown on the card)
    derek_target: str              # what serves this role once Derek exists
    legacy_provider_keys: Tuple[str, ...] = ()
    legacy_model_keys: Tuple[str, ...] = ()
    consumer: str = ""             # where the role is used (file hint)
    home_is_legacy: bool = False   # legacy key is the PERMANENT home (CHAT), not a deprecated alias


@dataclass
class ResolvedRole:
    """A role resolved against the live environment, with provenance."""
    role: str
    tier: str
    description: str
    derek_target: str
    provider: str = ""
    model: str = ""
    raw_model: str = ""            # pre-alias-resolution value
    model_source: str = ""         # env key the model came from
    provider_source: str = ""      # env key / "inferred from model"
    deprecated_source: bool = False
    deprecation_note: str = ""
    error: str = ""                # non-empty when resolution failed

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "tier": self.tier,
            "description": self.description,
            "derek_target": self.derek_target,
            "provider": self.provider,
            "model": self.model,
            "raw_model": self.raw_model,
            "model_source": self.model_source,
            "provider_source": self.provider_source,
            "deprecated_source": self.deprecated_source,
            "deprecation_note": self.deprecation_note,
            "canonical_model_env": f"ASTRA_STAGE_{self.role}_MODEL",
            "canonical_provider_env": f"ASTRA_STAGE_{self.role}_PROVIDER",
            "error": self.error,
        }


STAGE_ROLES: Dict[str, RoleSpec] = {
    "WEAVER_MAIN": RoleSpec(
        role="WEAVER_MAIN", tier="HEAVY",
        description="Turns your rambling into a structured job outline before SpecGate.",
        derek_target="Gemma 4 31B dense (RTX 4090, load-on-demand)",
        legacy_provider_keys=("WEAVER_PROVIDER",),
        legacy_model_keys=("WEAVER_MODEL",),
        consumer="app/llm/weaver_stream.py",
    ),
    "SPECGATE_MAIN": RoleSpec(
        role="SPECGATE_MAIN", tier="APEX",
        description="The architecture brain — synthesises the build spec from evidence. Stays frontier, even under Derek.",
        derek_target="Frontier API (permanently)",
        legacy_provider_keys=("SPEC_GATE_PROVIDER",),
        legacy_model_keys=("SPEC_GATE_MODEL", "OPENAI_SPEC_GATE_MODEL"),
        consumer="app/llm/spec_gate_stream.py -> app/pot_spec/grounded/",
    ),
    "SPECGATE_AGENT": RoleSpec(
        role="SPECGATE_AGENT", tier="HANDS",
        description="Research runners that fetch files and evidence so the SpecGate brain only synthesises.",
        derek_target="Gemma 4 26B MoE x2 (RTX 3090 pair)",
        consumer="app/pot_spec/grounded/agent_dispatch.py (Derek phase 4)",
    ),
    "BUILDER_MAIN": RoleSpec(
        role="BUILDER_MAIN", tier="HEAVY",
        description="The build integrator — verifies cross-segment wiring and runs the boot/build.",
        derek_target="Gemma 4 31B dense (RTX 4090, load-on-demand)",
        legacy_provider_keys=("ASTRA_V2_BUILDER_PROVIDER",),
        legacy_model_keys=("ASTRA_V2_BUILDER_MODEL",),
        consumer="app/pipeline_v2/agentic_builder.py",
    ),
    "BUILDER_WORKER": RoleSpec(
        role="BUILDER_WORKER", tier="HANDS",
        description="The hands that build each segment — fresh small context per segment.",
        derek_target="Gemma 4 26B MoE x2 (RTX 3090 pair)",
        consumer="app/pipeline_v2/segmented/ (Derek phase 5)",
    ),
    "CHECKOUT_JUDGE": RoleSpec(
        role="CHECKOUT_JUDGE", tier="HEAVY",
        description="Judges the eyes' evidence against the spec: does it work for a customer?",
        derek_target="Gemma 4 31B dense (RTX 4090, load-on-demand)",
        consumer="app/pipeline_v2/verifier_agent/ (Derek phase 6)",
    ),
    "CHECKOUT_EYES": RoleSpec(
        role="CHECKOUT_EYES", tier="SMALL",
        description="Cheap vision that drives the built app — screenshots, clicks, captures what happened.",
        derek_target="Gemma E-class small vision (3rd RTX 3090, personality card)",
        legacy_provider_keys=("ASTRA_V2_VERIFIER_PROVIDER",),
        legacy_model_keys=("ASTRA_V2_VERIFIER_MODEL",),
        consumer="app/pipeline_v2/checkout.py + verifier_agent perception (Derek phase 6)",
    ),
    # UTILITY roles (2026-07-04 live, Taz request): not build-pipeline tiers,
    # but exposed as cards so the everyday chat + debug models are tweakable.
    "CHAT_MAIN": RoleSpec(
        role="CHAT_MAIN", tier="UTILITY",
        description="The everyday conversational model — what you talk to in normal chat (base tier; the router may still escalate).",
        derek_target="Gemma E-class conversational/secretary (3rd RTX 3090, personality card)",
        legacy_provider_keys=("CHAT_PROVIDER",),
        legacy_model_keys=("CHAT_MODEL",),
        consumer="app/llm/routing/chat_model_selection.py get_role_model('CHAT')",
        home_is_legacy=True,
    ),
    "DEBUG_MAIN": RoleSpec(
        role="DEBUG_MAIN", tier="UTILITY",
        description="The debug assistant's chat brain — honours the DEBUG_PROVIDER openai/anthropic column switch.",
        derek_target="Host tooling — stays API or a heavy local model",
        consumer="app/debug/debug_model_config.py resolve_debug_model('chat')",
    ),
}

# Deprecation WARNs fire once per role per process.
_deprecation_warned: set = set()


def _infer_provider(model: str) -> str:
    m = model.lower()
    if "claude" in m or "opus" in m or "sonnet" in m or "haiku" in m or "fable" in m:
        return "anthropic"
    if "gemini" in m or "gemma" in m:
        return "google"
    if "gpt" in m or m.startswith(("o1", "o3", "o4")):
        return "openai"
    if "qwen" in m or "vllm" in m or "llama" in m or "mistral" in m:
        return "vllm_local"
    return ""


def _resolve_debug_main() -> ResolvedRole:
    """DEBUG_MAIN resolves through the debug provider-switch so the card shows
    the TRULY-active column (openai vs anthropic), never a lie. No canonical
    ASTRA_STAGE_ key — the debug system owns DEBUG_PROVIDER + DEBUG_*_MODEL[_ANTHROPIC]."""
    spec = STAGE_ROLES["DEBUG_MAIN"]
    r = ResolvedRole(
        role=spec.role, tier=spec.tier,
        description=spec.description, derek_target=spec.derek_target,
    )
    try:
        from app.debug.debug_model_config import resolve_debug_model
        provider, model = resolve_debug_model("chat")
        r.provider, r.model, r.raw_model = provider, model, model
        # Source var reflects the column ACTUALLY returned (resolve_debug_model
        # may key-gate-fall-back from anthropic to openai), so provider and
        # source never disagree.
        r.model_source = "DEBUG_CHAT_MODEL_ANTHROPIC" if provider == "anthropic" else "DEBUG_CHAT_MODEL"
        r.provider_source = "DEBUG_PROVIDER"
    except Exception as exc:
        r.error = f"debug model resolution failed: {exc}"
    return r


def resolve_stage_role(role: str) -> ResolvedRole:
    """
    Resolve one stage role from env. Raises StageRoleResolutionError with a
    message naming the exact env key when the model cannot be resolved.

    Precedence: ASTRA_STAGE_{ROLE}_MODEL > legacy alias vars (deprecation
    WARN) > error. NEVER a code default — stale defaults are hardcoding
    with extra steps.
    """
    key = role.upper().replace("-", "_").replace(" ", "_")
    spec = STAGE_ROLES.get(key)
    if spec is None:
        raise StageRoleResolutionError(
            f"Unknown stage role {role!r}. Known roles: {', '.join(sorted(STAGE_ROLES))}"
        )

    # DEBUG_MAIN has a provider-switch consumer — resolve via its own config.
    if key == "DEBUG_MAIN":
        resolved = _resolve_debug_main()
        if resolved.error:
            raise StageRoleResolutionError(resolved.error)
        return resolved

    resolved = ResolvedRole(
        role=spec.role, tier=spec.tier,
        description=spec.description, derek_target=spec.derek_target,
    )

    canonical_model_key = f"ASTRA_STAGE_{key}_MODEL"
    canonical_provider_key = f"ASTRA_STAGE_{key}_PROVIDER"

    if spec.home_is_legacy:
        # The legacy var IS the permanent home (e.g. CHAT_MODEL) — read it as
        # the primary, non-deprecated source; canonical_model_key still shown
        # so the UI knows the ASTRA_STAGE_ name exists, but it is not the home.
        model = model_source = provider = provider_source = ""
        for i, mk in enumerate(spec.legacy_model_keys):
            val = os.getenv(mk, "").strip()
            if val:
                model, model_source = val, mk
                pk = spec.legacy_provider_keys[i] if i < len(spec.legacy_provider_keys) else ""
                provider = os.getenv(pk, "").strip() if pk else ""
                provider_source = pk if provider else ""
                break
    else:
        model = os.getenv(canonical_model_key, "").strip()
        model_source = canonical_model_key
        provider = os.getenv(canonical_provider_key, "").strip()
        provider_source = canonical_provider_key if provider else ""

    if not model:
        for legacy_key in spec.legacy_model_keys:
            legacy_val = os.getenv(legacy_key, "").strip()
            if legacy_val:
                model = legacy_val
                model_source = legacy_key
                resolved.deprecated_source = True
                resolved.deprecation_note = (
                    f"{legacy_key} is a deprecated alias — set {canonical_model_key} instead"
                )
                if key not in _deprecation_warned:
                    _deprecation_warned.add(key)
                    logger.warning(
                        "[stage_roles] %s resolved from DEPRECATED %s=%s — "
                        "migrate to %s (same value).",
                        key, legacy_key, legacy_val, canonical_model_key,
                    )
                break

    if not model:
        raise StageRoleResolutionError(
            f"Stage role {key} has no model configured. Set {canonical_model_key} "
            f"in D:\\Orb\\.env (provider via {canonical_provider_key}; frontier "
            f"aliases like 'anthropic:frontier-opus-thinking' are accepted)."
            + (
                f" Deprecated fallbacks also empty: {', '.join(spec.legacy_model_keys)}."
                if spec.legacy_model_keys else ""
            )
        )

    if not provider:
        for legacy_key in spec.legacy_provider_keys:
            legacy_val = os.getenv(legacy_key, "").strip()
            if legacy_val:
                provider = legacy_val
                provider_source = legacy_key
                break

    resolved.raw_model = model
    try:
        from app.llm.frontier_models import resolve_model_alias
        model = resolve_model_alias(model)
    except Exception as exc:
        logger.debug("[stage_roles] alias resolution unavailable: %s", exc)

    if not model.strip():
        raise StageRoleResolutionError(
            f"Stage role {key}: {model_source}={resolved.raw_model!r} resolved to an "
            f"empty model id — check the frontier alias env it points at."
        )
    # Derek p7 review fix: an UNEXPANDED alias (resolver missing/failed or
    # unknown alias name) must fail loudly here, not at the provider API.
    if ":" in model and model.split(":", 1)[0] in ("openai", "anthropic", "google", "vllm_local"):
        raise StageRoleResolutionError(
            f"Stage role {key}: {model_source}={resolved.raw_model!r} is a frontier "
            f"alias that did not expand (still {model!r}) — check FRONTIER_* / "
            f"ASTRA_VERIFIER_FAMILY_MODEL env keys and the alias spelling."
        )

    if not provider:
        provider = _infer_provider(model)
        provider_source = "inferred from model" if provider else ""
    if not provider:
        raise StageRoleResolutionError(
            f"Stage role {key}: model {model!r} set via {model_source} but no "
            f"provider — set {canonical_provider_key}."
        )

    resolved.model = model
    resolved.provider = provider.lower()
    resolved.model_source = model_source
    resolved.provider_source = provider_source
    return resolved


def resolve_all_stage_roles() -> List[ResolvedRole]:
    """Resolve every role; failures land in .error instead of raising
    (the introspection endpoint must never be able to lie OR die)."""
    out: List[ResolvedRole] = []
    for key, spec in STAGE_ROLES.items():
        try:
            out.append(resolve_stage_role(key))
        except StageRoleResolutionError as exc:
            out.append(ResolvedRole(
                role=spec.role, tier=spec.tier,
                description=spec.description, derek_target=spec.derek_target,
                error=str(exc),
            ))
    return out


def validate_stage_roles_at_startup() -> int:
    """Boot-time check: log a clear ERROR banner for every unresolvable role.
    Returns the number of failures. Does not kill the backend — the chat
    assistant must still boot; the pipeline fails loudly at dispatch instead."""
    failures = 0
    for r in resolve_all_stage_roles():
        if r.error:
            failures += 1
            logger.error("[stage_roles] STARTUP CHECK FAILED — %s", r.error)
        elif r.deprecated_source:
            logger.warning("[stage_roles] %s", r.deprecation_note)
    if failures == 0:
        logger.info("[stage_roles] All %d stage roles resolve cleanly.", len(STAGE_ROLES))
    return failures


__all__ = [
    "STAGE_ROLES",
    "RoleSpec",
    "ResolvedRole",
    "StageRoleResolutionError",
    "resolve_stage_role",
    "resolve_all_stage_roles",
    "validate_stage_roles_at_startup",
]

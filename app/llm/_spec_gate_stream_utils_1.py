from __future__ import annotations
import json
import logging
import os
from app.llm.spec_gate_stream import logger
from sqlalchemy.orm import Session
from typing import Any, Dict, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
get_spec_gate_config = None
_FLOW_STATE_AVAILABLE = True
get_active_flow = None
specs_service = None


_USE_GROUNDED_SPEC_GATE = os.getenv("USE_GROUNDED_SPEC_GATE", "1") == "1"

def _safe_json_event(payload: Dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

def _resolve_spec_gate_model() -> tuple[str, str]:
    if not get_spec_gate_config:
        return "", ""
    cfg = get_spec_gate_config()
    return (cfg.provider or "", cfg.model or "")

def _get_weaver_job_description_from_flow(project_id: int) -> Optional[str]:
    """Get simple Weaver job description from flow state (v3.0)."""
    if not _FLOW_STATE_AVAILABLE or not get_active_flow:
        return None
    try:
        flow = get_active_flow(project_id)
        if flow:
            return getattr(flow, 'weaver_job_description', None)
    except Exception as e:
        logger.debug("[spec_gate_stream] get_active_flow failed: %s", e)
    return None

def _get_weaver_vision_context_from_flow(project_id: int) -> Optional[str]:
    """
    Get Weaver vision context from flow state (v2.3).
    
    v3.9.1: Vision context is extracted by Weaver from Gemini screenshot analysis
    and stored in flow state. This allows SpecGate classifier to identify
    USER-VISIBLE UI elements for intelligent refactor classification.
    """
    if not _FLOW_STATE_AVAILABLE or not get_active_flow:
        return None
    try:
        flow = get_active_flow(project_id)
        if flow:
            vision_ctx = getattr(flow, 'weaver_vision_context', None)
            if vision_ctx:
                logger.info(
                    "[spec_gate_stream] v2.3 Found vision context in flow state (%d chars)",
                    len(vision_ctx)
                )
            return vision_ctx
    except Exception as e:
        logger.debug("[spec_gate_stream] get_active_flow failed for vision context: %s", e)
    return None

def _load_latest_weaver_spec_json(db: Session, project_id: int) -> tuple[Optional[dict], dict]:
    """Load Weaver output - checks flow state first (v3.0), then DB."""
    
    # v3.0: First check flow state for simple Weaver job description
    job_description = _get_weaver_job_description_from_flow(project_id)
    if job_description:
        logger.info("[spec_gate_stream] Found Weaver job description in flow state (%d chars)", len(job_description))
        # Return job description wrapped in a format Spec Gate can use
        return {
            "job_description": job_description,
            "source": "weaver_simple",
            "title": "Job Description from Weaver",
        }, {"weaver_source": "flow_state"}
    
    # Fallback: Load from DB (v2.x behaviour)
    if not specs_service:
        return None, {}

    try:
        spec_rec = specs_service.get_latest_draft_spec(db, project_id)
        if not spec_rec:
            return None, {}

        content_json = getattr(spec_rec, "content_json", None)
        if not content_json:
            return None, {}

        try:
            weaver_spec = json.loads(content_json) if isinstance(content_json, str) else content_json
        except Exception as e:
            logger.warning("[spec_gate_stream] Failed to parse Weaver content_json: %s", e)
            return None, {}

        provenance = {
            "weaver_spec_id": getattr(spec_rec, "spec_id", None),
            "weaver_spec_hash": getattr(spec_rec, "spec_hash", None),
            "weaver_spec_version": getattr(spec_rec, "spec_version", None),
        }
        return (weaver_spec if isinstance(weaver_spec, dict) else None), provenance
    except Exception as e:
        logger.warning("[spec_gate_stream] Could not load latest Weaver draft spec: %s", e)
        return None, {}

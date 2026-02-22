import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


WEAVER_MAX_OUTPUT_TOKENS = int(os.getenv("WEAVER_MAX_OUTPUT_TOKENS", "15000"))

WEAVER_DELTA_FETCH_MULTIPLIER = int(os.getenv("WEAVER_DELTA_FETCH_MULTIPLIER", "4"))

def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "value"):
        try:
            return _to_jsonable(obj.value)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            data = {k: v for k, v in vars(obj).items() if not str(k).startswith("_")}
            return _to_jsonable(data)
        except Exception:
            pass
    return str(obj)

def _get_last_consumed_message_id_from_spec(db_spec: Any) -> Optional[int]:
    try:
        if hasattr(db_spec, "content_json") and db_spec.content_json:
            if isinstance(db_spec.content_json, dict):
                metadata = db_spec.content_json.get("metadata", {})
                if metadata and "weaver_last_consumed_message_id" in metadata:
                    val = metadata["weaver_last_consumed_message_id"]
                    if isinstance(val, int):
                        return val
                    if isinstance(val, str) and val.isdigit():
                        return int(val)
        if hasattr(db_spec, "source_message_ids") and db_spec.source_message_ids:
            ids = [int(x) for x in db_spec.source_message_ids if x]
            if ids:
                return max(ids)
        return None
    except Exception as e:
        logger.warning("[weaver_core] Failed to extract last_consumed_message_id: %s", e)
        return None

def parse_weaver_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    response_text = (response_text or "").strip()

    json_block_start = response_text.find("```json")
    if json_block_start != -1:
        json_block_end = response_text.find("```", json_block_start + 7)
        if json_block_end != -1:
            json_str = response_text[json_block_start + 7 : json_block_end].strip()
        else:
            return None, "Could not find closing ``` for JSON block"
    else:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            return None, "Could not find JSON in response"
        json_str = response_text[json_start:json_end].strip()

    try:
        spec_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    if not isinstance(spec_dict, dict):
        return None, "Top-level JSON must be an object"

    # Ensure required fields with defaults
    spec_dict.setdefault("steps", [])
    spec_dict.setdefault("weak_spots", [])
    spec_dict.setdefault("scope_constraints", [])
    spec_dict.setdefault("outputs", [])
    spec_dict.setdefault("acceptance_criteria", [])
    spec_dict.setdefault("execution_mode", None)  # v2.2: Bug 4 fix - backward compatible
    
    # CRITICAL: Ensure output file info is in acceptance_criteria
    # (acceptance_criteria survives DB serialization, outputs may not)
    outputs = spec_dict.get("outputs", [])
    content_verbatim = spec_dict.get("content_verbatim", "")
    location = spec_dict.get("location", "")
    
    if outputs:
        for out in outputs:
            name = out.get("name", "") if isinstance(out, dict) else str(out)
            if name:
                # Add an acceptance criterion that describes this output
                criterion = f"Output file '{name}'"
                if location:
                    criterion += f" at {location}"
                if content_verbatim:
                    criterion += f" contains: {content_verbatim[:100]}"
                # Avoid duplicates
                if criterion not in spec_dict["acceptance_criteria"]:
                    spec_dict["acceptance_criteria"].append(criterion)
    elif content_verbatim and location:
        # No outputs but we have content and location - synthesize acceptance criterion
        criterion = f"File at {location} contains exactly: {content_verbatim}"
        if criterion not in spec_dict["acceptance_criteria"]:
            spec_dict["acceptance_criteria"].append(criterion)

    # Log content preservation for debugging
    if spec_dict.get("content_verbatim"):
        logger.info("[weaver_core] ✓ content_verbatim: '%s'", spec_dict["content_verbatim"][:80])
    if spec_dict.get("location"):
        logger.info("[weaver_core] ✓ location: '%s'", spec_dict["location"])
    # v2.2: Log execution_mode for debugging
    if spec_dict.get("execution_mode"):
        logger.info("[weaver_core] ✓ execution_mode: '%s'", spec_dict["execution_mode"])

    summary_text = ""
    for marker in ("**Summary:**", "Summary:"):
        idx = response_text.find(marker)
        if idx != -1:
            summary_text = response_text[idx + len(marker):].strip()
            break

    return spec_dict, summary_text

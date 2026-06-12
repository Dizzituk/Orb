# FILE: app/self_model/identity.py
# Purpose: Tier 1 Identity Store — durable, structured, always-injected.
# Called-by: app.debug.executors.memory_tools, app.self_model.hooks, app.self_model.identity_capture, app.self_model.identity_format (+3 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Tier 1 Identity Store — durable, structured, always-injected.

This is the highest-priority memory layer. It holds biographical hard
facts about the user that should be present in EVERY system prompt,
regardless of session, project, or context.

Design rules:
  - Flat JSON, human-editable
  - One record per field, latest value + history
  - No confidence scoring (caller is responsible for getting it right)
  - No retrieval (always fully injected)
  - All writes audited with timestamp + source
  - Never destructively overwritten — old values archived in `history`

Schema:
{
  "fields": {
    "name":               {"value": "Taz", "set_at": "...", "source": "..."},
    "date_of_birth":      {"value": "1980-01-01", ...},
    "birthplace":         {"value": "Exeter, UK", ...},
    "current_location":   {"value": "Cornwall, UK", ...},
    ...
  },
  "history": [
    {"field": "current_location", "old": "Exeter", "new": "Cornwall",
     "changed_at": "...", "source": "..."}
  ]
}
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "self_model"
_IDENTITY_FILE = _DATA_DIR / "identity.json"

# Canonical identity fields. Adding a new field here means it gets a slot
# in the injected block and the UI tab. Order matters — this is the
# display order in the prompt.
#
# Health coaching fields appended 2026-05-25 — these are surfaced in
# every chat so ASTRA can engage coherently on training, nutrition, or
# wellbeing without re-asking baseline questions. See canonical_schema.py
# for the matching validators.
KNOWN_FIELDS: List[str] = [
    "name",
    "preferred_name",
    "date_of_birth",
    "birthplace",
    "current_location",
    "address",
    "residence_history",
    "nationality",
    "languages",
    "occupation",
    "email",
    "phone",
    # Health coaching block — kept together at the end so the injection
    # template can stylise it as one section if it ever needs to.
    "primary_sport",
    "training_goal_summary",
    "training_priorities",
    "nutrition_approach",
    "health_considerations",
]

# Fields whose VALUE is a list of structured entries rather than a
# scalar. These append/merge by entry rather than overwriting, and
# the identity block renders them as bulleted sub-entries.
#
# training_priorities and health_considerations are lists of plain
# strings rather than lists of dicts, but they live here so the
# injection layer renders them as bullets rather than as one long
# scalar value.
LIST_FIELDS: set = {
    "residence_history",
    "languages",
    "training_priorities",
    "health_considerations",
}

_lock = Lock()


# ─── Kill switch ──────────────────────────────────────────
# Phase 0 (2026-05-02): emergency global gate to stop runaway
# auto-writes corrupting identity.json. Set
# ASTRA_DISABLE_IDENTITY_AUTOWRITE=1 in the env to block every
# mutation EXCEPT explicit human/repair sources listed below.
# This is a temporary safety mechanism while Phase 3's arbiter
# is being built; once writers route through the arbiter the
# switch becomes redundant but is left in place as defence in
# depth.

_KILL_SWITCH_ALLOWLIST_PREFIXES = (
    "manual",                # default human-attributed source
    "user_manual",           # POST /self-model/identity from the UI
    "user_delete",           # DELETE /self-model/identity/{field} from the UI
    "user_confirmed",        # explicit user confirmation
    "user_correction",       # user fixed something via the API
    "audit_fix",             # repair scripts
    "claude_audit_fix",      # Claude-driven repair
    "manual_seed",           # manual seeding scripts
    "smoke_test",            # test harness
    "phase",                 # phase migration scripts (phase6_reseed etc.)
    "arbiter:confirmed",     # writes that came from accept_proposal()
)


def _kill_switch_blocks(source: str) -> bool:
    """
    Return True if the kill switch is engaged AND `source` is not on
    the human/repair allowlist. Returning True means the caller must
    refuse the mutation.
    """
    if os.getenv("ASTRA_DISABLE_IDENTITY_AUTOWRITE", "").lower() not in ("1", "true", "yes"):
        return False
    src = (source or "").strip().lower()
    return not any(src.startswith(prefix) for prefix in _KILL_SWITCH_ALLOWLIST_PREFIXES)


def _blocked_response(op: str, field: str, source: str) -> Dict[str, Any]:
    logger.warning(
        "[identity] kill switch blocked %s on %s (source=%s)",
        op, field, source,
    )
    return {
        "action": "blocked_kill_switch",
        "op": op,
        "field": field,
        "source": source,
        "reason": "ASTRA_DISABLE_IDENTITY_AUTOWRITE=1 and source not on allowlist",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IdentityField:
    value: Any
    set_at: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "set_at": self.set_at, "source": self.source}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IdentityField":
        return cls(
            value=d.get("value"),
            set_at=d.get("set_at", ""),
            source=d.get("source", ""),
        )


class IdentityStore:
    def __init__(self) -> None:
        self._fields: Dict[str, IdentityField] = {}
        self._history: List[Dict[str, Any]] = []
        self._ensure_storage()
        self._load()

    def _ensure_storage(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not _IDENTITY_FILE.exists():
            _IDENTITY_FILE.write_text(
                json.dumps({"fields": {}, "history": []}, indent=2),
                encoding="utf-8",
            )

    def _load(self) -> None:
        try:
            raw = json.loads(_IDENTITY_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("[identity] failed to load: %s", exc)
            raw = {"fields": {}, "history": []}
        self._fields = {
            k: IdentityField.from_dict(v) for k, v in raw.get("fields", {}).items()
        }
        self._history = list(raw.get("history", []))

    def _persist(self) -> None:
        payload = {
            "fields": {k: f.to_dict() for k, f in self._fields.items()},
            "history": self._history[-200:],  # cap history
        }
        _IDENTITY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ───── public API ─────────────────────────────────────────

    def get(self, field: str) -> Optional[IdentityField]:
        return self._fields.get(field)

    def get_value(self, field: str) -> Optional[Any]:
        f = self._fields.get(field)
        return f.value if f else None

    def all(self) -> Dict[str, IdentityField]:
        return dict(self._fields)

    def set(self, field: str, value: Any, source: str = "manual") -> Dict[str, Any]:
        """
        Set or update an identity field.
        Returns: {"action": "created"|"updated"|"unchanged", "field": ..., ...}
        """
        if _kill_switch_blocks(source):
            return _blocked_response("set", field, source)
        if not field or value in (None, "", []):
            return {"action": "skipped", "reason": "empty value or field"}

        with _lock:
            now = _utc_now()
            existing = self._fields.get(field)

            if existing and existing.value == value:
                # Same value — just bump timestamp, no history entry
                existing.set_at = now
                existing.source = source
                self._persist()
                logger.info("[identity] reaffirmed %s", field)
                return {"action": "unchanged", "field": field, "value": value}

            if existing:
                # Real change — record in history before overwriting
                self._history.append({
                    "field": field,
                    "old": existing.value,
                    "new": value,
                    "changed_at": now,
                    "source": source,
                })
                self._fields[field] = IdentityField(value=value, set_at=now, source=source)
                self._persist()
                logger.info(
                    "[identity] updated %s: %r -> %r (source=%s)",
                    field, existing.value, value, source,
                )
                return {
                    "action": "updated",
                    "field": field,
                    "old_value": existing.value,
                    "new_value": value,
                }

            # Brand new field
            self._fields[field] = IdentityField(value=value, set_at=now, source=source)
            self._persist()
            logger.info("[identity] created %s = %r (source=%s)", field, value, source)
            return {"action": "created", "field": field, "value": value}

    def delete(self, field: str, source: str = "manual") -> bool:
        """Remove a field. Logs to history. Returns True if removed."""
        if _kill_switch_blocks(source):
            _blocked_response("delete", field, source)
            return False
        with _lock:
            existing = self._fields.get(field)
            if not existing:
                return False
            self._history.append({
                "field": field,
                "old": existing.value,
                "new": None,
                "changed_at": _utc_now(),
                "source": source,
                "action": "deleted",
            })
            del self._fields[field]
            self._persist()
            logger.info("[identity] deleted %s (source=%s)", field, source)
            return True

    def append_to_list(
        self,
        field: str,
        entry: Dict[str, Any],
        dedupe_key: Optional[str] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        """
        Append an entry to a list-valued field (e.g. residence_history).

        If dedupe_key is provided, any existing entry with the same value
        for that key is replaced in-place rather than duplicated.

        Returns: {"action": "appended"|"updated"|"skipped", ...}
        """
        if _kill_switch_blocks(source):
            return _blocked_response("append_to_list", field, source)
        if field not in LIST_FIELDS:
            return {"action": "skipped", "reason": f"{field} is not a list field"}
        if not isinstance(entry, dict) or not entry:
            return {"action": "skipped", "reason": "entry must be a non-empty dict"}

        with _lock:
            now = _utc_now()
            existing = self._fields.get(field)
            current_list: List[Dict[str, Any]] = []
            if existing and isinstance(existing.value, list):
                current_list = list(existing.value)

            action = "appended"
            matched = False
            if dedupe_key and dedupe_key in entry:
                dedupe_val = str(entry[dedupe_key]).strip().lower()
                for i, item in enumerate(current_list):
                    if (isinstance(item, dict)
                            and str(item.get(dedupe_key, "")).strip().lower() == dedupe_val):
                        # Merge: new fields win, preserve old fields not in new entry
                        merged = {**item, **entry}
                        current_list[i] = merged
                        action = "updated"
                        matched = True
                        break
            if not matched:
                current_list.append(entry)

            self._history.append({
                "field": field,
                "action": action,
                "entry": entry,
                "changed_at": now,
                "source": source,
            })
            self._fields[field] = IdentityField(
                value=current_list, set_at=now, source=source,
            )
            self._persist()
            logger.info(
                "[identity] %s entry in %s: %r (source=%s)",
                action, field, entry, source,
            )
            return {
                "action": action,
                "field": field,
                "entry": entry,
                "list_size": len(current_list),
            }

    def remove_from_list(
        self,
        field: str,
        dedupe_key: str,
        dedupe_value: str,
        source: str = "manual",
    ) -> Dict[str, Any]:
        """Remove an entry from a list-valued field. Matches on dedupe_key/value."""
        if _kill_switch_blocks(source):
            return _blocked_response("remove_from_list", field, source)
        if field not in LIST_FIELDS:
            return {"action": "skipped", "reason": f"{field} is not a list field"}

        with _lock:
            existing = self._fields.get(field)
            if not existing or not isinstance(existing.value, list):
                return {"action": "skipped", "reason": "field not set or not a list"}

            dedupe_val = str(dedupe_value).strip().lower()
            current = list(existing.value)
            before = len(current)
            current = [
                item for item in current
                if not (isinstance(item, dict)
                        and str(item.get(dedupe_key, "")).strip().lower() == dedupe_val)
            ]
            removed = before - len(current)
            if removed == 0:
                return {"action": "skipped", "reason": "no matching entry"}

            now = _utc_now()
            self._history.append({
                "field": field,
                "action": "removed_entry",
                "matched_on": {dedupe_key: dedupe_value},
                "removed_count": removed,
                "changed_at": now,
                "source": source,
            })
            self._fields[field] = IdentityField(
                value=current, set_at=now, source=source,
            )
            self._persist()
            logger.info(
                "[identity] removed %d entries from %s matching %s=%r",
                removed, field, dedupe_key, dedupe_value,
            )
            return {
                "action": "removed",
                "field": field,
                "removed_count": removed,
                "list_size": len(current),
            }

    def history(self, field: Optional[str] = None) -> List[Dict[str, Any]]:
        if field is None:
            return list(self._history)
        return [h for h in self._history if h.get("field") == field]

    def summary(self) -> Dict[str, Any]:
        return {
            "total_fields": len(self._fields),
            "fields_set": sorted(self._fields.keys()),
            "history_entries": len(self._history),
            "storage_path": str(_IDENTITY_FILE),
        }


_store: Optional[IdentityStore] = None


def get_identity_store() -> IdentityStore:
    global _store
    if _store is None:
        _store = IdentityStore()
    return _store

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ScopeFlag:
    flag_id: str
    source_subsystem: str
    flagged_subsystem: str
    reason: str
    requires_approval: bool = True
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_scope_flags: Dict[str, ScopeFlag] = {}


def detect_scope_expansion(pass_result: Any, original_target: str) -> List[ScopeFlag]:
    touched = set(getattr(pass_result, "touched_subsystems", []) or [])
    target_chunks = getattr(pass_result, "target_chunks", []) or []
    for chunk in target_chunks:
        if isinstance(chunk, str) and "/" in chunk:
            touched.add(chunk.split("/", 1)[0])

    flags: List[ScopeFlag] = []
    for subsystem in sorted(touched):
        if subsystem and subsystem != original_target:
            flag = ScopeFlag(
                flag_id=str(uuid4()),
                source_subsystem=original_target,
                flagged_subsystem=subsystem,
                reason=f"The latest pass found evidence that {subsystem} is affecting {original_target}.",
                requires_approval=True,
            )
            _scope_flags[flag.flag_id] = flag
            flags.append(flag)
    return flags


def approve_scope_expansion(flag_id: str) -> ScopeFlag | None:
    flag = _scope_flags.get(flag_id)
    if flag:
        flag.status = "approved"
    return flag


def reject_scope_expansion(flag_id: str) -> ScopeFlag | None:
    flag = _scope_flags.get(flag_id)
    if flag:
        flag.status = "rejected"
    return flag


def list_scope_flags() -> List[ScopeFlag]:
    return list(_scope_flags.values())

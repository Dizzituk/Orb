# FILE: app/content/distribution/posting_drivers/results.py
# Purpose: PostResult value type shared across the posting drivers.
# Called-by: app.content.distribution.posting_drivers.driver_runner, .meta_driver, app.tools.social_posting_tools
# Depends-on: stdlib only
# Last-renovated: 2026-07-02
"""
PostResult — the single shape every posting driver returns, and every
chat tool relays verbatim to the model.

Deliberately flat and JSON-friendly: the chat tools hand it straight
back to a (possibly small, local) model, which then tells the user
"posted, here's the link" or "it failed at <step>, audit in <dir>".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PostResult:
    ok: bool
    platform: str
    permalink: Optional[str] = None
    audit_dir: Optional[str] = None
    failed_step: Optional[str] = None
    error: Optional[str] = None
    # Ordered breadcrumb of steps attempted, for the audit trail / debugging.
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def failure(
    platform: str,
    *,
    failed_step: str,
    error: str,
    audit_dir: Optional[str] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> PostResult:
    """Convenience constructor for a failed post."""
    return PostResult(
        ok=False,
        platform=platform,
        failed_step=failed_step,
        error=error,
        audit_dir=audit_dir,
        steps=steps or [],
    )

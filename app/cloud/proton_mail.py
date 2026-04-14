# FILE: app/cloud/proton_mail.py
"""
Proton Mail send stub.

v0.1 (2026-04-13): Stub so crash-report and other callers that import
`send_email` don't raise ImportError. Real Proton Mail integration is
not yet implemented — returns (False, "not implemented") instead of
throwing. When we get around to building real Proton Mail support
(or swap in another provider) we replace this file; callers don't need
to change.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Can be flipped on once a real implementation lands. For now, always False.
EMAIL_ENABLED = False


async def send_email(
    to_address: Optional[str],
    subject: str,
    body: str,
    attachments: Optional[list] = None,
) -> Tuple[bool, str]:
    """Send an email. Returns (success, message).

    Current implementation is a no-op stub — logs the attempt and returns
    False. Callers should handle False as "couldn't send, fall back to
    disk save" (which is what the bridge crash-report handler already does).
    """
    logger.info(
        "[proton_mail] send_email stub called: to=%r subject=%r body_len=%d (not sent)",
        to_address, subject[:60], len(body),
    )
    return False, "proton_mail send not implemented — stub returns False"

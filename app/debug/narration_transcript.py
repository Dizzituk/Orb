# FILE: app/debug/narration_transcript.py
# Purpose: Turn captured narration.intent/reflection lines into a short markdown
#          preamble prepended to the persisted assistant message, so the desktop
#          chat history shows the STEPS a phone-initiated debug run took, not
#          just its final answer (live-session plan §2.5, chat-tool-loop path
#          only -- the orchestrator path already folds its phase narration into
#          full_response, see debug_lock_stream.py's auto-route branch).
# Called-by: app.debug.debug_lock_stream
# Depends-on: stdlib only
# Last-renovated: 2026-07-01
from __future__ import annotations

from typing import List


def build_narration_preamble(lines: List[str]) -> str:
    """Plain GFM markdown, NOT raw HTML: orb-desktop's MessageBubble.tsx
    renders with react-markdown + remark-gfm only (no rehype-raw), so a
    <details>/<summary> block would render as literal escaped text or be
    silently dropped, not as a real collapsible section (verified against
    the actual component 2026-07-01). A bullet list + a rule is the reliable
    choice. Capped at 40 lines -- a long tool-heavy run's narration is a
    transcript, not a punch list."""
    if not lines:
        return ""
    body = "\n".join(f"- {l}" for l in lines[:40])
    return f"**Steps taken (phone debug session):**\n{body}\n\n---\n\n"

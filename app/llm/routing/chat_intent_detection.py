# FILE: app/llm/routing/chat_intent_detection.py
"""
Intent detection patterns and functions for chat routing.

Extracted from chat_routing.py during modularisation (2026-04-06).

This module provides regex-based intent detectors used by the chat routing
layer to decide model tier, tool access, and special-case routing (image gen,
APK builds, codebase exploration, file creation, etc.).

All functions are pure — they take a message string (and optionally a request
or DB session) and return a bool or simple value.  No LLM calls, no streaming.
"""

from __future__ import annotations

import re
from typing import Any

from app.memory import service as memory_service


# =============================================================================
# FILE CREATION INTENT  (v10.1)
# =============================================================================

_FILE_CREATION_PATTERNS = re.compile(
    r'(?:create|build|make|generate|write|design|return|produce|come\s+up\s+with|put\s+together)\s+'
    r'(?:me\s+)?(?:a\s+|an\s+|the\s+)?'
    r'(?:html|webpage|web\s*page|website|landing\s*page|page|file|document)',
    re.IGNORECASE,
)


def detect_file_creation_intent(message: str) -> bool:
    """Check if the user is asking for a file to be created."""
    return bool(_FILE_CREATION_PATTERNS.search(message))


# =============================================================================
# BUILDS CONTEXT  (v11.0)
# =============================================================================

_BUILDS_KEYWORDS = re.compile(
    r'\b(add|modify|update|extend|change|build|create|implement|work\s+on)\b'
    r'.*\b(app|copilot|co-?pilot|android|driver|feature|element|screen|module|component)\b',
    re.IGNORECASE,
)


def is_builds_context(req: Any) -> bool:
    """Check if the user is in a builds context (tab + project intent)."""
    ui_ctx = getattr(req, 'ui_context', None)
    on_builds_tab = (
        ui_ctx is not None
        and getattr(ui_ctx, 'job_type', '') == 'project_builds'
    )
    has_builds_intent = bool(_BUILDS_KEYWORDS.search(req.message))
    return on_builds_tab and has_builds_intent


# =============================================================================
# APK BUILD + DEPLOY INTENT  (v11.3)
# =============================================================================

# Question patterns — if the message looks like a question, it's not a request
_QUESTION_PATTERN = re.compile(
    r'^\s*(?:how|what|why|when|where|which|can\s+(?:you|we)|could\s+(?:you|we)|'
    r'would\s+it|is\s+(?:it|there|this)|are\s+(?:there|these)|do\s+(?:we|you)|'
    r'does\s+(?:it|this|the)|should\s+(?:we|I)|explain|tell\s+me\s+about|'
    r'what\s+(?:is|are|does|if)|describe)\b',
    re.IGNORECASE,
)

_BUILD_DEPLOY_PATTERN = re.compile(
    r'\b(build|create|make|compile|generate|assemble|deploy|upload|push|put|drop|send)\b'
    r'.{0,40}'
    r'\bapk\b',
    re.IGNORECASE,
)

_BUILD_DEPLOY_PATTERN_ALT = re.compile(
    r'\bapk\b'
    r'.{0,40}'
    r'\b(to\s+(?:the\s+)?(?:cloud|proton|drive|phone)|upload|deploy|install|push)\b',
    re.IGNORECASE,
)

_ASTRA_CMD_BUILD = re.compile(
    r'(?:astra|orb),?\s*(?:command:?)?\s*(?:build|create|make|compile|deploy|upload)\b.*\bapk\b',
    re.IGNORECASE,
)


def detect_build_deploy_intent(message: str) -> bool:
    """Check if the user is REQUESTING an APK build (not just discussing it).

    Returns True only for imperative requests containing 'apk'.
    Returns False for questions, general discussion, or sentences that
    just happen to mention build-related words near 'app' or 'phone'.
    """
    msg = message.strip()

    # Explicit Astra command — always trust
    if _ASTRA_CMD_BUILD.search(msg):
        return True

    # Questions are never build requests
    if _QUESTION_PATTERN.search(msg):
        return False

    return bool(_BUILD_DEPLOY_PATTERN.search(msg) or _BUILD_DEPLOY_PATTERN_ALT.search(msg))


# =============================================================================
# CODEBASE EXPLORATION  (v12.0)
# =============================================================================

_CODEBASE_EXPLORE_PATTERNS = re.compile(
    r'(?:'
    r'(?:inspect|examine|explore|look\s+at|have\s+a\s+look|read|review|scan|map|check)\s+'
    r'(?:\w+\s+){0,3}(?:codebase|code|source|app|project|architecture|files?|structure|tree)'
    r'|'
    r'(?:implementation|architecture|feature)\s*plan'
    r'|'
    r'(?:plan\s+(?:out|of\s+action)|spec\s+out|come\s+up\s+with\s+a\s+plan)'
    r'|'
    r'(?:what\s+(?:files?|code)\s+(?:exists?|is\s+there|do\s+we\s+have))'
    r'|'
    r'(?:current\s+(?:state|architecture|structure)\s+of)'
    r'|'
    r'(?:(?:every|each|all)\s+files?\b)'
    r')',
    re.IGNORECASE,
)


def detect_codebase_exploration(message: str) -> bool:
    """Check if the user wants ASTRA to explore/inspect a codebase.

    These requests need tool access to actually read files — without it
    the model will say 'I will inspect...' but never actually do it.
    """
    return bool(_CODEBASE_EXPLORE_PATTERNS.search(message))


# =============================================================================
# IMAGE GENERATION  (v10.3 / v3.1 / v3.2 / v10.5)
# =============================================================================
#
# v10.5 (2026-05-01): Added optional adjective slot before chart/graph/plot/
# diagram nouns so requests like "make me a bar chart", "pie chart",
# "line graph", "stacked column chart", "scatter plot" etc. trigger image
# generation instead of falling through to the document-creation path.
#
# Without this slot, the regex required the noun to sit immediately after
# the article ("make me a chart" matched, "make me a bar chart" did not),
# so chart-type-prefixed requests were silently treated as ordinary text
# document creation — which produced HTML reports instead of routing to
# gpt-image-2. See conversation 161 / 2026-05-01 for the failure trace.
#
# Adjective list covers the common chart families. Standalone nouns still
# match because the prefix group is optional.
# Optional chart-type adjective group. The `{0,3}` after the outer non-capture
# group matches 0–3 stacked adjectives, so:
#   "chart"               → 0 adjectives, noun matches "chart"
#   "bar chart"           → 1 adjective "bar ", noun matches "chart"
#   "stacked bar chart"   → 2 adjectives "stacked " + "bar ", noun matches
#   "horizontal grouped bar chart" → 3 adjectives stacked
# Capped at 3 to avoid pathological backtracking.
_CHART_ADJ_OPT = (
    r'(?:'
    r'(?:bar|pie|line|column|stacked|grouped|donut|doughnut|scatter|area|'
    r'bubble|gantt|waterfall|candlestick|horizontal|vertical|venn|tree|'
    r'network|flow|sankey|pivot|radar|heat|heat\s*map|funnel|polar|box|'
    r'violin|step)\s+'
    r'){0,3}'
)

# v10.6 (2026-06-10): Added square/card/poster/wallpaper/meme nouns and an
# optional "instagram/insta" prefix -- "make this quote into an Instagram
# Square" previously matched nothing and fell through to plain chat (see
# conversations 270/271, 2026-06-10 failure trace). The second alternation
# also gains "make" and tolerates a short noun phrase between the subject
# and "into" ("make this quote into...", "turn it all into...").
_IMAGE_NOUNS = (
    r'image|picture|photo|illustration|avatar|icon|graphic|artwork|'
    r'portrait|visual|banner|thumbnail|logo|cover|infographic|'
    r'(?:quote\s+)?card|square|poster|wallpaper|meme'
)

_IMAGE_GEN_PATTERNS = re.compile(
    r'(?:'
        r'(?:create|recreate|draw|redraw|make|remake|generate|regenerate|design|paint|sketch|render|produce|build|compile|visuali[sz]e|need|plot|put\s+together|do|show)\s+'
        r'(?:me\s+|yourself\s+)?(?:a\s+|an\s+|the\s+|another\s+|that\s+|this\s+|this\s+into\s+(?:a|an)\s+)?'
        r'(?:new\s+)?'
        r'(?:instagram\s+|insta\s+)?'
        r'(?:'
            + _IMAGE_NOUNS +
            r'|'
            + _CHART_ADJ_OPT + r'(?:chart|graph|plot|diagram)'
        + r')'
    r'|'
        r'(?:turn|convert|transform|make)\s+(?:this|that|it)(?:\s+[\w\'-]+){0,3}?\s+into\s+(?:a\s+|an\s+|the\s+)?'
        r'(?:instagram\s+|insta\s+)?'
        r'(?:'
            + _IMAGE_NOUNS +
            r'|'
            + _CHART_ADJ_OPT + r'(?:chart|graph|plot|diagram)'
        + r')'
    r')',
    re.IGNORECASE,
)


def detect_image_gen_intent(message: str) -> bool:
    """Check if the user is asking for an image to be generated."""
    return bool(_IMAGE_GEN_PATTERNS.search(message))


# =============================================================================
# IMAGE REFINEMENT  (v10.4)
# =============================================================================

# v10.6 (2026-06-10): added redesign/restyle/rework/resend, "send it
# again", and wrong-image phrasings -- "Redesign it please and send it
# again" previously matched nothing and routed to plain chat.
_IMAGE_REFINE_PATTERNS = re.compile(
    r"(?:change|modify|adjust|tweak|fix|redo|redesign|restyle|rework|recreate|regenerate|redraw|remake|re-?send|again\s+but|same\s+but|less\s+\w+|more\s+\w+|make\s+it|try\s+again|send\s+(?:it|that)\s+(?:to\s+me\s+)?again|another\s+(?:version|image|one|picture|photo|chart|graph|graphic|illustration|visual|infographic|diagram)|do\s+(?:that|this|it)\s+again|not\s+(?:quite|right)|(?:isn|wasn)'?t\s+the\s+(?:image|picture|one)|wrong\s+(?:image|picture|quote|text|colou?rs?)|too\s+\w+)",
    re.IGNORECASE,
)


def detect_image_refinement(message: str) -> bool:
    """Check if the user is asking to refine a previous image."""
    return bool(_IMAGE_REFINE_PATTERNS.search(message))


# Markers written by image_router into assistant messages on successful
# image generation. Detecting by content keeps this decoupled from any
# specific image-gen model name (gpt-image-2, gemini-2.5-flash-image,
# nano-banana-2, plotly/kaleido, etc.) so future model swaps don't break it.
_IMAGE_MSG_MARKERS = (
    "/output/images/",
    "Generated with ",
    "Generated image:",
    "rendered with Plotly",
    "[ASTRA_ARTIFACT:image:",   # bridge/phone image path (v2026-06-10)
)


def last_assistant_was_image(project_id: int, db) -> bool:
    """True if there is a recent image generation in this conversation.

    v2026-06-10 FIX (two dead-code bugs):
      1. Called memory_service.get_messages(), which does not exist --
         the AttributeError was silently swallowed and this ALWAYS
         returned False, killing refinement detection entirely.
      2. Even repaired, 'strictly the last assistant message' was wrong
         in practice: users chat about the image ('describe it for me')
         between generation and refinement. Scan the last few messages.

    Matches by content markers written by image_router / the bridge
    image path rather than by model name, so model swaps don't break it.
    """
    try:
        msgs = memory_service.list_messages(db, project_id, limit=8)
        for msg in reversed(msgs):
            if msg.role != 'assistant':
                continue
            content = msg.content or ""
            if any(marker in content for marker in _IMAGE_MSG_MARKERS):
                return True
    except Exception:
        pass
    return False

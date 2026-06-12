# FILE: app/llm/weaver_rules_engine.py
# Purpose: Weaver Classification Rules Engine — Job 8.
# Called-by: app.llm._weaver_stream_modes, app.llm.weaver_rules_inject
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Weaver Classification Rules Engine — Job 8.

Pre-classifies conversation content into Weaver output sections
using deterministic pattern matching. Reduces LLM workload by
providing already-sorted items that the LLM can verify rather
than classify from scratch.

Target: 70-75% of classification done by rules.
LLM handles: ambiguous intent, creative phrasing, multi-part
sentences that span categories.

v1.0 (2026-03-01): Initial rules engine.

Categories:
- key_requirement_functional: "I want X", "it should Y", "make it Z"
- key_requirement_technical: component names, function names, file refs
- design_preference: CSS variables, colour mentions, visual style
- constraint: "don't", "never", "not", "must not"
- established_fact: assistant codebase findings
- specgate_directive: code-answerable investigation tasks
- question_for_user: subjective/preference questions (rare)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RULES_ENGINE_BUILD_ID = "2026-03-01-v1.0-weaver-rules"


@dataclass
class ClassifiedItem:
    """A single classified piece of conversation content."""
    text: str
    category: str           # One of the category constants below
    confidence: float       # 0.0 - 1.0
    source_role: str        # "human" or "assistant"
    rule_name: str          # Which rule matched


# ─── Category constants ──────────────────────────────────────────────

CAT_REQ_FUNCTIONAL = "key_requirement_functional"
CAT_REQ_TECHNICAL = "key_requirement_technical"
CAT_DESIGN_PREF = "design_preference"
CAT_CONSTRAINT = "constraint"
CAT_ESTABLISHED_FACT = "established_fact"
CAT_SPECGATE = "specgate_directive"
CAT_QUESTION_USER = "question_for_user"
CAT_AMBIGUOUS = "ambiguous"


# ─── Rule definitions ────────────────────────────────────────────────
# Each rule: (regex, category, confidence, rule_name)
# Rules are applied per-sentence. First match wins per sentence.

_HUMAN_RULES: List[Tuple[re.Pattern, str, float, str]] = [
    # ── Constraints (check FIRST — "don't" overrides "I want") ────
    (re.compile(
        r"\b(?:don'?t|do\s+not|never|must\s+not|shouldn'?t|avoid|no\s+\w+ing)\b",
        re.I,
    ), CAT_CONSTRAINT, 0.85, "negation_constraint"),

    (re.compile(
        r"\b(?:must\s+be|has\s+to\s+be|needs?\s+to\s+be|required)\b",
        re.I,
    ), CAT_CONSTRAINT, 0.75, "hard_requirement_constraint"),

    # ── Functional requirements ───────────────────────────────────
    (re.compile(
        r"\b(?:I\s+want|I\s+need|I(?:'d|\s+would)\s+like|"
        r"it\s+should|make\s+it|add\s+(?:a|an|the)?|"
        r"include\s+(?:a|an)?|build\s+(?:a|an|me)?|"
        r"create\s+(?:a|an)?|implement|there\s+should\s+be)\b",
        re.I,
    ), CAT_REQ_FUNCTIONAL, 0.80, "user_want_statement"),

    (re.compile(
        r"\b(?:when\s+(?:the\s+user|I|you)\s+click|"
        r"clicking\s+(?:on|the)|tapping|"
        r"on\s+hover|scrolling|dragging)\b",
        re.I,
    ), CAT_REQ_FUNCTIONAL, 0.75, "interaction_requirement"),

    (re.compile(
        r"\b(?:show|display|render|list|present|fetch|load|save|store|calculate)\b"
        r".*\b(?:data|items?|results?|content|information)\b",
        re.I,
    ), CAT_REQ_FUNCTIONAL, 0.70, "data_action_requirement"),

    # ── Design preferences ────────────────────────────────────────
    (re.compile(
        r"--[\w-]+",  # CSS variable reference
    ), CAT_DESIGN_PREF, 0.90, "css_variable_mention"),

    (re.compile(
        r"\b(?:colou?r|font|theme|style|look|feel|aesthetic|visual|"
        r"dark\s+mode|light\s+mode|gradient|shadow|border|rounded|"
        r"animation|transition|opacity)\b",
        re.I,
    ), CAT_DESIGN_PREF, 0.70, "visual_style_mention"),

    # ── Technical requirements ────────────────────────────────────
    (re.compile(
        r"(?:\.tsx|\.ts|\.css|\.py|\.json)\b",
        re.I,
    ), CAT_REQ_TECHNICAL, 0.85, "file_extension_mention"),

    (re.compile(
        r"\b(?:component|endpoint|router|middleware|hook|context|"
        r"provider|reducer|state|props?|interface|type|module)\b",
        re.I,
    ), CAT_REQ_TECHNICAL, 0.65, "technical_term_mention"),

    # ── Questions / uncertainty ────────────────────────────────────
    (re.compile(
        r"\b(?:should\s+(?:it|I|we)|what\s+(?:do\s+you|should)|"
        r"(?:which|what)\s+(?:one|style|approach|way)|"
        r"or\s+(?:should|would|could))\b",
        re.I,
    ), CAT_QUESTION_USER, 0.60, "user_question_pattern"),
]

_ASSISTANT_RULES: List[Tuple[re.Pattern, str, float, str]] = [
    # ── Established facts from codebase ───────────────────────────
    (re.compile(
        r"\b(?:the\s+codebase\s+(?:uses?|has|contains)|"
        r"the\s+existing\s+(?:code|pattern|convention|system)|"
        r"I(?:'ve|\s+have)\s+(?:found|discovered|identified|noticed)|"
        r"the\s+current\s+(?:implementation|approach|pattern)|"
        r"looking\s+at\s+the\s+(?:code|source|files?)|"
        r"(?:found|discovered)\s+(?:in|at)\s+(?:the|your))\b",
        re.I,
    ), CAT_ESTABLISHED_FACT, 0.85, "codebase_analysis"),

    (re.compile(
        r"(?:src/|app/|components/|services/|utils/|lib/|hooks/)"
        r"[\w/.-]+\.(?:tsx?|py|css|json)",
    ), CAT_ESTABLISHED_FACT, 0.90, "file_path_reference"),

    (re.compile(
        r"\b(?:var\(--[\w-]+\))",
    ), CAT_ESTABLISHED_FACT, 0.90, "css_var_usage_fact"),

    (re.compile(
        r"\b(?:import|export)\s+(?:\{[^}]+\}|[\w]+)\s+from\s+['\"]",
    ), CAT_ESTABLISHED_FACT, 0.85, "import_statement_fact"),

    # ── SpecGate directives (assistant suggesting investigation) ──
    (re.compile(
        r"\b(?:we(?:'d|\s+should)\s+need\s+to\s+(?:check|determine|find|investigate)|"
        r"this\s+depends\s+on\s+(?:how|what|whether)|"
        r"(?:need|have)\s+to\s+(?:discover|scan|read)\s+the)\b",
        re.I,
    ), CAT_SPECGATE, 0.70, "investigation_suggestion"),
]


# ─── Sentence splitter ───────────────────────────────────────────────

_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"   # Split on sentence boundaries
    r"|(?<=\n)(?=\S)"            # or on newlines followed by content
)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like chunks for classification."""
    raw = _SENTENCE_RE.split(text)
    # Also split on bullet points and numbered lists
    result: List[str] = []
    for chunk in raw:
        sub = re.split(r"(?:^|\n)\s*[-•*]\s+|(?:^|\n)\s*\d+[.)]\s+", chunk)
        result.extend(s.strip() for s in sub if s.strip())
    return result


# ─── Main classification function ────────────────────────────────────

def classify_conversation(
    ramble_text: str,
    confidence_threshold: float = 0.65,
) -> Dict[str, List[ClassifiedItem]]:
    """Pre-classify conversation content into Weaver categories.

    Scans the formatted ramble text (with [Human]/[Assistant] markers)
    and applies rules to each sentence.

    Args:
        ramble_text: Formatted conversation from _format_ramble().
        confidence_threshold: Min confidence to accept a classification.

    Returns:
        Dict mapping category names to lists of ClassifiedItems.
        Also includes 'ambiguous' for sentences that didn't match.
    """
    classified: Dict[str, List[ClassifiedItem]] = {
        CAT_REQ_FUNCTIONAL: [],
        CAT_REQ_TECHNICAL: [],
        CAT_DESIGN_PREF: [],
        CAT_CONSTRAINT: [],
        CAT_ESTABLISHED_FACT: [],
        CAT_SPECGATE: [],
        CAT_QUESTION_USER: [],
        CAT_AMBIGUOUS: [],
    }

    # Parse into speaker blocks
    blocks = _parse_speaker_blocks(ramble_text)

    total_sentences = 0
    classified_count = 0

    for role, block_text in blocks:
        sentences = _split_into_sentences(block_text)
        rules = _HUMAN_RULES if role == "human" else _ASSISTANT_RULES

        for sentence in sentences:
            if len(sentence) < 5:
                continue
            total_sentences += 1

            matched = False
            for regex, category, confidence, rule_name in rules:
                if regex.search(sentence) and confidence >= confidence_threshold:
                    classified[category].append(ClassifiedItem(
                        text=sentence,
                        category=category,
                        confidence=confidence,
                        source_role=role,
                        rule_name=rule_name,
                    ))
                    matched = True
                    classified_count += 1
                    break  # First match wins

            if not matched:
                classified[CAT_AMBIGUOUS].append(ClassifiedItem(
                    text=sentence,
                    category=CAT_AMBIGUOUS,
                    confidence=0.0,
                    source_role=role,
                    rule_name="no_match",
                ))

    coverage = classified_count / total_sentences if total_sentences else 0
    logger.info(
        "[weaver_rules] Classified %d/%d sentences (%.0f%% coverage)",
        classified_count, total_sentences, coverage * 100,
    )

    return classified


def _parse_speaker_blocks(
    ramble_text: str,
) -> List[Tuple[str, str]]:
    """Parse ramble text into (role, text) tuples.

    Handles format: [Human]: text\n\n[Assistant]: text
    """
    blocks: List[Tuple[str, str]] = []
    current_role = "human"
    current_text: List[str] = []

    for line in ramble_text.split("\n"):
        if line.startswith("[Human]:"):
            if current_text:
                blocks.append((current_role, "\n".join(current_text)))
            current_role = "human"
            current_text = [line[len("[Human]:"):].strip()]
        elif line.startswith("[Assistant]:"):
            if current_text:
                blocks.append((current_role, "\n".join(current_text)))
            current_role = "assistant"
            current_text = [line[len("[Assistant]:"):].strip()]
        else:
            current_text.append(line)

    if current_text:
        blocks.append((current_role, "\n".join(current_text)))

    return blocks

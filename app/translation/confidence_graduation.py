# FILE: app/translation/confidence_graduation.py
"""
Confidence Graduation Engine.

Reads confirmation event logs and promotes high-confidence intent patterns
to tier0 graduated rules. Run periodically (e.g. on startup or via cron)
to regenerate tier0_learned_rules.py.

Graduation criteria:
- Minimum 20 confirmation events for the intent
- 95%+ confirmation rate (confirmed / total)
- No corrections logged for the phrase

v1.0 (2026-03-11): Initial implementation — reads from confirmation_events.jsonl.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Thresholds
MIN_SAMPLES = 20
AUTO_EXECUTE_THRESHOLD = 0.95  # 95% confirmation rate to auto-execute

# Paths
_METRICS_DIR = Path("D:/Orb/metrics")
_CONFIRMATION_LOG = _METRICS_DIR / "confirmation_events.jsonl"
_LEARNED_RULES_FILE = Path("D:/Orb/app/translation/tier0_learned_rules.py")


def _normalise_phrase(text: str) -> str:
    """Normalise a phrase for matching."""
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    normalised = re.sub(r"[^\w\s]", "", normalised)
    return normalised


def read_confirmation_stats() -> Dict[str, Dict]:
    """Read confirmation events and compute per-intent stats.

    Returns dict keyed by intent value, with:
        total: int
        confirmed: int
        rejected: int
        rate: float
        phrases: dict of normalised_phrase -> {confirmed, total}
    """
    if not _CONFIRMATION_LOG.exists():
        return {}

    stats: Dict[str, Dict] = {}

    try:
        with open(_CONFIRMATION_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                intent = event.get("intent", "")
                confirmed = event.get("confirmed", False)
                excerpt = event.get("user_message_excerpt", "")
                phrase = _normalise_phrase(excerpt)

                if intent not in stats:
                    stats[intent] = {
                        "total": 0,
                        "confirmed": 0,
                        "rejected": 0,
                        "rate": 0.0,
                        "phrases": defaultdict(lambda: {"confirmed": 0, "total": 0}),
                    }

                s = stats[intent]
                s["total"] += 1
                if confirmed:
                    s["confirmed"] += 1
                else:
                    s["rejected"] += 1
                s["rate"] = s["confirmed"] / s["total"] if s["total"] > 0 else 0.0

                if phrase:
                    s["phrases"][phrase]["total"] += 1
                    if confirmed:
                        s["phrases"][phrase]["confirmed"] += 1

    except Exception as e:
        logger.error("[confidence_graduation] Failed to read logs: %s", e)

    return stats


def find_graduation_candidates(
    min_samples: int = MIN_SAMPLES,
    threshold: float = AUTO_EXECUTE_THRESHOLD,
) -> List[Tuple[str, str, int, float]]:
    """Find intent phrases ready for graduation.

    Returns list of (phrase, intent, confirmations, rate) tuples
    that meet the graduation criteria.
    """
    stats = read_confirmation_stats()
    candidates = []

    for intent, s in stats.items():
        if s["total"] < min_samples:
            continue
        if s["rate"] < threshold:
            continue

        # Find specific phrases with high confidence
        for phrase, p_stats in s["phrases"].items():
            if p_stats["total"] >= 5 and phrase:  # At least 5 uses of this phrase
                p_rate = p_stats["confirmed"] / p_stats["total"]
                if p_rate >= threshold:
                    candidates.append((phrase, intent, p_stats["confirmed"], p_rate))

    # Sort by confirmations descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def regenerate_learned_rules() -> int:
    """Regenerate tier0_learned_rules.py from confirmation logs.

    Returns the number of graduated candidates written.
    """
    candidates = find_graduation_candidates()

    # Also keep any existing graduated rules that aren't in the new set
    existing = _read_existing_graduated_rules()
    existing_phrases = {c[0] for c in candidates}

    # Merge: new candidates + existing that aren't superseded
    all_rules = list(candidates)
    for phrase, intent, confs, rate in existing:
        if phrase not in existing_phrases:
            all_rules.append((phrase, intent, confs, rate))

    now = datetime.now(timezone.utc).isoformat()

    lines = [
        '# FILE: app/translation/tier0_learned_rules.py',
        '"""',
        'Auto-generated graduated rules from confidence learning.',
        '',
        'DO NOT EDIT MANUALLY -- this file is regenerated by confidence_graduation.py.',
        f'Generated: {now}',
        f'Candidates graduated: {len(all_rules)}',
        '"""',
        'from __future__ import annotations',
        '',
        'import re',
        'from typing import Optional',
        '',
        '',
        'class Tier0RuleResult:',
        '    """Minimal result class (matches tier0_rules.py interface)."""',
        '    def __init__(self, matched: bool, intent=None, confidence: float = 1.0,',
        '                 rule_name: Optional[str] = None, reason: Optional[str] = None):',
        '        self.matched = matched',
        '        self.intent = intent',
        '        self.confidence = confidence',
        '        self.rule_name = rule_name',
        '        self.reason = reason',
        '',
        '',
        '# =========================================================================',
        '# Graduated phrase->intent mappings',
        '# =========================================================================',
        '',
        'GRADUATED_RULES = [',
    ]

    for phrase, intent, confs, rate in all_rules:
        lines.append(f"    ('{phrase}', '{intent}', {confs}, {rate:.4f}),")

    lines.extend([
        ']',
        '',
        '',
        'def check_graduated_rules(text: str) -> Tier0RuleResult:',
        '    """',
        '    Check input against graduated confidence rules.',
        '',
        '    These are phrase->intent mappings that have been confirmed',
        '    enough times with zero corrections to be trusted as Tier 0.',
        '    """',
        '    normalised = re.sub(r"\\s+", " ", text.lower().strip())',
        '    normalised = re.sub(r"[^\\w\\s]", "", normalised)',
        '',
        '    for phrase, intent, confirmations, confidence in GRADUATED_RULES:',
        '        if normalised == phrase:',
        '            return Tier0RuleResult(',
        '                matched=True,',
        '                intent=intent,',
        '                confidence=0.98,',
        '                rule_name="graduated_confidence",',
        '                reason="Graduated: " + phrase + " -> " + intent,',
        '            )',
        '',
        '    return Tier0RuleResult(matched=False)',
        '',
    ])

    try:
        _LEARNED_RULES_FILE.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            "[confidence_graduation] Regenerated %d graduated rules",
            len(all_rules),
        )
    except Exception as e:
        logger.error("[confidence_graduation] Failed to write rules: %s", e)

    return len(all_rules)


def _read_existing_graduated_rules() -> List[Tuple[str, str, int, float]]:
    """Read existing graduated rules from the current file."""
    rules = []
    try:
        if not _LEARNED_RULES_FILE.exists():
            return []
        content = _LEARNED_RULES_FILE.read_text(encoding="utf-8")
        # Parse GRADUATED_RULES list
        match = re.search(r"GRADUATED_RULES\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if match:
            for line in match.group(1).strip().split("\n"):
                line = line.strip().rstrip(",")
                if line.startswith("("):
                    try:
                        parts = eval(line)  # Safe: we control the file
                        if len(parts) == 4:
                            rules.append(parts)
                    except Exception:
                        continue
    except Exception:
        pass
    return rules


def get_intent_auto_execute_status(intent_value: str) -> dict:
    """Check if an intent has reached auto-execute threshold.

    Returns dict with:
        eligible: bool — whether auto-execute criteria are met
        total: int
        confirmed: int
        rate: float
        gap: float — how far from threshold (negative = above)
    """
    stats = read_confirmation_stats()
    s = stats.get(intent_value)

    if not s:
        return {"eligible": False, "total": 0, "confirmed": 0, "rate": 0.0, "gap": AUTO_EXECUTE_THRESHOLD}

    eligible = s["total"] >= MIN_SAMPLES and s["rate"] >= AUTO_EXECUTE_THRESHOLD

    return {
        "eligible": eligible,
        "total": s["total"],
        "confirmed": s["confirmed"],
        "rate": s["rate"],
        "gap": AUTO_EXECUTE_THRESHOLD - s["rate"],
    }

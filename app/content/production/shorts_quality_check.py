# FILE: app/content/production/shorts_quality_check.py
"""
Shorts Quality Checker — AI-powered QA using Gemini 3.1 Pro Preview.

After shorts are cut, each one is fed back into Gemini Pro
for a thorough quality assessment. The AI watches the actual
short and scores it on multiple dimensions.

Quality gate:
- Score >= 7/10 → auto-approve for review
- Score 4-6 → flag for manual review with notes
- Score < 4 → auto-reject with explanation
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Use Gemini 3.1 Pro Preview for quality — best video understanding
QUALITY_MODEL = "gemini-2.0-flash"
PASS_THRESHOLD = 7
FLAG_THRESHOLD = 4


@dataclass
class QualityResult:
    """Quality assessment for a single short."""
    short_index: int
    title: str
    overall_score: float
    hook_strength: float
    pacing: float
    audio_clarity: float
    visual_quality: float
    standalone_coherence: float
    verdict: str  # "approve" | "flag" | "reject"
    issues: List[str]
    suggestions: List[str]
    summary: str


async def check_short_quality(
    video_path: str,
    title: str,
    short_index: int,
    original_prompt: str = "",
) -> Optional[QualityResult]:
    """
    Feed a cut short back into Gemini 3.1 Pro for quality assessment.

    Uploads the video and asks the model to evaluate it
    as a standalone piece of short-form content.
    """
    import httpx

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("[quality] No GOOGLE_API_KEY")
        return None

    if not os.path.exists(video_path):
        logger.error("[quality] Video not found: %s", video_path)
        return None

    # Upload the short to Gemini File API
    file_uri = await _upload_for_review(video_path, api_key)
    if not file_uri:
        return None

    prompt = _build_quality_prompt(title, original_prompt)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/{QUALITY_MODEL}:generateContent",
                params={"key": api_key},
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "fileData": {
                                        "mimeType": "video/mp4",
                                        "fileUri": file_uri,
                                    }
                                },
                                {"text": prompt},
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 2048,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract text from all parts (reasoning models may
            # have thinking parts before the text response)
            candidate = data.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])
            text_parts = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
            text = "\n".join(text_parts) if text_parts else ""

            logger.info(
                "[quality] Raw response for short %d (first 500 chars): %s",
                short_index, repr(text[:500]),
            )
            return _parse_quality_result(
                text, short_index, title
            )

    except Exception as e:
        logger.error("[quality] Gemini check failed: %s", e)
        return None


async def check_all_shorts(
    shorts: List[Dict[str, Any]],
    original_prompt: str = "",
) -> List[QualityResult]:
    """
    Quality check all shorts in a batch.

    Returns results for each short with verdict.
    """
    results = []
    for short in shorts:
        result = await check_short_quality(
            video_path=short["path"],
            title=short["title"],
            short_index=short["index"],
            original_prompt=original_prompt,
        )
        if result:
            results.append(result)
            logger.info(
                "[quality] Short %d '%s': %s (%.1f/10)",
                result.short_index,
                result.title,
                result.verdict,
                result.overall_score,
            )
        else:
            logger.warning(
                "[quality] Failed to check short %d '%s'",
                short["index"],
                short["title"],
            )

    return results


async def _upload_for_review(
    video_path: str, api_key: str,
) -> Optional[str]:
    """Upload a short video to Gemini File API for review."""
    import httpx
    import asyncio

    file_size = os.path.getsize(video_path)
    filename = os.path.basename(video_path)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Initiate upload
            init_resp = await client.post(
                "https://generativelanguage.googleapis.com/upload/"
                "v1beta/files",
                params={"key": api_key},
                headers={
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(
                        file_size
                    ),
                    "X-Goog-Upload-Header-Content-Type": "video/mp4",
                    "Content-Type": "application/json",
                },
                json={"file": {"display_name": f"qa_{filename}"}},
            )
            init_resp.raise_for_status()

            upload_url = init_resp.headers.get("X-Goog-Upload-URL")
            if not upload_url:
                return None

            with open(video_path, "rb") as f:
                video_data = f.read()

            upload_resp = await client.put(
                upload_url,
                headers={
                    "X-Goog-Upload-Command": "upload, finalize",
                    "X-Goog-Upload-Offset": "0",
                    "Content-Length": str(file_size),
                },
                content=video_data,
            )
            upload_resp.raise_for_status()
            result = upload_resp.json()

            file_uri = result.get("file", {}).get("uri")
            state = result.get("file", {}).get("state")

            # Wait for processing
            if state == "PROCESSING":
                file_name = result["file"]["name"]
                for _ in range(24):  # 2 min max
                    await asyncio.sleep(5)
                    check = await client.get(
                        f"https://generativelanguage.googleapis.com/"
                        f"v1beta/{file_name}",
                        params={"key": api_key},
                    )
                    check.raise_for_status()
                    info = check.json()
                    if info.get("state") == "ACTIVE":
                        return info.get("uri")
                    if info.get("state") == "FAILED":
                        return None

            return file_uri

    except Exception as e:
        logger.error("[quality] Upload failed: %s", e)
        return None


def _build_quality_prompt(
    title: str, original_prompt: str,
) -> str:
    """Build the quality assessment prompt."""
    context = ""
    if original_prompt:
        context = f"\nOriginal creator intent: {original_prompt}"

    return f"""You are a professional video editor and content quality reviewer. Watch this short-form video clip and evaluate it for publishing on YouTube Shorts.

Video title: {title}{context}

Score each dimension from 1-10 and provide an overall assessment.

Respond in this exact JSON format (no markdown, no backticks):
{{
    "overall_score": 7.5,
    "hook_strength": 8,
    "pacing": 7,
    "audio_clarity": 8,
    "visual_quality": 7,
    "standalone_coherence": 8,
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1"],
    "summary": "One sentence overall assessment"
}}

Scoring guide:
- hook_strength: Does the first 2-3 seconds grab attention?
- pacing: Is the rhythm engaging? Does it feel too fast/slow?
- audio_clarity: Is speech clear? Any background noise issues?
- visual_quality: Is the video sharp? Good framing?
- standalone_coherence: Does this make sense on its own without context?

Issues to flag:
- Clip starts/ends mid-sentence
- Awkward jump cuts
- Missing context that makes it confusing
- Dead air or long pauses
- Poor audio quality

Return ONLY valid JSON, no other text."""


def _parse_quality_result(
    text: str,
    short_index: int,
    title: str,
) -> Optional[QualityResult]:
    """Parse the quality check response.

    Robust parser that handles:
    - Markdown fences (```json ... ```)
    - Thinking/preamble text before JSON
    - Trailing text after JSON
    - Newlines and whitespace in strings
    """
    import re

    cleaned = text.strip()

    # Strategy 1: Extract JSON from markdown fences
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        cleaned, re.DOTALL,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Strategy 2: Find the first { and last } to extract JSON object
    if not cleaned.startswith("{"):
        brace_start = cleaned.find("{")
        if brace_start >= 0:
            # Find matching closing brace
            depth = 0
            brace_end = -1
            for i in range(brace_start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = i
                        break
            if brace_end > brace_start:
                cleaned = cleaned[brace_start:brace_end + 1]

    # Strategy 3: Fix common JSON issues from LLMs
    # Replace smart quotes
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    # Fix trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        data = json.loads(cleaned)

        overall = float(data.get("overall_score", 5))

        # Determine verdict
        if overall >= PASS_THRESHOLD:
            verdict = "approve"
        elif overall >= FLAG_THRESHOLD:
            verdict = "flag"
        else:
            verdict = "reject"

        return QualityResult(
            short_index=short_index,
            title=title,
            overall_score=overall,
            hook_strength=float(data.get("hook_strength", 5)),
            pacing=float(data.get("pacing", 5)),
            audio_clarity=float(data.get("audio_clarity", 5)),
            visual_quality=float(data.get("visual_quality", 5)),
            standalone_coherence=float(
                data.get("standalone_coherence", 5)
            ),
            verdict=verdict,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            summary=data.get("summary", ""),
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error("[quality] Parse failed: %s (cleaned: %s)", e, repr(cleaned[:300]))

        # Fallback: extract scores with regex
        try:
            overall = _extract_number(cleaned, "overall_score")
            if overall is not None:
                hook = _extract_number(cleaned, "hook_strength") or 5
                pacing = _extract_number(cleaned, "pacing") or 5
                audio = _extract_number(cleaned, "audio_clarity") or 5
                visual = _extract_number(cleaned, "visual_quality") or 5
                standalone = _extract_number(cleaned, "standalone_coherence") or 5

                if overall >= PASS_THRESHOLD:
                    verdict = "approve"
                elif overall >= FLAG_THRESHOLD:
                    verdict = "flag"
                else:
                    verdict = "reject"

                # Extract issues and summary with regex
                issues = _extract_string_list(cleaned, "issues")
                suggestions = _extract_string_list(cleaned, "suggestions")
                summary = _extract_string_value(cleaned, "summary") or ""

                logger.info(
                    "[quality] Regex fallback succeeded: %.1f/10 (%s)",
                    overall, verdict,
                )

                return QualityResult(
                    short_index=short_index,
                    title=title,
                    overall_score=overall,
                    hook_strength=hook,
                    pacing=pacing,
                    audio_clarity=audio,
                    visual_quality=visual,
                    standalone_coherence=standalone,
                    verdict=verdict,
                    issues=issues,
                    suggestions=suggestions,
                    summary=summary,
                )
        except Exception as e2:
            logger.error("[quality] Regex fallback also failed: %s", e2)

        return None


def _extract_number(text: str, key: str) -> Optional[float]:
    """Extract a numeric value for a JSON key using regex."""
    import re
    match = re.search(
        rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text
    )
    if match:
        return float(match.group(1))
    return None


def _extract_string_value(text: str, key: str) -> Optional[str]:
    """Extract a string value for a JSON key using regex."""
    import re
    match = re.search(
        rf'"{key}"\s*:\s*"([^"]*)"', text
    )
    if match:
        return match.group(1)
    return None


def _extract_string_list(text: str, key: str) -> List[str]:
    """Extract a list of strings for a JSON key using regex."""
    import re
    match = re.search(
        rf'"{key}"\s*:\s*\[(.*?)\]', text, re.DOTALL
    )
    if match:
        items = re.findall(r'"([^"]*)"', match.group(1))
        return items
    return []





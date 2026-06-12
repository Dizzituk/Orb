# Purpose: llm scraper
# Called-by: app.education.service
# Depends-on: app.education.scraper, app.education.scraper_schemas, app.llm.clients
# Last-renovated: 2026-06-11
from __future__ import annotations

import json
import logging
import re
from html import unescape
from typing import List
from urllib.parse import urlparse

import httpx

from app.education.scraper import scrape_coursera_course
from app.education.scraper_schemas import ScrapedCourseData, ScrapedSubCourse, ScrapedModuleItem
from app.llm.clients import async_call_openai

logger = logging.getLogger(__name__)


def _clean_html(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<(nav|footer|header|aside)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = unescape(html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def _extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


async def scrape_coursera_with_llm(url: str) -> ScrapedCourseData:
    parsed = urlparse(url)
    if "coursera.org" not in parsed.netloc.lower():
        raise ValueError("Only Coursera URLs are supported right now.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
        cleaned = _clean_html(response.text)[:18000]

        system_prompt = (
            "You extract structured education data from Coursera pages. "
            "Return valid JSON only. Capture professional certificate hierarchies as sub_courses."
        )
        user_prompt = (
            "Extract the course into this JSON schema: "
            "{title?: string, skills: string[], tools: string[], details: object, "
            "sub_courses: [{title: string, description?: string, modules: [{title: string, description?: string}]}]}. "
            "If the page is a professional certificate, include every sub-course you can identify. "
            "details should include useful metadata like level, duration, rating, schedule, certificate_type when present.\n\n"
            f"URL: {url}\n\nPAGE_TEXT:\n{cleaned}"
        )
        content, _usage = await async_call_openai(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1,
        )
        payload_text = _extract_json_block(content or "")
        if not payload_text:
            raise ValueError("LLM did not return JSON")
        payload = json.loads(payload_text)
        data = ScrapedCourseData.model_validate(payload)
        if data.sub_courses:
            return data
        raise ValueError("LLM returned no sub-courses")
    except Exception as exc:
        logger.warning("LLM scraper failed, falling back to regex scraper: %s", exc)
        flat_modules = scrape_coursera_course(url)
        return ScrapedCourseData(
            skills=[],
            tools=[],
            details={"fallback": True},
            sub_courses=[
                ScrapedSubCourse(
                    title="Course Curriculum",
                    description="Fallback extraction from page structure.",
                    modules=[
                        ScrapedModuleItem(title=m.title, description=m.description)
                        for m in flat_modules
                    ],
                )
            ],
        )


__all__ = ["scrape_coursera_with_llm"]

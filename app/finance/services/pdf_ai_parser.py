# FILE: app/finance/services/pdf_ai_parser.py
"""
AI-powered PDF statement parser using OpenAI vision (model from OPENAI_VISION_MODEL env).

Used as a fallback when pdfplumber table/text extraction fails,
typically for scanned PDFs or complex layouts.
Converts each page to an image and sends to OpenAI vision for extraction.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AIParseResult:
    """Result from AI vision parsing."""
    transactions: list[dict]  # [{date, description, amount, is_credit}]
    confidence: float = 0.0
    pages_processed: int = 0
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def parse_pdf_with_vision(pdf_path: str | Path) -> AIParseResult:
    """Parse a credit card statement PDF using OpenAI vision.
    
    Converts pages to images, sends to OpenAI vision with structured
    extraction prompt, returns parsed transactions.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        return AIParseResult(
            transactions=[], confidence=0,
            warnings=["pdf2image not installed. Run: pip install pdf2image"]
        )

    pdf_path = Path(pdf_path)
    result = AIParseResult(transactions=[])

    try:
        images = convert_from_path(str(pdf_path), dpi=200)
    except Exception as e:
        result.warnings.append(f"Failed to convert PDF to images: {e}")
        return result

    result.pages_processed = len(images)
    all_transactions = []

    for page_num, image in enumerate(images, 1):
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()

        # Call OpenAI vision
        txs = _call_vision_api(b64, page_num)
        if txs:
            all_transactions.extend(txs)

    result.transactions = all_transactions
    result.confidence = 0.85 if all_transactions else 0.0
    return result


def _call_vision_api(image_b64: str, page_num: int) -> list[dict]:
    """Send a page image to OpenAI vision for transaction extraction."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("[ai_parse] No OPENAI_API_KEY set")
        return []

    client = openai.OpenAI(api_key=api_key)

    system_prompt = """You are a UK credit card statement parser. Extract ALL transactions 
from this statement page image. Return ONLY valid JSON — no markdown, no explanation.

Return format:
{
  "transactions": [
    {
      "date": "DD/MM/YYYY",
      "description": "merchant or transaction description",
      "amount": 12.34,
      "is_credit": false
    }
  ]
}

Rules:
- date: UK format DD/MM/YYYY
- amount: always positive number, no currency symbols
- is_credit: true for payments/refunds, false for purchases
- Include ALL transactions, even small ones
- Skip summary rows (opening balance, closing balance, interest)
- If no transactions on this page, return {"transactions": []}"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Extract transactions from page {page_num}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=4000,
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        # Clean any markdown wrapping
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return data.get("transactions", [])

    except Exception as e:
        logger.warning("[ai_parse] Vision API error page %d: %s", page_num, e)
        return []

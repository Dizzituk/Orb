# FILE: app/finance/services/van_pdf_parser.py
"""
Parse van finance agreement PDFs using AI vision (OCR).
Extracts: purchase price, deposit, finance amount, APR, monthly payment,
total payments, vehicle description, registration, provider.
"""

import base64
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def parse_van_finance_pdf(file_bytes: bytes, filename: str = "agreement.pdf") -> dict:
    """Parse a van finance agreement PDF using OpenAI vision OCR.
    
    Returns dict with extracted fields:
    - vehicle_description, registration, purchase_price, deposit_paid,
      finance_amount, apr, monthly_payment, total_payments, 
      finance_provider, agreement_date
    """
    import httpx

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set — needed for PDF OCR")

    # Convert PDF pages to images for vision
    images_b64 = await _pdf_to_images(file_bytes)
    if not images_b64:
        raise ValueError("Could not extract images from PDF")

    logger.info("[van_pdf] Sending %d page(s) to OpenAI vision", len(images_b64))

    # Build the vision prompt
    content = [
        {
            "type": "text",
            "text": (
                "This is a van finance / hire purchase agreement PDF. "
                "Extract the following details as JSON. Use null for any field you cannot find.\n\n"
                "Required fields:\n"
                '- "vehicle_description": string (make, model, year if shown)\n'
                '- "registration": string (vehicle registration/plate number)\n'
                '- "purchase_price": number (cash price of vehicle in GBP)\n'
                '- "deposit_paid": number (deposit/advance payment in GBP)\n'
                '- "finance_amount": number (total amount of credit/loan in GBP)\n'
                '- "apr": number (APR percentage, e.g. 41.9)\n'
                '- "monthly_payment": number (monthly instalment in GBP)\n'
                '- "total_payments": number (total number of monthly payments)\n'
                '- "total_payable": number (total amount payable over agreement)\n'
                '- "finance_provider": string (lender name e.g. Moneybarn)\n'
                '- "agreement_date": string (date agreement was signed, YYYY-MM-DD)\n'
                '- "interest_total": number (total charge for credit in GBP)\n\n'
                "Return ONLY valid JSON, no markdown, no explanation."
            ),
        }
    ]

    # Add each page as an image
    for i, img_b64 in enumerate(images_b64[:4]):  # Max 4 pages
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high",
            },
        })

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini"),
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1000,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    raw = data["choices"][0]["message"]["content"].strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    parsed = json.loads(raw)
    logger.info("[van_pdf] Extracted: %s", json.dumps(parsed, indent=2))
    return parsed


async def _pdf_to_images(file_bytes: bytes) -> list[str]:
    """Convert PDF bytes to list of base64-encoded PNG images."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Fallback: try pdf2image
        return await _pdf_to_images_fallback(file_bytes)

    images = []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        for page_num in range(min(doc.page_count, 4)):
            page = doc[page_num]
            # Render at 2x for OCR quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            images.append(base64.b64encode(img_bytes).decode())
        doc.close()
    finally:
        os.unlink(tmp_path)

    return images


async def _pdf_to_images_fallback(file_bytes: bytes) -> list[str]:
    """Fallback using pdfplumber + PIL if PyMuPDF not available."""
    import pdfplumber
    from io import BytesIO

    images = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages[:4]:
            img = page.to_image(resolution=200)
            buf = BytesIO()
            img.save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode())

    return images

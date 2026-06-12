# FILE: scripts/legal_case/_extraction_prompt.py
# Purpose: Prompt used to classify and extract fields from a delivery-work screenshot.
# Called-by: scripts.legal_case.legal_case_extractor
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Prompt used to classify and extract fields from a delivery-work screenshot.

Kept in its own file so prompt tuning doesn't require touching the
orchestration logic, and so the same prompt can later be promoted into
ASTRA's archetype library as a reusable structured-extraction recipe.
"""
from __future__ import annotations

EXTRACTION_PROMPT = """You are analysing a screenshot from a UK self-employed delivery driver's phone or van dashcam. The image will be used as LEGAL EVIDENCE in an employment-status dispute (Leigh Day Solicitors).

Accuracy matters more than confidence. If you can't read a value clearly, return null. Do not guess.

Classify this screenshot into ONE of these types (pick the single best fit):
- route_summary_finished   : End-of-day summary showing total stops delivered, parcel counts, completion rate. Usually the "Tour Completed" style screen.
- route_summary_inprogress : Mid-route view showing partial completion (some stops done, others still pending).
- route_map                : Map view of the delivery route, stops, or geographic area.
- depot_photo              : Photo of the depot, lorry, yard, cages, or loading area.
- parcel_photo             : Photo of parcels, including TEAM LIFT / oversized items, damaged parcels, or bulky loads.
- working_time_warning     : App warning or screen about working-time limits, duty hours, or driving time.
- message_thread           : Text message, WhatsApp, or in-app chat with management, DSP operator, or other drivers.
- payslip_or_rate          : Payslip, rate statement, or earnings screen.
- fuel_receipt             : Fuel receipt, fuel pump, or filling-station evidence.
- other_work               : Clearly work-related but none of the above (e.g. PDA app home screen).
- not_work                 : NOT work-related (code editor, social media, system settings, personal photos, memes).

Then extract any VISIBLE fields. Return STRICTLY JSON matching this schema (use null for anything not visible — do not invent):

{
  "type": "<one of the classes above>",
  "confidence": "high" | "medium" | "low",
  "date_visible": "<date shown in the image, in any format, or null>",
  "time_visible": "<time shown in the image, or null>",
  "fields": {
    "stops_assigned":  <integer or null>,
    "stops_delivered": <integer or null>,
    "parcels":         <integer or null>,
    "failed":          <integer or null>,
    "hours_worked":    <float or null>,
    "route_name":      "<string or null, e.g. TO09>"
  },
  "notable_text": "<any short quotation that is directly relevant as evidence, verbatim from the image, or null>",
  "summary": "<one concise factual sentence describing what is visible on screen>"
}

RULES:
- Return ONLY the JSON object. No preamble. No code fences. No commentary.
- For not_work classification, still fill type/confidence/summary; everything else null.
- Do not speculate about context you cannot see. If the date isn't on screen, date_visible is null even if the filename hints at it.
- notable_text must be a direct readable quote from the screen, not a paraphrase.
""".strip()

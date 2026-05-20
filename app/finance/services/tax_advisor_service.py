# FILE: app/finance/services/tax_advisor_service.py
"""
AI-powered tax advisor for HMRC deductibility analysis.
Uses LLM to check transactions against UK self-employed tax rules.

Handles:
- HP/finance payment splitting (interest vs capital)
- Mixed-use asset apportionment
- Mileage rate vs actual costs method conflict detection
- Credit card payment → underlying spend mapping
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

HMRC_RULES_CONTEXT = """
You are an expert UK tax advisor for self-employed sole traders (delivery drivers).
Apply these HMRC rules precisely:

DEDUCTION METHODS (choose ONE per tax year - cannot mix):
1. MILEAGE RATE METHOD (simplified expenses):
   - 45p/mile first 10,000 business miles, 25p/mile after
   - This rate ALREADY INCLUDES: fuel, insurance, repairs, servicing, depreciation,
     finance payments (HP/PCP interest AND capital), tyres, MOT, road tax
   - If using this method, you CANNOT also claim actual vehicle costs separately
   - You CAN still claim: parking, tolls, congestion charges (these are separate)

2. ACTUAL COSTS METHOD:
   - Claim real costs: fuel receipts, insurance, repairs, servicing, road tax, MOT
   - PLUS capital allowances (AIA) on vehicle purchase price
   - PLUS interest portion of HP/PCP payments (NOT the capital repayment)
   - Must apportion for personal use (e.g. 80% business = claim 80%)
   - Requires detailed records of every cost

VEHICLE FINANCE (HP - Hire Purchase):
- Monthly payment = capital repayment + interest
- Under MILEAGE METHOD: cannot claim separately (already included in 45p rate)
- Under ACTUAL COSTS: can claim ONLY the interest portion as a business expense
- The van itself can qualify for Annual Investment Allowance (AIA) - claim full
  purchase price (up to £1M) as capital allowance in year of purchase
- If mixed use, reduce AIA claim by personal use percentage

CREDIT CARD PAYMENTS:
- Paying off a credit card is NOT an expense itself
- The individual purchases ON the credit card are the actual expenses
- Must look at what was bought, not the credit card payment
- Interest on business credit card borrowing IS deductible

FOOD AND DRINK:
- Can claim reasonable costs when working away from normal workplace
- HMRC benchmark rates: £5 (5-10 hrs), £10 (10+ hrs), £25 (overnight)
- OR claim actual costs with receipts (must be reasonable)
- Cannot claim everyday food/snacks at regular workplace

HOME OFFICE (simplified):
- 25-50 hrs/month at home: £10/month
- 51-100 hrs/month: £18/month  
- 101+ hrs/month: £26/month

CLOTHING:
- Only uniforms, protective clothing, branded workwear
- Cannot claim everyday clothing even if worn for work

IMPORTANT: Always flag if the user appears to be mixing mileage rate and actual costs,
as this is a common error that HMRC will reject.
"""


@dataclass
class TaxAdvice:
    """Result from AI tax analysis of a transaction."""
    transaction_description: str
    is_deductible: bool
    deductible_amount: float
    deductible_percentage: float
    category_suggestion: str
    expense_scope: str  # business | personal | mixed
    reasoning: str
    hmrc_rule_reference: str
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0


class TaxAdvisorService:
    """AI-powered HMRC tax deductibility analyser."""

    def __init__(self):
        self._client = AsyncOpenAI()

    async def analyse_transaction(
        self,
        description: str,
        amount: float,
        merchant: Optional[str] = None,
        user_context: Optional[str] = None,
        existing_method: str = "mileage",
    ) -> TaxAdvice:
        """
        Analyse a single transaction for HMRC deductibility.
        
        Args:
            description: Bank statement description
            amount: Transaction amount (positive)
            merchant: Merchant name if known
            user_context: Additional context from user (e.g. "this is my van HP payment")
            existing_method: Current expense method (mileage or actual_costs)
        """
        prompt = f"""Analyse this transaction for a UK self-employed delivery driver:

Transaction: {description}
Amount: £{amount:.2f}
Merchant: {merchant or 'Unknown'}
User says: {user_context or 'No additional context'}
Current expense method: {existing_method}

Respond in JSON only (no markdown, no backticks):
{{
  "is_deductible": true/false,
  "deductible_percentage": 0-100,
  "deductible_amount": number,
  "category_suggestion": "fuel|van_maintenance|van_insurance|phone_bill|api_costs|parking_tolls|food_on_road|clothing|home_office|van_finance_interest|capital_allowance|personal|other_business",
  "expense_scope": "business|personal|mixed",
  "reasoning": "Clear explanation of why this is/isn't deductible",
  "hmrc_rule_reference": "Which HMRC rule applies",
  "warnings": ["any important warnings or things to watch out for"],
  "confidence": 0.0-1.0
}}"""

        try:
            model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.4-mini")
            # GPT-5.x / o-series reasoning models reject `temperature` and
            # `max_tokens` — they use `max_completion_tokens` and the default
            # temperature only. Detect by model name prefix.
            def _is_reasoning(m: str) -> bool:
                m = (m or "").lower()
                return m.startswith(("gpt-5", "o1", "o3", "o4"))

            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": HMRC_RULES_CONTEXT},
                    {"role": "user", "content": prompt},
                ],
            }
            if _is_reasoning(model):
                kwargs["max_completion_tokens"] = 800
            else:
                kwargs["max_tokens"] = 800
                kwargs["temperature"] = 0.1

            response = await self._client.chat.completions.create(**kwargs)

            raw = response.choices[0].message.content.strip()
            # Strip any markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)

            return TaxAdvice(
                transaction_description=description,
                is_deductible=data.get("is_deductible", False),
                deductible_amount=data.get("deductible_amount", 0.0),
                deductible_percentage=data.get("deductible_percentage", 0),
                category_suggestion=data.get("category_suggestion", "other_business"),
                expense_scope=data.get("expense_scope", "personal"),
                reasoning=data.get("reasoning", ""),
                hmrc_rule_reference=data.get("hmrc_rule_reference", ""),
                warnings=data.get("warnings", []),
                confidence=data.get("confidence", 0.5),
            )

        except Exception as e:
            logger.error("[tax_advisor] AI analysis failed: %s", e)
            return TaxAdvice(
                transaction_description=description,
                is_deductible=False,
                deductible_amount=0.0,
                deductible_percentage=0,
                category_suggestion="unknown",
                expense_scope="unknown",
                reasoning=f"AI analysis failed: {str(e)}",
                hmrc_rule_reference="",
                warnings=["AI analysis failed — review manually"],
                confidence=0.0,
            )

    async def analyse_batch(
        self,
        transactions: list[dict],
        existing_method: str = "mileage",
    ) -> list[TaxAdvice]:
        """Analyse multiple transactions. Batches to avoid rate limits."""
        results = []
        for tx in transactions:
            result = await self.analyse_transaction(
                description=tx.get("description", ""),
                amount=tx.get("amount", 0.0),
                merchant=tx.get("merchant_name"),
                user_context=tx.get("user_context"),
                existing_method=existing_method,
            )
            results.append(result)
        return results

    async def check_method_conflicts(
        self,
        claimed_expenses: list[dict],
        using_mileage: bool = True,
    ) -> list[str]:
        """Check if the user is accidentally mixing mileage and actual costs."""
        conflicts = []
        if using_mileage:
            vehicle_categories = {
                "fuel", "van_maintenance", "van_insurance",
                "van_finance_interest", "road_tax",
            }
            for exp in claimed_expenses:
                cat = exp.get("category", "")
                if cat in vehicle_categories:
                    conflicts.append(
                        f"⚠️ '{exp.get('description', cat)}' (£{exp.get('amount', 0):.2f}) "
                        f"— You're using the mileage rate method, which already covers "
                        f"{cat.replace('_', ' ')}. Claiming this separately would be "
                        f"double-dipping. Either switch to actual costs method or remove this."
                    )
        return conflicts

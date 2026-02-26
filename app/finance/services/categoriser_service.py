"""
Auto-categorisation engine for transactions.
Learns merchant patterns from user confirmations.

Layered approach:
1. Exact learned pattern match (highest confidence)
2. Substring pattern match
3. Keyword hints (built-in merchant knowledge)
4. Amount heuristics (fuel range, food range, etc.)
5. Fall back to asking user
"""

from dataclasses import dataclass
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.finance.models import MerchantPattern, ExpenseCategory


# ── Result container ──────────────────────────────────────────────

@dataclass
class CategorisationResult:
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    expense_scope: str = "business"
    confidence: float = 0.0
    match_method: str = "none"
    needs_user_confirmation: bool = True
    suggested_merchant_name: Optional[str] = None


# ── Built-in keyword hints ────────────────────────────────────────
# Maps keywords found in bank descriptions to category names + scope

KEYWORD_HINTS: list[dict] = [
    # Fuel
    {"keywords": ["shell", "bp ", "esso", "texaco", "jet ", "gulf", "murco",
                   "sainsbury", "tesco fuel", "asda fuel", "morrisons fuel",
                   "total energies", "applegreen", "certas"],
     "category": "fuel", "scope": "business", "confidence": 0.90},

    # Food on road
    {"keywords": ["greggs", "subway", "mcdonald", "burger king", "kfc",
                   "costa", "starbucks", "pret a manger", "leon ", "eat.",
                   "just eat", "deliveroo", "uber eats", "dominos",
                   "co-op", "spar ", "one stop", "nisa "],
     "category": "food_on_road", "scope": "business", "confidence": 0.70},

    # Drinks on road
    {"keywords": ["vending", "coffee machine"],
     "category": "drinks_on_road", "scope": "business", "confidence": 0.65},

    # Groceries (personal)
    {"keywords": ["tesco", "asda", "aldi", "lidl", "morrisons",
                   "waitrose", "m&s food", "iceland", "farmfoods"],
     "category": "groceries", "scope": "personal", "confidence": 0.70},

    # Van maintenance
    {"keywords": ["halfords", "kwik fit", "national tyres", "ats euromaster",
                   "mr clutch", "formula one", "eurocar parts", "gsc mot",
                   "garage", "mechanic", "mot "],
     "category": "van_maintenance", "scope": "business", "confidence": 0.75},

    # Parking & tolls
    {"keywords": ["ncp ", "ringo", "paybyphone", "justpark", "parkopedia",
                   "dart charge", "m6 toll", "congestion", "parking"],
     "category": "parking_tolls", "scope": "business", "confidence": 0.80},

    # Phone
    {"keywords": ["ee ", "vodafone", "o2 ", "three ", "giffgaff", "tesco mobile",
                   "sky mobile", "virgin mobile"],
     "category": "phone_bill", "scope": "mixed", "confidence": 0.85},

    # Software & subscriptions
    {"keywords": ["spotify", "netflix", "disney+", "amazon prime", "youtube",
                   "apple.com", "google play"],
     "category": "personal_subscriptions", "scope": "personal", "confidence": 0.85},

    {"keywords": ["openai", "anthropic", "github", "digitalocean", "aws ",
                   "azure", "google cloud", "vercel", "netlify", "render"],
     "category": "api_costs", "scope": "business", "confidence": 0.90},

    # Insurance
    {"keywords": ["admiral", "direct line", "aviva", "axa", "rac ", "aa ",
                   "green flag"],
     "category": "van_insurance", "scope": "business", "confidence": 0.75},

    # Transfers (not categorised as expense)
    {"keywords": ["revolut", "monzo", "starling", "transfer to", "transfer from",
                   "standing order"],
     "category": "_transfer", "scope": "transfer", "confidence": 0.90},

    # ATM withdrawals
    {"keywords": ["cash withdrawal", "atm", "cash machine", "cashpoint"],
     "category": "_cash", "scope": "personal", "confidence": 0.60},

    # Van finance / HP payments
    {"keywords": ["moneybarn", "money barn", "black horse", "close brothers",
                   "oodle", "zuto", "carcraft finance", "motofinance"],
     "category": "van_finance", "scope": "business", "confidence": 0.85},

    # Credit card payments (flag as needing drill-down)
    {"keywords": ["credit card payment", "card payment", "barclaycard",
                   "capital one", "aqua card", "vanquis", "halifax card",
                   "natwest credit"],
     "category": "_credit_card_payment", "scope": "mixed", "confidence": 0.80},

    # Income sources
    {"keywords": ["yodel", "arrow xl", "yodel delivery", "evri", "hermes",
                   "amazon flex", "dpd", "royal mail"],
     "category": "delivery_income", "scope": "income", "confidence": 0.95},
]


# ── Amount-based heuristics ───────────────────────────────────────

AMOUNT_HINTS: list[dict] = [
    {"min": 30.0, "max": 90.0, "category": "fuel", "scope": "business",
     "confidence": 0.30, "description_must_not": ["tesco", "asda"]},
    {"min": 2.0, "max": 12.0, "category": "food_on_road", "scope": "business",
     "confidence": 0.20},
]


# ── Core categoriser ─────────────────────────────────────────────

def categorise_transaction(
    db: Session,
    description: str,
    amount: float,
    merchant_raw: Optional[str] = None,
) -> CategorisationResult:
    """
    Attempt to auto-categorise a transaction.
    Returns result with confidence score and whether user needs to confirm.
    """
    text = (merchant_raw or description or "").lower().strip()

    if not text:
        return CategorisationResult(match_method="none", needs_user_confirmation=True)

    # Layer 1: Learned patterns (exact match first, then substring)
    result = _match_learned_patterns(db, text)
    if result and result.confidence >= 0.90:
        return result
    if result and result.confidence >= 0.60:
        best_so_far = result
    else:
        best_so_far = None

    # Layer 2: Built-in keyword hints
    keyword_result = _match_keyword_hints(db, text)
    if keyword_result and (not best_so_far or keyword_result.confidence > best_so_far.confidence):
        best_so_far = keyword_result

    # Layer 3: Amount heuristics (only if nothing better found)
    if not best_so_far or best_so_far.confidence < 0.50:
        amount_result = _match_amount_hints(db, text, abs(amount))
        if amount_result and (not best_so_far or amount_result.confidence > best_so_far.confidence):
            best_so_far = amount_result

    if best_so_far:
        # High confidence: auto-apply, medium: suggest, low: ask
        best_so_far.needs_user_confirmation = best_so_far.confidence < 0.80
        return best_so_far

    return CategorisationResult(match_method="none", needs_user_confirmation=True)


# ── Layer 1: Learned patterns ────────────────────────────────────

def _match_learned_patterns(db: Session, text: str) -> Optional[CategorisationResult]:
    """Check against user-confirmed merchant patterns."""
    patterns = (
        db.query(MerchantPattern)
        .order_by(MerchantPattern.confidence_score.desc())
        .all()
    )

    for pattern in patterns:
        pat = pattern.merchant_pattern.lower()
        if pat == text or pat in text:
            category = db.query(ExpenseCategory).get(pattern.category_id)
            if not category:
                continue

            # Update match stats
            pattern.match_count = (pattern.match_count or 0) + 1
            pattern.last_matched = func.now()
            db.commit()

            return CategorisationResult(
                category_id=pattern.category_id,
                category_name=category.name,
                expense_scope=pattern.default_scope or "business",
                confidence=pattern.confidence_score,
                match_method="learned_pattern",
                needs_user_confirmation=pattern.confidence_score < 0.80,
                suggested_merchant_name=pattern.merchant_display_name,
            )

    return None


# ── Layer 2: Keyword hints ───────────────────────────────────────

def _match_keyword_hints(db: Session, text: str) -> Optional[CategorisationResult]:
    """Check against built-in keyword dictionary."""
    best = None

    for hint in KEYWORD_HINTS:
        for keyword in hint["keywords"]:
            if keyword.lower() in text:
                cat_name = hint["category"]

                # Skip transfer/cash pseudo-categories
                if cat_name.startswith("_"):
                    return CategorisationResult(
                        category_id=None,
                        category_name=cat_name,
                        expense_scope=hint["scope"],
                        confidence=hint["confidence"],
                        match_method="keyword_hint",
                        needs_user_confirmation=hint["confidence"] < 0.80,
                        suggested_merchant_name=keyword.strip(),
                    )

                category = (
                    db.query(ExpenseCategory)
                    .filter(ExpenseCategory.name == cat_name)
                    .first()
                )
                if not category:
                    continue

                if not best or hint["confidence"] > best.confidence:
                    best = CategorisationResult(
                        category_id=category.id,
                        category_name=cat_name,
                        expense_scope=hint["scope"],
                        confidence=hint["confidence"],
                        match_method="keyword_hint",
                        needs_user_confirmation=hint["confidence"] < 0.80,
                        suggested_merchant_name=keyword.strip(),
                    )
                break  # first keyword match per hint is enough

    return best


# ── Layer 3: Amount heuristics ───────────────────────────────────

def _match_amount_hints(
    db: Session, text: str, amount: float
) -> Optional[CategorisationResult]:
    """Last resort: guess based on transaction amount."""
    for hint in AMOUNT_HINTS:
        if hint["min"] <= amount <= hint["max"]:
            # Check exclusion list
            exclusions = hint.get("description_must_not", [])
            if any(ex in text for ex in exclusions):
                continue

            category = (
                db.query(ExpenseCategory)
                .filter(ExpenseCategory.name == hint["category"])
                .first()
            )
            if not category:
                continue

            return CategorisationResult(
                category_id=category.id,
                category_name=hint["category"],
                expense_scope=hint["scope"],
                confidence=hint["confidence"],
                match_method="amount_heuristic",
                needs_user_confirmation=True,
            )

    return None


# ── Learning: record user confirmations ──────────────────────────

def confirm_categorisation(
    db: Session,
    merchant_raw: str,
    category_id: int,
    expense_scope: str,
    display_name: Optional[str] = None,
) -> MerchantPattern:
    """
    Record a user-confirmed categorisation to improve future matching.
    If pattern exists, boost confidence. If new, create with base confidence.
    """
    text = merchant_raw.lower().strip()

    existing = (
        db.query(MerchantPattern)
        .filter(MerchantPattern.merchant_pattern == text)
        .first()
    )

    if existing:
        # Boost confidence (cap at 0.99)
        existing.confidence_score = min(0.99, existing.confidence_score + 0.05)
        existing.category_id = category_id
        existing.default_scope = expense_scope
        existing.match_count = (existing.match_count or 0) + 1
        if display_name:
            existing.merchant_display_name = display_name
        db.commit()
        return existing

    category = db.query(ExpenseCategory).get(category_id)
    new_pattern = MerchantPattern(
        merchant_pattern=text,
        merchant_display_name=display_name or merchant_raw,
        category_id=category_id,
        default_scope=expense_scope,
        confidence_score=0.75,  # first confirmation = decent confidence
        match_count=1,
    )
    db.add(new_pattern)
    db.commit()
    db.refresh(new_pattern)
    return new_pattern






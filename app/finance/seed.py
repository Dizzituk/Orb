# FILE: app/finance/seed.py
# Purpose: Seed data for the finance module.
# Called-by: main
# Depends-on: app.finance.models
# Last-renovated: 2026-06-11
"""
Seed data for the finance module.
Pre-configured expense categories for a self-employed delivery driver.
Aligned with HMRC SA103 self-employment supplementary pages.
"""
import logging
from datetime import date
from sqlalchemy.orm import Session
from app.finance.models import ExpenseCategory

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = [
    # Business: Vehicle
    {"name": "fuel", "display_name": "Fuel", "hmrc_category": "Car, van and travel expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "fuel-pump", "colour": "#E74C3C", "sort_order": 1},
    {"name": "vehicle_maintenance", "display_name": "Van Maintenance & Repairs", "hmrc_category": "Car, van and travel expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "wrench", "colour": "#E67E22", "sort_order": 2},
    {"name": "vehicle_insurance", "display_name": "Van Insurance", "hmrc_category": "Car, van and travel expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "shield", "colour": "#F39C12", "sort_order": 3},
    {"name": "mot_tax", "display_name": "MOT & Road Tax", "hmrc_category": "Car, van and travel expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "clipboard-check", "colour": "#D35400", "sort_order": 4},
    {"name": "parking", "display_name": "Parking & Tolls", "hmrc_category": "Car, van and travel expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "parking", "colour": "#C0392B", "sort_order": 5},
    # Business: Food
    {"name": "food_on_road", "display_name": "Food (On the Road)", "hmrc_category": "Other allowable business expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "utensils", "colour": "#27AE60", "sort_order": 10},
    {"name": "drinks_on_road", "display_name": "Drinks (On the Road)", "hmrc_category": "Other allowable business expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "coffee", "colour": "#2ECC71", "sort_order": 11},
    # Business: Phone
    {"name": "phone_bill", "display_name": "Phone Bill", "hmrc_category": "Phone, fax, stationery and other office costs", "default_scope": "mixed", "is_deductible": True, "deductible_percentage": 75.0, "icon": "smartphone", "colour": "#3498DB", "sort_order": 20},
    {"name": "phone_accessories", "display_name": "Phone Mount / Accessories", "hmrc_category": "Phone, fax, stationery and other office costs", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "smartphone", "colour": "#2980B9", "sort_order": 21},
    # Business: Tech
    {"name": "software_subscriptions", "display_name": "Software & Subscriptions (Business)", "hmrc_category": "Other allowable business expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "code", "colour": "#9B59B6", "sort_order": 30},
    {"name": "api_costs", "display_name": "API & Cloud Costs (ASTRA)", "hmrc_category": "Other allowable business expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "cloud", "colour": "#8E44AD", "sort_order": 31},
    {"name": "hardware", "display_name": "Computer / Hardware", "hmrc_category": "Other allowable business expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "laptop", "colour": "#7D3C98", "sort_order": 32},
    # Business: Other
    {"name": "work_clothing", "display_name": "Work Clothing / PPE", "hmrc_category": "Other allowable business expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "shirt", "colour": "#1ABC9C", "sort_order": 40},
    {"name": "accountant", "display_name": "Accountant Fees", "hmrc_category": "Accountancy, legal and other professional fees", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "calculator", "colour": "#16A085", "sort_order": 50},
    {"name": "insurance_pl", "display_name": "Public Liability Insurance", "hmrc_category": "Other allowable business expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "shield-check", "colour": "#148F77", "sort_order": 51},
    # Personal
    {"name": "rent", "display_name": "Rent", "hmrc_category": None, "default_scope": "personal", "is_deductible": False, "deductible_percentage": 0.0, "icon": "home", "colour": "#95A5A6", "sort_order": 100},
    {"name": "personal_insurance", "display_name": "Personal Insurance", "hmrc_category": None, "default_scope": "personal", "is_deductible": False, "deductible_percentage": 0.0, "icon": "shield", "colour": "#7F8C8D", "sort_order": 101},
    {"name": "groceries", "display_name": "Groceries", "hmrc_category": None, "default_scope": "personal", "is_deductible": False, "deductible_percentage": 0.0, "icon": "shopping-cart", "colour": "#BDC3C7", "sort_order": 102},
    {"name": "personal_subscriptions", "display_name": "Personal Subscriptions", "hmrc_category": None, "default_scope": "personal", "is_deductible": False, "deductible_percentage": 0.0, "icon": "tv", "colour": "#AAB7B8", "sort_order": 103},
    {"name": "family_payments", "display_name": "Family Payments", "hmrc_category": None, "default_scope": "personal", "is_deductible": False, "deductible_percentage": 0.0, "icon": "users", "colour": "#85929E", "sort_order": 104},
    {"name": "entertainment", "display_name": "Entertainment & Leisure", "hmrc_category": None, "default_scope": "personal", "is_deductible": False, "deductible_percentage": 0.0, "icon": "film", "colour": "#AEB6BF", "sort_order": 105},
    {"name": "other_personal", "display_name": "Other (Personal)", "hmrc_category": None, "default_scope": "personal", "is_deductible": False, "deductible_percentage": 0.0, "icon": "circle", "colour": "#D5D8DC", "sort_order": 199},
    {"name": "other_business", "display_name": "Other (Business)", "hmrc_category": "Other allowable business expenses", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "briefcase", "colour": "#5D6D7E", "sort_order": 200},
    # Income (not deductible — this is revenue, not spend)
    {"name": "delivery_income", "display_name": "Delivery Income (Yodel)", "hmrc_category": "Turnover", "default_scope": "business", "is_deductible": False, "deductible_percentage": 0.0, "icon": "truck", "colour": "#2E86AB", "sort_order": 300},
    # Van finance split (actual-costs method only)
    {"name": "van_hp_interest", "display_name": "Van HP Interest", "hmrc_category": "Interest on bank and other loans", "default_scope": "business", "is_deductible": True, "deductible_percentage": 100.0, "icon": "percent", "colour": "#8B4513", "sort_order": 60},
    {"name": "van_hp_capital", "display_name": "Van HP Capital (not deductible)", "hmrc_category": None, "default_scope": "business", "is_deductible": False, "deductible_percentage": 0.0, "icon": "archive", "colour": "#A0522D", "sort_order": 61},
]


def seed_finance_data(db: Session) -> dict:
    """Seed expense categories. Safe to call multiple times."""
    result = {"categories_created": 0}

    existing = {c.name for c in db.query(ExpenseCategory.name).all()}
    for cat_data in DEFAULT_CATEGORIES:
        if cat_data["name"] not in existing:
            db.add(ExpenseCategory(**cat_data))
            result["categories_created"] += 1

    db.commit()
    logger.info("[finance_seed] Seeded: %s", result)
    return result

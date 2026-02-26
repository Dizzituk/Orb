# FILE: app/finance/engines/hmrc_tax_engine.py
"""
HMRC tax calculation engine for UK self-employed.
Handles income tax, NI Class 2 & 4, mileage deductions, food allowance.

Tax rates isolated in TaxYearConfig dataclass — update once per April.
Rates below are for 2025-26. Verify at gov.uk before each new tax year.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class TaxYearConfig:
    """All HMRC rates and thresholds for a tax year."""
    tax_year: str = "2025-26"
    start_date: date = date(2025, 4, 6)
    end_date: date = date(2026, 4, 5)

    # Income Tax
    personal_allowance: float = 12_570.0
    personal_allowance_taper: float = 100_000.0
    basic_rate_limit: float = 50_270.0
    higher_rate_limit: float = 125_140.0
    basic_rate: float = 0.20
    higher_rate: float = 0.40
    additional_rate: float = 0.45

    # NI Self-Employed
    ni_class2_weekly: float = 3.45
    ni_class2_threshold: float = 12_570.0
    ni_class4_lower: float = 12_570.0
    ni_class4_upper: float = 50_270.0
    ni_class4_main_rate: float = 0.06
    ni_class4_additional_rate: float = 0.02

    # Mileage
    mileage_first_10k: float = 0.45
    mileage_after_10k: float = 0.25
    mileage_threshold: float = 10_000.0

    # Food allowance benchmarks
    meal_5_10_hours: float = 5.0
    meal_10_plus_hours: float = 10.0
    meal_overnight: float = 25.0

    # MTD deadlines
    mtd_q1_end: date = date(2025, 7, 5)
    mtd_q2_end: date = date(2025, 10, 5)
    mtd_q3_end: date = date(2026, 1, 5)
    mtd_q4_end: date = date(2026, 4, 5)
    mtd_q1_deadline: date = date(2025, 8, 7)
    mtd_q2_deadline: date = date(2025, 11, 7)
    mtd_q3_deadline: date = date(2026, 2, 7)
    mtd_q4_deadline: date = date(2026, 5, 7)


@dataclass
class TaxBreakdown:
    """Complete tax calculation result."""
    tax_year: str
    gross_income: float = 0.0
    total_allowable_expenses: float = 0.0
    taxable_profit: float = 0.0
    personal_allowance_used: float = 0.0
    basic_rate_tax: float = 0.0
    higher_rate_tax: float = 0.0
    additional_rate_tax: float = 0.0
    total_income_tax: float = 0.0
    ni_class2: float = 0.0
    ni_class4_main: float = 0.0
    ni_class4_additional: float = 0.0
    total_ni: float = 0.0
    total_tax_liability: float = 0.0
    effective_tax_rate: float = 0.0
    weekly_tax_aside: float = 0.0
    total_business_miles: float = 0.0
    mileage_deduction: float = 0.0
    food_allowance_days: int = 0
    food_allowance_total: float = 0.0
    home_office_weekly: float = 0.0
    home_office_total: float = 0.0
    recorded_expenses: float = 0.0


class HMRCTaxEngine:
    """Calculates tax for self-employed sole trader."""

    def __init__(self, config: Optional[TaxYearConfig] = None):
        self.config = config or TaxYearConfig()

    def calculate_income_tax(self, taxable_profit: float) -> dict:
        c = self.config
        pa = c.personal_allowance
        if taxable_profit > c.personal_allowance_taper:
            pa = max(0, pa - (taxable_profit - c.personal_allowance_taper) / 2)

        taxable = max(0, taxable_profit - pa)
        basic_band = min(taxable, c.basic_rate_limit - c.personal_allowance)
        basic_tax = max(0, basic_band) * c.basic_rate

        higher_start = c.basic_rate_limit - c.personal_allowance
        higher_band = min(max(0, taxable - higher_start), c.higher_rate_limit - c.basic_rate_limit)
        higher_tax = higher_band * c.higher_rate

        add_start = c.higher_rate_limit - c.personal_allowance
        add_band = max(0, taxable - add_start)
        add_tax = add_band * c.additional_rate

        return {
            "personal_allowance_used": pa,
            "basic_rate_tax": round(basic_tax, 2),
            "higher_rate_tax": round(higher_tax, 2),
            "additional_rate_tax": round(add_tax, 2),
            "total_income_tax": round(basic_tax + higher_tax + add_tax, 2),
        }

    def calculate_ni(self, taxable_profit: float) -> dict:
        c = self.config
        class2 = c.ni_class2_weekly * 52 if taxable_profit >= c.ni_class2_threshold else 0.0

        class4_main = 0.0
        class4_add = 0.0
        if taxable_profit > c.ni_class4_lower:
            main_band = min(taxable_profit - c.ni_class4_lower, c.ni_class4_upper - c.ni_class4_lower)
            class4_main = main_band * c.ni_class4_main_rate
            if taxable_profit > c.ni_class4_upper:
                class4_add = (taxable_profit - c.ni_class4_upper) * c.ni_class4_additional_rate

        return {
            "ni_class2": round(class2, 2),
            "ni_class4_main": round(class4_main, 2),
            "ni_class4_additional": round(class4_add, 2),
            "total_ni": round(class2 + class4_main + class4_add, 2),
        }

    def calculate_mileage(self, business_miles: float) -> dict:
        c = self.config
        if business_miles <= c.mileage_threshold:
            deduction = business_miles * c.mileage_first_10k
        else:
            deduction = (c.mileage_threshold * c.mileage_first_10k
                         + (business_miles - c.mileage_threshold) * c.mileage_after_10k)
        return {"business_miles": business_miles, "mileage_deduction": round(deduction, 2)}

    def calculate_food_allowance(self, hours_away: float, overnight: bool = False) -> float:
        c = self.config
        if overnight:
            return c.meal_overnight
        if hours_away >= 10:
            return c.meal_10_plus_hours
        if hours_away >= 5:
            return c.meal_5_10_hours
        return 0.0

    def calculate_full(
        self, gross_income: float, total_expenses: float,
        business_miles: float = 0.0, weeks_elapsed: int = 52,
    ) -> TaxBreakdown:
        mileage = self.calculate_mileage(business_miles)
        total_deductions = total_expenses + mileage["mileage_deduction"]
        taxable_profit = max(0, gross_income - total_deductions)
        it = self.calculate_income_tax(taxable_profit)
        ni = self.calculate_ni(taxable_profit)
        total_liability = it["total_income_tax"] + ni["total_ni"]

        if 0 < weeks_elapsed < 52:
            weekly_aside = (total_liability * (52 / weeks_elapsed)) / 52
        else:
            weekly_aside = total_liability / 52

        eff_rate = (total_liability / gross_income * 100) if gross_income > 0 else 0.0

        return TaxBreakdown(
            tax_year=self.config.tax_year,
            gross_income=round(gross_income, 2),
            total_allowable_expenses=round(total_deductions, 2),
            taxable_profit=round(taxable_profit, 2),
            personal_allowance_used=it["personal_allowance_used"],
            basic_rate_tax=it["basic_rate_tax"],
            higher_rate_tax=it["higher_rate_tax"],
            additional_rate_tax=it["additional_rate_tax"],
            total_income_tax=it["total_income_tax"],
            ni_class2=ni["ni_class2"],
            ni_class4_main=ni["ni_class4_main"],
            ni_class4_additional=ni["ni_class4_additional"],
            total_ni=ni["total_ni"],
            total_tax_liability=round(total_liability, 2),
            effective_tax_rate=round(eff_rate, 1),
            weekly_tax_aside=round(weekly_aside, 2),
            total_business_miles=mileage["business_miles"],
            mileage_deduction=mileage["mileage_deduction"],
        )

    def get_current_quarter(self, check_date: Optional[date] = None) -> dict:
        c = self.config
        d = check_date or date.today()
        quarters = [
            (1, c.start_date, c.mtd_q1_end, c.mtd_q1_deadline),
            (2, c.mtd_q1_end, c.mtd_q2_end, c.mtd_q2_deadline),
            (3, c.mtd_q2_end, c.mtd_q3_end, c.mtd_q3_deadline),
            (4, c.mtd_q3_end, c.mtd_q4_end, c.mtd_q4_deadline),
        ]
        for qnum, qs, qe, qd in quarters:
            if qs <= d <= qe:
                return {"quarter": qnum, "start": str(qs), "end": str(qe),
                        "deadline": str(qd), "days_until_deadline": max(0, (qd - d).days)}
        return {"quarter": None, "message": "Date outside current tax year"}


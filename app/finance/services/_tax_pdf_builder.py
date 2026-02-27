# FILE: app/finance/services/_tax_pdf_builder.py
"""
Builds the PDF tax cover sheet using ReportLab.

Called by tax_export_service.generate_tax_pdf().
Renders to a BytesIO buffer — no disk I/O.
"""
from __future__ import annotations

import io
from datetime import date
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Colours ──
NAVY = HexColor("#1B2A4A")
DARK_BLUE = HexColor("#2C3E6B")
ACCENT = HexColor("#3B82F6")
LIGHT_BG = HexColor("#F1F5F9")
GREY = HexColor("#64748B")
WHITE_BG = HexColor("#FFFFFF")
ALT_BG = HexColor("#F8FAFC")
BORDER = HexColor("#CBD5E1")


def _fm(val) -> str:
    """Format as GBP."""
    return f"\u00a3{float(val or 0):,.2f}"


def _n(val) -> str:
    """Format integer with commas."""
    return f"{int(val or 0):,}"


def build_tax_pdf(
    buf: io.BytesIO,
    tax: dict,
    work: dict,
    van: Optional[dict],
    tax_year: str,
) -> None:
    """Write a professional PDF tax cover sheet into buf."""
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm,
    )
    styles = getSampleStyleSheet()
    W = A4[0] - 40*mm

    # Custom styles
    title_s = ParagraphStyle('T', parent=styles['Title'],
        fontSize=22, textColor=NAVY, spaceAfter=2*mm, fontName='Helvetica-Bold')
    sub_s = ParagraphStyle('Sub', parent=styles['Normal'],
        fontSize=11, textColor=GREY, spaceAfter=6*mm)
    h2_s = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=14, textColor=NAVY, spaceBefore=8*mm, spaceAfter=4*mm,
        fontName='Helvetica-Bold')
    h3_s = ParagraphStyle('H3', parent=styles['Heading3'],
        fontSize=11, textColor=DARK_BLUE, spaceBefore=4*mm, spaceAfter=2*mm,
        fontName='Helvetica-Bold')
    body_s = ParagraphStyle('B', parent=styles['Normal'],
        fontSize=9.5, textColor=black, leading=13)
    small_s = ParagraphStyle('Sm', parent=styles['Normal'],
        fontSize=8, textColor=GREY, leading=10)

    story = []
    generated = date.today().strftime("%d %B %Y")
    is_actual = tax.get("cost_method") == "actual_costs"
    vc = tax.get("vehicle_costs") or {}

    # ─── HEADER ───
    header = Table([[
        Paragraph("SELF-EMPLOYMENT TAX PACK", title_s),
        Paragraph(f"Tax Year {tax_year}", ParagraphStyle('TR', parent=styles['Normal'],
            fontSize=12, textColor=ACCENT, alignment=TA_RIGHT, fontName='Helvetica-Bold'))
    ]], colWidths=[W*0.65, W*0.35])
    header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('LINEBELOW', (0,0), (-1,0), 2, ACCENT),
    ]))
    story.append(header)
    story.append(Spacer(1, 3*mm))

    method_label = "Actual Costs" if is_actual else "Mileage"
    story.append(Paragraph(
        f"Prepared: {generated}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Method: <b>{method_label}</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Status: <font color='#D97706'><b>IN PROGRESS</b></font>",
        sub_s))

    # ─── KEY FIGURES ───
    story.append(Paragraph("KEY FIGURES", h2_s))

    rows = [
        ["", "Amount", "Notes"],
        ["Gross Income (Turnover)", _fm(tax.get("gross_income")), "Yodel delivery earnings"],
        ["Other Business Expenses", _fm(tax.get("recorded_expenses")), "Non-vehicle deductible costs"],
    ]

    if is_actual and vc:
        rows.append(["Vehicle Running Costs", _fm(vc.get("total_running_costs")),
                      "Fuel, insurance, repairs, HP interest etc."])
        if vc.get("aia", 0) > 0:
            rows.append(["Annual Investment Allowance", _fm(vc.get("aia")),
                          "Van purchase — first year 100% deduction"])
    else:
        if tax.get("mileage_deduction", 0) > 0:
            miles = tax.get("total_business_miles", 0)
            rows.append(["Mileage Allowance", _fm(tax.get("mileage_deduction")),
                          f"{_n(miles)} business miles"])

    if tax.get("home_office_total", 0) > 0:
        rows.append(["Use of Home as Office", _fm(tax.get("home_office_total")),
                      f"{_fm(tax.get('home_office_weekly'))}/wk simplified rate"])

    rows.append(["Total Allowable Expenses", _fm(tax.get("total_allowable_expenses")), ""])
    rows.append(["", "", ""])
    rows.append(["Taxable Profit", _fm(tax.get("taxable_profit")), "Income minus deductions"])
    rows.append(["Personal Allowance", _fm(tax.get("personal_allowance_used")), ""])
    rows.append(["", "", ""])
    rows.append(["Income Tax", _fm(tax.get("total_income_tax")), ""])
    rows.append(["NI Class 2", _fm(tax.get("ni_class2")), ""])
    rows.append(["NI Class 4", _fm(tax.get("ni_class4_main", 0) + tax.get("ni_class4_additional", 0)), ""])
    rows.append(["TOTAL TAX LIABILITY", _fm(tax.get("total_tax_liability")),
                  f"Effective rate: {tax.get('effective_tax_rate', 0)}%"])

    key_table = Table(rows, colWidths=[W*0.40, W*0.22, W*0.38])
    tbl_style = [
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('TEXTCOLOR', (0,0), (-1,0), NAVY),
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('LINEBELOW', (0,0), (-1,0), 1, BORDER),
        ('LINEBELOW', (0,-1), (-1,-1), 1.5, NAVY),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (0,-1), 8),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,-1), (-1,-1), LIGHT_BG),
    ]
    key_table.setStyle(TableStyle(tbl_style))
    story.append(key_table)

    # ─── VEHICLE COSTS (actual method) ───
    if is_actual and vc:
        story.append(Paragraph("VEHICLE RUNNING COSTS", h2_s))

        vc_rows = [["Expense", "Amount"]]
        for label, key in [
            ("Fuel", "fuel"), ("HP Interest (deductible)", "hp_interest"),
            ("Insurance", "insurance"), ("Road Tax (DVLA)", "road_tax"),
            ("Repairs & Tyres", "repairs"), ("Servicing", "servicing"),
        ]:
            val = vc.get(key, 0)
            if val > 0:
                vc_rows.append([label, _fm(val)])
        vc_rows.append(["Running Total", _fm(vc.get("total_running_costs"))])
        if vc.get("aia", 0) > 0:
            vc_rows.append(["Annual Investment Allowance (Van)", _fm(vc.get("aia"))])

        vc_table = Table(vc_rows, colWidths=[W*0.65, W*0.35])
        vc_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9.5),
            ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
            ('ALIGN', (1,0), (1,-1), 'RIGHT'),
            ('LINEBELOW', (0,0), (-1,0), 0.5, BORDER),
            ('LINEABOVE', (0,-1), (-1,-1), 0.5, BORDER),
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(vc_table)

    # ─── MILEAGE INFO ───
    if tax.get("total_business_miles", 0) > 0:
        story.append(Spacer(1, 4*mm))
        if is_actual:
            story.append(Paragraph(
                f"<b>{_n(tax.get('total_business_miles'))} business miles tracked</b> "
                f"(not claimed — using actual costs method)",
                body_s))
        else:
            story.append(Paragraph("MILEAGE ALLOWANCE", h2_s))
            mi_rows = [
                ["", "Miles", "Rate", "Claim"],
                ["First 10,000", "10,000", "45p/mile", _fm(min(tax.get("total_business_miles", 0), 10000) * 0.45)],
            ]
            extra = max(0, tax.get("total_business_miles", 0) - 10000)
            if extra > 0:
                mi_rows.append(["Remaining", _n(extra), "25p/mile", _fm(extra * 0.25)])
            mi_rows.append(["TOTAL", _n(tax.get("total_business_miles")), "", _fm(tax.get("mileage_deduction"))])
            mi_table = Table(mi_rows, colWidths=[W*0.30, W*0.22, W*0.22, W*0.26])
            mi_table.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9.5),
                ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
                ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
                ('LINEBELOW', (0,0), (-1,0), 0.5, BORDER),
                ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(mi_table)

    # ─── PAGE 2 ───
    story.append(PageBreak())

    # ─── VAN / HP ───
    if van:
        story.append(Paragraph("VEHICLE &amp; HIRE PURCHASE", h2_s))
        van_rows = [
            ["Vehicle", van.get("description", "")],
            ["Purchase Price", _fm(van.get("purchase_price"))],
            ["Deposit", _fm(van.get("deposit"))],
            ["Finance Amount", _fm(van.get("finance_amount"))],
            ["APR", f"{van.get('apr', 0)}%"],
            ["Monthly Payment", _fm(van.get("monthly_payment"))],
            ["Payments", f"{van.get('payments_made', 0)} of {van.get('total_payments', 0)}"],
            ["Provider", van.get("provider", "")],
            ["Business Use", f"{van.get('business_use_pct', 100):.0f}%"],
            ["Cost Method", van.get("cost_method", "").replace("_", " ").title()],
        ]
        van_table = Table(van_rows, colWidths=[W*0.35, W*0.65])
        van_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9.5),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE_BG, ALT_BG]),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ]))
        story.append(van_table)
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            "<b>HP Tax Treatment:</b> Only the interest portion is deductible. "
            "Capital repayments are NOT deductible. The van purchase price may "
            "qualify for Annual Investment Allowance — discuss with your accountant.",
            body_s))

    # ─── WORK SUMMARY ───
    story.append(Paragraph("WORK LOG SUMMARY", h2_s))
    period = f"{work.get('first_date', 'N/A')} to {work.get('last_date', 'N/A')}"
    work_rows = [
        ["Period", period],
        ["Days Worked", str(work.get("days", 0))],
        ["Total Deliveries", _n(work.get("deliveries"))],
        ["Total Hours", f"{work.get('hours', 0):,.0f}"],
        ["Average per Hour", _fm(work.get("avg_per_hour"))],
        ["Total Gross (logs)", _fm(work.get("gross"))],
        ["Total Gross Income", _fm(tax.get("gross_income"))],
    ]
    work_table = Table(work_rows, colWidths=[W*0.40, W*0.60])
    work_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [WHITE_BG, ALT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
    ]))
    story.append(work_table)

    # ─── NOTES ───
    story.append(Paragraph("IMPORTANT NOTES", h2_s))
    notes = [
        "<b>Year In Progress:</b> All figures are interim — tax year ends 5 April.",
        "<b>Data Sources:</b> OCR screenshot processing, bank statement imports, credit card parsing via ASTRA.",
    ]
    if is_actual:
        notes.append(
            "<b>Actual Costs Method:</b> Vehicle running costs claimed directly. "
            "Mileage tracked but not claimed. Cannot switch mid-year."
        )
    notes.append(
        "<b>HP Capital Allowances:</b> Van purchase may be eligible for AIA "
        "(100% first-year). Separate from HP interest deduction."
    )
    notes.append(
        "<b>Disclaimer:</b> Auto-generated by ASTRA. Not professional tax advice — "
        "consult your accountant before filing."
    )
    for note in notes:
        story.append(Paragraph(f"\u2022 {note}", ParagraphStyle('Note', parent=body_s,
            leftIndent=8, spaceAfter=3*mm)))

    # ─── FOOTER ───
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph(
        f"Generated by ASTRA Financial Module on {generated}. "
        f"HMRC {tax_year} rates applied.",
        ParagraphStyle('Ft', parent=small_s, alignment=TA_CENTER)))

    doc.build(story)

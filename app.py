"""Professional Streamlit experience for lending and debt-planning simulations."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.components.debt_repayment import (
    DebtAccount,
    DebtRepaymentRequest,
    simulate_debt_repayment,
)
from src.components.loan_simulator import (
    EMPLOYMENT_TYPES,
    PRODUCT_POLICIES,
    AssetProfile,
    LoanApplication,
    simulate_loan,
)
from src.components.repayment_stress import estimate_repayment_stress


st.set_page_config(
    page_title="ClearPath | Credit Decision Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


PRODUCT_LABEL_TO_CODE = {
    policy.label: code for code, policy in PRODUCT_POLICIES.items()
}
CURRENCY_SYMBOLS = {"INR (₹)": "₹", "USD ($)": "$", "GBP (£)": "£"}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #13243a;
            --muted: #607086;
            --navy: #0c2748;
            --blue: #1769aa;
            --teal: #0b8a83;
            --line: #dbe4ee;
            --canvas: #f4f7fb;
            --card: #ffffff;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(244, 247, 251, 0.86); }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
        .block-container {
            max-width: 1240px;
            padding-top: 2.1rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.02em; }
        .brand-lockup { padding: .35rem .15rem 1rem; }
        .brand-mark {
            display: inline-grid; place-items: center; width: 38px; height: 38px;
            border-radius: 11px; background: linear-gradient(135deg, #1769aa, #0b8a83);
            color: white; font-weight: 800; margin-right: 9px;
        }
        .brand-name { color: var(--ink); font-size: 1.08rem; font-weight: 750; }
        .brand-note { color: var(--muted); font-size: .78rem; margin: .4rem 0 0 3.2rem; }
        .hero {
            background: radial-gradient(circle at 90% 15%, #1c5f86 0, #102f53 34%, #0b213d 76%);
            padding: 2.55rem 2.7rem; border-radius: 22px; color: white;
            box-shadow: 0 18px 45px rgba(19, 43, 73, .14); margin-bottom: 1.55rem;
        }
        .hero .eyebrow, .section-kicker {
            text-transform: uppercase; letter-spacing: .13em; font-size: .72rem;
            font-weight: 750; color: #76d6ce;
        }
        .hero h1 { color: white; font-size: 2.55rem; line-height: 1.08; margin: .55rem 0 .75rem; }
        .hero p { color: #dce9f6; font-size: 1.08rem; max-width: 760px; margin: 0; }
        .trust-row { margin-top: 1.45rem; display: flex; flex-wrap: wrap; gap: .7rem; }
        .trust-chip {
            border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.08);
            border-radius: 999px; padding: .36rem .72rem; font-size: .78rem; color: #eff7ff;
        }
        .section-intro { margin: .4rem 0 1.15rem; }
        .section-intro h2 { font-size: 1.8rem; margin: .3rem 0 .3rem; }
        .section-intro p { color: var(--muted); margin: 0; max-width: 820px; }
        .feature-card {
            background: var(--card); border: 1px solid var(--line); border-radius: 16px;
            padding: 1.25rem 1.25rem 1.1rem; min-height: 190px;
            box-shadow: 0 5px 18px rgba(32, 56, 85, .045);
        }
        .feature-icon {
            display: grid; place-items: center; width: 38px; height: 38px; border-radius: 10px;
            background: #eaf3fb; color: var(--blue); font-weight: 800; margin-bottom: .8rem;
        }
        .feature-card h3 { font-size: 1.02rem; margin: 0 0 .45rem; }
        .feature-card p { color: var(--muted); font-size: .9rem; line-height: 1.5; margin: 0; }
        .definition-card {
            background: var(--card); border: 1px solid var(--line); border-radius: 14px;
            padding: 1.05rem 1.1rem; min-height: 142px; height: 100%;
            box-shadow: 0 4px 14px rgba(32,56,85,.04);
        }
        .definition-card .definition-label {
            color: var(--blue); font-size: .79rem; font-weight: 750; margin-bottom: .55rem;
        }
        .definition-card .definition-copy {
            color: var(--ink); font-size: 1rem; font-weight: 650; line-height: 1.35;
            white-space: normal; overflow-wrap: anywhere;
        }
        .plain-note {
            padding: .9rem 1rem; background: #edf7f5; border: 1px solid #c9e7e2;
            border-radius: 12px; color: #175d59; margin: .65rem 0 1rem;
        }
        .caution-note {
            padding: .9rem 1rem; background: #fff8e8; border: 1px solid #f0d899;
            border-radius: 12px; color: #705319; margin: .65rem 0 1rem;
        }
        .result-banner {
            border-radius: 15px; padding: 1.1rem 1.25rem; margin: .4rem 0 1.15rem;
            border: 1px solid #bddfd8; background: #ecf8f5;
        }
        .result-banner.attention { border-color: #eed391; background: #fff8e8; }
        .result-banner.stop { border-color: #e8babb; background: #fff1f1; }
        .result-banner strong { display: block; color: var(--ink); margin-bottom: .2rem; }
        .result-banner span { color: var(--muted); }
        [data-testid="stMetric"] {
            background: white; border: 1px solid var(--line); border-radius: 14px;
            padding: 1rem 1.05rem; box-shadow: 0 4px 14px rgba(32,56,85,.04);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--ink); letter-spacing: -0.035em; }
        [data-testid="stForm"] {
            background: white; border: 1px solid var(--line); border-radius: 18px;
            padding: 1.25rem 1.35rem 1.4rem;
        }
        .stButton > button, .stFormSubmitButton > button {
            border-radius: 10px; min-height: 2.8rem; font-weight: 700;
        }
        .stFormSubmitButton > button[kind="primary"] {
            background: linear-gradient(90deg, #1769aa, #0b8a83); border: none;
        }
        [data-baseweb="tab-list"] { gap: .4rem; }
        [data-baseweb="tab"] { background: white; border-radius: 10px; padding: .45rem .9rem; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        .metric-explainer { color: var(--muted); font-size: .82rem; line-height: 1.38; margin-top: -.45rem; }
        .footer-note { color: var(--muted); font-size: .8rem; text-align: center; padding-top: 2rem; }
        @media (max-width: 700px) {
            .hero { padding: 1.7rem 1.35rem; }
            .hero h1 { font-size: 2rem; }
            .block-container { padding-top: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _money(value: float, symbol: str) -> str:
    return f"{symbol}{value:,.0f}"


def _months_label(months: int | None) -> str:
    if months is None:
        return "Not reached"
    years, remaining = divmod(months, 12)
    if years and remaining:
        return f"{years}y {remaining}m"
    if years:
        return f"{years} year{'s' if years != 1 else ''}"
    return f"{remaining} month{'s' if remaining != 1 else ''}"


def _payoff_date_label(months: int | None) -> str:
    if months is None:
        return "Increase the payment budget"
    today = date.today()
    month_index = today.month - 1 + months
    year = today.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1).strftime("%b %Y")


def _term_label(term: int) -> str:
    years = term / 12
    year_word = "year" if years == 1 else "years"
    return f"{term} months ({years:g} {year_word})"


def _section_intro(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
          <div class="section-kicker">{kicker}</div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
          <div class="feature-icon">{icon}</div>
          <h3>{title}</h3>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _definition_card(label: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="definition-card">
          <div class="definition-label">{label}</div>
          <div class="definition-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_home() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Responsible lending intelligence · Portfolio demonstration</div>
          <h1>Make borrowing decisions easier to understand.</h1>
          <p>Explore loan affordability, compare repayment structures, and build a practical debt-payoff plan—all from transparent inputs you control.</p>
          <div class="trust-row">
            <span class="trust-chip">No bureau lookup</span>
            <span class="trust-chip">Explainable calculations</span>
            <span class="trust-chip">Calibrated ML disclosed separately</span>
            <span class="trust-chip">No personal data stored</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    _section_intro(
        "Two financial journeys",
        "A clear answer first. The supporting detail when you need it.",
        "The experience separates affordability policy, historical ML context, and debt-planning mathematics so each metric has one understandable purpose.",
    )
    first, second, third = st.columns(3)
    with first:
        _feature_card(
            "01",
            "Check loan affordability",
            "See selected-term loan capacity, monthly payment fit, cash-flow buffers, and the exact checks behind the result.",
        )
    with second:
        _feature_card(
            "02",
            "Build a debt payoff plan",
            "Compare highest-interest-first and smallest-balance-first plans using balances, APRs, minimums, and your extra budget.",
        )
    with third:
        _feature_card(
            "03",
            "Understand the model",
            "Inspect where ML is used, what its probability means, how it was validated, and why it never overrides affordability.",
        )

    st.markdown("### What makes a metric useful?")
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    with metric_1:
        _definition_card("Maximum affordable EMI", "The monthly loan payment your submitted cash flow can support.")
    with metric_2:
        _definition_card("Cash-flow buffer", "Income remaining after essentials, existing debts, and the proposed EMI.")
    with metric_3:
        _definition_card("Payoff horizon", "The estimated time required to repay all submitted debt balances.")
    with metric_4:
        _definition_card("Interest saved", "The estimated borrowing cost avoided by paying debt faster.")
    st.markdown(
        '<div class="plain-note"><strong>Plain-language design:</strong> every primary number is paired with what it means, what changes it, and whether it is a calculation, policy assumption, or ML estimate.</div>',
        unsafe_allow_html=True,
    )


def _loan_status_banner(result: dict[str, object]) -> None:
    status = str(result["plan_status"])
    if status == "SELECTED_PLAN_FITS":
        css_class = "result-banner"
        title = "The selected loan plan fits the displayed affordability limits"
        detail = "The requested EMI stays within both the debt-service ceiling and the monthly safety-buffer ceiling."
    elif status in {
        "LOWER_AMOUNT_REQUIRED",
        "LONGER_TERM_REQUIRED",
        "LOWER_AMOUNT_AND_TERM_ADJUSTMENT",
        "REQUEST_DOES_NOT_FIT_AVAILABLE_TERMS",
    }:
        css_class = "result-banner attention"
        title = str(result["plan_status_label"])
        detail = "The current structure does not fit, but the calculated amount and term alternatives below show the closest workable route."
    else:
        css_class = "result-banner stop"
        title = "Basic product requirements are not currently met"
        detail = "Review age at maturity, income stability, product limit, and monthly repayment capacity in the decision details."
    st.markdown(
        f'<div class="{css_class}"><strong>{title}</strong><span>{detail}</span></div>',
        unsafe_allow_html=True,
    )


def _render_loan_results(result: dict[str, object], symbol: str) -> None:
    st.divider()
    _section_intro(
        "Your result",
        "Loan affordability snapshot",
        "This is an educational policy simulation—not an approval, offer, or bureau decision.",
    )
    _loan_status_banner(result)

    fit_col, requested_col, capacity_col, gap_col = st.columns(4)
    fit_col.metric("Loan Plan Fit", f"{float(result['loan_plan_fit_pct']):.0f}%")
    requested_col.metric("Requested-plan EMI", _money(float(result["requested_emi"]), symbol))
    capacity_col.metric("Maximum affordable EMI", _money(float(result["max_affordable_emi"]), symbol))
    gap_col.metric("Monthly EMI gap", _money(float(result["emi_shortfall"]), symbol))
    explainers = st.columns(4)
    explainers[0].markdown('<div class="metric-explainer">Share of the requested EMI supported by current cash flow. It is not an approval probability.</div>', unsafe_allow_html=True)
    explainers[1].markdown('<div class="metric-explainer">Payment for the amount, selected term, and displayed product rate.</div>', unsafe_allow_html=True)
    explainers[2].markdown('<div class="metric-explainer">Lower of debt-service capacity and cash flow after the required safety buffer.</div>', unsafe_allow_html=True)
    explainers[3].markdown('<div class="metric-explainer">Exact amount by which the requested EMI exceeds the calculated ceiling.</div>', unsafe_allow_html=True)

    fit = float(result["loan_plan_fit_pct"])
    st.progress(fit / 100, text=f"Selected-plan support: {fit:.0f}%")

    overview_tab, plans_tab, resilience_tab, checks_tab, model_tab = st.tabs(
        ["Plan & gaps", "Repayment options", "Financial resilience", "Decision details", "ML transparency"]
    )
    with overview_tab:
        selected_col, longest_col, reduction_col = st.columns(3)
        selected_col.metric(
            f"Maximum principal at {int(result['selected_term_months'])} months",
            _money(float(result["max_principal_selected_term"]), symbol),
        )
        longest_col.metric(
            "Maximum principal at longest eligible term",
            _money(float(result["max_principal_longest_term"]), symbol),
        )
        reduction_col.metric(
            "Amount reduction needed at selected term",
            _money(float(result["amount_reduction_required"]), symbol),
        )
        remaining_1, remaining_2 = st.columns(2)
        remaining_1.metric("Cash remaining after requested EMI", _money(float(result["cash_remaining"]), symbol))
        remaining_2.metric("Cash after required safety buffer", _money(float(result["cash_remaining_after_buffer"]), symbol))
        supporting_term = result["shortest_supporting_term_months"]
        if supporting_term is not None and int(supporting_term) != int(result["selected_term_months"]):
            st.info(
                f"The full request first fits at {int(supporting_term)} months, with an estimated EMI of "
                f"{_money(float(result['alternative_term_emi']), symbol)} and approximately "
                f"{_money(float(result['incremental_interest_for_alternative'] or 0), symbol)} more total interest than the selected term."
            )
        st.markdown("#### Recommended next steps")
        recommendations = result["recommendations"]
        if recommendations:
            for index, recommendation in enumerate(recommendations, start=1):
                st.markdown(f"**{index}. {recommendation['action']}**  \n{recommendation['detail']}")
        else:
            st.info("No specific recommendation was generated for this scenario.")
        st.markdown(
            '<div class="plain-note"><strong>How to use the ceiling:</strong> calculated capacity is a limit, not a target. A lower payment can leave more room for emergencies and changing expenses.</div>',
            unsafe_allow_html=True,
        )

    with plans_tab:
        plans = pd.DataFrame(result["repayment_plans"])
        if plans.empty:
            st.warning("No repayment term fits the configured age-at-maturity limit.")
        else:
            display = plans.rename(
                columns={
                    "term_months": "Term",
                    "monthly_emi": "Monthly payment",
                    "total_interest": "Total interest",
                    "total_repayment": "Total repaid",
                    "cash_remaining": "Money left monthly",
                    "cash_remaining_after_buffer": "After safety buffer",
                    "total_debt_service_ratio": "Debt-to-income after loan",
                    "affordable": "Fits policy",
                    "preferred": "Your selection",
                }
            )
            display["Term"] = display["Term"].map(lambda value: f"{value} months")
            for column in ("Monthly payment", "Total interest", "Total repaid", "Money left monthly", "After safety buffer"):
                display[column] = display[column].map(lambda value: _money(float(value), symbol))
            display["Debt-to-income after loan"] = display["Debt-to-income after loan"].map(lambda value: f"{value:.1%}")
            display["Fits policy"] = display["Fits policy"].map({True: "Yes", False: "No"})
            display["Your selection"] = display["Your selection"].map({True: "Selected", False: ""})
            st.dataframe(display, width="stretch", hide_index=True)
            st.caption("Shorter terms usually cost less overall but require a higher monthly payment. Longer terms reverse that trade-off.")

    with resilience_tab:
        foundation = result["financial_foundation"]
        assets = result["asset_resilience"]
        stress = result["stress_test"]
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Financial foundation", f"{int(foundation['score'])}/100", str(foundation["band"]).title())
        coverage = assets["emergency_coverage_months"]
        f2.metric("Emergency coverage", "N/A" if coverage is None else f"{float(coverage):.1f} months")
        f3.metric("Adjusted emergency liquidity", _money(float(assets["adjusted_emergency_liquidity"]), symbol))
        f4.metric("Rate-stress result", str(stress["resilience"]).title())
        st.caption("Assets strengthen resilience indicators only. They never increase maximum affordable EMI.")
        stress_1, stress_2, stress_3 = st.columns(3)
        stress_1.metric("Stressed rate", f"{float(stress['stressed_annual_interest_rate_percent']):.2f}%")
        stress_2.metric("Stressed EMI", _money(float(stress["stressed_emi"]), symbol))
        stress_3.metric("Cash after buffer under stress", _money(float(stress["stressed_cash_remaining_after_buffer"]), symbol))
        with st.expander("How the Financial Foundation indicator is built"):
            breakdown = pd.DataFrame(
                {"Component": foundation["breakdown"].keys(), "Points": foundation["breakdown"].values()}
            )
            st.bar_chart(breakdown.set_index("Component"), horizontal=True)
            st.caption(str(foundation["methodology"]))
        guidance = result["product_guidance"]
        if guidance["guidance_type"] in {"HOME_LTV", "VEHICLE_LTV"} and guidance["requested_ltv"] is not None:
            st.markdown("#### Down-payment and LTV guidance")
            g1, g2, g3 = st.columns(3)
            g1.metric("Requested LTV", f"{float(guidance['requested_ltv']):.1%}")
            g2.metric("Illustrative LTV guide", f"{float(guidance['recommended_ltv']):.0%}")
            g3.metric("Additional down payment", _money(float(guidance["additional_down_payment_needed"]), symbol))
        elif guidance["guidance_type"] == "PERSONAL_LOAN_SALARY_MULTIPLE":
            st.markdown("#### Personal-loan size guidance")
            st.write(
                f"The request is **{float(guidance['loan_to_monthly_income_multiple']):.1f}× monthly income**, "
                f"classified as **{str(guidance['salary_multiple_band']).replace('_', ' ').title()}** for this planning simulation."
            )

    with checks_tab:
        ratio_1, ratio_2, ratio_3 = st.columns(3)
        ratio_1.metric("Debt payments today", f"{float(result['current_debt_service_ratio']):.1%} of income")
        ratio_2.metric("Debt payments after loan", f"{float(result['requested_total_debt_service_ratio']):.1%} of income")
        ratio_3.metric("Essential expenses", f"{float(result['essential_expense_ratio']):.1%} of income")
        st.caption("Lower ratios generally leave more flexibility, but the relevant limits depend on product policy and verified circumstances.")
        checks = pd.DataFrame(result["policy_checks"])
        checks["Outcome"] = checks["passed"].map({True: "Within guide", False: "Needs attention"})
        checks = checks.rename(columns={"label": "What was checked", "observed": "Your value", "limit": "Illustrative limit"})
        st.dataframe(checks[["What was checked", "Outcome", "Your value", "Illustrative limit"]], width="stretch", hide_index=True)
        st.caption(f"Combined planning summary: {str(result['combined_decision_summary']).replace('_', ' ').title()}")

    with model_tab:
        stress = result["repayment_stress"]
        if not stress["available"]:
            st.info(str(stress["unavailable_reason"]))
        else:
            probability = float(stress["calibrated_payment_difficulty_probability"])
            probability_col, band_col = st.columns(2)
            probability_col.metric("Historical payment-difficulty estimate", f"{probability:.1%}")
            band_col.metric("Historical comparison band", str(stress["historical_stress_band"]).replace("_", " ").title())
            st.markdown(
                '<div class="caution-note"><strong>Important:</strong> this probability compares the form inputs with historical Home Credit outcomes. It does not decide eligibility and is not a bureau score.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Factors that moved the model estimate")
            for reason in stress["reason_codes"]:
                direction = "raised" if reason["direction"] == "increases_stress" else "lowered"
                st.markdown(f"- **{reason['label']}** {direction} the model estimate.")
            with st.expander("Model validation details"):
                final_metrics = stress["final_test_metrics"]
                m1, m2, m3 = st.columns(3)
                m1.metric("ROC-AUC", f"{float(final_metrics['roc_auc']):.3f}", help="How well the model ranks higher-risk cases above lower-risk cases.")
                m2.metric("PR-AUC", f"{float(final_metrics['pr_auc']):.3f}", help="Ranking quality focused on the less common payment-difficulty class.")
                m3.metric("Brier score", f"{float(final_metrics['brier_score']):.4f}", help="Probability error; lower is better.")
                st.caption(str(stress["data_scope_note"]))


def _render_loan_journey(symbol: str) -> None:
    _section_intro(
        "Borrowing",
        "Loan affordability explorer",
        "Enter a simple monthly picture. The simulator calculates an explainable ceiling, compares repayment options, and shows exactly which checks shaped the result.",
    )
    st.markdown(
        '<div class="caution-note"><strong>Before you begin:</strong> use recurring take-home income and realistic essential expenses. The result is only as useful as the submitted scenario.</div>',
        unsafe_allow_html=True,
    )
    with st.form("loan_simulation_form"):
        st.markdown("### 1 · Loan request")
        request_1, request_2 = st.columns(2)
        product_label = request_1.selectbox(
            "Loan product",
            options=list(PRODUCT_LABEL_TO_CODE),
            help="Each product has an illustrative rate, term range, and affordability policy.",
        )
        product_code = PRODUCT_LABEL_TO_CODE[product_label]
        policy = PRODUCT_POLICIES[product_code]
        requested_loan_amount = request_2.number_input(
            "Amount you want to borrow",
            min_value=1.0,
            max_value=float(policy.maximum_principal * 2),
            value=min(500_000.0, float(policy.maximum_principal)),
            step=10_000.0,
        )
        st.markdown("### 2 · About your income")
        applicant_1, applicant_2, applicant_3 = st.columns(3)
        age = applicant_1.number_input("Age", min_value=18, max_value=80, value=32, step=1)
        employment_type = applicant_2.selectbox("Primary income source", options=EMPLOYMENT_TYPES)
        income_stability_years = applicant_3.number_input("Years with stable income", min_value=0.0, max_value=60.0, value=4.0, step=0.5)
        st.markdown("### 3 · Monthly commitments")
        cash_1, cash_2, cash_3 = st.columns(3)
        monthly_net_income = cash_1.number_input("Take-home income", min_value=1.0, value=100_000.0, step=1_000.0)
        monthly_essential_expenses = cash_2.number_input(
            "Essential living expenses",
            min_value=0.0,
            value=40_000.0,
            step=1_000.0,
            help="Housing, food, utilities, transport, dependants, and other recurring essentials. Exclude debt payments.",
        )
        existing_monthly_debt_payments = cash_3.number_input(
            "Existing debt payments",
            min_value=0.0,
            value=10_000.0,
            step=1_000.0,
            help="Total scheduled monthly payments for existing loans and credit cards.",
        )
        eligible_terms = [term for term in policy.permitted_terms_months if age + term / 12 <= policy.maximum_age_at_maturity]
        preferred_term_months = st.selectbox(
            "Preferred repayment term",
            options=eligible_terms or list(policy.permitted_terms_months),
            index=0,
            format_func=_term_label,
        )
        purchase_property_value = 0.0
        vehicle_purchase_price = 0.0
        available_down_payment = 0.0
        if product_code == "home_loan":
            st.markdown("### 4 · Property and down payment")
            purchase_1, purchase_2 = st.columns(2)
            purchase_property_value = purchase_1.number_input(
                "Property purchase value (optional)", min_value=0.0, value=0.0, step=100_000.0
            )
            available_down_payment = purchase_2.number_input(
                "Available down payment (optional)", min_value=0.0, value=0.0, step=50_000.0
            )
        elif product_code == "vehicle_loan":
            st.markdown("### 4 · Vehicle and down payment")
            purchase_1, purchase_2 = st.columns(2)
            vehicle_purchase_price = purchase_1.number_input(
                "Vehicle purchase price (optional)", min_value=0.0, value=0.0, step=25_000.0
            )
            available_down_payment = purchase_2.number_input(
                "Available down payment (optional)", min_value=0.0, value=0.0, step=25_000.0
            )
        with st.expander("Assets & financial resilience (optional)"):
            st.caption("These values affect resilience and down-payment guidance only. They never raise affordable EMI.")
            asset_1, asset_2, asset_3 = st.columns(3)
            asset_cash = asset_1.number_input("Cash / savings", min_value=0.0, value=0.0, step=10_000.0)
            fixed_deposit = asset_2.number_input("Fixed deposits", min_value=0.0, value=0.0, step=10_000.0)
            debt_fund = asset_3.number_input("Debt funds", min_value=0.0, value=0.0, step=10_000.0)
            asset_4, asset_5, asset_6 = st.columns(3)
            equity = asset_4.number_input("Listed equity / equity funds", min_value=0.0, value=0.0, step=10_000.0)
            gold = asset_5.number_input("Gold", min_value=0.0, value=0.0, step=10_000.0)
            epf_ppf = asset_6.number_input("EPF / PPF", min_value=0.0, value=0.0, step=10_000.0)
            asset_7, asset_8, asset_9 = st.columns(3)
            property_value = asset_7.number_input("Existing property value", min_value=0.0, value=0.0, step=100_000.0)
            property_liability = asset_8.number_input("Property loan outstanding", min_value=0.0, value=0.0, step=100_000.0)
            nps = asset_9.number_input("NPS", min_value=0.0, value=0.0, step=10_000.0)
            asset_10, asset_11 = st.columns(2)
            vehicle_value = asset_10.number_input("Existing vehicle value", min_value=0.0, value=0.0, step=25_000.0)
            vehicle_liability = asset_11.number_input("Vehicle loan outstanding", min_value=0.0, value=0.0, step=25_000.0)
        submitted = st.form_submit_button("Calculate my affordability", type="primary", width="stretch")

    if submitted:
        try:
            application_input = LoanApplication(
                product_type=product_code,
                age=age,
                employment_type=employment_type,
                income_stability_years=income_stability_years,
                monthly_net_income=monthly_net_income,
                monthly_essential_expenses=monthly_essential_expenses,
                existing_monthly_debt_payments=existing_monthly_debt_payments,
                requested_loan_amount=requested_loan_amount,
                preferred_term_months=preferred_term_months,
                assets=AssetProfile(
                    cash=asset_cash,
                    fixed_deposit=fixed_deposit,
                    debt_fund=debt_fund,
                    equity=equity,
                    gold=gold,
                    property_value=property_value,
                    property_liability=property_liability,
                    vehicle_value=vehicle_value,
                    vehicle_liability=vehicle_liability,
                    epf_ppf=epf_ppf,
                    nps=nps,
                ),
                purchase_property_value=purchase_property_value,
                vehicle_purchase_price=vehicle_purchase_price,
                available_down_payment=available_down_payment,
            )
            result = simulate_loan(application_input)
            result["repayment_stress"] = estimate_repayment_stress(application_input, float(result["requested_emi"]))
            st.session_state["loan_simulation_result"] = result
        except ValueError as error:
            st.error(f"Please review the submitted information: {error}")
    if "loan_simulation_result" in st.session_state:
        _render_loan_results(st.session_state["loan_simulation_result"], symbol)


def _render_debt_results(selected: dict[str, object], comparison: dict[str, object], symbol: str) -> None:
    st.divider()
    _section_intro(
        "Your plan",
        "Debt-free roadmap",
        "The estimate assumes no new borrowing or missed payments and keeps your submitted monthly budget constant.",
    )
    if selected["payoff_possible"]:
        st.markdown(
            f'<div class="result-banner"><strong>Your current plan reaches a zero balance around {_payoff_date_label(selected["estimated_payoff_months"])}</strong><span>{selected["guidance"]}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-banner stop"><strong>This payment plan needs attention</strong><span>{selected["guidance"]}</span></div>',
            unsafe_allow_html=True,
        )
    payoff_col, payment_col, interest_col, savings_col = st.columns(4)
    payoff_col.metric("Estimated time to debt-free", _months_label(selected["estimated_payoff_months"]))
    payment_col.metric("Monthly plan payment", _money(float(selected["monthly_plan_payment"]), symbol))
    interest_col.metric("Estimated interest", _money(float(selected["estimated_total_interest"]), symbol))
    savings = selected["estimated_interest_saved"]
    savings_col.metric("Interest avoided", _money(float(savings), symbol) if savings is not None else "Not available")
    explainers = st.columns(4)
    explainers[0].markdown(f'<div class="metric-explainer">Estimated finish: {_payoff_date_label(selected["estimated_payoff_months"])}.</div>', unsafe_allow_html=True)
    explainers[1].markdown('<div class="metric-explainer">All submitted minimums plus your extra monthly amount.</div>', unsafe_allow_html=True)
    explainers[2].markdown('<div class="metric-explainer">Interest expected from now until the final simulated payment.</div>', unsafe_allow_html=True)
    explainers[3].markdown('<div class="metric-explainer">Compared with paying only the submitted fixed minimums.</div>', unsafe_allow_html=True)
    roadmap_tab, compare_tab, schedule_tab, assumptions_tab = st.tabs(["Action plan", "Strategy comparison", "Balance timeline", "Assumptions"])
    with roadmap_tab:
        st.markdown(f"#### Start with: {selected['first_priority_debt']}")
        st.write(str(selected["guidance"]))
        order = pd.DataFrame(selected["payoff_order"])
        if not order.empty:
            order = order.rename(columns={"debt": "Debt", "payoff_month": "Estimated payoff", "annual_interest_rate_percent": "APR"})
            order["Estimated payoff"] = order["Estimated payoff"].map(_months_label)
            order["APR"] = order["APR"].map(lambda value: f"{value:.2f}%")
            st.dataframe(order, width="stretch", hide_index=True)
        st.markdown(
            '<div class="plain-note"><strong>Rollover rule:</strong> when one balance reaches zero, keep paying the same total monthly budget and redirect the freed payment to the next debt.</div>',
            unsafe_allow_html=True,
        )
    with compare_tab:
        comparison_rows = [
            {
                "Method": plan["strategy_label"],
                "First focus": plan["first_priority_debt"],
                "Debt-free in": _months_label(plan["estimated_payoff_months"]),
                "Estimated interest": _money(float(plan["estimated_total_interest"]), symbol),
            }
            for plan in (selected, comparison)
        ]
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
        st.markdown(
            "- **Highest interest first (avalanche):** normally minimizes total interest.\n"
            "- **Smallest balance first (snowball):** creates earlier account closures, which some people find easier to sustain."
        )
    with schedule_tab:
        schedule = pd.DataFrame(selected["schedule"])
        if not schedule.empty:
            chart = schedule.set_index("month")[["remaining_balance"]].rename(columns={"remaining_balance": "Remaining debt"})
            st.area_chart(chart, color="#1769AA")
            milestones = schedule[
                (schedule["month"] == 0) | (schedule["month"] % 12 == 0) | (schedule.index == schedule.index[-1])
            ].copy()
            milestones = milestones.rename(
                columns={
                    "month": "Month",
                    "remaining_balance": "Balance remaining",
                    "payment": "Payment that month",
                    "cumulative_interest": "Interest paid to date",
                }
            )
            for column in ("Balance remaining", "Payment that month", "Interest paid to date"):
                milestones[column] = milestones[column].map(lambda value: _money(float(value), symbol))
            st.dataframe(milestones[["Month", "Balance remaining", "Payment that month", "Interest paid to date"]], width="stretch", hide_index=True)
    with assumptions_tab:
        for assumption in selected["assumptions"]:
            st.markdown(f"- {assumption}")
        st.markdown(
            '<div class="caution-note"><strong>Honest limitation:</strong> this is repayment mathematics, not ML. Without historical payment behaviour, account statements, fees, and future rate changes, the project cannot truthfully predict whether someone will follow the plan.</div>',
            unsafe_allow_html=True,
        )


def _render_debt_journey(symbol: str) -> None:
    _section_intro(
        "Repayment",
        "Multi-debt payoff planner",
        "Add current balances, APRs, and minimum payments. Then see how an extra monthly amount changes your payoff date and total interest.",
    )
    st.markdown(
        '<div class="plain-note"><strong>Best use:</strong> model credit cards, personal loans, or other fixed-rate balances. Enter APR as an annual percentage, such as 24 for 24%.</div>',
        unsafe_allow_html=True,
    )
    with st.form("debt_repayment_form"):
        setup_1, setup_2 = st.columns([1, 2])
        debt_count = setup_1.selectbox("Number of debts", options=list(range(1, 7)), index=2)
        strategy_label = setup_2.radio(
            "Payoff method",
            options=["Highest interest first", "Smallest balance first"],
            horizontal=True,
            help="Highest interest first usually minimizes cost. Smallest balance first may provide quicker visible wins.",
        )
        st.markdown("### Your debts")
        debt_values: list[tuple[str, float, float, float]] = []
        defaults = [
            ("Credit card", 120_000.0, 24.0, 6_000.0),
            ("Personal loan", 240_000.0, 12.0, 8_000.0),
            ("Store card", 45_000.0, 30.0, 3_000.0),
            ("Vehicle loan", 400_000.0, 10.0, 9_000.0),
            ("Education loan", 300_000.0, 9.0, 6_000.0),
            ("Other debt", 75_000.0, 15.0, 3_000.0),
        ]
        for index in range(debt_count):
            name_default, balance_default, rate_default, minimum_default = defaults[index]
            st.markdown(f"**Debt {index + 1}**")
            debt_1, debt_2, debt_3, debt_4 = st.columns([1.35, 1, 1, 1])
            name = debt_1.text_input("Account name", value=name_default, key=f"debt_name_{index}")
            balance = debt_2.number_input("Current balance", min_value=1.0, value=balance_default, step=1_000.0, key=f"debt_balance_{index}")
            rate = debt_3.number_input("APR (%)", min_value=0.0, max_value=100.0, value=rate_default, step=0.5, key=f"debt_rate_{index}")
            minimum = debt_4.number_input("Minimum payment", min_value=1.0, value=minimum_default, step=500.0, key=f"debt_minimum_{index}")
            debt_values.append((name, balance, rate, minimum))
        st.markdown("### Monthly payoff budget")
        extra_monthly_payment = st.number_input(
            "Extra amount available after all minimum payments",
            min_value=0.0,
            value=5_000.0,
            step=500.0,
            help="This amount is directed to one priority debt. Cleared minimum payments are then rolled forward automatically.",
        )
        submitted = st.form_submit_button("Build my payoff plan", type="primary", width="stretch")
    if submitted:
        try:
            debts = tuple(
                DebtAccount(name=name, balance=balance, annual_interest_rate=rate / 100, minimum_payment=minimum)
                for name, balance, rate, minimum in debt_values
            )
            strategy = "avalanche" if strategy_label == "Highest interest first" else "snowball"
            selected = simulate_debt_repayment(DebtRepaymentRequest(debts=debts, extra_monthly_payment=extra_monthly_payment, strategy=strategy))
            comparison_strategy = "snowball" if strategy == "avalanche" else "avalanche"
            comparison = simulate_debt_repayment(DebtRepaymentRequest(debts=debts, extra_monthly_payment=extra_monthly_payment, strategy=comparison_strategy))
            st.session_state["debt_repayment_result"] = {"selected": selected, "comparison": comparison}
        except ValueError as error:
            st.error(f"Please review the debt details: {error}")
    if "debt_repayment_result" in st.session_state:
        stored = st.session_state["debt_repayment_result"]
        _render_debt_results(stored["selected"], stored["comparison"], symbol)


def _render_transparency() -> None:
    _section_intro(
        "Methodology",
        "What is calculated, what is learned, and what is assumed",
        "A trustworthy financial interface should make those boundaries visible instead of presenting every number as an opaque score.",
    )
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        _feature_card("ƒx", "Deterministic affordability", "Loan Plan Fit, selected-term capacity, EMI gaps, alternatives, liquidity, and policy checks are reproduced exactly from submitted values and displayed assumptions.")
    with col_2:
        _feature_card("ML", "Historical stress lens", "A calibrated XGBoost model estimates payment difficulty for historically similar form-aligned profiles. It cannot approve or reject a loan.")
    with col_3:
        _feature_card("↘", "Debt payoff mathematics", "The planner amortizes submitted balances month by month. It does not predict behaviour, income shocks, fees, or future interest-rate changes.")
    st.markdown("### Metrics translated into everyday language")
    translations = pd.DataFrame(
        [
            ("Loan Plan Fit", "How much of the requested EMI the submitted cash flow can support", "Primary deterministic ratio; not an approval probability"),
            ("Selected-term maximum principal", "The largest principal supported at the term the user actually chose", "Policy and cash-flow calculation"),
            ("Financial Foundation", "Pre-loan cash flow, debt, expenses, liquidity, and stability", "Supporting planning indicator; not a credit score"),
            ("Emergency coverage", "How many months of post-loan commitments adjusted liquid assets could cover", "Asset-resilience calculation; never increases EMI capacity"),
            ("Historical stress probability", "How often similar profiles had the dataset's payment-difficulty outcome", "Calibrated ML estimate"),
            ("Debt-free horizon", "How long the submitted payoff budget takes to clear every balance", "Month-by-month amortization"),
            ("Interest avoided", "Estimated cost difference versus paying only fixed minimums", "Scenario comparison"),
        ],
        columns=["Metric", "Plain meaning", "Source"],
    )
    st.dataframe(translations, width="stretch", hide_index=True)
    st.markdown("### What a real financial institution would add")
    st.markdown(
        "Verified income and liabilities, customer consent, credit-bureau data, transaction history, fraud/KYC/AML controls, approved product pricing, model-risk validation, fairness testing, audit logs, security controls, human-review workflows, and ongoing outcome monitoring."
    )
    st.markdown(
        '<div class="caution-note"><strong>Portfolio position:</strong> the application demonstrates responsible product and engineering boundaries. It does not reproduce any bank or card issuer’s proprietary underwriting or collections system.</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
              <span class="brand-mark">CP</span><span class="brand-name">ClearPath Credit</span>
              <div class="brand-note">Decision intelligence studio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        navigation = st.radio(
            "Explore",
            options=["Overview", "Loan affordability", "Debt payoff planner", "Model transparency"],
            label_visibility="collapsed",
        )
        st.divider()
        currency = st.selectbox("Display currency", options=list(CURRENCY_SYMBOLS))
        symbol = CURRENCY_SYMBOLS[currency]
        st.caption("Currency changes display symbols only. It does not change rates, policy assumptions, or calculations.")
        st.divider()
        st.caption("Educational portfolio application · Self-declared inputs · No account data stored")
    if navigation == "Overview":
        _render_home()
    elif navigation == "Loan affordability":
        _render_loan_journey(symbol)
    elif navigation == "Debt payoff planner":
        _render_debt_journey(symbol)
    else:
        _render_transparency()
    st.markdown(
        '<div class="footer-note">ClearPath Credit is a portfolio simulation. Results are educational estimates, not financial advice, approval, or a lender offer.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

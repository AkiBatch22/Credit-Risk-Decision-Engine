"""Streamlit dashboard for hybrid affordability and repayment-stress simulation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.loan_simulator import (
    EMPLOYMENT_TYPES,
    PRODUCT_POLICIES,
    LoanApplication,
    simulate_loan,
)
from src.components.repayment_stress import estimate_repayment_stress


st.set_page_config(
    page_title="Loan Eligibility & Repayment Simulator",
    page_icon="🏦",
    layout="wide",
)


PRODUCT_LABEL_TO_CODE = {
    policy.label: code for code, policy in PRODUCT_POLICIES.items()
}


def _money(value: float) -> str:
    return f"{value:,.2f} units"


def _status_message(result: dict[str, object]) -> None:
    status = str(result["eligibility_status"])
    if status == "ELIGIBLE":
        st.success(
            "The requested loan is affordable under the selected illustrative policy and submitted values."
        )
    elif status == "ELIGIBLE_FOR_LOWER_AMOUNT_OR_DIFFERENT_TERM":
        st.warning(
            "The requested structure is not currently affordable, but the submitted values support "
            "a lower amount or another repayment term."
        )
    else:
        st.error(
            "The submitted values do not currently satisfy the simulator's basic affordability or "
            "product criteria. Review the specific actions below."
        )


def render_applicant_dashboard(result: dict[str, object]) -> None:
    st.subheader("Your Affordability Summary")
    _status_message(result)
    st.caption(f"Application ID: {result['application_id']}")

    score_col, eligible_col, emi_col, surplus_col = st.columns(4)
    score_col.metric(
        "Financial Readiness Score",
        f"{result['financial_readiness_score']}/100",
        help="An input-only affordability score—not a bureau credit score.",
    )
    eligible_col.metric(
        "Maximum Eligible Loan",
        _money(float(result["maximum_eligible_loan_amount"])),
    )
    emi_col.metric(
        "Maximum Affordable EMI",
        _money(float(result["maximum_affordable_monthly_emi"])),
    )
    surplus_col.metric(
        "Surplus After Preferred EMI",
        _money(float(result["post_preferred_emi_monthly_surplus"])),
    )
    st.caption(
        f"Readiness band: {str(result['financial_readiness_band']).replace('_', ' ').title()}. "
        "This is a transparent financial-readiness indicator calculated only from the submitted values."
    )

    st.markdown("#### Historical repayment-stress estimate")
    stress = result["repayment_stress"]
    if not stress["available"]:
        st.info(str(stress["unavailable_reason"]))
    else:
        probability = float(stress["calibrated_payment_difficulty_probability"])
        probability_col, band_col = st.columns(2)
        probability_col.metric(
            "Calibrated Payment-Difficulty Probability",
            f"{probability:.1%}",
            help=(
                "A historical comparison trained on Home Credit outcomes using only fields "
                "derivable from this form. It is not a bureau score or approval decision."
            ),
        )
        band_col.metric(
            "Historical Stress Band",
            str(stress["historical_stress_band"]).replace("_", " ").title(),
        )
        st.caption(str(stress["role_in_decision"]))
        with st.expander("Why the ML estimate moved in this direction"):
            for reason in stress["reason_codes"]:
                direction = (
                    "increased the model estimate"
                    if reason["direction"] == "increases_stress"
                    else "reduced the model estimate"
                )
                st.markdown(f"- **{reason['label']}** {direction}.")
            st.caption(
                "Reason directions are local model contributions before probability calibration; "
                "they are explanatory, not instructions or adverse-action reasons."
            )

    st.markdown("#### What you can do next")
    recommendations = result["recommendations"]
    if recommendations:
        for recommendation in recommendations:
            st.markdown(
                f"**{recommendation['action']}**  \n{recommendation['detail']}"
            )
    else:
        st.info("No specific recommendation was generated for this scenario.")

    st.markdown("#### Repayment plan comparison")
    plans = pd.DataFrame(result["repayment_plans"])
    if plans.empty:
        st.warning("No repayment term fits the configured maximum age-at-maturity rule.")
    else:
        display_plans = plans.rename(
            columns={
                "term_months": "Term (months)",
                "monthly_emi": "Monthly EMI",
                "total_interest": "Total Interest",
                "total_repayment": "Total Repayment",
                "post_emi_monthly_surplus": "Post-EMI Surplus",
                "total_debt_service_ratio": "Total Debt Ratio",
                "affordable": "Affordable",
                "preferred": "Preferred",
            }
        )
        display_plans["Total Debt Ratio"] = display_plans["Total Debt Ratio"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(display_plans, width="stretch", hide_index=True)
        st.caption(
            "A shorter term usually reduces total interest but increases the monthly EMI. A longer "
            "term usually lowers the EMI but increases total interest."
        )


def render_bank_dashboard(result: dict[str, object]) -> None:
    st.subheader("Decision Insights")
    st.caption(
        "This view keeps deterministic affordability checks separate from the optional historical "
        "ML signal. It is suitable for explaining the simulation, not for making a regulated lending decision."
    )

    ratio_1, ratio_2, ratio_3, requested = st.columns(4)
    ratio_1.metric(
        "Existing Debt-Service Ratio",
        f"{float(result['current_debt_service_ratio']):.1%}",
    )
    ratio_2.metric(
        "Debt Ratio With Requested Loan",
        f"{float(result['requested_total_debt_service_ratio']):.1%}",
    )
    ratio_3.metric(
        "Essential Expense Ratio",
        f"{float(result['essential_expense_ratio']):.1%}",
    )
    requested.metric(
        "Preferred-Plan EMI",
        _money(float(result["preferred_plan_monthly_emi"])),
    )

    st.markdown("#### Policy checks")
    checks = pd.DataFrame(result["policy_checks"])
    checks["Result"] = checks["passed"].map({True: "PASS", False: "FAIL"})
    checks = checks.rename(
        columns={
            "code": "Rule Code",
            "label": "Check",
            "observed": "Observed",
            "limit": "Policy Limit",
        }
    )
    st.dataframe(
        checks[["Rule Code", "Check", "Result", "Observed", "Policy Limit"]],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Readiness score breakdown")
    breakdown = pd.DataFrame(
        {
            "Component": result["readiness_score_breakdown"].keys(),
            "Points": result["readiness_score_breakdown"].values(),
        }
    )
    st.bar_chart(breakdown.set_index("Component"))

    stress = result["repayment_stress"]
    st.markdown("#### ML model card snapshot")
    if stress["available"]:
        final_metrics = stress["final_test_metrics"]
        model_1, model_2, model_3, model_4 = st.columns(4)
        model_1.metric("Model", str(stress["model_name"]).replace("_", " ").title())
        model_2.metric("Final-test ROC-AUC", f"{float(final_metrics['roc_auc']):.3f}")
        model_3.metric("Final-test PR-AUC", f"{float(final_metrics['pr_auc']):.3f}")
        model_4.metric("Final-test Brier", f"{float(final_metrics['brier_score']):.4f}")
        st.caption(str(stress["data_scope_note"]))
    else:
        st.info(str(stress["unavailable_reason"]))

    assumptions = result["policy_assumptions"]
    with st.expander("Illustrative product-policy assumptions"):
        st.json(
            {
                "Product": assumptions["label"],
                "Illustrative annual rate": f"{assumptions['annual_interest_rate_percent']}%",
                "Maximum total debt-service ratio": (
                    f"{assumptions['maximum_debt_service_ratio_percent']}%"
                ),
                "Minimum residual-income buffer": (
                    f"{assumptions['minimum_residual_income_ratio_percent']}%"
                ),
                "Minimum income stability": (
                    f"{assumptions['minimum_income_stability_years']} years"
                ),
                "Maximum age at maturity": assumptions["maximum_age_at_maturity"],
                "Maximum product principal": assumptions["maximum_principal"],
                "Permitted terms": assumptions["permitted_terms_months"],
            }
        )


st.title("Loan Eligibility & Repayment Simulator")
st.markdown(
    "Explore how current income, essential expenses, existing debt, income stability, and loan "
    "structure affect affordability. The same result is explained from both the applicant and bank perspectives."
)
st.warning(
    "Educational simulation only. It uses self-declared inputs and illustrative product policies. "
    "It does not access a credit bureau, calculate an official credit score, verify income, or make a lending decision."
)

with st.form("loan_simulation_form"):
    st.subheader("Loan Request")
    request_1, request_2 = st.columns(2)
    product_label = request_1.selectbox(
        "Loan Product",
        options=list(PRODUCT_LABEL_TO_CODE),
        help="Choose the type of loan to apply the corresponding illustrative rate, term, and affordability policy.",
    )
    product_code = PRODUCT_LABEL_TO_CODE[product_label]
    policy = PRODUCT_POLICIES[product_code]
    requested_loan_amount = request_2.number_input(
        "Requested Loan Amount",
        min_value=1.0,
        max_value=float(policy.maximum_principal * 2),
        value=min(500_000.0, float(policy.maximum_principal)),
        step=10_000.0,
        help="Enter the amount you would like to borrow. Values are shown in neutral currency units.",
    )

    applicant_1, applicant_2, applicant_3 = st.columns(3)
    age = applicant_1.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=32,
        step=1,
        help="Enter age in completed years. The term must end before the product's maximum maturity age.",
    )
    employment_type = applicant_2.selectbox(
        "Income Source",
        options=EMPLOYMENT_TYPES,
        help="Select the primary recurring source used to support repayment.",
    )
    income_stability_years = applicant_3.number_input(
        "Income Stability (years)",
        min_value=0.0,
        max_value=60.0,
        value=4.0,
        step=0.5,
        help="Enter how long the current income source has been continuous or stable.",
    )

    st.subheader("Monthly Cash Flow")
    cash_1, cash_2, cash_3 = st.columns(3)
    monthly_net_income = cash_1.number_input(
        "Monthly Net Income",
        min_value=1.0,
        value=100_000.0,
        step=1_000.0,
        help="Enter recurring take-home income after tax and mandatory deductions.",
    )
    monthly_essential_expenses = cash_2.number_input(
        "Monthly Essential Expenses",
        min_value=0.0,
        value=40_000.0,
        step=1_000.0,
        help="Include housing, food, utilities, transport, dependants, and other recurring essentials. Exclude existing loan payments entered separately.",
    )
    existing_monthly_debt_payments = cash_3.number_input(
        "Existing Monthly Debt Payments",
        min_value=0.0,
        value=10_000.0,
        step=1_000.0,
        help="Enter the total scheduled monthly payment for current loans, cards, or other debts.",
    )

    eligible_terms = [
        term
        for term in policy.permitted_terms_months
        if age + term / 12 <= policy.maximum_age_at_maturity
    ]
    preferred_term_months = st.selectbox(
        "Preferred Repayment Term",
        options=eligible_terms or list(policy.permitted_terms_months),
        index=0,
        format_func=lambda term: f"{term} months ({term / 12:g} years)",
        help="Compare this preferred term with the alternative EMI plans shown after simulation.",
    )
    submitted = st.form_submit_button("Simulate Loan Eligibility", type="primary", width="stretch")

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
        )
        result = simulate_loan(application_input)
        result["repayment_stress"] = estimate_repayment_stress(
            application_input,
            float(result["preferred_plan_monthly_emi"]),
        )
        st.session_state["loan_simulation_result"] = result
    except ValueError as error:
        st.error(f"Please review the submitted information: {error}")

if "loan_simulation_result" in st.session_state:
    st.divider()
    applicant_tab, bank_tab = st.tabs(["Applicant Dashboard", "Decision Insights"])
    with applicant_tab:
        render_applicant_dashboard(st.session_state["loan_simulation_result"])
    with bank_tab:
        render_bank_dashboard(st.session_state["loan_simulation_result"])

with st.expander("How maximum eligibility is calculated"):
    st.markdown(
        "The simulator calculates one EMI capacity from the product's maximum total debt-service "
        "ratio and another from income remaining after essential expenses, existing debt, and a "
        "residual-income buffer. The lower capacity is converted to principal using the standard "
        "amortizing-loan formula and the longest age-permitted term, then capped by the illustrative "
        "product maximum. This deterministic eligibility calculation does not use the ML probability."
    )

st.info(
    "A real bank would additionally verify identity, income, employment, existing obligations, "
    "fraud and compliance checks, collateral, product rules, and credit-bureau history. Eligibility "
    "here means only that the submitted numbers pass the displayed illustrative affordability rules. "
    "The separate ML estimate is a portfolio demonstration trained on historical outcomes."
)

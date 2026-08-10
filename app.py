"""Streamlit interface for the deployable credit-risk decision engine."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.components.feature_contract import (
    FEATURE_BY_API_NAME,
    UNKNOWN_CATEGORY,
    application_to_model_input,
    friendly_explanation_name,
)
from src.pipeline.prediction_pipeline import PredictionPipeline


st.set_page_config(
    page_title="Credit Risk Decision Engine",
    page_icon="📊",
    layout="wide",
)


SAMPLE_APPLICATION: dict[str, Any] = {
    "age": 34,
    "years_employed": 5.0,
    "family_members": 2.0,
    "number_of_children": 0,
    "annual_income": 202_500.0,
    "requested_loan_amount": 406_597.5,
    "loan_annuity": 24_700.5,
    "goods_purchase_price": 351_000.0,
    "credit_product_type": "Cash loans",
    "income_type": "Working",
    "housing_situation": "House / apartment",
    "owns_car": "No",
    "owns_property": "Yes",
}


@st.cache_resource(show_spinner=False)
def load_pipeline() -> PredictionPipeline:
    """Load fitted deployable artifacts once per Streamlit process."""

    return PredictionPipeline()


def build_model_input_from_form(**application_values: Any) -> dict[str, Any]:
    """Validate and translate human-readable fields to the production model schema."""

    return application_to_model_input(application_values)


def predict_applicants(
    applicants: dict[str, Any] | pd.DataFrame,
    *,
    include_explanations: bool,
) -> pd.DataFrame | None:
    try:
        with st.spinner("Loading fitted artifacts and assessing credit risk..."):
            return load_pipeline().predict(
                applicants,
                include_explanations=include_explanations,
            )
    except FileNotFoundError:
        st.error(
            "Deployable model artifacts were not found. Run the documented full training pipeline "
            "before starting the application."
        )
    except (TypeError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as error:
        st.error(f"The application could not be processed. Please review the inputs: {error}")
    except Exception as error:  # pragma: no cover - defensive UI boundary
        st.error(f"Prediction failed. Check the local artifacts and application schema: {error}")
    return None


def render_results(results: pd.DataFrame, source_label: str) -> None:
    st.header("Assessment Result")
    st.caption(f"Latest assessment source: {source_label}")

    for _, row in results.iterrows():
        application_id = str(row["application_id"])
        with st.container(border=True):
            st.metric("Application ID", application_id)
            if "source_record_id" in row and pd.notna(row["source_record_id"]):
                st.caption(f"Historical source record: {row['source_record_id']}")

            expected_loss = row.get("expected_loss")
            expected_loss_text = (
                "Unavailable"
                if expected_loss is None or pd.isna(expected_loss)
                else f"{float(expected_loss):,.2f} dataset currency units"
            )
            probability_col, band_col, recommendation_col, loss_col = st.columns(4)
            probability_col.metric(
                "Probability of Payment Difficulty",
                f"{float(row['probability']):.1%}",
            )
            band_col.metric("Risk Band", str(row["risk_band"]).replace("_", " ").title())
            recommendation_col.metric(
                "Recommendation",
                str(row["recommendation"]).replace("_", " ").title(),
            )
            loss_col.metric("Expected Loss", expected_loss_text)

            missing_count = int(row.get("missing_input_feature_count", 0) or 0)
            if missing_count:
                st.caption(
                    f"{missing_count} optional contract value(s) were not provided and were handled "
                    "by the fitted preprocessing pipeline."
                )

            reasons = row.get("top_risk_reasons")
            if isinstance(reasons, list) and reasons:
                st.markdown("#### Key Risk Drivers")
                reason_rows = []
                for reason in reasons:
                    feature = reason.get("feature", "Feature")
                    label = friendly_explanation_name(feature)
                    reason_rows.append(
                        {
                            "Risk driver": label,
                            "Effect on estimated risk": str(
                                reason.get("direction", "model influence")
                            ).replace("_", " ").title(),
                            "Model contribution": f"{float(reason.get('contribution', 0)):+.4f}",
                        }
                    )
                st.dataframe(pd.DataFrame(reason_rows), width="stretch", hide_index=True)


def _spec(api_name: str):
    return FEATURE_BY_API_NAME[api_name]


st.title("Credit Risk Decision Engine")
st.markdown(
    "A deployment-shaped portfolio demonstration using only information that can be collected "
    "or deterministically derived when a completely new application arrives."
)
st.warning(
    "Educational decision-support demo only. It is not suitable for real lending decisions and "
    "requires human review, fairness testing, legal review, monitoring, and governance."
)

st.header("Applicant Details")
with st.form("applicant_assessment_form"):
    st.subheader("Personal Information")
    personal_1, personal_2 = st.columns(2)
    age = personal_1.number_input(
        _spec("age").label,
        min_value=float(_spec("age").minimum),
        max_value=float(_spec("age").maximum),
        value=float(SAMPLE_APPLICATION["age"]),
        step=float(_spec("age").step),
        help=_spec("age").help,
    )
    years_employed = personal_2.number_input(
        _spec("years_employed").label,
        min_value=float(_spec("years_employed").minimum),
        max_value=float(_spec("years_employed").maximum),
        value=float(SAMPLE_APPLICATION["years_employed"]),
        step=float(_spec("years_employed").step),
        help=_spec("years_employed").help,
    )
    personal_3, personal_4 = st.columns(2)
    family_members = personal_3.number_input(
        _spec("family_members").label,
        min_value=float(_spec("family_members").minimum),
        max_value=float(_spec("family_members").maximum),
        value=float(SAMPLE_APPLICATION["family_members"]),
        step=float(_spec("family_members").step),
        help=_spec("family_members").help,
    )
    number_of_children = personal_4.number_input(
        _spec("number_of_children").label,
        min_value=int(_spec("number_of_children").minimum),
        max_value=int(_spec("number_of_children").maximum),
        value=int(SAMPLE_APPLICATION["number_of_children"]),
        step=int(_spec("number_of_children").step),
        help=_spec("number_of_children").help,
    )

    st.subheader("Financial Information")
    financial_1, financial_2 = st.columns(2)
    annual_income = financial_1.number_input(
        _spec("annual_income").label,
        min_value=float(_spec("annual_income").minimum),
        max_value=float(_spec("annual_income").maximum),
        value=float(SAMPLE_APPLICATION["annual_income"]),
        step=float(_spec("annual_income").step),
        help=_spec("annual_income").help,
    )
    requested_loan_amount = financial_2.number_input(
        _spec("requested_loan_amount").label,
        min_value=float(_spec("requested_loan_amount").minimum),
        max_value=float(_spec("requested_loan_amount").maximum),
        value=float(SAMPLE_APPLICATION["requested_loan_amount"]),
        step=float(_spec("requested_loan_amount").step),
        help=_spec("requested_loan_amount").help,
    )
    financial_3, financial_4 = st.columns(2)
    loan_annuity = financial_3.number_input(
        _spec("loan_annuity").label,
        min_value=float(_spec("loan_annuity").minimum),
        max_value=float(_spec("loan_annuity").maximum),
        value=float(SAMPLE_APPLICATION["loan_annuity"]),
        step=float(_spec("loan_annuity").step),
        help=_spec("loan_annuity").help,
    )
    goods_purchase_price = financial_4.number_input(
        _spec("goods_purchase_price").label,
        min_value=float(_spec("goods_purchase_price").minimum),
        max_value=float(_spec("goods_purchase_price").maximum),
        value=float(SAMPLE_APPLICATION["goods_purchase_price"]),
        step=float(_spec("goods_purchase_price").step),
        help=_spec("goods_purchase_price").help,
    )

    st.subheader("Application Information")
    application_1, application_2, application_3 = st.columns(3)
    credit_product_type = application_1.selectbox(
        _spec("credit_product_type").label,
        options=list(_spec("credit_product_type").allowed_values),
        help=_spec("credit_product_type").help,
    )
    income_type = application_2.selectbox(
        _spec("income_type").label,
        options=[*_spec("income_type").allowed_values, UNKNOWN_CATEGORY],
        index=list(_spec("income_type").allowed_values).index(SAMPLE_APPLICATION["income_type"]),
        help=_spec("income_type").help,
    )
    housing_situation = application_3.selectbox(
        _spec("housing_situation").label,
        options=[*_spec("housing_situation").allowed_values, UNKNOWN_CATEGORY],
        index=list(_spec("housing_situation").allowed_values).index(
            SAMPLE_APPLICATION["housing_situation"]
        ),
        help=_spec("housing_situation").help,
    )
    application_4, application_5 = st.columns(2)
    owns_car = application_4.selectbox(
        _spec("owns_car").label,
        options=list(_spec("owns_car").allowed_values),
        help=_spec("owns_car").help,
    )
    owns_property = application_5.selectbox(
        _spec("owns_property").label,
        options=list(_spec("owns_property").allowed_values),
        index=1,
        help=_spec("owns_property").help,
    )

    include_explanations = st.checkbox("Show model reason codes", value=True)
    assess_submitted = st.form_submit_button(
        "Assess Credit Risk",
        width="stretch",
        type="primary",
    )

if assess_submitted:
    form_values = {
        "age": age,
        "years_employed": years_employed,
        "family_members": family_members,
        "number_of_children": number_of_children,
        "annual_income": annual_income,
        "requested_loan_amount": requested_loan_amount,
        "loan_annuity": loan_annuity,
        "goods_purchase_price": goods_purchase_price,
        "credit_product_type": credit_product_type,
        "income_type": income_type,
        "housing_situation": housing_situation,
        "owns_car": owns_car,
        "owns_property": owns_property,
    }
    try:
        model_input = build_model_input_from_form(**form_values)
        prediction_results = predict_applicants(
            model_input,
            include_explanations=include_explanations,
        )
        if prediction_results is not None:
            st.session_state["latest_prediction_results"] = prediction_results
            st.session_state["latest_prediction_source"] = "guided applicant form"
    except ValueError as error:
        st.error(f"Please review the application details: {error}")

if "latest_prediction_results" in st.session_state:
    render_results(
        st.session_state["latest_prediction_results"],
        st.session_state.get("latest_prediction_source", "applicant input"),
    )

with st.expander("Advanced input options"):
    st.caption(
        "Use friendly JSON for one new applicant or upload a CSV containing the deployable raw "
        "feature contract. Historical SK_ID_CURR values are retained only as source_record_id."
    )
    with st.form("advanced_prediction_form"):
        advanced_mode = st.radio("Input format", ["JSON", "CSV upload"], horizontal=True)
        advanced_payload: str | Any
        if advanced_mode == "JSON":
            advanced_payload = st.text_area(
                "New-applicant JSON",
                value=json.dumps(SAMPLE_APPLICATION, indent=2),
                height=360,
            )
        else:
            advanced_payload = st.file_uploader("Historical or application CSV", type=["csv"])
        advanced_explanations = st.checkbox(
            "Include model reason codes",
            value=True,
            key="advanced_explanations",
        )
        advanced_submitted = st.form_submit_button("Run Advanced Prediction", width="stretch")

    if advanced_submitted:
        try:
            if advanced_mode == "JSON":
                parsed_applicants: dict[str, Any] | pd.DataFrame = json.loads(advanced_payload)
                source = "advanced new-applicant JSON"
            else:
                if advanced_payload is None:
                    raise ValueError("Choose a CSV file before running the prediction.")
                parsed_applicants = pd.read_csv(advanced_payload)
                source = "advanced CSV upload"
            advanced_results = predict_applicants(
                parsed_applicants,
                include_explanations=advanced_explanations,
            )
            if advanced_results is not None:
                st.session_state["latest_prediction_results"] = advanced_results
                st.session_state["latest_prediction_source"] = source
                st.rerun()
        except (TypeError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as error:
            st.error(f"The advanced input is invalid: {error}")

with st.expander("How the assessment works"):
    st.markdown(
        "The deployable model intentionally sacrifices performance associated with unavailable "
        "historical features so every predictor can be reproduced for a new applicant. The fitted "
        "model returns a calibrated probability of payment difficulty. Saved policy thresholds "
        "produce the risk band and recommendation. Expected loss is PD × illustrative LGD × "
        "requested credit and is reported in neutral dataset currency units. Reason codes describe "
        "model influence, not causation."
    )

st.info(
    "Sensitive and proxy attributes require jurisdiction-specific legal and fairness review. This "
    "portfolio system deliberately excludes gender, family status, education, occupation, and "
    "unreproducible external scores from its recommended deployable contract."
)

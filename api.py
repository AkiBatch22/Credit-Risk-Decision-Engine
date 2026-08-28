"""FastAPI service for hybrid affordability and repayment-stress simulations."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.components.loan_simulator import LoanApplication, simulate_loan
from src.components.repayment_stress import (
    estimate_repayment_stress,
    repayment_stress_available,
)


class LoanSimulationRequest(BaseModel):
    """Self-declared values used by the deterministic affordability simulator."""

    model_config = ConfigDict(extra="forbid")

    product_type: Literal["personal_loan", "vehicle_loan", "home_loan"]
    age: int = Field(ge=18, le=80, description="Applicant age in completed years.")
    employment_type: Literal[
        "Salaried",
        "Self-employed",
        "Business owner",
        "Retired / pension income",
        "Other regular income",
    ]
    income_stability_years: float = Field(
        ge=0,
        le=60,
        description="Years the current income source has remained stable.",
    )
    monthly_net_income: float = Field(
        gt=0,
        description="Monthly take-home or otherwise available recurring income.",
    )
    monthly_essential_expenses: float = Field(
        ge=0,
        description="Essential recurring living expenses excluding existing loan payments.",
    )
    existing_monthly_debt_payments: float = Field(
        ge=0,
        description="Total scheduled monthly repayments for existing debts.",
    )
    requested_loan_amount: float = Field(gt=0)
    preferred_term_months: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_cash_flow(self) -> "LoanSimulationRequest":
        if self.monthly_essential_expenses > self.monthly_net_income * 3:
            raise ValueError(
                "Monthly essential expenses appear inconsistent with monthly net income"
            )
        return self

    def to_domain(self) -> LoanApplication:
        return LoanApplication(**self.model_dump())


class LoanSimulationResponse(BaseModel):
    application_id: str
    eligibility_status: str
    eligible_for_requested_loan: bool
    financial_readiness_score: int
    financial_readiness_band: str
    maximum_eligible_loan_amount: float
    maximum_affordable_monthly_emi: float
    requested_loan_amount: float
    preferred_term_months: int
    preferred_plan_monthly_emi: float
    preferred_plan_total_interest: float
    post_preferred_emi_monthly_surplus: float
    current_debt_service_ratio: float
    requested_total_debt_service_ratio: float
    essential_expense_ratio: float
    readiness_score_breakdown: dict[str, float]
    policy_checks: list[dict[str, Any]]
    repayment_plans: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    policy_assumptions: dict[str, Any]
    repayment_stress: dict[str, Any]
    disclaimer: str


def create_app() -> FastAPI:
    application = FastAPI(
        title="Loan Eligibility and Repayment Simulator",
        version="4.0.0",
        description=(
            "Deterministic affordability simulation with a separately disclosed, calibrated "
            "historical repayment-stress estimate. It does not use a bureau credit score."
        ),
    )

    @application.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "status": "healthy",
            "service": "loan-simulator",
            "repayment_stress_model_available": repayment_stress_available(),
        }

    @application.get("/products")
    def products() -> dict[str, list[str]]:
        return {
            "product_types": ["personal_loan", "vehicle_loan", "home_loan"],
        }

    @application.post("/simulate", response_model=LoanSimulationResponse)
    def simulate(request: LoanSimulationRequest) -> LoanSimulationResponse:
        try:
            application_input = request.to_domain()
            result = simulate_loan(application_input)
            result["repayment_stress"] = estimate_repayment_stress(
                application_input,
                float(result["preferred_plan_monthly_emi"]),
            )
            return LoanSimulationResponse(**result)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail="Loan simulation could not be completed",
            ) from error

    return application


app = create_app()

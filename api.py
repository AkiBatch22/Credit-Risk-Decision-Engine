"""FastAPI service for hybrid affordability and repayment-stress simulations."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.components.debt_repayment import (
    DebtAccount,
    DebtRepaymentRequest,
    simulate_debt_repayment,
)
from src.components.loan_simulator import AssetProfile, LoanApplication, simulate_loan
from src.components.repayment_stress import (
    estimate_repayment_stress,
    repayment_stress_available,
)


class AssetPosition(BaseModel):
    """Optional self-declared assets used only for resilience indicators."""

    model_config = ConfigDict(extra="forbid")

    cash: float = Field(default=0, ge=0)
    fixed_deposit: float = Field(default=0, ge=0)
    debt_fund: float = Field(default=0, ge=0)
    equity: float = Field(default=0, ge=0)
    gold: float = Field(default=0, ge=0)
    property_value: float = Field(default=0, ge=0)
    property_liability: float = Field(default=0, ge=0)
    vehicle_value: float = Field(default=0, ge=0)
    vehicle_liability: float = Field(default=0, ge=0)
    epf_ppf: float = Field(default=0, ge=0)
    nps: float = Field(default=0, ge=0)


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
    assets: AssetPosition = Field(default_factory=AssetPosition)
    purchase_property_value: float = Field(default=0, ge=0)
    vehicle_purchase_price: float = Field(default=0, ge=0)
    available_down_payment: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_cash_flow(self) -> "LoanSimulationRequest":
        if self.monthly_essential_expenses > self.monthly_net_income * 3:
            raise ValueError(
                "Monthly essential expenses appear inconsistent with monthly net income"
            )
        return self

    def to_domain(self) -> LoanApplication:
        values = self.model_dump()
        values["assets"] = AssetProfile(**values["assets"])
        return LoanApplication(**values)


class LoanSimulationResponse(BaseModel):
    application_id: str
    plan_status: str
    plan_status_label: str
    combined_decision_summary: str
    eligible_for_requested_loan: bool
    loan_plan_fit_pct: float
    requested_principal: float
    requested_emi: float
    max_affordable_emi: float
    emi_shortfall: float
    max_principal_selected_term: float
    max_principal_longest_term: float
    amount_reduction_required: float
    selected_term_months: int
    shortest_supporting_term_months: int | None
    longest_age_eligible_term_months: int | None
    alternative_term_emi: float | None
    selected_term_total_interest: float
    alternative_term_total_interest: float | None
    incremental_interest_for_alternative: float | None
    cash_remaining: float
    required_residual_buffer: float
    cash_remaining_after_buffer: float
    debt_service_capacity: float
    cash_flow_capacity: float
    current_debt_service_ratio: float
    requested_total_debt_service_ratio: float
    essential_expense_ratio: float
    request_fits_any_available_term: bool
    financial_foundation: dict[str, Any]
    asset_resilience: dict[str, Any]
    stress_test: dict[str, Any]
    product_guidance: dict[str, Any]
    policy_checks: list[dict[str, Any]]
    repayment_plans: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    policy_assumptions: dict[str, Any]
    preferred_plan_monthly_emi: float
    maximum_affordable_monthly_emi: float
    maximum_eligible_loan_amount: float
    repayment_stress: dict[str, Any]
    disclaimer: str


class DebtAccountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    balance: float = Field(gt=0)
    annual_interest_rate_percent: float = Field(ge=0, le=100)
    minimum_payment: float = Field(gt=0)

    def to_domain(self) -> DebtAccount:
        return DebtAccount(
            name=self.name,
            balance=self.balance,
            annual_interest_rate=self.annual_interest_rate_percent / 100,
            minimum_payment=self.minimum_payment,
        )


class DebtRepaymentSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    debts: list[DebtAccountRequest] = Field(min_length=1, max_length=10)
    extra_monthly_payment: float = Field(ge=0)
    strategy: Literal["avalanche", "snowball"] = "avalanche"

    def to_domain(self) -> DebtRepaymentRequest:
        return DebtRepaymentRequest(
            debts=tuple(debt.to_domain() for debt in self.debts),
            extra_monthly_payment=self.extra_monthly_payment,
            strategy=self.strategy,
        )


class DebtRepaymentSimulationResponse(BaseModel):
    status: str
    strategy: str
    strategy_label: str
    starting_balance: float
    minimum_payment_total: float
    extra_monthly_payment: float
    monthly_plan_payment: float
    first_priority_debt: str
    payoff_possible: bool
    estimated_payoff_months: int | None
    estimated_total_interest: float
    estimated_total_paid: float
    estimated_months_saved: int | None
    estimated_interest_saved: float | None
    minimum_only_payoff_months: int | None
    minimum_only_total_interest: float | None
    payoff_order: list[dict[str, Any]]
    schedule: list[dict[str, Any]]
    debts: list[dict[str, Any]]
    guidance: str
    assumptions: list[str]
    disclaimer: str


def create_app() -> FastAPI:
    application = FastAPI(
        title="ClearPath Credit Decision Studio",
        version="6.0.0",
        description=(
            "Deterministic affordability simulation with a separately disclosed, calibrated "
            "historical repayment-stress estimate and a deterministic multi-debt payoff planner. "
            "It does not use a bureau credit score."
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
                float(result["requested_emi"]),
            )
            return LoanSimulationResponse(**result)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail="Loan simulation could not be completed",
            ) from error

    @application.post(
        "/debt-repayment/simulate",
        response_model=DebtRepaymentSimulationResponse,
    )
    def simulate_debt_repayment_plan(
        request: DebtRepaymentSimulationRequest,
    ) -> DebtRepaymentSimulationResponse:
        try:
            result = simulate_debt_repayment(request.to_domain())
            return DebtRepaymentSimulationResponse(**result)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail="Debt repayment simulation could not be completed",
            ) from error

    return application


app = create_app()

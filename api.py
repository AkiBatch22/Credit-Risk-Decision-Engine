"""FastAPI service for deployable, artifact-backed credit-risk predictions."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.components.feature_contract import FEATURE_BY_API_NAME
from src.pipeline.prediction_pipeline import PredictionPipeline


def _description(api_name: str) -> str:
    return FEATURE_BY_API_NAME[api_name].help


class ApplicantRequest(BaseModel):
    """New-application schema containing only reproducible inference-time values."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=18, le=75, description=_description("age"))
    years_employed: float = Field(ge=0, le=60, description=_description("years_employed"))
    family_members: float = Field(ge=1, le=20, description=_description("family_members"))
    number_of_children: int = Field(ge=0, le=19, description=_description("number_of_children"))
    annual_income: float = Field(ge=0, description=_description("annual_income"))
    requested_loan_amount: float = Field(gt=0, description=_description("requested_loan_amount"))
    loan_annuity: float = Field(ge=0, description=_description("loan_annuity"))
    goods_purchase_price: float = Field(ge=0, description=_description("goods_purchase_price"))
    credit_product_type: Literal["Cash loans", "Revolving loans"] = Field(
        description=_description("credit_product_type")
    )
    income_type: Literal[
        "Businessman",
        "Commercial associate",
        "Maternity leave",
        "Pensioner",
        "State servant",
        "Student",
        "Unemployed",
        "Working",
    ] | None = Field(default=None, description=_description("income_type"))
    housing_situation: Literal[
        "Co-op apartment",
        "House / apartment",
        "Municipal apartment",
        "Office apartment",
        "Rented apartment",
        "With parents",
    ] | None = Field(default=None, description=_description("housing_situation"))
    owns_car: bool = Field(description=_description("owns_car"))
    owns_property: bool = Field(description=_description("owns_property"))
    include_explanation: bool = False
    top_n_reasons: int = Field(default=5, ge=1, le=10)

    def application_values(self) -> dict[str, Any]:
        values = self.model_dump(
            exclude={"include_explanation", "top_n_reasons"}
        )
        values["owns_car"] = "Yes" if self.owns_car else "No"
        values["owns_property"] = "Yes" if self.owns_property else "No"
        return values


class PredictionResponse(BaseModel):
    application_id: str
    calibrated_probability: float
    probability_percent: float
    risk_band: str
    recommendation: str
    expected_loss: float
    top_risk_reasons: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str


@lru_cache(maxsize=1)
def get_prediction_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


def create_app(
    pipeline_factory: Callable[[], PredictionPipeline] = get_prediction_pipeline,
) -> FastAPI:
    application = FastAPI(
        title="Credit Risk Decision Engine",
        version="2.0.0",
        description=(
            "Educational deployment-shaped API using only reproducible application-time features."
        ),
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: ApplicantRequest) -> PredictionResponse:
        try:
            if request.years_employed > max(request.age - 14, 0):
                raise ValueError("Years Employed is not plausible relative to Age")
            if request.number_of_children > request.family_members:
                raise ValueError("Number of Children cannot exceed Family Members")
            frame = pipeline_factory().predict(
                request.application_values(),
                include_explanations=request.include_explanation,
                top_n_reasons=request.top_n_reasons,
            )
            row = frame.iloc[0]
            return PredictionResponse(
                application_id=str(row["application_id"]),
                calibrated_probability=float(row["probability"]),
                probability_percent=float(row["probability_percent"]),
                risk_band=str(row["risk_band"]),
                recommendation=str(row["recommendation"]),
                expected_loss=float(row["expected_loss"]),
                top_risk_reasons=list(row.get("top_risk_reasons", [])),
                disclaimer=(
                    "Educational prototype only. Outputs and economic assumptions are not real lending policy."
                ),
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail="Model artifacts are not available") from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail="Prediction could not be completed") from error

    return application


app = create_app()

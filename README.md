# Hybrid Loan Eligibility & Repayment Simulator

A transparent bank-loan affordability simulation with maximum-loan estimation, repayment plans, actionable guidance, and a separately disclosed calibrated ML estimate of historical payment difficulty.

> Educational simulation only. It is not a credit bureau, official credit score, loan offer, underwriting policy, or regulated lending decision.

## What the production-facing application does

The Streamlit application and FastAPI service use only values submitted for the current inference request:

- loan product and requested amount;
- applicant age;
- income source and years of income stability;
- monthly net income;
- monthly essential expenses;
- existing monthly debt payments;
- preferred repayment term.

The simulator returns:

- maximum affordable monthly EMI;
- maximum eligible loan amount;
- eligibility of the requested amount and term;
- a 0–100 Financial Readiness Score with component breakdown;
- debt-service, expense, and residual-income metrics;
- explicit policy-check results;
- alternative EMI plans with total interest and total repayment;
- quantified actions that could improve eligibility.
- a calibrated historical payment-difficulty probability and stress band;
- local ML reason directions and an embedded final-test model-card snapshot.

The eligibility and maximum-loan calculations remain deterministic and do not use the ML probability. The separate ML model was trained on historical Home Credit outcomes, but inference uses no historical applicant lookup, `SK_ID_CURR`, `EXT_SOURCE`, or bureau score.

## Where machine learning is used

The optional **Repayment-Stress Estimate** answers a narrower question than eligibility: how often did historically similar Home Credit application profiles experience the dataset's `TARGET=1` payment-difficulty outcome?

Its inference contract is deliberately limited to fields derivable from the simple form:

- age;
- income source;
- income stability;
- submitted monthly income;
- requested loan amount;
- calculated EMI for the preferred plan;
- derived loan-to-income and EMI-to-income ratios.

Expenses, current debts, loan product, and the readiness score are not fed into ML because equivalent training fields are not reliably available in `application_train.csv`. They still drive the deterministic affordability result.

Training uses a stratified train/calibration/final-test design:

1. Candidate selection is performed with out-of-fold predictions only inside the training partition.
2. The selected base estimator is fitted on training data only.
3. Sigmoid probability calibration is fitted on a separate calibration partition.
4. The final test is evaluated once after model and calibration choices are fixed.

This prevents the final test from becoming a repeatedly consulted model-selection set. Run the dedicated full-data pipeline with:

```bash
python -m src.pipeline.stress_training_pipeline
```

Generated ML artifacts stay under ignored `artifacts/`; the raw dataset stays under ignored `data/raw/` and is never modified.

## Why this is not called a credit score

A real credit score normally depends on verified repayment and credit-account history that this project does not possess. Calling an input-only estimate a credit score would be misleading.

The application therefore reports a **Financial Readiness Score**. It is derived transparently from:

- pre-loan cash-flow surplus;
- existing debt burden;
- debt burden after the requested loan;
- income-source stability;
- essential-expense balance.

It measures the strength of the submitted affordability scenario only. It does not estimate historical repayment behavior.

## Affordability methodology

The engine calculates two independent monthly capacities:

```text
Debt-service capacity
    = monthly income × maximum debt-service ratio − existing debt payments

Cash-flow capacity
    = monthly income − essential expenses − existing debt payments
      − required residual-income buffer

Maximum affordable EMI
    = minimum(debt-service capacity, cash-flow capacity)
```

The maximum EMI is converted to principal with the standard amortizing-loan present-value formula using the longest permitted term that satisfies the maximum age-at-maturity rule. The result is capped by the illustrative product maximum.

Each requested repayment plan is checked against:

- age at maturity;
- minimum income-source stability;
- maximum total debt-service ratio;
- minimum post-loan residual-income buffer;
- product maximum principal;
- calculated affordable principal.

Possible statuses are:

- `ELIGIBLE`
- `ELIGIBLE_FOR_LOWER_AMOUNT_OR_DIFFERENT_TERM`
- `NOT_CURRENTLY_ELIGIBLE`

These describe the simulator outcome—not a bank approval or rejection.

## Illustrative product policies

The policies in `src/components/loan_simulator.py` are configuration examples rather than current market offers:

- Personal Loan: 14% illustrative annual rate, 12–60 months, 40% total debt-service ceiling.
- Vehicle Loan: 10% illustrative annual rate, 12–84 months, 45% ceiling.
- Home Loan: 8.5% illustrative annual rate, 60–240 months, 50% ceiling.

Every policy also defines a residual-income buffer, income-stability requirement, age-at-maturity limit, and product amount ceiling. A real institution would replace these values with approved, version-controlled policies and validated pricing.

## Applicant dashboard

The applicant view explains:

- whether the requested structure fits the current inputs;
- maximum simulated eligibility;
- maximum affordable EMI;
- readiness score and band;
- post-EMI monthly surplus;
- alternative repayment terms;
- total interest and repayment;
- calculated steps to improve affordability.

Recommendations are based on the actual shortfall. Examples include reducing the requested amount, lowering existing monthly debt, increasing stable verified income, improving expense headroom, building a longer income history, or selecting a different term.

## Decision Insights dashboard

The bank view exposes:

- current and requested total debt-service ratios;
- essential-expense ratio;
- preferred-plan EMI;
- every policy rule, observed value, limit, and pass/fail result;
- readiness-score component points;
- all product-policy assumptions.
- the calibrated historical stress probability and band;
- final-test ROC-AUC, PR-AUC, and Brier score;
- model version, calibration method, local reason directions, and scope limitations.

This separation makes the simulation auditable and avoids hiding policy logic inside a model probability. The ML estimate cannot change eligibility, the maximum loan amount, EMI plans, or pricing.

## API

Start the service:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Example request:

```json
{
  "product_type": "personal_loan",
  "age": 34,
  "employment_type": "Salaried",
  "income_stability_years": 5,
  "monthly_net_income": 100000,
  "monthly_essential_expenses": 30000,
  "existing_monthly_debt_payments": 5000,
  "requested_loan_amount": 300000,
  "preferred_term_months": 36
}
```

Use `POST /simulate`. The response includes a `repayment_stress` object. If artifacts are not installed it returns an explicit `available: false` state instead of failing the affordability simulation. `GET /products` lists product identifiers, while `GET /health` reports service and ML-artifact availability.

## Streamlit

```bash
streamlit run app.py
```

The guided form uses neutral currency units because no specific country, institution, or currency has been established.

## Historical ML research

The five Home Credit notebooks and training/prediction modules remain in the repository as a separate research track. They demonstrate leakage-safe splitting, fold-fitted preprocessing, XGBoost/LightGBM/Logistic Regression comparison, calibration, threshold analysis, and feature-availability auditing.

Historical benchmark results are retained for research integrity. The original 13-field research/serving model still does not determine the application-facing result because it requires fields intentionally removed from the simplified experience.

- the dataset describes historical Home Credit applicants rather than the current user;
- no bureau or past repayment history is submitted;
- hidden defaults would make its prediction misleading.

The new form-aligned model is separately trained for the secondary repayment-stress panel. It uses fewer features, so its measured performance—not the older benchmark—must be used when describing the live ML estimate.

## Project structure

```text
.
├── app.py                              # Applicant and decision-insights dashboards
├── api.py                              # Hybrid simulation API
├── src/components/loan_simulator.py    # Affordability, score, plans, recommendations
├── src/components/repayment_stress.py  # Form-aligned ML inference and reason directions
├── src/components/feature_contract.py  # Historical ML application feature audit
├── src/pipeline/stress_training_pipeline.py
├── src/pipeline/                        # Preserved research ML training and inference
├── notebooks/                           # Home Credit research notebooks
├── docs/feature_availability_audit.md
└── tests/                               # Deterministic simulator and ML tests
```

## Validation

```bash
python -m compileall app.py api.py src tests
python -m pytest -q
python -m pip check
```

Tests cover EMI mathematics, principal recovery, eligibility, recommendations, deterministic outputs, form-to-training feature alignment, probability bands, reason codes, missing-artifact behavior, API validation, and the preserved ML research components.

## What a real bank would still require

A production lending system would additionally require:

- identity, consent, KYC, fraud, sanctions, and AML controls;
- verified income and employment;
- verified existing obligations and bureau history;
- product, collateral, down-payment, and loan-purpose rules;
- institution-specific pricing and cost-of-risk models;
- case persistence and an underwriter workflow;
- adverse-action and appeal processes;
- access controls, encryption, retention policies, and audit logging;
- fairness, calibration, drift, stress, and outcome monitoring;
- legal, compliance, model-risk, and independent validation approval.

Eligibility in this project means only that submitted values satisfy displayed illustrative affordability rules.

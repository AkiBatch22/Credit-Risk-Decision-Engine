# ClearPath Credit Decision Studio

A professional, explainable financial decision experience with loan-affordability simulation, repayment-plan comparison, multi-debt payoff planning, actionable guidance, and a separately disclosed calibrated ML estimate of historical payment difficulty.

> Educational simulation only. It is not a credit bureau, official credit score, loan offer, underwriting policy, or regulated lending decision.

## What the production-facing application does

The Streamlit application and FastAPI service use only values submitted for the current inference request:

- loan product and requested amount;
- applicant age;
- income source and years of income stability;
- monthly net income;
- monthly essential expenses;
- existing monthly debt payments;
- preferred repayment term;
- optional asset, collateral-value, liability, and down-payment details.

The simulator returns:

- Loan Plan Fit: the percentage of requested EMI supported by current cash flow;
- requested EMI, maximum affordable EMI, and the exact EMI shortfall;
- maximum principal at the selected term and, separately, at the longest age-eligible term;
- the shortest available term that supports the request, where one exists;
- a supporting 0–100 Financial Foundation indicator with component breakdown;
- adjusted emergency liquidity and months of post-loan commitment coverage;
- rate-stress, LTV/down-payment, and personal-loan salary-multiple guidance;
- debt-service, expense, and residual-income metrics;
- explicit policy-check results;
- alternative EMI plans with total interest and total repayment;
- quantified actions that could make the requested structure fit.
- a calibrated historical payment-difficulty probability and stress band;
- local ML reason directions and an embedded final-test model-card snapshot.

The separate debt-payoff journey accepts up to ten balances with their APRs and minimum payments. It returns:

- a debt-free horizon and estimated finish month;
- the total monthly plan payment and estimated lifetime interest;
- interest and time saved versus paying only the submitted fixed minimums;
- highest-interest-first (avalanche) and smallest-balance-first (snowball) comparisons;
- a payoff order, annual milestones, and month-by-month balance schedule;
- explicit assumptions and an honest warning when the submitted payments do not reduce the debt.

Debt repayment is deterministic amortization—not ML. The project has no behavioural repayment history, future rate changes, fees, or account statements that would justify predicting whether a person will follow the plan.

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

Expenses, current debts, loan product, assets, and the Financial Foundation indicator are not fed into ML because equivalent training fields are not reliably available in `application_train.csv`. They still drive deterministic affordability or resilience diagnostics as documented below.

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

The primary result is therefore **Loan Plan Fit**, not a credit score. It is simply:

```text
Loan Plan Fit = min(maximum affordable EMI ÷ requested EMI, 1) × 100
```

The application also reports a supporting **Financial Foundation** indicator derived transparently from:

- pre-loan cash-flow surplus;
- existing debt burden;
- essential-expense burden;
- adjusted emergency liquidity;
- income-source stability;

It measures pre-loan financial footing only. Neither metric estimates historical repayment behavior, acts as a bureau score, or represents approval probability.

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

The maximum EMI is converted to principal with the standard amortizing-loan present-value formula. The headline maximum uses the **selected term**, so it answers the same scenario the user entered. A distinct longest-age-eligible-term maximum is displayed only as an alternative. Both are capped by the illustrative product maximum.

Optional assets never increase affordable EMI. Each asset is converted to adjusted emergency liquidity using a disclosed accessibility weight: cash 100%, fixed deposits 95%, debt funds 85%, gold 70%, listed equity 60%, property equity 25%, vehicle equity 10%, EPF/PPF 15%, and NPS 0%. Secured-asset equity is floored at zero when liabilities exceed value. Emergency coverage divides adjusted liquidity by post-loan monthly commitments.

The Financial Foundation indicator weights pre-loan cash-flow strength (30 points), existing debt burden (20), essential expenses (15), emergency liquidity (25), and income stability (10). It is a planning aid—not a model score or lending decision.

The rate-stress scenario increases the displayed personal-loan rate by 3 percentage points and secured-loan rates by 2 points. It reports the recalculated EMI and remaining cash; it does not change the base result. Home and vehicle journeys add illustrative LTV/down-payment guidance, while personal loans show the request as a multiple of monthly income.

Each requested repayment plan is checked against:

- age at maturity;
- minimum income-source stability;
- maximum total debt-service ratio;
- minimum post-loan residual-income buffer;
- product maximum principal;
- calculated affordable principal.

Possible plan statuses are:

- `SELECTED_PLAN_FITS`
- `LONGER_TERM_REQUIRED`
- `LOWER_AMOUNT_REQUIRED`
- `LOWER_AMOUNT_AND_TERM_ADJUSTMENT`
- `REQUEST_DOES_NOT_FIT_AVAILABLE_TERMS`
- `BASIC_PRODUCT_REQUIREMENTS_NOT_MET`

These describe the simulator outcome—not a bank approval or rejection.

## Illustrative product policies

The policies in `src/components/loan_simulator.py` are configuration examples rather than current market offers:

- Personal Loan: 12.75% illustrative annual rate, 12–60 months, 40% total debt-service ceiling.
- Vehicle Loan: 8.5% illustrative annual rate, 12–84 months, 45% ceiling and 85% illustrative LTV guide.
- Home Loan: 7.75% illustrative annual rate, 60–360 months, 50% ceiling and 80% illustrative LTV guide.

Every policy also defines a residual-income buffer, income-stability requirement, age-at-maturity limit, and product amount ceiling. A real institution would replace these values with approved, version-controlled policies and validated pricing.

## Applicant dashboard

The applicant view explains:

- whether the requested structure fits the current inputs;
- selected-term and longest-term capacity (clearly separated);
- maximum affordable EMI;
- Loan Plan Fit and exact EMI gap;
- Financial Foundation and emergency-coverage indicators;
- cash remaining before and after the safety buffer;
- alternative repayment terms;
- total interest and repayment;
- calculated steps to make the requested structure fit;
- product-specific LTV/down-payment or salary-multiple guidance.

Recommendations are based on the actual shortfall. Examples include reducing the requested amount, lowering existing monthly debt, increasing stable verified income, improving expense headroom, building a longer income history, or selecting a different term.

## Decision Insights dashboard

The bank view exposes:

- current and requested total debt-service ratios;
- essential-expense ratio;
- requested-plan EMI, affordable EMI, and exact gap;
- every policy rule, observed value, limit, and pass/fail result;
- Financial Foundation component points and disclosed asset-liquidity weights;
- base and stressed repayment scenarios;
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
  "preferred_term_months": 36,
  "assets": {
    "cash": 250000,
    "fixed_deposit": 100000,
    "nps": 500000
  }
}
```

Use `POST /simulate`. The response includes a `repayment_stress` object. If artifacts are not installed it returns an explicit `available: false` state instead of failing the affordability simulation. `GET /products` lists product identifiers, while `GET /health` reports service and ML-artifact availability.

## Streamlit

```bash
streamlit run app.py
```

The professional multi-page experience includes Overview, Loan Affordability, Debt Payoff Planner, and Model Transparency journeys. INR, USD, and GBP are display choices only; changing the symbol does not alter calculations or assumptions.

## Debt payoff API

Use `POST /debt-repayment/simulate` with one to ten debts:

```json
{
  "debts": [
    {
      "name": "Credit card",
      "balance": 120000,
      "annual_interest_rate_percent": 24,
      "minimum_payment": 6000
    },
    {
      "name": "Personal loan",
      "balance": 240000,
      "annual_interest_rate_percent": 12,
      "minimum_payment": 8000
    }
  ],
  "extra_monthly_payment": 5000,
  "strategy": "avalanche"
}
```

The API keeps the total submitted payment budget constant after a debt is cleared and rolls the freed payment into the next priority account.

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
├── src/components/loan_simulator.py    # Affordability, resilience, stress, plans, guidance
├── src/components/debt_repayment.py    # Avalanche/snowball payoff planning
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

Tests cover EMI mathematics, principal recovery, Loan Plan Fit boundaries, selected-term caps, exact gaps, alternative terms, age limits, asset invariants and liquidity weights, emergency coverage, LTV, salary bands, rate stress, deterministic outputs, ML separation, API/domain consistency, and the preserved ML research components.

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

Plan fit in this project means only that submitted values satisfy displayed illustrative affordability rules.

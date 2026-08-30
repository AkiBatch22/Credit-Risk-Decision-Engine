# Loan Affordability Methodology

This document defines the production-facing deterministic methodology. It is an educational simulation built from self-declared values, not a lender decision, credit score, or offer.

## Decision layers

1. **Loan Plan Fit** is the primary answer to whether the requested amount and selected term fit the displayed cash-flow limits.
2. **Financial Foundation** describes pre-loan financial footing. It supports interpretation but cannot override Plan Fit.
3. **Asset Resilience** estimates how long adjusted liquid resources could cover post-loan commitments. Assets never increase affordable EMI.
4. **Rate Stress** recalculates the requested EMI at a higher illustrative rate. It is a sensitivity test, not a second decision.
5. **Product guidance** adds LTV/down-payment context for secured loans or a salary-multiple signal for personal loans.
6. **Historical ML stress** is a separate calibrated comparison with Home Credit outcomes. It cannot change deterministic fields or statuses.

## Monthly repayment capacity

```text
debt_service_capacity
  = income × product DSR ceiling − existing debt payments

cash_flow_capacity
  = income − essential expenses − existing debt payments
    − (income × required residual-buffer ratio)

maximum_affordable_EMI
  = max(min(debt_service_capacity, cash_flow_capacity), 0)
```

Requested EMI uses the standard level-payment amortization formula at the configured annual rate and the selected term. Loan Plan Fit is capped at 100%:

```text
Loan Plan Fit = min(maximum_affordable_EMI / requested_EMI, 1) × 100
EMI shortfall = max(requested_EMI − maximum_affordable_EMI, 0)
```

The headline principal ceiling converts maximum affordable EMI back to principal at the **selected term**. The longest eligible term is calculated separately so a longer term cannot silently inflate the headline answer.

## Assets and emergency liquidity

Only non-negative equity is used for existing property and vehicles:

```text
asset equity = max(market value − outstanding liability, 0)
```

Adjusted liquidity uses these transparent weights:

- Cash: 100%
- Fixed deposits: 95%
- Debt funds: 85%
- Gold: 70%
- Listed equity/equity funds: 60%
- Property equity: 25%
- Vehicle equity: 10%
- EPF/PPF: 15%
- NPS: 0%

Emergency coverage is adjusted liquidity divided by essential expenses, existing debt payments, and requested EMI. Coverage bands are Critical (<1 month), Weak (<3), Moderate (<6), Strong (<12), and Very Strong (12+).

The zero NPS weight is intentional: it acknowledges ownership while excluding a generally inaccessible retirement balance from near-term emergency liquidity.

## Financial Foundation

The indicator totals 100 points:

- Pre-loan cash-flow strength: 30
- Existing debt burden: 20
- Essential-expense burden: 15
- Emergency liquidity: 25
- Income stability: 10

Bands are Strong (80+), Stable (65–79), Limited (45–64), and Weak (<45). Because liquidity contributes only 25 points, assets cannot conceal an unaffordable requested EMI.

## Alternative terms and status

Every age-eligible product term is recalculated for the full requested amount. The engine finds the shortest term whose EMI fits. It also reports the extra total interest versus the selected term.

- `SELECTED_PLAN_FITS`: requested EMI fits and basic requirements pass.
- `LONGER_TERM_REQUIRED`: the full request fits at a longer available term.
- `LOWER_AMOUNT_REQUIRED`: the selected term is already the longest practical term, but supports a smaller amount.
- `LOWER_AMOUNT_AND_TERM_ADJUSTMENT`: a longer term raises capacity but still does not support the full request.
- `REQUEST_DOES_NOT_FIT_AVAILABLE_TERMS`: no available restructuring route is produced.
- `BASIC_PRODUCT_REQUIREMENTS_NOT_MET`: age, stability, product cap, or positive repayment capacity fails.

## Product guidance

- Home loans use an illustrative 80% LTV guide.
- Vehicle loans use an illustrative 85% LTV guide.
- Personal loans show requested principal as a multiple of monthly income: Relatively Conservative (≤6×), Moderate (≤8×), Stretched (≤10×), or Very High (>10×).

LTV guidance is not a substitute for verified valuation, lien priority, product policy, or collateral eligibility.

## Implementation boundaries

- Pure deterministic helpers and configuration live in `src/components/loan_simulator.py`.
- Request validation and serialization live in `api.py`.
- Presentation and plain-language explanations live in `app.py`.
- Historical ML remains in `src/components/repayment_stress.py` and its training pipeline.
- Multi-debt avalanche/snowball mathematics remains independent in `src/components/debt_repayment.py`.

Changing any rate, ceiling, liquidity weight, band, or score weight is a policy change and should be versioned, reviewed, and covered by tests.

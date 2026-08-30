"""Transparent multi-debt repayment planning.

The planner uses only balances, interest rates, minimum payments, and the extra
monthly amount supplied by the user. It is deliberately deterministic: no ML
model or historical borrower outcome is used to predict repayment behaviour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


RepaymentStrategy = Literal["avalanche", "snowball"]


@dataclass(frozen=True)
class DebtAccount:
    """One revolving or instalment debt included in a payoff scenario."""

    name: str
    balance: float
    annual_interest_rate: float
    minimum_payment: float


@dataclass(frozen=True)
class DebtRepaymentRequest:
    """Inputs for an accelerated debt-repayment simulation."""

    debts: tuple[DebtAccount, ...]
    extra_monthly_payment: float
    strategy: RepaymentStrategy = "avalanche"
    maximum_months: int = 600


def validate_debt_repayment_request(request: DebtRepaymentRequest) -> None:
    if not 1 <= len(request.debts) <= 10:
        raise ValueError("Enter between 1 and 10 debts")
    if request.strategy not in {"avalanche", "snowball"}:
        raise ValueError("Strategy must be avalanche or snowball")
    if request.extra_monthly_payment < 0:
        raise ValueError("Extra monthly payment cannot be negative")
    if not 1 <= request.maximum_months <= 1200:
        raise ValueError("Maximum simulation horizon must be between 1 and 1200 months")

    normalized_names: list[str] = []
    for debt in request.debts:
        name = debt.name.strip()
        if not name:
            raise ValueError("Every debt needs a name")
        normalized_names.append(name.casefold())
        if debt.balance <= 0:
            raise ValueError(f"{name}: balance must be positive")
        if not 0 <= debt.annual_interest_rate <= 1:
            raise ValueError(f"{name}: annual interest rate must be between 0% and 100%")
        if debt.minimum_payment <= 0:
            raise ValueError(f"{name}: minimum payment must be positive")
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("Debt names must be unique")


def _priority_order(
    debts: tuple[DebtAccount, ...],
    balances: list[float],
    strategy: RepaymentStrategy,
) -> list[int]:
    active = [index for index, balance in enumerate(balances) if balance > 0.005]
    if strategy == "avalanche":
        return sorted(
            active,
            key=lambda index: (
                -debts[index].annual_interest_rate,
                balances[index],
                debts[index].name.casefold(),
            ),
        )
    return sorted(
        active,
        key=lambda index: (
            balances[index],
            -debts[index].annual_interest_rate,
            debts[index].name.casefold(),
        ),
    )


def _run_schedule(
    debts: tuple[DebtAccount, ...],
    *,
    extra_monthly_payment: float,
    strategy: RepaymentStrategy,
    maximum_months: int,
    roll_over_freed_payments: bool,
) -> dict[str, Any]:
    balances = [debt.balance for debt in debts]
    starting_total = sum(balances)
    committed_budget = sum(debt.minimum_payment for debt in debts) + extra_monthly_payment
    cumulative_interest = 0.0
    cumulative_paid = 0.0
    schedule: list[dict[str, float | int]] = [
        {
            "month": 0,
            "remaining_balance": round(starting_total, 2),
            "payment": 0.0,
            "principal_paid": 0.0,
            "interest_charged": 0.0,
            "cumulative_interest": 0.0,
        }
    ]
    payoff_order: list[dict[str, Any]] = []

    for month in range(1, maximum_months + 1):
        opening_balances = balances.copy()
        interest_charged = [
            balance * debts[index].annual_interest_rate / 12
            for index, balance in enumerate(balances)
        ]
        amounts_due = [
            balance + interest
            for balance, interest in zip(balances, interest_charged, strict=True)
        ]
        payments = [0.0 for _ in debts]

        for index, amount_due in enumerate(amounts_due):
            if amount_due > 0.005:
                payments[index] = min(debts[index].minimum_payment, amount_due)

        if roll_over_freed_payments:
            available_extra = max(committed_budget - sum(payments), 0.0)
        else:
            available_extra = extra_monthly_payment

        for index in _priority_order(debts, amounts_due, strategy):
            if available_extra <= 0.005:
                break
            remaining_due = max(amounts_due[index] - payments[index], 0.0)
            additional_payment = min(remaining_due, available_extra)
            payments[index] += additional_payment
            available_extra -= additional_payment

        balances = [
            max(amount_due - payment, 0.0)
            for amount_due, payment in zip(amounts_due, payments, strict=True)
        ]
        month_interest = sum(interest_charged)
        month_payment = sum(payments)
        principal_paid = max(sum(opening_balances) - sum(balances), 0.0)
        cumulative_interest += month_interest
        cumulative_paid += month_payment

        for index, (opening, closing) in enumerate(
            zip(opening_balances, balances, strict=True)
        ):
            if opening > 0.005 and closing <= 0.005:
                payoff_order.append(
                    {
                        "debt": debts[index].name.strip(),
                        "payoff_month": month,
                        "annual_interest_rate_percent": round(
                            debts[index].annual_interest_rate * 100,
                            2,
                        ),
                    }
                )

        schedule.append(
            {
                "month": month,
                "remaining_balance": round(sum(balances), 2),
                "payment": round(month_payment, 2),
                "principal_paid": round(principal_paid, 2),
                "interest_charged": round(month_interest, 2),
                "cumulative_interest": round(cumulative_interest, 2),
            }
        )
        if sum(balances) <= 0.005:
            return {
                "payoff_possible": True,
                "payoff_months": month,
                "total_interest": round(cumulative_interest, 2),
                "total_paid": round(cumulative_paid, 2),
                "payoff_order": payoff_order,
                "schedule": schedule,
                "remaining_balance": 0.0,
            }

        if month >= 24:
            previous_balance = float(schedule[-13]["remaining_balance"])
            if sum(balances) >= previous_balance - 0.01:
                break

    return {
        "payoff_possible": False,
        "payoff_months": None,
        "total_interest": round(cumulative_interest, 2),
        "total_paid": round(cumulative_paid, 2),
        "payoff_order": payoff_order,
        "schedule": schedule,
        "remaining_balance": round(sum(balances), 2),
    }


def simulate_debt_repayment(request: DebtRepaymentRequest) -> dict[str, Any]:
    """Return a payoff plan and minimum-payment comparison for submitted debts."""

    validate_debt_repayment_request(request)
    accelerated = _run_schedule(
        request.debts,
        extra_monthly_payment=request.extra_monthly_payment,
        strategy=request.strategy,
        maximum_months=request.maximum_months,
        roll_over_freed_payments=True,
    )
    baseline = _run_schedule(
        request.debts,
        extra_monthly_payment=0.0,
        strategy=request.strategy,
        maximum_months=min(max(request.maximum_months, 600), 1200),
        roll_over_freed_payments=False,
    )

    starting_balance = round(sum(debt.balance for debt in request.debts), 2)
    minimum_payment_total = round(
        sum(debt.minimum_payment for debt in request.debts),
        2,
    )
    monthly_plan_payment = round(
        minimum_payment_total + request.extra_monthly_payment,
        2,
    )
    first_target_index = _priority_order(
        request.debts,
        [debt.balance for debt in request.debts],
        request.strategy,
    )[0]

    months_saved = None
    interest_saved = None
    if accelerated["payoff_possible"] and baseline["payoff_possible"]:
        months_saved = max(
            int(baseline["payoff_months"]) - int(accelerated["payoff_months"]),
            0,
        )
        interest_saved = round(
            max(float(baseline["total_interest"]) - float(accelerated["total_interest"]), 0.0),
            2,
        )

    if accelerated["payoff_possible"]:
        status = "ON_TRACK"
        guidance = (
            f"Pay every minimum, then direct the remaining monthly budget to "
            f"{request.debts[first_target_index].name.strip()}. Roll each cleared payment "
            "into the next priority debt."
        )
    else:
        status = "PAYMENT_PLAN_NEEDS_ATTENTION"
        guidance = (
            "The submitted monthly budget did not reduce all balances within the simulation "
            "horizon. Increase the payment budget, verify rates and minimums, or seek qualified "
            "debt guidance before balances compound further."
        )

    return {
        "status": status,
        "strategy": request.strategy,
        "strategy_label": "Highest interest first" if request.strategy == "avalanche" else "Smallest balance first",
        "starting_balance": starting_balance,
        "minimum_payment_total": minimum_payment_total,
        "extra_monthly_payment": round(request.extra_monthly_payment, 2),
        "monthly_plan_payment": monthly_plan_payment,
        "first_priority_debt": request.debts[first_target_index].name.strip(),
        "payoff_possible": accelerated["payoff_possible"],
        "estimated_payoff_months": accelerated["payoff_months"],
        "estimated_total_interest": accelerated["total_interest"],
        "estimated_total_paid": accelerated["total_paid"],
        "estimated_months_saved": months_saved,
        "estimated_interest_saved": interest_saved,
        "minimum_only_payoff_months": baseline["payoff_months"],
        "minimum_only_total_interest": (
            baseline["total_interest"] if baseline["payoff_possible"] else None
        ),
        "payoff_order": accelerated["payoff_order"],
        "schedule": accelerated["schedule"],
        "debts": [asdict(debt) for debt in request.debts],
        "guidance": guidance,
        "assumptions": [
            "Interest is compounded monthly from the submitted annual rates.",
            "Minimum payments are treated as fixed monthly amounts until each debt is cleared.",
            "The accelerated plan keeps the submitted total monthly budget constant and rolls freed payments forward.",
            "Fees, penalties, promotional-rate changes, new borrowing, and missed payments are excluded.",
        ],
        "disclaimer": (
            "Deterministic planning estimate based only on submitted debt details. It is not "
            "credit counselling, a lender quote, or a prediction that payments will be made."
        ),
    }


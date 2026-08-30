import pytest

from src.components.debt_repayment import (
    DebtAccount,
    DebtRepaymentRequest,
    simulate_debt_repayment,
)


def request(**overrides):
    values = {
        "debts": (
            DebtAccount("Credit card", 120_000, 0.24, 6_000),
            DebtAccount("Personal loan", 240_000, 0.12, 8_000),
        ),
        "extra_monthly_payment": 5_000,
        "strategy": "avalanche",
    }
    values.update(overrides)
    return DebtRepaymentRequest(**values)


def test_avalanche_plan_returns_payoff_schedule_and_savings():
    result = simulate_debt_repayment(request())

    assert result["status"] == "ON_TRACK"
    assert result["first_priority_debt"] == "Credit card"
    assert result["monthly_plan_payment"] == 19_000
    assert result["estimated_payoff_months"] > 0
    assert result["estimated_months_saved"] > 0
    assert result["estimated_interest_saved"] > 0
    assert result["schedule"][0]["remaining_balance"] == 360_000
    assert result["schedule"][-1]["remaining_balance"] == 0
    assert {item["debt"] for item in result["payoff_order"]} == {
        "Credit card",
        "Personal loan",
    }


def test_snowball_targets_smallest_balance_first():
    result = simulate_debt_repayment(
        request(
            debts=(
                DebtAccount("Higher balance", 200_000, 0.30, 10_000),
                DebtAccount("Quick win", 40_000, 0.10, 2_000),
            ),
            strategy="snowball",
        )
    )

    assert result["strategy_label"] == "Smallest balance first"
    assert result["first_priority_debt"] == "Quick win"


def test_zero_interest_plan_preserves_payment_identity():
    result = simulate_debt_repayment(
        request(
            debts=(DebtAccount("Family loan", 12_000, 0.0, 1_000),),
            extra_monthly_payment=0,
        )
    )

    assert result["estimated_payoff_months"] == 12
    assert result["estimated_total_interest"] == 0
    assert result["estimated_total_paid"] == pytest.approx(12_000)


@pytest.mark.parametrize(
    "invalid_request, message",
    [
        (request(debts=()), "between 1 and 10"),
        (request(extra_monthly_payment=-1), "cannot be negative"),
        (
            request(
                debts=(
                    DebtAccount("Card", 10_000, 0.20, 500),
                    DebtAccount("card", 20_000, 0.10, 500),
                )
            ),
            "unique",
        ),
        (
            request(debts=(DebtAccount("Card", 10_000, 1.01, 500),)),
            "between 0% and 100%",
        ),
    ],
)
def test_invalid_debt_inputs_are_rejected(invalid_request, message):
    with pytest.raises(ValueError, match=message):
        simulate_debt_repayment(invalid_request)


import numpy as np

from src.components.risk_policy import RiskPolicy, apply_risk_policy, calculate_expected_loss


def test_risk_policy_assigns_decisions_bands_and_expected_loss():
    policy = RiskPolicy(approve_threshold=0.08, reject_threshold=0.15)
    result = apply_risk_policy([0.03, 0.10, 0.25], exposure=[1000, 1000, 1000], policy=policy)
    assert result["recommendation"].tolist() == ["APPROVE", "MANUAL_REVIEW", "REJECT"]
    assert result["risk_band"].tolist() == ["LOW", "HIGH", "VERY_HIGH"]
    np.testing.assert_allclose(result["expected_loss"], [18, 60, 150])


def test_expected_loss_formula_and_validation():
    assert calculate_expected_loss(0.2, 1000, 0.6) == 120
    try:
        calculate_expected_loss(1.1, 1000, 0.6)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("invalid probability should fail")

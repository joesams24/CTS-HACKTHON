def calculate_roi(
    base_cost,
    intervention_cost,
    risk_probability,
    risk_reduction
):
    """
    ROI = Cost avoided - intervention cost
    """

    expected_cost_before = base_cost * risk_probability
    expected_cost_after = base_cost * (risk_probability * (1 - risk_reduction))

    savings = expected_cost_before - expected_cost_after
    roi = savings - intervention_cost

    return {
        "expected_cost_before": round(expected_cost_before, 2),
        "expected_cost_after": round(expected_cost_after, 2),
        "savings": round(savings, 2),
        "roi": round(roi, 2)
    }

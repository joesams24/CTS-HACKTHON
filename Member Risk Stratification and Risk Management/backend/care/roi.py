def calculate_roi(
    base_cost,
    intervention_cost,
    risk_probability,
    risk_reduction
):
    """
    ROI expressed as percentage.

    ROI (%) = ((Savings - Intervention Cost) / Intervention Cost) * 100
    """

    expected_cost_before = base_cost * risk_probability
    expected_cost_after = base_cost * (risk_probability * (1 - risk_reduction))

    savings = expected_cost_before - expected_cost_after

    net_benefit = savings - intervention_cost

    roi_percent = (
        (net_benefit / intervention_cost) * 100
        if intervention_cost > 0 else 0
    )

    return {
        "expected_cost_before": round(expected_cost_before, 2),
        "expected_cost_after": round(expected_cost_after, 2),
        "savings": round(savings, 2),
        "net_benefit": round(net_benefit, 2),
        "roi_percent": round(roi_percent, 2)
    }

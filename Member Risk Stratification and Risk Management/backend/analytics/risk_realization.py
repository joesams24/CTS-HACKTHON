def realized_savings_factor(avg_risk_probability: float) -> float:
    """
    Caps how much of modeled savings are actually realized.
    Industry-standard conservative assumptions.
    """

    if avg_risk_probability < 0.15:
        return 0.60   # low-risk: many false positives
    elif avg_risk_probability < 0.30:
        return 0.70   # medium-risk
    else:
        return 0.75   # very high-risk (still capped)

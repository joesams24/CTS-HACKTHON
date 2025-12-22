def acute_realization_factor(window_days: int) -> float:
    """
    Controls how quickly avoided acute-event savings are realized.
    Prevents unrealistic short-term ROI.
    """
    if window_days <= 30:
        return 0.25
    elif window_days <= 60:
        return 0.5
    else:
        return 1.0


def recommended_intervention_policy(window_days: int):
    """
    Returns recommended intervention strategy and rationale
    based on time horizon.
    """

    if window_days <= 30:
        return {
            "recommendation": "Stabilize only Very High risk members",
            "eligible_tiers": ["Very High"],
            "rationale": (
                "Short horizons do not allow sufficient time for avoided "
                "catastrophic events to fully realize as savings. "
                "Intervening broadly increases cost without proportional benefit."
            )
        }

    elif window_days <= 60:
        return {
            "recommendation": "Full Very High + partial High intervention",
            "eligible_tiers": ["Very High", "High"],
            "rationale": (
                "At mid-term horizons, early avoidance of acute events begins "
                "to offset intervention costs. Targeted expansion improves ROI "
                "while controlling spend."
            )
        }

    else:
        return {
            "recommendation": "Aggressive Very High + High intervention",
            "eligible_tiers": ["Very High", "High"],
            "rationale": (
                "Longer horizons allow compounding benefits from avoided acute "
                "events and chronic stabilization. Broader intervention "
                "maximizes long-term value."
            )
        }

def window_effectiveness_multiplier(window_days: int) -> float:
    """
    Models intervention fatigue / saturation over time.
    """
    if window_days <= 30:
        return 1.00
    elif window_days <= 60:
        return 0.85
    else:
        return 0.70

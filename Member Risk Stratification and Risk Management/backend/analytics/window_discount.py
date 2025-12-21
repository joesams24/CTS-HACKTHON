def window_discount_factor(window_days: int) -> float:
    """
    Discount factor for savings based on time horizon.
    Longer windows are discounted due to uncertainty and delay.
    """

    if window_days <= 30:
        return 0.70
    elif window_days <= 60:
        return 0.85
    else:
        return 1.00

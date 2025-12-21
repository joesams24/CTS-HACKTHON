def simulate_deterioration(base_prob: float, window_days: int, base_window: int = 30):
    """
    Simulates cumulative deterioration risk over time.
    Uses compounding survival probability.
    """
    if base_prob <= 0:
        return 0.0

    time_factor = window_days / base_window
    adjusted_prob = 1 - (1 - base_prob) ** time_factor

    return min(adjusted_prob, 0.999)

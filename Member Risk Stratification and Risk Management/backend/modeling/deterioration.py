def simulate_deterioration(base_probability: float, window_days: int) -> float:
    """
    Simulates health deterioration over a time window.
    Uses non-linear amplification to preserve ranking.
    """

    severity_map = {
        30: 2.0,   # urgent
        60: 1.5,   # moderate
        90: 1.2    # mild
    }

    severity = severity_map.get(window_days, 1.0)

    # Non-linear amplification
    adjusted = 1 - (1 - base_probability) ** severity

    return min(max(adjusted, 0.0), 1.0)

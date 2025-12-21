def diminishing_effectiveness(
    base_effectiveness,
    coverage_ratio,
    knee=0.6,
    decay=0.5
):
    """
    Models diminishing returns after a coverage threshold.

    knee = % coverage where decay starts
    decay = strength of diminishing effect
    """

    if coverage_ratio <= knee:
        return base_effectiveness

    excess = coverage_ratio - knee
    penalty = decay * excess

    effective = base_effectiveness * max(0.1, 1 - penalty)
    return round(effective, 3)

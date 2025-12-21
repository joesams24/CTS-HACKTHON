import numpy as np

def assign_fixed_tiers(probabilities, cutoffs):
    """
    Assign tiers using precomputed cutoffs (frozen baseline tiers)
    """
    tiers = []

    for p in probabilities:
        if p >= cutoffs["Very High"]["min"]:
            tiers.append("Very High")
        elif p >= cutoffs["High"]["min"]:
            tiers.append("High")
        elif p >= cutoffs["Medium"]["min"]:
            tiers.append("Medium")
        elif p >= cutoffs["Low"]["min"]:
            tiers.append("Low")
        else:
            tiers.append("Very Low")

    return tiers


def assign_quantile_tiers(probabilities):
    """
    Assigns risk tiers based on population quantiles.
    Returns:
        tiers: list[str]
        cutoffs: dict (for charts & audit)
    """

    probs = np.array(probabilities)

    # Quantile cutoffs
    q10 = np.quantile(probs, 0.10)
    q30 = np.quantile(probs, 0.30)
    q70 = np.quantile(probs, 0.70)
    q90 = np.quantile(probs, 0.90)

    tiers = []
    for p in probs:
        if p >= q90:
            tiers.append("Very High")
        elif p >= q70:
            tiers.append("High")
        elif p >= q30:
            tiers.append("Medium")
        elif p >= q10:
            tiers.append("Low")
        else:
            tiers.append("Very Low")

    cutoffs = {
        "Very Low": {"max": round(q10, 4)},
        "Low": {"min": round(q10, 4), "max": round(q30, 4)},
        "Medium": {"min": round(q30, 4), "max": round(q70, 4)},
        "High": {"min": round(q70, 4), "max": round(q90, 4)},
        "Very High": {"min": round(q90, 4)}
    }

    return tiers, cutoffs

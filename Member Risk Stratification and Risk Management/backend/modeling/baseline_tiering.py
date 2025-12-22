# modeling/baseline_tiering.py

def assign_tiers_from_cutoffs(probabilities, cutoffs):
    """
    Assign risk tiers using fixed (baseline) cutoffs.
    """
    tiers = []

    for p in probabilities:
        if p <= cutoffs["Very Low"]["max"]:
            tiers.append("Very Low")
        elif p <= cutoffs["Low"]["max"]:
            tiers.append("Low")
        elif p <= cutoffs["Medium"]["max"]:
            tiers.append("Medium")
        elif p <= cutoffs["High"]["max"]:
            tiers.append("High")
        else:
            tiers.append("Very High")

    return tiers

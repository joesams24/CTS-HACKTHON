from collections import defaultdict

def build_migration_matrix(
    tiers_from,
    tiers_to,
    ordered_tiers=None
):
    """
    Builds a tier-to-tier migration matrix.
    """

    if ordered_tiers is None:
        ordered_tiers = [
            "Very Low", "Low", "Medium", "High", "Very High"
        ]

    matrix = {
        t: {t2: 0 for t2 in ordered_tiers}
        for t in ordered_tiers
    }

    for t_from, t_to in zip(tiers_from, tiers_to):
        matrix[t_from][t_to] += 1

    return matrix

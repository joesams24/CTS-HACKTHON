from collections import defaultdict
import numpy as np


def calculate_roi_per_tier(
    probabilities,
    tiers
):
    """
    Aggregates population risk statistics per tier.
    Financial logic is intentionally NOT handled here.
    """

    tier_probs = defaultdict(list)

    # Group probabilities by tier
    for p, t in zip(probabilities, tiers):
        tier_probs[t].append(p)

    tier_roi = {}

    for tier, probs in tier_probs.items():
        avg_prob = float(np.mean(probs))
        member_count = len(probs)

        expected_cost_before = avg_prob * 100_000  # informational only

        tier_roi[tier] = {
            "member_count": member_count,
            "avg_risk_probability": round(avg_prob, 4),
            "expected_cost_before": round(expected_cost_before, 2),
            "expected_cost_after": round(expected_cost_before, 2),
            "savings": 0,
            "net_benefit": 0,
            "roi_percent": 0,
            "eligible": tier in ["High", "Very High"]
        }

    return tier_roi

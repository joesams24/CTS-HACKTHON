# -----------------------------------------
# Map MODEL tiers → BUSINESS tiers
# -----------------------------------------
TIER_MAPPING = {
    "Very Low": "Tier 1 (Very Low)",
    "Low": "Tier 2 (Low)",
    "Medium": "Tier 3 (Medium)",
    "High": "Tier 4 (High)",
    "Very High": "Tier 5 (Very High)",
}


INTERVENTIONS = {
    "Tier 1 (Very Low)": {
        "intervention": "Self-care education",
        "cost": 500,
        "risk_reduction": 0.02
    },
    "Tier 2 (Low)": {
        "intervention": "Telephonic coaching",
        "cost": 1500,
        "risk_reduction": 0.05
    },
    "Tier 3 (Medium)": {
        "intervention": "Nurse follow-up",
        "cost": 4000,
        "risk_reduction": 0.10
    },
    "Tier 4 (High)": {
        "intervention": "Care manager assignment",
        "cost": 8000,
        "risk_reduction": 0.20
    },
    "Tier 5 (Very High)": {
        "intervention": "Intensive care program",
        "cost": 15000,
        "risk_reduction": 0.35
    }
}


def get_intervention(model_risk_tier: str):
    """
    Converts model risk tier → business tier → intervention
    """
    business_tier = TIER_MAPPING.get(model_risk_tier)

    if not business_tier:
        # Defensive fallback (never crash prod systems)
        return {
            "intervention": "No intervention",
            "cost": 0,
            "risk_reduction": 0.0
        }

    return INTERVENTIONS[business_tier]

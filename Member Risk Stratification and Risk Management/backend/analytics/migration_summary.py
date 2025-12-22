ORDERED_TIERS = ["Very Low", "Low", "Medium", "High", "Very High"]

HIGH_RISK = {"High", "Very High"}
LOW_RISK = {"Very Low", "Low", "Medium"}

def summarize_migration(matrix):
    inflow_high = 0
    outflow_low = 0
    upward_moves = 0
    downward_moves = 0

    for from_tier, row in matrix.items():
        for to_tier, count in row.items():
            if from_tier != to_tier:
                if from_tier in LOW_RISK and to_tier in HIGH_RISK:
                    inflow_high += count
                if from_tier in HIGH_RISK and to_tier in LOW_RISK:
                    outflow_low += count

                if ORDERED_TIERS.index(to_tier) > ORDERED_TIERS.index(from_tier):
                    upward_moves += count
                else:
                    downward_moves += count

    return {
        "net_new_high_risk_members": inflow_high,
        "net_recovered_members": outflow_low,
        "total_upward_moves": upward_moves,
        "total_downward_moves": downward_moves
    }

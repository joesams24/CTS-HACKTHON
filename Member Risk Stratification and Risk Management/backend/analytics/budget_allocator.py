def apply_budget_cap(
    tier,
    members,
    cost_per_member,
    max_budget,
    priority_weight=1.0
):
    """
    Caps spend per tier based on remaining budget.
    Higher priority tiers should be processed first.
    """

    required_budget = members * cost_per_member * priority_weight

    if max_budget <= 0:
        return 0, 0

    if required_budget <= max_budget:
        return members, required_budget

    # Partial coverage if budget insufficient
    affordable_members = int(max_budget // cost_per_member)
    actual_spend = affordable_members * cost_per_member

    return affordable_members, actual_spend

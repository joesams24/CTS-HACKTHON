import random
import math


def _catastrophic_jump(base_prob: float, treated: bool) -> float:
    """
    Rare acute deterioration event.
    Represents hospitalization / sudden failure / escalation.
    """

    if base_prob < 0.12:
        return 0.0

    if treated:
        # Treatment greatly reduces probability & severity
        if random.random() < 0.006:   # ↑ from 0.004
            return random.uniform(0.01, 0.03)
    else:
        if random.random() < 0.025:   # ↑ from 0.015
            return random.uniform(0.03, 0.07)

    return 0.0


def simulate_deterioration(
    base_prob: float,
    window_days: int,
    population_mean: float = 0.11,
    treated: bool = False,
    seed: int | None = None
) -> float:
    """
    Realistic bidirectional risk evolution with:
    - Mean reversion
    - Time-dependent drift
    - Treatment awareness
    - Bounded stochastic noise
    - Catastrophic event avoidance
    """

    if seed is not None:
        random.seed(seed)

    # -----------------------------
    # 1. Time scaling
    # -----------------------------
    time_factor = math.log1p(window_days / 30)

    # -----------------------------
    # 2. Mean reversion
    # -----------------------------
    reversion_strength = 0.15
    mean_pull = reversion_strength * (population_mean - base_prob)

    # -----------------------------
    # 3. State-dependent drift
    # -----------------------------
    base_drift = 0.01 * time_factor
    severity_multiplier = 1 + base_prob
    drift = base_drift * severity_multiplier

    # -----------------------------
    # 4. Treatment awareness
    # -----------------------------
    if treated:
        drift -= 0.04 * time_factor  # stronger drift suppression

        if base_prob < 0.05:
            drift *= 0.3  # diminishing returns at low risk

    # -----------------------------
    # 5. Stochastic noise
    # -----------------------------
    noise_scale = 0.012 + 0.02 * base_prob
    noise = random.gauss(0, noise_scale)

    # -----------------------------
    # 6. Catastrophic event jump
    # -----------------------------
    catastrophe = _catastrophic_jump(base_prob, treated)

    # -----------------------------
    # 7. Final update
    # -----------------------------
    new_prob = (
        base_prob
        + mean_pull
        + drift
        + noise
        + catastrophe
    )

    # -----------------------------
    # 8. Hard bounds
    # -----------------------------
    return min(max(new_prob, 0.001), 0.999)

import numpy as np

def probability_to_score(prob):
    return int(prob * 100)

def assign_percentile_tiers(probabilities):
    probs = np.array(probabilities)

    p20 = np.percentile(probs, 20)
    p40 = np.percentile(probs, 40)
    p60 = np.percentile(probs, 60)
    p80 = np.percentile(probs, 80)

    tiers = []
    for p in probs:
        if p <= p20:
            tiers.append("Very Low")
        elif p <= p40:
            tiers.append("Low")
        elif p <= p60:
            tiers.append("Medium")
        elif p <= p80:
            tiers.append("High")
        else:
            tiers.append("Very High")

    return tiers

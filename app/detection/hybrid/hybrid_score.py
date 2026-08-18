from typing import Dict


def calculate_weighted_score(
    scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:

    if not scores:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for detector, score in scores.items():

        weight = weights.get(
            detector,
            0.0,
        )

        if weight <= 0:
            continue

        weighted_sum += (
            score * weight
        )

        total_weight += weight

    if total_weight == 0:
        return 0.0

    result = (
        weighted_sum
        / total_weight
    )

    return min(
        max(result, 0.0),
        1.0,
    )


def classify_hybrid_score(
    score: float,
) -> str:

    if score < 0.3:
        return "normal"

    if score < 0.6:
        return "potential"

    if score < 0.8:
        return "likely_anomaly"

    return "high_confidence"
from typing import Any


def calculate_rule_score(
    value: Any,
    rules: dict,
) -> float:
    """
    Calculate normalized anomaly score for rule violation.

    Score:
        0.0 -> valid
        1.0 -> extremely severe violation
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0

    min_value = rules.get("min")
    max_value = rules.get("max")

    # ==========================================
    # MIN RULE
    # ==========================================

    if min_value is not None and value < min_value:

        distance = min_value - value

        # Scale according to the valid range
        scale = max(
            abs(min_value),
            10.0
        )

        score = distance / scale

        return min(
            1.0,
            max(0.0, score)
        )

    # ==========================================
    # MAX RULE
    # ==========================================

    if max_value is not None and value > max_value:

        distance = value - max_value

        scale = max(
            abs(max_value),
            10.0
        )

        score = distance / scale

        return min(
            1.0,
            max(0.0, score)
        )

    return 0.0


def calculate_final_score(
    scores: list[float],
) -> float:
    """
    Combine multiple detector scores.

    Current strategy:
        final_score = maximum detector score

    This ensures that a strong anomaly
    detected by one detector is not weakened
    by another detector.
    """

    valid_scores = [
        float(score)
        for score in scores
        if score is not None
    ]

    if not valid_scores:
        return 0.0

    return min(
        1.0,
        max(valid_scores)
    )


def calculate_severity(
    score: float,
) -> str:
    """
    Convert normalized anomaly score to severity.

    < 0.40 -> low
    < 0.70 -> medium
    < 0.90 -> high
    >= 0.90 -> critical
    """

    score = max(
        0.0,
        min(1.0, float(score))
    )

    if score < 0.40:
        return "low"

    if score < 0.70:
        return "medium"

    if score < 0.90:
        return "high"

    return "critical"
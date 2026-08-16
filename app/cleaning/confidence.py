def calculate_cleaning_confidence(
    anomaly: dict,
) -> float:

    anomaly_types = set(
        anomaly.get("anomaly_types", [])
    )

    detectors = set(
        anomaly.get("detectors", [])
    )

    score = float(
        anomaly.get("score", 0.0)
    )

    # ==========================================
    # INVALID VALUE
    # ==========================================

    if "invalid_value" in anomaly_types:

        # Rule-based semantic/domain violation
        # is highly reliable for automatic cleaning.

        if "rule" in detectors:
            return 1.0

        if "hybrid" in detectors:
            return 1.0

        return min(
            1.0,
            0.80 + 0.20 * score
        )

    # ==========================================
    # OUTLIER
    # ==========================================

    if "outlier" in anomaly_types:

        # Statistical outliers may be valid.
        # Therefore confidence must be lower
        # even when anomaly score is very high.

        if "statistical" in detectors:

            return min(
                0.85,
                0.50 + 0.35 * score
            )

        return min(
            0.80,
            0.50 + 0.30 * score
        )

    # ==========================================
    # UNKNOWN
    # ==========================================

    return 0.30
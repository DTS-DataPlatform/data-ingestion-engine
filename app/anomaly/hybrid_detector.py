from app.detection.models import AnomalyRecord


def detect_hybrid_anomalies(
    rule_anomalies: list[AnomalyRecord],
    statistical_anomalies: list[AnomalyRecord],
) -> list[AnomalyRecord]:

    grouped = {}

    # ==========================================
    # 1. GROUP RULE ANOMALIES
    # ==========================================

    for anomaly in rule_anomalies:

        key = (
            anomaly.row_index,
            anomaly.column,
        )

        grouped.setdefault(
            key,
            {
                "rule": None,
                "statistical": None,
            }
        )

        grouped[key]["rule"] = anomaly

    # ==========================================
    # 2. GROUP STATISTICAL ANOMALIES
    # ==========================================

    for anomaly in statistical_anomalies:

        key = (
            anomaly.row_index,
            anomaly.column,
        )

        grouped.setdefault(
            key,
            {
                "rule": None,
                "statistical": None,
            }
        )

        grouped[key]["statistical"] = anomaly

    # ==========================================
    # 3. MERGE
    # ==========================================

    final_anomalies = []

    for item in grouped.values():

        rule = item["rule"]
        statistical = item["statistical"]

        # ======================================
        # RULE + STATISTICAL
        # ======================================

        if rule is not None and statistical is not None:

            score = max(
                float(rule.score or 0.0),
                float(statistical.score or 0.0),
            )

            final_anomalies.append(
                AnomalyRecord(
                    row_index=rule.row_index,
                    column=rule.column,
                    value=rule.value,

                    # Hybrid anomaly
                    anomaly_type="invalid_value",

                    detector="hybrid",

                    score=score,

                    method="rule+IQR",

                    severity=(
                        "critical"
                        if score >= 0.90
                        else "high"
                        if score >= 0.70
                        else "medium"
                        if score >= 0.40
                        else "low"
                    ),

                    reason=(
                        f"{rule.reason}; "
                        f"also detected as statistical outlier "
                        f"({statistical.reason})"
                    ),
                )
            )

        # ======================================
        # ONLY RULE
        # ======================================

        elif rule is not None:

            final_anomalies.append(
                AnomalyRecord(
                    row_index=rule.row_index,
                    column=rule.column,
                    value=rule.value,

                    anomaly_type=rule.anomaly_type,

                    detector="rule",

                    score=float(rule.score or 0.0),

                    method=rule.method,

                    severity=rule.severity,

                    reason=rule.reason,
                )
            )

        # ======================================
        # ONLY STATISTICAL
        # ======================================

        elif statistical is not None:

            final_anomalies.append(
                AnomalyRecord(
                    row_index=statistical.row_index,
                    column=statistical.column,
                    value=statistical.value,

                    anomaly_type=statistical.anomaly_type,

                    detector="statistical",

                    score=float(
                        statistical.score or 0.0
                    ),

                    method=statistical.method,

                    severity=statistical.severity,

                    reason=statistical.reason,
                )
            )

    return final_anomalies
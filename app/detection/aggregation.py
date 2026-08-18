from collections import defaultdict

from app.anomaly.models import AnomalyRecord


def aggregate_anomalies(
    anomalies: list[AnomalyRecord],
) -> list[AnomalyRecord]:
    """
    Merge anomalies detected by multiple detectors.

    Anomalies are grouped by row and column.

    Example:

        row 10, age, IQR
        row 10, age, Z-score
        row 10, age, LOF

    becomes:

        row 10, age, hybrid
    """

    if not anomalies:
        return []

    groups = defaultdict(list)

    # ==========================================================
    # 1. GROUP ANOMALIES
    # ==========================================================

    for anomaly in anomalies:

        key = (
            anomaly.row_index,
            anomaly.column,
        )

        groups[key].append(anomaly)

    aggregated = []

    # ==========================================================
    # 2. MERGE EACH GROUP
    # ==========================================================

    for (row_index, column), records in groups.items():

        if len(records) == 1:

            aggregated.append(
                records[0]
            )

            continue

        # ======================================================
        # MULTIPLE DETECTORS
        # ======================================================

        detectors = sorted(
            {
                record.detector
                for record in records
            }
        )

        methods = sorted(
            {
                record.method
                for record in records
            }
        )

        anomaly_types = sorted(
            {
                record.anomaly_type
                for record in records
            }
        )

        # Highest score
        max_score = max(
            record.score
            for record in records
        )

        # Highest severity
        severity_order = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }

        severity = max(
            records,
            key=lambda record:
                severity_order.get(
                    record.severity,
                    0,
                ),
        ).severity

        # Use first non-null value
        value = records[0].value

        # Combine reasons
        reasons = []

        for record in records:

            if record.reason:
                reasons.append(
                    record.reason
                )

        reason = (
            "Detected by multiple methods: "
            + ", ".join(detectors)
            + ". "
            + " | ".join(reasons)
        )

        aggregated.append(
            AnomalyRecord(
                row_index=row_index,

                column=column,

                value=value,

                anomaly_type=(
                    anomaly_types[0]
                ),

                detector="hybrid",

                score=float(max_score),

                method="+".join(
                    methods
                ),

                severity=severity,

                reason=reason,
            )
        )

    return aggregated
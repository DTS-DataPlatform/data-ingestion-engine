from collections import defaultdict

from .models import AnomalyRecord

from .scoring import (
    calculate_final_score,
    calculate_severity,
)


def deduplicate_anomalies(
    anomalies: list[AnomalyRecord],
) -> list[dict]:

    groups = defaultdict(list)

    # ==========================================
    # GROUP
    # ==========================================

    for anomaly in anomalies:

        key = (
            anomaly.row_index,
            anomaly.column,
            str(anomaly.value),
        )

        groups[key].append(anomaly)

    final_anomalies = []

    # ==========================================
    # MERGE
    # ==========================================

    for key, records in groups.items():

        row_index = key[0]
        column = key[1]

        value = records[0].value

        # ======================================
        # DETECTORS
        # ======================================

        detectors = sorted(
            set(
                record.detector
                for record in records
            )
        )

        # ======================================
        # METHODS
        # ======================================

        methods = sorted(
            set(
                record.method
                for record in records
            )
        )

        # ======================================
        # TYPES
        # ======================================

        anomaly_types = sorted(
            set(
                record.anomaly_type
                for record in records
            )
        )

        # ======================================
        # REASONS
        # ======================================

        reasons = []

        for record in records:

            if record.reason not in reasons:

                reasons.append(
                    record.reason
                )

        # ======================================
        # SCORES
        # ======================================

        scores = [
            record.score
            for record in records
            if record.score is not None
        ]

        final_score = calculate_final_score(
            scores
        )

        # ======================================
        # SEVERITY
        # ======================================

        severity = calculate_severity(
            final_score
        )

        # ======================================
        # FINAL ANOMALY
        # ======================================

        anomaly = {

            "row_index": row_index,

            "column": column,

            "value": value,

            "anomaly_types": anomaly_types,

            "detectors": detectors,

            "methods": methods,

            "score": final_score,

            "severity": severity,

            "reasons": reasons,

            "detection_count": len(records),
        }

        final_anomalies.append(
            anomaly
        )

    return final_anomalies
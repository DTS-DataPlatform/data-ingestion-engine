from collections import defaultdict

from .models import HybridAnomalyRecord


def aggregate_anomalies(
    anomalies,
    selected_detectors,
):
    """
    Aggregate anomaly records produced by
    multiple baseline detectors.

    Parameters
    ----------
    anomalies:
        Raw anomaly records produced by
        baseline detectors.

    selected_detectors:
        Detectors that were actually executed.

    Returns
    -------
    list[HybridAnomalyRecord]
    """

    # ==========================================================
    # 1. GROUP ANOMALIES
    # ==========================================================

    groups = defaultdict(list)

    for anomaly in anomalies:

        key = (
            anomaly.row_index,
            anomaly.column,
        )

        groups[key].append(
            anomaly
        )

    # ==========================================================
    # 2. TOTAL DETECTORS
    # ==========================================================

    total_detectors = len(
        selected_detectors
    )

    if total_detectors == 0:
        return []

    # ==========================================================
    # 3. BUILD HYBRID RESULTS
    # ==========================================================

    hybrid_results = []

    for (
        row_index,
        column,
    ), records in groups.items():

        # ------------------------------------------
        # Unique detector names
        # ------------------------------------------

        detector_names = list(
            dict.fromkeys(
                record.detector
                for record in records
            )
        )

        detector_count = len(
            detector_names
        )

        # ------------------------------------------
        # Agreement ratio
        # ------------------------------------------

        agreement_ratio = (
            detector_count
            / total_detectors
        )

        # ------------------------------------------
        # Confidence
        # ------------------------------------------

        confidence = agreement_ratio

        # ------------------------------------------
        # Severity
        # ------------------------------------------

        if confidence >= 0.8:

            severity = "critical"

        elif confidence >= 0.6:

            severity = "high"

        elif confidence >= 0.4:

            severity = "medium"

        else:

            severity = "low"

        # ------------------------------------------
        # Representative value
        # ------------------------------------------

        value = records[0].value

        # ------------------------------------------
        # Anomaly type
        # ------------------------------------------

        anomaly_types = list(
            dict.fromkeys(
                record.anomaly_type
                for record in records
            )
        )

        if len(anomaly_types) == 1:

            anomaly_type = (
                anomaly_types[0]
            )

        else:

            anomaly_type = (
                "multidetector_outlier"
            )

        # ------------------------------------------
        # Explanation
        # ------------------------------------------

        reason = (
            f"Detected by "
            f"{detector_count}/"
            f"{total_detectors} "
            f"selected detectors: "
            f"{', '.join(detector_names)}. "
            f"Agreement ratio="
            f"{agreement_ratio:.2f}."
        )

        # ------------------------------------------
        # Create record
        # ------------------------------------------

        hybrid_results.append(
            HybridAnomalyRecord(

                row_index=row_index,

                column=column,

                value=value,

                detectors=detector_names,

                detector_count=detector_count,

                total_detectors=(
                    total_detectors
                ),

                agreement_ratio=(
                    agreement_ratio
                ),

                confidence=confidence,

                anomaly_type=(
                    anomaly_type
                ),

                severity=severity,

                reason=reason,
            )
        )

    return hybrid_results
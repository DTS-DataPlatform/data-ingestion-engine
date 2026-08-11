from collections import defaultdict

from .models import HybridAnomalyRecord


COLUMN_LEVEL_DETECTORS = {
    "iqr",
    "zscore",
}

MULTIVARIATE_DETECTORS = {
    "isolation_forest",
    "lof",
    "dbscan",
}


def _normalize_detector_name(
    detector: str,
) -> str:

    return detector.lower().strip()


def _get_detection_scope(
    detector: str,
) -> str:

    detector = _normalize_detector_name(
        detector
    )

    if detector in COLUMN_LEVEL_DETECTORS:
        return "column"

    if detector in MULTIVARIATE_DETECTORS:
        return "multivariate"

    return "unknown"


def aggregate_anomalies(
    anomalies,
    selected_detectors,
):
    """
    Aggregate anomaly records from all
    selected detectors.

    Important:
    total_detectors represents the total
    number of selected detectors, not only
    the detectors that produced an anomaly.
    """

    # ==========================================================
    # 1. NORMALIZE SELECTED DETECTORS
    # ==========================================================

    selected_detectors = [
        _normalize_detector_name(
            detector
        )
        for detector in selected_detectors
    ]

    # Remove duplicates but preserve order
    selected_detectors = list(
        dict.fromkeys(
            selected_detectors
        )
    )

    total_detectors = len(
        selected_detectors
    )

    if total_detectors == 0:
        return []

    # ==========================================================
    # 2. GROUP ANOMALIES
    # ==========================================================

    groups = defaultdict(list)

    for anomaly in anomalies:

        detector = _normalize_detector_name(
            anomaly.detector
        )

        scope = _get_detection_scope(
            detector
        )

        # ------------------------------------------------------
        # Column-level anomaly
        # ------------------------------------------------------

        if scope == "column":

            key = (
                "column",
                anomaly.row_index,
                anomaly.column,
            )

        # ------------------------------------------------------
        # Multivariate anomaly
        # ------------------------------------------------------

        elif scope == "multivariate":

            key = (
                "multivariate",
                anomaly.row_index,
            )

        else:

            # Unknown detector:
            # keep original location
            key = (
                "unknown",
                anomaly.row_index,
                anomaly.column,
            )

        groups[key].append(
            anomaly
        )

    # ==========================================================
    # 3. BUILD RESULTS
    # ==========================================================

    hybrid_results = []

    for key, records in groups.items():

        scope = key[0]

        # ------------------------------------------------------
        # Location
        # ------------------------------------------------------

        if scope == "column":

            row_index = key[1]

            column = key[2]

        elif scope == "multivariate":

            row_index = key[1]

            column = "__multivariate__"

        else:

            row_index = key[1]

            column = key[2]

        # ------------------------------------------------------
        # Unique detectors
        # ------------------------------------------------------

        detector_names = list(
            dict.fromkeys(
                _normalize_detector_name(
                    record.detector
                )
                for record in records
            )
        )

        detector_count = len(
            detector_names
        )

        # ------------------------------------------------------
        # IMPORTANT
        #
        # denominator = ALL selected detectors
        # ------------------------------------------------------

        agreement_ratio = (
            detector_count
            / total_detectors
        )

        # ------------------------------------------------------
        # Confidence
        # ------------------------------------------------------

        confidence = agreement_ratio

        # ------------------------------------------------------
        # Severity
        # ------------------------------------------------------

        if confidence >= 0.8:

            severity = "critical"

        elif confidence >= 0.6:

            severity = "high"

        elif confidence >= 0.4:

            severity = "medium"

        else:

            severity = "low"

        # ------------------------------------------------------
        # Representative value
        # ------------------------------------------------------

        value = records[0].value

        # ------------------------------------------------------
        # Anomaly type
        # ------------------------------------------------------

        anomaly_types = list(
            dict.fromkeys(
                record.anomaly_type
                for record in records
            )
        )

        if scope == "multivariate":

            if len(anomaly_types) == 1:

                anomaly_type = (
                    anomaly_types[0]
                )

            else:

                anomaly_type = (
                    "multidetector_outlier"
                )

            # --------------------------------------------------
            # Multivariate explanation
            # --------------------------------------------------

            reason = (
                f"Row detected as anomalous "
                f"by {detector_count}/"
                f"{total_detectors} "
                f"selected multivariate "
                f"detectors: "
                f"{', '.join(detector_names)}. "
                f"Agreement ratio="
                f"{agreement_ratio:.2f}."
            )

        else:

            if len(anomaly_types) == 1:

                anomaly_type = (
                    anomaly_types[0]
                )

            else:

                anomaly_type = (
                    "multidetector_outlier"
                )

            # --------------------------------------------------
            # Column explanation
            # --------------------------------------------------

            reason = (
                f"Detected by "
                f"{detector_count}/"
                f"{total_detectors} "
                f"selected detectors: "
                f"{', '.join(detector_names)}. "
                f"Agreement ratio="
                f"{agreement_ratio:.2f}."
            )

        # ------------------------------------------------------
        # Create result
        # ------------------------------------------------------

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
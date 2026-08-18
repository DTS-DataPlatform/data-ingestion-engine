from collections import defaultdict


def normalize_detector_name(anomaly):
    """
    Convert anomaly method name into the canonical
    detector name used by the hybrid scoring system.
    """

    method = str(
        anomaly.method
    ).lower().strip()

    mapping = {
        "iqr": "iqr",

        "z-score": "zscore",
        "zscore": "zscore",

        "isolation forest": "isolation_forest",
        "isolationforest": "isolation_forest",
        "isolation_forest": "isolation_forest",

        "lof": "lof",

        "dbscan": "dbscan",
    }

    return mapping.get(
        method,
        method,
    )


def aggregate_detector_scores(
    anomalies,
    selected_detectors,
):
    """
    Align normalized anomaly scores produced by
    multiple detectors.

    Anomalies are grouped by:

        (row_index, column)

    Each detector gets one score slot.

    If a detector did not detect the anomaly,
    its score is None.

    Parameters
    ----------
    anomalies:
        Iterable of AnomalyRecord objects.

    selected_detectors:
        List of detectors actually executed.

    Returns
    -------
    list[dict]
        Aligned anomaly records.
    """

    # ==========================================================
    # 1. NORMALIZE SELECTED DETECTOR NAMES
    # ==========================================================

    normalized_selected = []

    for detector in selected_detectors:

        detector = str(
            detector
        ).lower().strip()

        if detector not in normalized_selected:

            normalized_selected.append(
                detector
            )

    total_detectors = len(
        normalized_selected
    )

    if total_detectors == 0:
        return []

    # ==========================================================
    # 2. GROUP ANOMALIES
    # ==========================================================

    groups = defaultdict(list)

    for anomaly in anomalies:

        key = (
            int(anomaly.row_index),
            anomaly.column,
        )

        groups[key].append(
            anomaly
        )

    # ==========================================================
    # 3. BUILD ALIGNED RESULTS
    # ==========================================================

    results = []

    for (
        row_index,
        column,
    ), records in groups.items():

        # ------------------------------------------------------
        # Initialize every detector with None
        # ------------------------------------------------------

        scores = {
            detector: None
            for detector
            in normalized_selected
        }

        # ------------------------------------------------------
        # Store detector metadata
        # ------------------------------------------------------

        detector_metadata = {
            detector: None
            for detector
            in normalized_selected
        }

        # ------------------------------------------------------
        # Fill detected detector scores
        # ------------------------------------------------------

        for record in records:

            detector = normalize_detector_name(
                record
            )

            if detector not in scores:
                continue

            scores[detector] = float(
                record.score
            )

            detector_metadata[detector] = {
                "severity": record.severity,
                "method": record.method,
                "anomaly_type": (
                    record.anomaly_type
                ),
            }

        # ------------------------------------------------------
        # Detectors that actually detected
        # ------------------------------------------------------

        detected_by = [
            detector
            for detector, score
            in scores.items()
            if score is not None
        ]

        detector_count = len(
            detected_by
        )

        # ------------------------------------------------------
        # Agreement ratio
        # ------------------------------------------------------

        agreement_ratio = (
            detector_count
            / total_detectors
        )

        # ------------------------------------------------------
        # Representative value
        # ------------------------------------------------------

        value = records[0].value

        # ------------------------------------------------------
        # Create aligned record
        # ------------------------------------------------------

        results.append(
            {
                "row_index": row_index,

                "column": column,

                "value": value,

                "scores": scores,

                "detector_metadata": (
                    detector_metadata
                ),

                "detected_by": detected_by,

                "detector_count": (
                    detector_count
                ),

                "total_detectors": (
                    total_detectors
                ),

                "agreement_ratio": (
                    float(agreement_ratio)
                ),
            }
        )

    # ==========================================================
    # 4. SORT BY ROW / COLUMN
    # ==========================================================

    results.sort(
        key=lambda item: (
            item["row_index"],
            str(item["column"]),
        )
    )

    return results
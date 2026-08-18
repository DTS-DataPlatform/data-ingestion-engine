from collections import defaultdict


COLUMN_LEVEL_DETECTORS = {
    "iqr",
    "zscore",
}


MULTIVARIATE_DETECTORS = {
    "isolation_forest",
    "lof",
    "dbscan",
}


def normalize_detector_name(
    detector: str,
) -> str:

    return detector.lower().strip()


def is_column_level_detector(
    detector: str,
) -> bool:

    return (
        normalize_detector_name(detector)
        in COLUMN_LEVEL_DETECTORS
    )


def is_multivariate_detector(
    detector: str,
) -> bool:

    return (
        normalize_detector_name(detector)
        in MULTIVARIATE_DETECTORS
    )


def collect_detector_evidence(
    anomalies,
    selected_detectors,
):
    """
    Separate anomaly evidence into:

    1. column-level evidence
    2. multivariate evidence

    Returns
    -------
    dict
    """

    column_evidence = defaultdict(list)

    multivariate_evidence = defaultdict(list)

    for anomaly in anomalies:

        detector = normalize_detector_name(
            anomaly.detector
        )

        # ------------------------------------------
        # Column-level
        # ------------------------------------------

        if is_column_level_detector(
            detector
        ):

            key = (
                anomaly.row_index,
                anomaly.column,
            )

            column_evidence[key].append(
                anomaly
            )

        # ------------------------------------------
        # Multivariate
        # ------------------------------------------

        elif is_multivariate_detector(
            detector
        ):

            key = anomaly.row_index

            multivariate_evidence[key].append(
                anomaly
            )

    return {
        "column": dict(
            column_evidence
        ),
        "multivariate": dict(
            multivariate_evidence
        ),
    }
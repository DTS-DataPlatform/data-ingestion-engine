"""
Score normalization utilities for hybrid anomaly detection.

Different anomaly detectors produce scores on different scales.

Examples:

    IQR:
        0, 1, 3, 10, 96, ...

    Z-score:
        0, 1, 2, 3, 5, ...

    LOF:
        around 1.0 = normal
        > 1.0 = increasingly anomalous

The hybrid engine needs to transform these scores
into a common [0, 1] scale before combining them.
"""


# ==========================================================
# GENERIC VALIDATION
# ==========================================================


def _clamp(
    value: float,
) -> float:
    """
    Clamp a value into [0, 1].
    """

    return min(
        max(
            float(value),
            0.0,
        ),
        1.0,
    )


# ==========================================================
# IQR
# ==========================================================


def normalize_iqr_score(
    score: float,
) -> float:
    """
    Normalize an IQR anomaly score into [0, 1].

    The IQR detector currently uses:

        score = distance / IQR

    Therefore:

        score = 0
            -> normal

        score > 0
            -> increasingly anomalous

    We use a saturating transformation:

        normalized = score / (score + 1)

    Examples
    --------
    0   -> 0.00
    1   -> 0.50
    3   -> 0.75
    9   -> 0.90
    99  -> 0.99
    """

    if score <= 0:
        return 0.0

    normalized = (
        score
        / (score + 1.0)
    )

    return _clamp(
        normalized
    )


# ==========================================================
# Z-SCORE
# ==========================================================


def normalize_zscore_score(
    score: float,
) -> float:
    """
    Normalize an absolute Z-score into [0, 1].

    Z-score detector uses:

        absolute_z = abs(z)

    Therefore:

        0 -> normal
        larger |Z| -> more anomalous

    Transformation:

        normalized = |Z| / (|Z| + 1)
    """

    if score <= 0:
        return 0.0

    normalized = (
        score
        / (score + 1.0)
    )

    return _clamp(
        normalized
    )


# ==========================================================
# LOF
# ==========================================================


def normalize_lof_score(
    score: float,
) -> float:
    """
    Normalize a Local Outlier Factor score into [0, 1].

    LOF interpretation:

        LOF ~= 1
            -> normal

        LOF > 1
            -> increasingly anomalous

    Therefore:

        score <= 1
            -> 0

        score > 1
            -> (score - 1) / score

    Examples
    --------
    1.0 -> 0.00
    1.5 -> 0.33
    2.0 -> 0.50
    3.0 -> 0.67
    5.0 -> 0.80
    """

    if score <= 1.0:
        return 0.0

    normalized = (
        (score - 1.0)
        / score
    )

    return _clamp(
        normalized
    )


# ==========================================================
# ISOLATION FOREST
# ==========================================================


def normalize_isolation_forest_score(
    score: float,
) -> float:
    """
    Normalize an Isolation Forest anomaly score.

    IMPORTANT
    ---------
    The exact interpretation depends on what the detector
    returns.

    This function assumes the input score has already been
    transformed so that:

        larger score = more anomalous

    and is approximately in:

        [0, 1]

    Therefore this function mainly guarantees that the score
    is safely bounded.

    If the Isolation Forest detector uses a different score
    representation, this function should be adjusted to that
    representation.
    """

    return _clamp(
        score
    )


# ==========================================================
# SEMANTIC SCORE
# ==========================================================


def normalize_semantic_score(
    score: float,
) -> float:
    """
    Normalize semantic anomaly evidence.

    Semantic score is expected to already represent:

        0.0 -> no semantic violation
        1.0 -> strong semantic violation

    Any value outside [0, 1] is clamped.
    """

    return _clamp(
        score
    )


# ==========================================================
# GENERIC NORMALIZER
# ==========================================================
def normalize_dbscan_score(
    score: float,
) -> float:
    """
    Normalize DBSCAN anomaly score
    to [0, 1].
    """

    score = float(score)

    return min(
        max(score, 0.0),
        1.0,
    )

def normalize_detector_score(
    detector: str,
    score: float,
) -> float:
    """
    Normalize a detector score according to detector type.

    Parameters
    ----------
    detector:
        Detector name.

    score:
        Raw detector score.

    Returns
    -------
    float
        Normalized anomaly score in [0, 1].
    """

    detector = detector.lower().strip()

    # ==========================================
    # IQR
    # ==========================================

    if detector == "iqr":

        return normalize_iqr_score(
            score
        )

    # ==========================================
    # Z-SCORE
    # ==========================================

    if detector == "zscore":

        return normalize_zscore_score(
            score
        )

    # ==========================================
    # LOF
    # ==========================================

    if detector == "lof":

        return normalize_lof_score(
            score
        )

    # ==========================================
    # ISOLATION FOREST
    # ==========================================

    if detector in {
        "isolation_forest",
        "isolationforest",
    }:

        return normalize_isolation_forest_score(
            score
        )

    # ==========================================
    # DBSCAN
    # ==========================================

    if detector == "dbscan":

        return normalize_dbscan_score(
            score
        )

    # ==========================================
    # SEMANTIC
    # ==========================================

    if detector == "semantic":

        return normalize_semantic_score(
            score
        )

    # ==========================================
    # UNKNOWN DETECTOR
    # ==========================================

    raise ValueError(
        f"Unsupported detector: {detector}"
    )
    
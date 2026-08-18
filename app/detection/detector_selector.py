def select_detectors(
    characteristics,
):

    detectors = []

    # ==========================================
    # IQR
    # ==========================================

    if characteristics.numeric_columns > 0:

        detectors.append("iqr")

    # ==========================================
    # Z-SCORE
    # ==========================================

    if (
        characteristics.numeric_columns > 0
        and not characteristics.skewed
    ):

        detectors.append("zscore")

    # ==========================================
    # ISOLATION FOREST
    # ==========================================

    if (
        characteristics.rows >= 50
        and characteristics.numeric_columns > 0
    ):

        detectors.append(
            "isolation_forest"
        )

    # ==========================================
    # LOF
    # ==========================================

    if (
        characteristics.rows >= 20
        and characteristics.numeric_columns >= 2
    ):

        detectors.append(
            "lof"
        )

    # ==========================================
    # DBSCAN
    # ==========================================

    if (
        characteristics.rows >= 20
        and characteristics.numeric_columns >= 2
    ):

        detectors.append(
            "dbscan"
        )

    return detectors
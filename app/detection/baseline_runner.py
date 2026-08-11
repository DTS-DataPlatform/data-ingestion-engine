from typing import Any

from app.detection.detector_selector import select_detectors

from app.detection.baseline.iqr_detector import (
    detect_iqr_anomalies,
)

from app.detection.baseline.zscore_detector import (
    detect_zscore_anomalies,
)

from app.detection.baseline.isolation_forest_detector import (
    detect_isolation_forest_anomalies,
)

from app.detection.baseline.lof_detector import (
    detect_lof_anomalies,
)

from app.detection.baseline.dbscan_detector import (
    detect_dbscan_anomalies,
)

from app.detection.hybrid.aggregator import (
    aggregate_anomalies,
)

def run_baseline_detection(
    df,
    profiles,
    characteristics,
) -> dict[str, Any]:

    # ==========================================================
    # 1. SELECT DETECTORS
    # ==========================================================

    selected_detectors = select_detectors(
        characteristics
    )

    # ==========================================================
    # 2. INITIAL RESULT
    # ==========================================================

    results = {
        "selected_detectors": selected_detectors,
        "raw_anomalies": [],
        "anomalies": [],
    }

    # ==========================================================
    # 3. IQR
    # ==========================================================

    if "iqr" in selected_detectors:

        results["iqr"] = detect_iqr_anomalies(
            df,
            profiles,
        )

    # ==========================================================
    # 4. Z-SCORE
    # ==========================================================

    if "zscore" in selected_detectors:

        results["zscore"] = detect_zscore_anomalies(
            df,
            profiles,
        )

    # ==========================================================
    # 5. ISOLATION FOREST
    # ==========================================================

    if "isolation_forest" in selected_detectors:

        results["isolation_forest"] = (
            detect_isolation_forest_anomalies(
                df,
                profiles,
            )
        )

    # ==========================================================
    # 6. LOF
    # ==========================================================

    if "lof" in selected_detectors:

        results["lof"] = detect_lof_anomalies(
            df,
            profiles,
        )

    # ==========================================================
    # 7. DBSCAN
    # ==========================================================

    if "dbscan" in selected_detectors:

        results["dbscan"] = detect_dbscan_anomalies(
            df,
            profiles,
        )

    # ==========================================================
    # 8. COLLECT RAW ANOMALIES
    # ==========================================================

    raw_anomalies = []

    for detector_name in (
        "iqr",
        "zscore",
        "isolation_forest",
        "lof",
        "dbscan",
    ):

        raw_anomalies.extend(
            results.get(
                detector_name,
                [],
            )
        )

    results["raw_anomalies"] = raw_anomalies

    # ==========================================================
    # 9. BASELINE ANOMALIES
    # ==========================================================

    results["anomalies"] = list(
        raw_anomalies
    )
    
    hybrid_anomalies = aggregate_anomalies(
    raw_anomalies,
    selected_detectors,
)

    results["hybrid_anomalies"] = (
        hybrid_anomalies
    )

    return results
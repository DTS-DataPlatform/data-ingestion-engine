from app.detection.baseline.iqr_detector import detect_iqr_anomalies
from app.detection.baseline.zscore_detector import detect_zscore_anomalies

# Sau này import thêm
# from app.detection.baseline.isolation_forest_detector import detect_isolation_forest_anomalies
# from app.detection.baseline.lof_detector import detect_lof_anomalies
# from app.detection.baseline.dbscan_detector import detect_dbscan_anomalies


def run_selected_detectors(
    df,
    profiles,
    selected_detectors,
):
    anomalies = []

    for detector in selected_detectors:

        # ==========================================
        # IQR
        # ==========================================

        if detector == "iqr":

            result = detect_iqr_anomalies(
                df,
                profiles,
            )

            anomalies.extend(result)

        # ==========================================
        # Z-SCORE
        # ==========================================

        elif detector == "zscore":

            result = detect_zscore_anomalies(
                df,
                profiles,
            )

            anomalies.extend(result)

        # ==========================================
        # ISOLATION FOREST
        # ==========================================

        elif detector == "isolation_forest":

            # sẽ implement sau
            continue

        # ==========================================
        # LOF
        # ==========================================

        elif detector == "lof":

            # sẽ implement sau
            continue

        # ==========================================
        # DBSCAN
        # ==========================================

        elif detector == "dbscan":

            # sẽ implement sau
            continue

    return anomalies
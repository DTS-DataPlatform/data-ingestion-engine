import pandas as pd

from ..models import AnomalyRecord


def detect_zscore_anomalies(
    df: pd.DataFrame,
    profiles,
    threshold: float = 2.0,
):
    anomalies = []

    for profile in profiles:

        # ==========================================
        # Only numerical columns
        # ==========================================

        if profile.mean is None:
            continue

        numeric_series = pd.to_numeric(
            df[profile.name],
            errors="coerce",
        )

        clean_series = numeric_series.dropna()

        # ==========================================
        # Dataset too small
        # ==========================================

        if len(clean_series) < 4:
            continue

        # ==========================================
        # Mean / Standard Deviation
        # ==========================================

        mean = clean_series.mean()
        std = clean_series.std()

        # ==========================================
        # Constant column
        # ==========================================

        if std == 0 or pd.isna(std):
            continue

        # ==========================================
        # Detect anomalies
        # ==========================================

        for index, value in numeric_series.items():

            # Ignore missing values
            if pd.isna(value):
                continue

            # --------------------------------------
            # Calculate Z-score
            # --------------------------------------

            z_score = (value - mean) / std

            absolute_z = abs(z_score)

            # --------------------------------------
            # Check threshold
            # --------------------------------------

            if absolute_z <= threshold:
                continue

            # ======================================
            # Severity
            # ======================================

            if absolute_z < 4:
                severity = "medium"

            elif absolute_z < 6:
                severity = "high"

            else:
                severity = "critical"

            # ======================================
            # Create anomaly record
            # ======================================

            anomalies.append(
                AnomalyRecord(
                    row_index=int(index),
                    column=profile.name,
                    value=float(value),
                    anomaly_type="outlier",
                    detector="zscore",
                    score=float(absolute_z),
                    method="Z-score",
                    severity=severity,
                    reason=(
                        f"|Z|={absolute_z:.2f} "
                        f"> threshold={threshold:.2f}"
                    ),
                )
            )

    return anomalies
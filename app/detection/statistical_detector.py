import pandas as pd

from .models import AnomalyRecord


def detect_iqr_anomalies(
    df: pd.DataFrame,
    profiles
):

    anomalies = []

    for profile in profiles:

        # Chỉ xử lý numerical columns
        if profile.mean is None:
            continue

        numeric_series = pd.to_numeric(
            df[profile.name],
            errors="coerce"
        )

        clean_series = numeric_series.dropna()

        if len(clean_series) < 4:
            continue

        # =========================
        # Calculate IQR
        # =========================

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        # =========================
        # Detect anomalies
        # =========================

        for index, value in numeric_series.items():

            if pd.isna(value):
                continue

            if (
                value < lower_bound
                or value > upper_bound
            ):

                # Distance outside boundary
                if value < lower_bound:

                    distance = (
                        lower_bound - value
                    )

                else:

                    distance = (
                        value - upper_bound
                    )

                # Simple normalized score
                score = (
                    distance / iqr
                )
                if score < 0.5:
                    severity = "low"
                elif score < 3:
                    severity = "medium"
                else:
                    severity = "high"
                
                anomalies.append(
                    AnomalyRecord(
                        row_index=int(index),

                        column=profile.name,

                        value=float(value),

                        anomaly_type="outlier",

                        detector="statistical",

                        score=float(score),

                        method="IQR",
                        severity=severity,

                        reason=(
                            f"Value is outside "
                            f"IQR bounds "
                            f"[{lower_bound:.2f}, "
                            f"{upper_bound:.2f}]"
                        ),
                    )
                )

    return anomalies
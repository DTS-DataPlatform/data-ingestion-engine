import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..models import AnomalyRecord


def detect_dbscan_anomalies(
    df: pd.DataFrame,
    profiles,
    eps: float = 0.5,
    min_samples: int = 5,
):
    anomalies = []

    # ==========================================
    # Numerical columns
    # ==========================================

    numeric_columns = []

    for profile in profiles:

        if profile.mean is not None:
            numeric_columns.append(
                profile.name
            )

    if not numeric_columns:
        return anomalies

    numeric_df = (
        df[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
    )

    if len(numeric_df) < min_samples:
        return anomalies

    # ==========================================
    # Standardization
    # ==========================================

    scaler = StandardScaler()

    X = scaler.fit_transform(
        numeric_df
    )

    # ==========================================
    # DBSCAN
    # ==========================================

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = model.fit_predict(X)

    # ==========================================
    # Noise = -1
    # ==========================================

    for position, label in enumerate(
        labels
    ):

        if label != -1:
            continue

        row_index = numeric_df.index[
            position
        ]

        anomalies.append(
            AnomalyRecord(
                row_index=int(row_index),
                column="__row__",
                value=numeric_df.loc[
                    row_index
                ].to_dict(),
                anomaly_type="noise",
                detector="clustering",
                score=1.0,
                method="DBSCAN",
                severity="medium",
                reason=(
                    "DBSCAN classified "
                    "this row as noise."
                ),
            )
        )

    return anomalies
import pandas as pd

from sklearn.ensemble import IsolationForest

from ..models import AnomalyRecord


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    profiles,
    contamination: float = "auto",
    random_state: int = 42,
):
    anomalies = []

    # ==========================================
    # Select numerical columns
    # ==========================================

    numeric_columns = []

    for profile in profiles:

        if profile.mean is not None:

            numeric_columns.append(
                profile.name
            )

    if not numeric_columns:
        return anomalies

    # ==========================================
    # Build numerical dataframe
    # ==========================================

    numeric_df = df[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Isolation Forest không xử lý NaN
    numeric_df = numeric_df.dropna()

    if len(numeric_df) < 5:
        return anomalies

    # ==========================================
    # Model
    # ==========================================

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )

    predictions = model.fit_predict(
        numeric_df
    )

    decision_scores = (
        model.decision_function(
            numeric_df
        )
    )

    # ==========================================
    # Detect anomalies
    # ==========================================

    for position, prediction in enumerate(
        predictions
    ):

        if prediction != -1:
            continue

        row_index = numeric_df.index[
            position
        ]

        score = float(
            -decision_scores[position]
        )

        if score < 0.1:
            severity = "low"

        elif score < 0.3:
            severity = "medium"

        elif score < 0.5:
            severity = "high"

        else:
            severity = "critical"

        # Một row có thể bất thường
        # do nhiều numerical columns.
        # Tạo anomaly cho row đó.

        values = numeric_df.loc[
            row_index
        ].to_dict()

        anomalies.append(
            AnomalyRecord(
                row_index=int(row_index),
                column="__row__",
                value=values,
                anomaly_type="multivariate_outlier",
                detector="machine_learning",
                score=score,
                method="IsolationForest",
                severity=severity,
                reason=(
                    "Isolation Forest "
                    "identified this row "
                    "as difficult to isolate."
                ),
            )
        )

    return anomalies
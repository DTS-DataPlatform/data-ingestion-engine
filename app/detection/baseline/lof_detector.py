import pandas as pd

from sklearn.neighbors import LocalOutlierFactor

from app.anomaly.models import AnomalyRecord


def detect_lof_anomalies(
    df: pd.DataFrame,
    profiles,
):
    """
    Detect multivariate anomalies using
    Local Outlier Factor (LOF).

    LOF works on multiple numerical columns.

    Strategy:
    - Automatically select numerical columns.
    - Remove rows containing NaN in selected features.
    - Adapt n_neighbors according to dataset size.
    - Use contamination="auto".
    - Convert detected observations into AnomalyRecord.
    """

    anomalies = []

    # ==========================================================
    # 1. COLLECT NUMERIC COLUMNS
    # ==========================================================

    numeric_columns = []

    for profile in profiles:

        if profile.mean is None:
            continue

        numeric_series = pd.to_numeric(
            df[profile.name],
            errors="coerce",
        )

        numeric_ratio = (
            numeric_series.notna().mean()
        )

        if numeric_ratio >= 0.8:
            numeric_columns.append(
                profile.name
            )

    # LOF needs at least 2 dimensions
    if len(numeric_columns) < 2:
        return anomalies

    # ==========================================================
    # 2. BUILD NUMERIC MATRIX
    # ==========================================================

    numeric_df = (
        df[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # LOF cannot handle NaN
    valid_mask = (
        numeric_df.notna().all(axis=1)
    )

    clean_df = numeric_df.loc[
        valid_mask
    ]

    # Need enough observations
    if len(clean_df) < 4:
        return anomalies

    # ==========================================================
    # 3. SELECT NUMBER OF NEIGHBORS
    # ==========================================================

    n_samples = len(clean_df)

    # ----------------------------------------------------------
    # Small dataset
    # ----------------------------------------------------------
    #
    # For small datasets, using too many neighbors makes LOF
    # behave almost globally instead of locally.
    #
    # Example:
    #
    # 7 samples -> 3 neighbors
    # 10 samples -> 4 neighbors
    #
    # ----------------------------------------------------------

    if n_samples <= 10:

        n_neighbors = max(
            2,
            min(
                3,
                n_samples - 1,
            ),
        )

    else:

        n_neighbors = min(
            20,
            max(
                5,
                int(n_samples * 0.1),
            ),
            n_samples - 1,
        )

    # ==========================================================
    # 4. RUN LOF
    # ==========================================================

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination="auto",
    )

    predictions = model.fit_predict(
        clean_df
    )

    # negative_outlier_factor_:
    #
    # normal point  -> approximately -1
    # anomaly       -> significantly below -1
    #
    lof_scores = (
        -model.negative_outlier_factor_
    )

    # ==========================================================
    # 5. BUILD ANOMALY RECORDS
    # ==========================================================

    clean_indices = clean_df.index

    for position, prediction in enumerate(
        predictions
    ):

        # +1 = normal
        # -1 = anomaly

        if prediction != -1:
            continue

        row_index = int(
            clean_indices[position]
        )

        score = float(
            lof_scores[position]
        )

        # ======================================================
        # 6. SEVERITY
        # ======================================================

        if score >= 3:
            severity = "critical"

        elif score >= 2:
            severity = "high"

        elif score >= 1.5:
            severity = "medium"

        else:
            severity = "low"

        # ======================================================
        # 7. COLLECT FEATURE VALUES
        # ======================================================

        values = {}

        for column in numeric_columns:

            values[column] = df.loc[
                row_index,
                column,
            ]

        # ======================================================
        # 8. CREATE ANOMALY RECORD
        # ======================================================

        anomalies.append(
            AnomalyRecord(
                row_index=row_index,

                column="__multivariate__",

                value=values,

                anomaly_type="outlier",

                detector="lof",

                score=score,

                method="LOF",

                severity=severity,

                reason=(
                    "Multivariate observation has "
                    "high local outlier factor "
                    f"(LOF={score:.3f}) "
                    f"using features "
                    f"{numeric_columns}."
                ),
            )
        )

    return anomalies
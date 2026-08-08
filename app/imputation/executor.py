import pandas as pd


SUPPORTED_STRATEGIES = {
    "mean",
    "median",
    "mode",
    "ffill",
    "bfill",
    "skip",
    "review",
}


def execute_imputation(
    dataframe: pd.DataFrame,
    column=None,
    strategy=None,
    recommendations=None,
):
    """
    Execute missing-value imputation.

    Supported strategies:
        - mean
        - median
        - mode
        - ffill
        - bfill
        - skip
        - review

    Direct mode:
        execute_imputation(
            df,
            column="age",
            strategy="median",
        )

    Recommendation mode:
        execute_imputation(
            df,
            recommendations,
        )

    Or:
        execute_imputation(
            df,
            recommendations=recommendations,
        )

    Returns
    -------
    Direct mode:
        pd.DataFrame

    Recommendation mode:
        tuple[pd.DataFrame, list[dict]]
    """

    # ==========================================================
    # COPY ORIGINAL DATAFRAME
    # ==========================================================

    cleaned_df = dataframe.copy()

    # ==========================================================
    # DETECT RECOMMENDATION MODE
    # ==========================================================

    if (
        recommendations is None
        and isinstance(column, (list, tuple))
    ):
        recommendations = column
        column = None

    # ==========================================================
    # DIRECT MODE
    # ==========================================================

    if column is not None:

        # ------------------------------------------------------
        # COLUMN NOT FOUND
        # ------------------------------------------------------

        if column not in cleaned_df.columns:
            return cleaned_df

        # ------------------------------------------------------
        # SKIP / REVIEW
        # ------------------------------------------------------

        if strategy in ("skip", "review"):
            return cleaned_df

        # ------------------------------------------------------
        # UNKNOWN STRATEGY
        # ------------------------------------------------------

        if strategy not in SUPPORTED_STRATEGIES:
            return cleaned_df

        series = cleaned_df[column]

        # ------------------------------------------------------
        # MEAN
        # ------------------------------------------------------

        if strategy == "mean":

            if not pd.api.types.is_numeric_dtype(series):
                return cleaned_df

            non_null = series.dropna()

            if non_null.empty:
                return cleaned_df

            fill_value = series.mean()

            cleaned_df[column] = series.fillna(
                fill_value
            )

        # ------------------------------------------------------
        # MEDIAN
        # ------------------------------------------------------

        elif strategy == "median":

            if not pd.api.types.is_numeric_dtype(series):
                return cleaned_df

            non_null = series.dropna()

            if non_null.empty:
                return cleaned_df

            fill_value = series.median()

            cleaned_df[column] = series.fillna(
                fill_value
            )

        # ------------------------------------------------------
        # MODE
        # ------------------------------------------------------

        elif strategy == "mode":

            modes = series.mode()

            if modes.empty:
                return cleaned_df

            fill_value = modes.iloc[0]

            cleaned_df[column] = series.fillna(
                fill_value
            )

        # ------------------------------------------------------
        # FORWARD FILL
        # ------------------------------------------------------

        elif strategy == "ffill":

            cleaned_df[column] = (
                series.ffill()
            )

        # ------------------------------------------------------
        # BACKWARD FILL
        # ------------------------------------------------------

        elif strategy == "bfill":

            cleaned_df[column] = (
                series.bfill()
            )

        # ------------------------------------------------------
        # SAFETY
        # ------------------------------------------------------

        else:
            return cleaned_df

        return cleaned_df

    # ==========================================================
    # NO RECOMMENDATIONS
    # ==========================================================

    if recommendations is None:
        return cleaned_df, []

    logs = []

    # ==========================================================
    # PROCESS EACH RECOMMENDATION
    # ==========================================================

    for recommendation in recommendations:

        # ======================================================
        # GET RECOMMENDATION DATA
        # ======================================================

        if isinstance(recommendation, dict):

            rec_column = recommendation.get(
                "column"
            )

            rec_strategy = recommendation.get(
                "strategy"
            )

            confidence = recommendation.get(
                "confidence",
                None,
            )

            reason = recommendation.get(
                "reason",
                "",
            )

        else:

            rec_column = recommendation.column

            rec_strategy = recommendation.strategy

            confidence = getattr(
                recommendation,
                "confidence",
                None,
            )

            reason = getattr(
                recommendation,
                "reason",
                "",
            )

        # ======================================================
        # COLUMN NOT FOUND
        # ======================================================

        if rec_column not in cleaned_df.columns:

            logs.append(
                {
                    "column": rec_column,
                    "strategy": rec_strategy,
                    "action": "skip",
                    "status": "skipped",
                    "reason": "Column not found.",
                }
            )

            continue

        # ======================================================
        # SKIP / REVIEW
        # ======================================================

        if rec_strategy in (
            "skip",
            "review",
        ):

            logs.append(
                {
                    "column": rec_column,
                    "strategy": rec_strategy,
                    "action": rec_strategy,
                    "status": (
                        "review_required"
                        if rec_strategy == "review"
                        else "skipped"
                    ),
                    "reason": reason,
                    "confidence": confidence,
                }
            )

            continue

        # ======================================================
        # UNKNOWN STRATEGY
        # ======================================================

        if rec_strategy not in SUPPORTED_STRATEGIES:

            logs.append(
                {
                    "column": rec_column,
                    "strategy": rec_strategy,
                    "action": "skip",
                    "status": "skipped",
                    "reason": (
                        "Unknown imputation strategy."
                    ),
                    "confidence": confidence,
                }
            )

            continue

        # ======================================================
        # GET SERIES
        # ======================================================

        series = cleaned_df[rec_column]

        # ======================================================
        # COUNT MISSING VALUES
        # ======================================================

        missing_count = int(
            series.isna().sum()
        )

        # ======================================================
        # NOTHING TO IMPUTE
        # ======================================================

        if missing_count == 0:

            logs.append(
                {
                    "column": rec_column,
                    "strategy": rec_strategy,
                    "action": "skip",
                    "status": "skipped",
                    "filled_count": 0,
                    "reason": (
                        "Column contains no "
                        "missing values."
                    ),
                    "confidence": confidence,
                }
            )

            continue

        # ======================================================
        # MEAN
        # ======================================================

        if rec_strategy == "mean":

            if not pd.api.types.is_numeric_dtype(
                series
            ):

                logs.append(
                    {
                        "column": rec_column,
                        "strategy": rec_strategy,
                        "action": "skip",
                        "status": "skipped",
                        "filled_count": 0,
                        "reason": (
                            "Mean imputation requires "
                            "a numeric column."
                        ),
                        "confidence": confidence,
                    }
                )

                continue

            non_null = series.dropna()

            if non_null.empty:

                logs.append(
                    {
                        "column": rec_column,
                        "strategy": rec_strategy,
                        "action": "skip",
                        "status": "skipped",
                        "filled_count": 0,
                        "reason": (
                            "No valid values available "
                            "to calculate mean."
                        ),
                        "confidence": confidence,
                    }
                )

                continue

            fill_value = series.mean()

            cleaned_df[rec_column] = (
                series.fillna(fill_value)
            )

        # ======================================================
        # MEDIAN
        # ======================================================

        elif rec_strategy == "median":

            if not pd.api.types.is_numeric_dtype(
                series
            ):

                logs.append(
                    {
                        "column": rec_column,
                        "strategy": rec_strategy,
                        "action": "skip",
                        "status": "skipped",
                        "filled_count": 0,
                        "reason": (
                            "Median imputation requires "
                            "a numeric column."
                        ),
                        "confidence": confidence,
                    }
                )

                continue

            non_null = series.dropna()

            if non_null.empty:

                logs.append(
                    {
                        "column": rec_column,
                        "strategy": rec_strategy,
                        "action": "skip",
                        "status": "skipped",
                        "filled_count": 0,
                        "reason": (
                            "No valid values available "
                            "to calculate median."
                        ),
                        "confidence": confidence,
                    }
                )

                continue

            fill_value = series.median()

            cleaned_df[rec_column] = (
                series.fillna(fill_value)
            )

        # ======================================================
        # MODE
        # ======================================================

        elif rec_strategy == "mode":

            modes = series.mode()

            if modes.empty:

                logs.append(
                    {
                        "column": rec_column,
                        "strategy": rec_strategy,
                        "action": "skip",
                        "status": "skipped",
                        "filled_count": 0,
                        "reason": (
                            "No valid mode available."
                        ),
                        "confidence": confidence,
                    }
                )

                continue

            fill_value = modes.iloc[0]

            cleaned_df[rec_column] = (
                series.fillna(fill_value)
            )

        # ======================================================
        # FORWARD FILL
        # ======================================================

        elif rec_strategy == "ffill":

            cleaned_df[rec_column] = (
                series.ffill()
            )

            fill_value = "forward_fill"

        # ======================================================
        # BACKWARD FILL
        # ======================================================

        elif rec_strategy == "bfill":

            cleaned_df[rec_column] = (
                series.bfill()
            )

            fill_value = "backward_fill"

        # ======================================================
        # SAFETY
        # ======================================================

        else:

            logs.append(
                {
                    "column": rec_column,
                    "strategy": rec_strategy,
                    "action": "skip",
                    "status": "skipped",
                    "filled_count": 0,
                    "reason": (
                        "Unsupported imputation strategy."
                    ),
                    "confidence": confidence,
                }
            )

            continue

        # ======================================================
        # CALCULATE ACTUAL FILLED COUNT
        # ======================================================

        remaining_missing = int(
            cleaned_df[rec_column]
            .isna()
            .sum()
        )

        filled_count = (
            missing_count
            - remaining_missing
        )

        # ======================================================
        # LOG RESULT
        # ======================================================

        logs.append(
            {
                "column": rec_column,
                "strategy": rec_strategy,
                "action": "imputed",
                "status": "imputed",
                "fill_value": fill_value,
                "filled_count": filled_count,
                "remaining_missing": (
                    remaining_missing
                ),
                "confidence": confidence,
                "reason": reason,
            }
        )

    # ==========================================================
    # RETURN
    # ==========================================================

    return cleaned_df, logs
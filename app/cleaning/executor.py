import pandas as pd


def execute_cleaning(
    dataframe: pd.DataFrame,
    recommendations: list,
) -> tuple[pd.DataFrame, list[dict]]:

    cleaned_df = dataframe.copy()

    cleaning_log = []

    for recommendation in recommendations:

        # ======================================================
        # GET RECOMMENDATION DATA
        # ======================================================

        if isinstance(recommendation, dict):

            row_index = recommendation.get("row_index")
            column = recommendation.get("column")
            action = recommendation.get("action")

            reason = recommendation.get(
                "reason",
                "",
            )

            confidence = recommendation.get(
                "confidence",
                None,
            )

        else:

            row_index = recommendation.row_index
            column = recommendation.column
            action = recommendation.action

            reason = getattr(
                recommendation,
                "reason",
                "",
            )

            confidence = getattr(
                recommendation,
                "confidence",
                None,
            )

        # ======================================================
        # COLUMN NOT FOUND
        # ======================================================

        if column not in cleaned_df.columns:

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "action": action,
                    "status": "skipped",
                    "reason": "Column not found.",
                }
            )

            continue

        # ======================================================
        # ROW NOT FOUND
        # ======================================================

        if row_index not in cleaned_df.index:

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "action": action,
                    "status": "skipped",
                    "reason": "Row not found.",
                }
            )

            continue

        # ======================================================
        # REPLACE WITH MISSING
        # ======================================================

        if action == "replace_with_missing":

            original_value = cleaned_df.loc[
                row_index,
                column,
            ]

            cleaned_df.loc[
                row_index,
                column,
            ] = pd.NA

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "action": action,
                    "status": "cleaned",
                    "original_value": original_value,
                    "new_value": None,
                    "confidence": confidence,
                    "reason": reason,
                }
            )

        # ======================================================
        # REVIEW
        # ======================================================

        elif action == "review":

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "action": action,
                    "status": "review_required",
                    "original_value": cleaned_df.loc[
                        row_index,
                        column,
                    ],
                    "confidence": confidence,
                    "reason": reason,
                }
            )

        # ======================================================
        # KEEP
        # ======================================================

        elif action == "keep":

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "action": action,
                    "status": "kept",
                    "original_value": cleaned_df.loc[
                        row_index,
                        column,
                    ],
                    "confidence": confidence,
                    "reason": reason,
                }
            )

        # ======================================================
        # UNKNOWN ACTION
        # ======================================================

        else:

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "action": action,
                    "status": "skipped",
                    "reason": (
                        f"Unknown cleaning action: {action}"
                    ),
                }
            )

    return cleaned_df, cleaning_log
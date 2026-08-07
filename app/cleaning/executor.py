import pandas as pd


def execute_cleaning(
    dataframe: pd.DataFrame,
    recommendations: list,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Execute cleaning recommendations on a copy
    of the original dataframe.

    Supported actions:
        - replace_with_missing
        - review
        - keep

    Returns:
        cleaned_dataframe
        cleaning_log
    """

    cleaned_df = dataframe.copy()

    cleaning_log = []

    for recommendation in recommendations:

        row_index = recommendation.row_index
        column = recommendation.column
        value = recommendation.value
        action = recommendation.action

        # ==========================================
        # SAFETY CHECK
        # ==========================================

        if row_index not in cleaned_df.index:
            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "old_value": value,
                    "new_value": None,
                    "action": action,
                    "status": "skipped",
                    "reason": "Row index does not exist",
                }
            )

            continue

        if column not in cleaned_df.columns:
            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "old_value": value,
                    "new_value": None,
                    "action": action,
                    "status": "skipped",
                    "reason": "Column does not exist",
                }
            )

            continue

        # ==========================================
        # REPLACE WITH MISSING
        # ==========================================

        if action == "replace_with_missing":

            old_value = cleaned_df.at[
                row_index,
                column
            ]

            cleaned_df.at[
                row_index,
                column
            ] = pd.NA

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "old_value": old_value,
                    "new_value": pd.NA,
                    "action": action,
                    "status": "cleaned",
                    "reason": recommendation.reason,
                }
            )

        # ==========================================
        # REVIEW
        # ==========================================

        elif action == "review":

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "old_value": value,
                    "new_value": value,
                    "action": action,
                    "status": "review_required",
                    "reason": recommendation.reason,
                }
            )

        # ==========================================
        # KEEP
        # ==========================================

        elif action == "keep":

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "old_value": value,
                    "new_value": value,
                    "action": action,
                    "status": "kept",
                    "reason": recommendation.reason,
                }
            )

        # ==========================================
        # UNKNOWN ACTION
        # ==========================================

        else:

            cleaning_log.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "old_value": value,
                    "new_value": value,
                    "action": action,
                    "status": "skipped",
                    "reason": (
                        f"Unsupported cleaning action: "
                        f"{action}"
                    ),
                }
            )

    return cleaned_df, cleaning_log

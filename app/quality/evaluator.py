import pandas as pd


def calculate_missing_ratio(
    dataframe: pd.DataFrame,
) -> float:
    """
    Calculate percentage of missing cells.

    Returns:
        Ratio from 0.0 to 1.0
    """

    total_cells = (
        dataframe.shape[0]
        * dataframe.shape[1]
    )

    if total_cells == 0:
        return 0.0

    missing_cells = dataframe.isna().sum().sum()

    return (
        float(missing_cells)
        / float(total_cells)
    )


def calculate_quality_score(
    df: pd.DataFrame,
) -> float:

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 100.0

    missing_cells = df.isna().sum().sum()

    valid_cells = total_cells - missing_cells

    score = (
        valid_cells / total_cells
    ) * 100

    return round(score, 10)
def evaluate_quality(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Evaluate basic data quality.
    """

    rows = dataframe.shape[0]
    columns = dataframe.shape[1]

    missing_cells = int(
        dataframe.isna().sum().sum()
    )

    total_cells = rows * columns

    missing_ratio = calculate_missing_ratio(
        dataframe
    )

    quality_score = calculate_quality_score(
        dataframe
    )

    return {
        "rows": rows,
        "columns": columns,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_ratio": missing_ratio,
        "quality_score": quality_score,
    }


def compare_quality(
    before: dict,
    after: dict,
) -> dict:
    """
    Compare data quality before and after cleaning.
    """

    return {
        "missing_cells_change": (
            after["missing_cells"]
            - before["missing_cells"]
        ),

        "missing_ratio_change": (
            after["missing_ratio"]
            - before["missing_ratio"]
        ),

        "quality_score_change": (
            after["quality_score"]
            - before["quality_score"]
        ),

        "rows_change": (
            after["rows"]
            - before["rows"]
        ),

        "columns_change": (
            after["columns"]
            - before["columns"]
        ),
    }

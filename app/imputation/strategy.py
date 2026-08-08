import pandas as pd


def select_numeric_strategy(
    series: pd.Series,
) -> str:

    non_null = series.dropna()

    if len(non_null) == 0:
        return "skip"

    if len(non_null) < 3:
        return "median"

    skewness = float(
        non_null.skew()
    )

    if abs(skewness) > 1.0:
        return "median"

    return "mean"


def select_categorical_strategy(
    series: pd.Series,
) -> str:

    non_null = series.dropna()

    if len(non_null) == 0:
        return "skip"

    return "mode"
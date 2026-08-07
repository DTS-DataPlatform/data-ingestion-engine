import pandas as pd


def get_numeric_ratio(
    series: pd.Series
) -> float:

    non_null = series.dropna()

    if len(non_null) == 0:
        return 0.0

    converted = pd.to_numeric(
        non_null,
        errors="coerce"
    )

    numeric_ratio = converted.notna().mean()

    return float(numeric_ratio)


def is_numeric_column(
    series: pd.Series,
    threshold: float = 0.8
) -> bool:

    # Trường hợp Pandas đã nhận dạng là numeric
    if pd.api.types.is_numeric_dtype(series):
        return True

    # Trường hợp dtype là object/string
    # nhưng phần lớn giá trị thực chất là số
    numeric_ratio = get_numeric_ratio(series)

    return numeric_ratio >= threshold
import pandas as pd


def analyze_missing_values(
    dataframe: pd.DataFrame,
) -> list[dict]:

    results = []

    total_rows = len(dataframe)

    if total_rows == 0:
        return results

    for column in dataframe.columns:

        missing_count = int(
            dataframe[column].isna().sum()
        )

        if missing_count == 0:
            continue

        missing_ratio = (
            missing_count / total_rows
        )

        series = dataframe[column]

        results.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_ratio": missing_ratio,
                "dtype": str(series.dtype),
                "unique_count": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    return results

def analyze_row_missingness(
    dataframe: pd.DataFrame,
) -> list[dict]:

    if dataframe.empty:
        return []

    missing_count = dataframe.isna().sum(axis=1)

    total_columns = len(dataframe.columns)

    results = []

    for row_index, count in missing_count.items():

        count = int(count)

        results.append(
            {
                "row_index": row_index,
                "missing_count": count,
                "missing_ratio": (
                    count / total_columns
                    if total_columns > 0
                    else 0.0
                ),
            }
        )

    return results

def analyze_missing_patterns(
    dataframe: pd.DataFrame,
) -> list[dict]:

    if dataframe.empty:
        return []

    missing_matrix = dataframe.isna()

    patterns = (
        missing_matrix
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
    )

    counts = patterns.value_counts()

    results = []

    for pattern, count in counts.items():

        results.append(
            {
                "pattern": pattern,
                "count": int(count),
                "ratio": float(
                    count / len(dataframe)
                ),
            }
        )

    return results

def analyze_missing_correlation(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if dataframe.empty:
        return pd.DataFrame()

    missing_matrix = (
        dataframe.isna()
        .astype(int)
    )

    return missing_matrix.corr()

def analyze_group_missingness(
    dataframe: pd.DataFrame,
    group_column: str,
    target_column: str,
) -> list[dict]:

    if group_column not in dataframe.columns:
        return []

    if target_column not in dataframe.columns:
        return []

    grouped = (
        dataframe
        .groupby(group_column)[target_column]
        .apply(lambda x: x.isna().mean())
    )

    results = []

    for group_value, missing_ratio in grouped.items():

        results.append(
            {
                "group_column": group_column,
                "group_value": group_value,
                "target_column": target_column,
                "missing_ratio": float(
                    missing_ratio
                ),
            }
        )

    return results



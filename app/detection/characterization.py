from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DatasetCharacteristics:

    rows: int
    columns: int

    numeric_columns: int
    categorical_columns: int

    numeric_ratio: float

    missing_ratio: float

    mean_skewness: float

    high_dimensional: bool

    skewed: bool


def characterize_dataset(
    df: pd.DataFrame,
    profiles,
) -> DatasetCharacteristics:

    rows = len(df)

    columns = len(df.columns)

    numeric_columns = 0

    categorical_columns = 0

    skewness_values = []

    # ==========================================
    # Analyze profiles
    # ==========================================

    for profile in profiles:

        if profile.mean is not None:

            numeric_columns += 1

            numeric = pd.to_numeric(
                df[profile.name],
                errors="coerce",
            ).dropna()

            if len(numeric) >= 3:

                skew = numeric.skew()

                if not pd.isna(skew):

                    skewness_values.append(
                        abs(float(skew))
                    )

        else:

            categorical_columns += 1

    # ==========================================
    # Ratios
    # ==========================================

    if columns > 0:

        numeric_ratio = (
            numeric_columns / columns
        )

    else:

        numeric_ratio = 0.0

    # ==========================================
    # Missing ratio
    # ==========================================

    total_cells = rows * columns

    if total_cells > 0:

        missing_ratio = (
            df.isna().sum().sum()
            / total_cells
        )

    else:

        missing_ratio = 0.0

    # ==========================================
    # Skewness
    # ==========================================

    if skewness_values:

        mean_skewness = float(
            np.mean(
                skewness_values
            )
        )

    else:

        mean_skewness = 0.0

    # ==========================================
    # Characteristics
    # ==========================================

    high_dimensional = (
        numeric_columns >= 10
    )

    skewed = (
        mean_skewness >= 1.0
    )

    return DatasetCharacteristics(
        rows=rows,
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_ratio=numeric_ratio,
        missing_ratio=missing_ratio,
        mean_skewness=mean_skewness,
        high_dimensional=high_dimensional,
        skewed=skewed,
    )
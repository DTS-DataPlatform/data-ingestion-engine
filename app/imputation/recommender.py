import pandas as pd

from .models import ImputationRecommendation
from .strategy import (
    select_numeric_strategy,
    select_categorical_strategy,
)


def recommend_imputation(
    dataframe: pd.DataFrame,
) -> list[ImputationRecommendation]:

    recommendations = []

    total_rows = len(dataframe)

    if total_rows == 0:
        return recommendations

    for column in dataframe.columns:

        series = dataframe[column]

        missing_count = int(
            series.isna().sum()
        )

        if missing_count == 0:
            continue

        missing_ratio = (
            missing_count / total_rows
        )

        # ======================================
        # ALL VALUES MISSING
        # ======================================

        if missing_count == total_rows:

            recommendations.append(
                ImputationRecommendation(
                    column=column,
                    strategy="skip",
                    confidence=1.0,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    reason=(
                        "Column contains only "
                        "missing values."
                    ),
                    requires_review=True,
                )
            )

            continue

        # ======================================
        # HIGH MISSING RATIO
        # ======================================

        if missing_ratio >= 0.50:

            recommendations.append(
                ImputationRecommendation(
                    column=column,
                    strategy="review",
                    confidence=0.90,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    reason=(
                        "Missing ratio is too high "
                        "for automatic imputation."
                    ),
                    requires_review=True,
                )
            )

            continue

        # ======================================
        # NUMERIC
        # ======================================

        if pd.api.types.is_numeric_dtype(series):

            strategy = select_numeric_strategy(
                series
            )

            confidence = (
                0.90
                if strategy == "median"
                else 0.85
            )

            reason = (
                "Numeric column with "
                "moderate missing ratio. "
                f"Selected {strategy} based "
                "on distribution."
            )

            recommendations.append(
                ImputationRecommendation(
                    column=column,
                    strategy=strategy,
                    confidence=confidence,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    reason=reason,
                )
            )

        # ======================================
        # CATEGORICAL
        # ======================================

        else:

            strategy = select_categorical_strategy(
                series
            )

            recommendations.append(
                ImputationRecommendation(
                    column=column,
                    strategy=strategy,
                    confidence=0.85,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    reason=(
                        "Categorical column. "
                        "Mode is used as the "
                        "default imputation strategy."
                    ),
                )
            )

    return recommendations
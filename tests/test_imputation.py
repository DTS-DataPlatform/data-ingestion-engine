import pandas as pd

from app.imputation.executor import execute_imputation
from app.imputation.recommender import ImputationRecommendation


# ==========================================================
# FIXTURES
# ==========================================================


def create_dataframe():
    return pd.DataFrame(
        {
            "age": [20, 30, None, 40, None],
            "salary": [1000, None, 3000, None, 5000],
            "city": ["Da Nang", None, "Hue", None, "Da Nang"],
            "score": [10, None, 30, 40, None],
        }
    )


# ==========================================================
# DIRECT MODE
# ==========================================================


def test_direct_mean_imputation():
    df = pd.DataFrame(
        {
            "age": [20, 30, None, 40],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="mean",
    )

    expected_mean = (20 + 30 + 40) / 3

    assert result["age"].isna().sum() == 0
    assert result.loc[2, "age"] == expected_mean


def test_direct_median_imputation():
    df = pd.DataFrame(
        {
            "age": [10, 20, None, 30, 40],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="median",
    )

    assert result["age"].isna().sum() == 0
    assert result.loc[2, "age"] == 25


def test_direct_mode_imputation():
    df = pd.DataFrame(
        {
            "city": [
                "Da Nang",
                "Hue",
                "Da Nang",
                None,
                "Da Nang",
            ]
        }
    )

    result = execute_imputation(
        df,
        column="city",
        strategy="mode",
    )

    assert result["city"].isna().sum() == 0
    assert result.loc[3, "city"] == "Da Nang"


def test_direct_ffill_imputation():
    df = pd.DataFrame(
        {
            "age": [20, None, None, 40],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="ffill",
    )

    assert result["age"].tolist() == [
        20,
        20,
        20,
        40,
    ]


def test_direct_bfill_imputation():
    df = pd.DataFrame(
        {
            "age": [20, None, None, 40],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="bfill",
    )

    assert result["age"].tolist() == [
        20,
        40,
        40,
        40,
    ]


# ==========================================================
# DIRECT MODE - SKIP / REVIEW
# ==========================================================


def test_direct_skip_does_not_modify_data():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="skip",
    )

    pd.testing.assert_frame_equal(result, df)


def test_direct_review_does_not_modify_data():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="review",
    )

    pd.testing.assert_frame_equal(result, df)


# ==========================================================
# DIRECT MODE - INVALID INPUT
# ==========================================================


def test_direct_non_existing_column_does_not_modify_data():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    result = execute_imputation(
        df,
        column="salary",
        strategy="mean",
    )

    pd.testing.assert_frame_equal(result, df)


def test_direct_unknown_strategy_does_not_modify_data():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="unknown",
    )

    pd.testing.assert_frame_equal(result, df)


# ==========================================================
# DIRECT MODE - TYPE VALIDATION
# ==========================================================


def test_direct_mean_on_non_numeric_column_is_skipped():
    df = pd.DataFrame(
        {
            "city": ["Da Nang", None, "Hue"],
        }
    )

    result = execute_imputation(
        df,
        column="city",
        strategy="mean",
    )

    pd.testing.assert_frame_equal(result, df)


def test_direct_median_on_non_numeric_column_is_skipped():
    df = pd.DataFrame(
        {
            "city": ["Da Nang", None, "Hue"],
        }
    )

    result = execute_imputation(
        df,
        column="city",
        strategy="median",
    )

    pd.testing.assert_frame_equal(result, df)


# ==========================================================
# DIRECT MODE - ALL VALUES MISSING
# ==========================================================


def test_direct_mean_when_all_values_are_missing():
    df = pd.DataFrame(
        {
            "age": [None, None, None],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="mean",
    )

    assert result["age"].isna().sum() == 3


def test_direct_median_when_all_values_are_missing():
    df = pd.DataFrame(
        {
            "age": [None, None, None],
        }
    )

    result = execute_imputation(
        df,
        column="age",
        strategy="median",
    )

    assert result["age"].isna().sum() == 3


# ==========================================================
# ORIGINAL DATAFRAME MUST NOT BE MODIFIED
# ==========================================================


def test_direct_imputation_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "age": [20, None, 40],
        }
    )

    original = df.copy(deep=True)

    result = execute_imputation(
        df,
        column="age",
        strategy="mean",
    )

    pd.testing.assert_frame_equal(df, original)

    assert result is not df


# ==========================================================
# RECOMMENDATION MODE - DICT
# ==========================================================


def test_recommendation_mean_imputation():
    df = pd.DataFrame(
        {
            "age": [20, 30, None, 40],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "mean",
            "confidence": 0.95,
            "reason": "Numeric column with low missing ratio.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    assert result["age"].isna().sum() == 0

    assert len(logs) == 1

    assert logs[0]["column"] == "age"
    assert logs[0]["strategy"] == "mean"
    assert logs[0]["action"] == "imputed"
    assert logs[0]["status"] == "imputed"

    assert logs[0]["filled_count"] == 1
    assert logs[0]["remaining_missing"] == 0
    assert logs[0]["confidence"] == 0.95


def test_recommendation_median_imputation():
    df = pd.DataFrame(
        {
            "age": [10, 20, None, 30, 40],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "median",
            "confidence": 0.9,
            "reason": "Robust to outliers.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    assert result.loc[2, "age"] == 25

    assert logs[0]["status"] == "imputed"
    assert logs[0]["filled_count"] == 1


def test_recommendation_mode_imputation():
    df = pd.DataFrame(
        {
            "city": [
                "Da Nang",
                "Hue",
                "Da Nang",
                None,
            ]
        }
    )

    recommendations = [
        {
            "column": "city",
            "strategy": "mode",
            "confidence": 0.88,
            "reason": "Categorical column.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    assert result.loc[3, "city"] == "Da Nang"

    assert logs[0]["status"] == "imputed"
    assert logs[0]["filled_count"] == 1


def test_recommendation_ffill_imputation():
    df = pd.DataFrame(
        {
            "score": [10, None, None, 40],
        }
    )

    recommendations = [
        {
            "column": "score",
            "strategy": "ffill",
            "confidence": 0.8,
            "reason": "Sequential data.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    assert result["score"].tolist() == [
        10,
        10,
        10,
        40,
    ]

    assert logs[0]["filled_count"] == 2
    assert logs[0]["remaining_missing"] == 0


def test_recommendation_bfill_imputation():
    df = pd.DataFrame(
        {
            "score": [10, None, None, 40],
        }
    )

    recommendations = [
        {
            "column": "score",
            "strategy": "bfill",
            "confidence": 0.8,
            "reason": "Sequential data.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    assert result["score"].tolist() == [
        10,
        40,
        40,
        40,
    ]

    assert logs[0]["filled_count"] == 2
    assert logs[0]["remaining_missing"] == 0


# ==========================================================
# RECOMMENDATION MODE - SKIP / REVIEW
# ==========================================================


def test_recommendation_skip():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "skip",
            "confidence": 0.5,
            "reason": "Missing ratio is acceptable.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["column"] == "age"
    assert logs[0]["strategy"] == "skip"
    assert logs[0]["action"] == "skip"
    assert logs[0]["status"] == "skipped"


def test_recommendation_review():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "review",
            "confidence": 0.4,
            "reason": "Low confidence recommendation.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["action"] == "review"
    assert logs[0]["status"] == "review_required"
    assert logs[0]["confidence"] == 0.4
    assert logs[0]["reason"] == (
        "Low confidence recommendation."
    )


# ==========================================================
# RECOMMENDATION MODE - INVALID
# ==========================================================


def test_recommendation_column_not_found():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    recommendations = [
        {
            "column": "salary",
            "strategy": "mean",
            "confidence": 0.9,
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["column"] == "salary"
    assert logs[0]["action"] == "skip"
    assert logs[0]["status"] == "skipped"
    assert logs[0]["reason"] == "Column not found."


def test_recommendation_unknown_strategy():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "unknown",
            "confidence": 0.7,
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["action"] == "skip"
    assert logs[0]["status"] == "skipped"
    assert logs[0]["reason"] == (
        "Unknown imputation strategy."
    )


def test_recommendation_mean_on_non_numeric_column():
    df = pd.DataFrame(
        {
            "city": ["Da Nang", None, "Hue"],
        }
    )

    recommendations = [
        {
            "column": "city",
            "strategy": "mean",
            "confidence": 0.9,
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["action"] == "skip"
    assert logs[0]["status"] == "skipped"
    assert logs[0]["filled_count"] == 0


def test_recommendation_median_on_non_numeric_column():
    df = pd.DataFrame(
        {
            "city": ["Da Nang", None, "Hue"],
        }
    )

    recommendations = [
        {
            "column": "city",
            "strategy": "median",
            "confidence": 0.9,
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["action"] == "skip"
    assert logs[0]["status"] == "skipped"


# ==========================================================
# NO MISSING VALUES
# ==========================================================


def test_recommendation_with_no_missing_values():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "mean",
            "confidence": 0.95,
            "reason": "Numeric column.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["action"] == "skip"
    assert logs[0]["status"] == "skipped"
    assert logs[0]["filled_count"] == 0
    assert logs[0]["reason"] == (
        "Column contains no missing values."
    )


# ==========================================================
# RECOMMENDATION MODE - DATACLASS OBJECT
# ==========================================================


def test_recommendation_dataclass_object():
    df = pd.DataFrame(
        {
            "age": [20, 30, None, 40],
        }
    )

    recommendations = [
        ImputationRecommendation(
            column="age",
            strategy="median",
            confidence=0.92,
            missing_count=1,
            missing_ratio=0.25,
            reason="Median is robust to outliers.",
        )
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    assert result["age"].isna().sum() == 0
    assert result.loc[2, "age"] == 30

    assert logs[0]["column"] == "age"
    assert logs[0]["strategy"] == "median"
    assert logs[0]["status"] == "imputed"
    assert logs[0]["confidence"] == 0.92
    assert logs[0]["reason"] == (
        "Median is robust to outliers."
    )


# ==========================================================
# RECOMMENDATION MODE - COLUMN LIST POSITIONAL ARGUMENT
# ==========================================================


def test_recommendation_passed_as_second_positional_argument():
    df = pd.DataFrame(
        {
            "age": [20, 30, None, 40],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "mean",
            "confidence": 0.9,
            "reason": "Numeric column.",
        }
    ]

    result, logs = execute_imputation(
        df,
        recommendations,
    )

    assert result["age"].isna().sum() == 0
    assert logs[0]["status"] == "imputed"


# ==========================================================
# MULTIPLE RECOMMENDATIONS
# ==========================================================


def test_multiple_recommendations():
    df = pd.DataFrame(
        {
            "age": [20, None, 40],
            "salary": [1000, None, 3000],
            "city": ["Da Nang", None, "Hue"],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "median",
            "confidence": 0.9,
            "reason": "Numeric data.",
        },
        {
            "column": "salary",
            "strategy": "mean",
            "confidence": 0.95,
            "reason": "Numeric data.",
        },
        {
            "column": "city",
            "strategy": "mode",
            "confidence": 0.85,
            "reason": "Categorical data.",
        },
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    assert result["age"].isna().sum() == 0
    assert result["salary"].isna().sum() == 0
    assert result["city"].isna().sum() == 0

    assert len(logs) == 3

    assert logs[0]["status"] == "imputed"
    assert logs[1]["status"] == "imputed"
    assert logs[2]["status"] == "imputed"


# ==========================================================
# MIXED RECOMMENDATIONS
# ==========================================================


def test_mixed_recommendations():
    df = pd.DataFrame(
        {
            "age": [20, None, 40],
            "salary": [1000, None, 3000],
            "city": ["Da Nang", None, "Hue"],
            "score": [10, None, 30],
        }
    )

    recommendations = [
        {
            "column": "age",
            "strategy": "median",
            "confidence": 0.9,
            "reason": "Use median.",
        },
        {
            "column": "salary",
            "strategy": "skip",
            "confidence": 0.5,
            "reason": "Skip this column.",
        },
        {
            "column": "city",
            "strategy": "review",
            "confidence": 0.4,
            "reason": "Needs human review.",
        },
        {
            "column": "score",
            "strategy": "unknown",
            "confidence": 0.2,
            "reason": "Invalid strategy.",
        },
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    # age should be imputed
    assert result["age"].isna().sum() == 0

    # other columns should remain unchanged
    assert result["salary"].isna().sum() == 1
    assert result["city"].isna().sum() == 1
    assert result["score"].isna().sum() == 1

    assert len(logs) == 4

    assert logs[0]["status"] == "imputed"
    assert logs[1]["status"] == "skipped"
    assert logs[2]["status"] == "review_required"
    assert logs[3]["status"] == "skipped"


# ==========================================================
# ORIGINAL DATAFRAME - RECOMMENDATION MODE
# ==========================================================


def test_recommendation_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "age": [20, None, 40],
            "city": ["Da Nang", None, "Hue"],
        }
    )

    original = df.copy(deep=True)

    recommendations = [
        {
            "column": "age",
            "strategy": "mean",
            "confidence": 0.9,
        },
        {
            "column": "city",
            "strategy": "mode",
            "confidence": 0.9,
        },
    ]

    result, logs = execute_imputation(
        df,
        recommendations=recommendations,
    )

    pd.testing.assert_frame_equal(df, original)

    assert result is not df
    assert len(logs) == 2
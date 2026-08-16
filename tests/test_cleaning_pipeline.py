import pandas as pd

from app.cleaning.executor import execute_cleaning
from app.imputation.executor import execute_imputation


def test_cleaning_pipeline_reduces_missing_values():

    df = pd.DataFrame(
        {
            "age": [20, -10, 30, None, 40],
            "salary": [1000, 2000, None, 3000, 4000],
        }
    )

    # ------------------------------------------
    # STEP 1: Cleaning invalid value
    # ------------------------------------------

    recommendations = [
        {
            "row_index": 1,
            "column": "age",
            "action": "replace_with_missing",
        }
    ]

    cleaned_df, cleaning_log = execute_cleaning(
        df,
        recommendations,
    )

    # -10 must become missing
    assert pd.isna(cleaned_df.loc[1, "age"])

    # ------------------------------------------
    # STEP 2: Imputation
    # ------------------------------------------

    imputed_df = execute_imputation(
        cleaned_df,
        column="age",
        strategy="median",
    )

    # Missing age should be filled
    assert imputed_df["age"].isna().sum() == 0

    # Salary still has one missing value
    assert imputed_df["salary"].isna().sum() == 1


def test_imputation_does_not_change_row_count():

    df = pd.DataFrame(
        {
            "age": [20, None, 30],
            "salary": [1000, 2000, None],
        }
    )

    original_rows = len(df)

    result = execute_imputation(
        df,
        column="age",
        strategy="median",
    )

    assert len(result) == original_rows


def test_imputation_does_not_modify_original_dataframe():

    df = pd.DataFrame(
        {
            "age": [20, None, 40],
        }
    )

    original = df.copy()

    execute_imputation(
        df,
        column="age",
        strategy="median",
    )

    pd.testing.assert_frame_equal(
        df,
        original,
    )
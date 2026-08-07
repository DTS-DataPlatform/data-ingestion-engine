import pandas as pd

from app.cleaning.executor import execute_cleaning


class Recommendation:

    def __init__(
        self,
        row_index,
        column,
        value,
        action,
        reason="test",
    ):
        self.row_index = row_index
        self.column = column
        self.value = value
        self.action = action
        self.reason = reason


def test_replace_with_missing():

    df = pd.DataFrame(
        {
            "age": [20, -10, 30],
            "salary": [1000, 2000, 3000],
        }
    )

    recommendation = Recommendation(
        row_index=1,
        column="age",
        value=-10,
        action="replace_with_missing",
    )

    cleaned_df, log = execute_cleaning(
        df,
        [recommendation],
    )

    assert pd.isna(
        cleaned_df.loc[1, "age"]
    )

    assert (
        cleaned_df.loc[0, "age"]
        == 20
    )

    assert (
        cleaned_df.loc[2, "age"]
        == 30
    )

    assert log[0]["status"] == "cleaned"


def test_original_dataframe_is_not_modified():

    df = pd.DataFrame(
        {
            "age": [20, -10, 30]
        }
    )

    recommendation = Recommendation(
        row_index=1,
        column="age",
        value=-10,
        action="replace_with_missing",
    )

    cleaned_df, _ = execute_cleaning(
        df,
        [recommendation],
    )

    assert df.loc[1, "age"] == -10

    assert pd.isna(
        cleaned_df.loc[1, "age"]
    )


def test_review_does_not_modify_value():

    df = pd.DataFrame(
        {
            "height": [170, 500, 180]
        }
    )

    recommendation = Recommendation(
        row_index=1,
        column="height",
        value=500,
        action="review",
    )

    cleaned_df, log = execute_cleaning(
        df,
        [recommendation],
    )

    assert (
        cleaned_df.loc[1, "height"]
        == 500
    )

    assert (
        log[0]["status"]
        == "review_required"
    )


def test_keep_does_not_modify_value():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40]
        }
    )

    recommendation = Recommendation(
        row_index=1,
        column="age",
        value=30,
        action="keep",
    )

    cleaned_df, log = execute_cleaning(
        df,
        [recommendation],
    )

    assert (
        cleaned_df.loc[1, "age"]
        == 30
    )

    assert (
        log[0]["status"]
        == "kept"
    )


def test_missing_column_is_skipped():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40]
        }
    )

    recommendation = Recommendation(
        row_index=1,
        column="salary",
        value=-100,
        action="replace_with_missing",
    )

    cleaned_df, log = execute_cleaning(
        df,
        [recommendation],
    )

    assert (
        log[0]["status"]
        == "skipped"
    )

    assert (
        cleaned_df.equals(df)
    )


def test_missing_row_is_skipped():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40]
        }
    )

    recommendation = Recommendation(
        row_index=100,
        column="age",
        value=-10,
        action="replace_with_missing",
    )

    cleaned_df, log = execute_cleaning(
        df,
        [recommendation],
    )

    assert (
        log[0]["status"]
        == "skipped"
    )

    assert (
        cleaned_df.equals(df)
    )

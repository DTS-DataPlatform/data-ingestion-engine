import pandas as pd

from app.pipeline.service import run_pipeline


def create_dataset():

    return pd.DataFrame(
        {
            "customer_id": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],

            "age": [
                20,
                25,
                None,
                30,
                999,
                35,
            ],

            "salary": [
                10_000_000,
                12_000_000,
                None,
                15_000_000,
                100_000_000,
                None,
            ],

            "city": [
                "Da Nang",
                "Hue",
                None,
                "Da Nang",
                "Da Nang",
                None,
            ],
        }
    )


def test_pipeline_returns_result():

    df = create_dataset()

    result = run_pipeline(df)

    assert result is not None

    assert result.original_dataframe is not None

    assert result.cleaned_dataframe is not None


def test_pipeline_does_not_modify_original():

    df = create_dataset()

    original = df.copy()

    run_pipeline(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_pipeline_detects_anomalies():

    df = create_dataset()

    result = run_pipeline(df)

    assert len(
        result.anomalies
    ) > 0


def test_pipeline_generates_imputation_recommendations():

    df = create_dataset()

    result = run_pipeline(df)

    assert len(
        result.imputation_recommendations
    ) > 0


def test_pipeline_generates_quality_report():

    df = create_dataset()

    result = run_pipeline(df)

    assert result.quality_report is not None


def test_pipeline_removes_missing_values_when_safe():

    df = create_dataset()

    result = run_pipeline(df)

    cleaned = result.cleaned_dataframe

    assert (
        cleaned.isna().sum().sum()
        == 0
    )
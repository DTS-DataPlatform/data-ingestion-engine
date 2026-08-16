import pandas as pd
import pytest

from app.quality.report import (
    build_quality_report,
)


def test_report_contains_dataset_dimensions():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "salary": [1000, 2000, 3000],
        }
    )

    report = build_quality_report(
        dataframe=df,
        anomalies=[],
        cleaning_results=[],
    )

    assert report.rows == 3
    assert report.columns == 2
    assert report.total_cells == 6


def test_report_contains_missing_information():

    df = pd.DataFrame(
        {
            "age": [20, None, 40],
            "salary": [1000, 2000, None],
        }
    )

    report = build_quality_report(
        dataframe=df,
        anomalies=[],
        cleaning_results=[],
    )

    assert report.missing_cells == 2

    assert report.missing_ratio == pytest.approx(
        2 / 6
    )

    assert report.quality_score == pytest.approx(
        (4 / 6) * 100
    )


def test_report_contains_anomaly_count():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    anomalies = [
        {
            "row_index": 1,
            "column": "age",
            "value": -10,
            "anomaly_types": [
                "invalid_value"
            ],
            "severity": "critical",
        },
        {
            "row_index": 2,
            "column": "age",
            "value": 1000,
            "anomaly_types": [
                "outlier"
            ],
            "severity": "high",
        },
    ]

    report = build_quality_report(
        dataframe=df,
        anomalies=anomalies,
        cleaning_results=[],
    )

    assert report.total_anomalies == 2


def test_report_contains_severity_counts():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    anomalies = [
        {
            "row_index": 1,
            "column": "age",
            "value": -10,
            "anomaly_types": [
                "invalid_value"
            ],
            "severity": "critical",
        },
        {
            "row_index": 2,
            "column": "age",
            "value": 1000,
            "anomaly_types": [
                "outlier"
            ],
            "severity": "high",
        },
    ]

    report = build_quality_report(
        dataframe=df,
        anomalies=anomalies,
        cleaning_results=[],
    )

    assert report.severity_counts == {
        "critical": 1,
        "high": 1,
    }


def test_report_contains_cleaning_statistics():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    cleaning_results = [

        {
            "action": "replace_with_missing"
        },

        {
            "action": "review"
        },

        {
            "action": "skip"
        },
    ]

    report = build_quality_report(
        dataframe=df,
        anomalies=[],
        cleaning_results=cleaning_results,
    )

    assert report.cleaned_count == 1
    assert report.review_count == 1
    assert report.skipped_count == 1


def test_report_contains_column_issues():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "salary": [1000, 2000, 3000],
        }
    )

    anomalies = [

        {
            "row_index": 1,
            "column": "age",
            "value": -10,
            "anomaly_types": [
                "invalid_value"
            ],
            "severity": "critical",
        },

        {
            "row_index": 2,
            "column": "age",
            "value": 1000,
            "anomaly_types": [
                "outlier"
            ],
            "severity": "high",
        },
    ]

    report = build_quality_report(
        dataframe=df,
        anomalies=anomalies,
        cleaning_results=[],
    )

    assert report.column_issues["age"][
        "anomalies"
    ] == 2

    assert report.column_issues["age"][
        "types"
    ] == {
        "invalid_value": 1,
        "outlier": 1,
    }


def test_quality_score_is_bounded():

    df = pd.DataFrame(
        {
            "age": [20, None, 40],
        }
    )

    report = build_quality_report(
        dataframe=df,
        anomalies=[],
        cleaning_results=[],
    )

    assert 0 <= report.quality_score <= 100
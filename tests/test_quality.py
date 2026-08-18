import pandas as pd
import pytest
from app.quality.models import QualityReport
from app.quality.report import build_quality_report
from app.detection.models import AnomalyRecord


def test_quality_report_basic():

    df = pd.DataFrame(
        {
            "age": [20, None, 30],
            "salary": [100, 200, None],
        }
    )

    report = build_quality_report(
        dataframe=df,
        anomalies=[],
        cleaning_results=[],
    )

    assert isinstance(
        report,
        QualityReport,
    )

    assert report.rows == 3

    assert report.columns == 2

    assert report.total_cells == 6

    assert report.missing_cells == 2

    assert report.missing_ratio == 2 / 6

    assert report.total_anomalies == 0

    assert report.quality_score == pytest.approx(
    100 * (
        0.5 * (4 / 6)
        + 0.5 * 1.0
    )
)


def test_quality_report_anomaly_record():

    df = pd.DataFrame(
        {
            "age": [20, 25, 999],
        }
    )

    anomalies = [
        AnomalyRecord(
            row_index=2,
            column="age",
            value=999,
            anomaly_type="invalid_value",
            detector="hybrid",
            score=0.95,
            method="rule+IQR",
            severity="critical",
            reason=(
                "Age exceeds maximum "
                "allowed value."
            ),
        )
    ]

    report = build_quality_report(
        dataframe=df,
        anomalies=anomalies,
        cleaning_results=[],
    )

    assert report.total_anomalies == 1

    assert report.severity_counts == {
        "critical": 1
    }

    assert report.issue_counts == {
        "invalid_value": 1
    }

    assert (
        report.column_issues["age"][
            "anomalies"
        ]
        == 1
    )

    assert (
        report.column_issues["age"][
            "types"
        ]
        == {
            "invalid_value": 1
        }
    )

    assert (
        report.column_issues["age"][
            "severities"
        ]
        == {
            "critical": 1
        }
    )

    assert report.quality_score < 100


def test_quality_report_cleaning():

    df = pd.DataFrame(
        {
            "age": [20, 25, 999],
        }
    )

    cleaning_results = [

        {
            "column": "age",
            "action": "replace_with_missing",
            "status": "cleaned",
        },

        {
            "column": "salary",
            "action": "review",
            "status": "review_required",
        },

        {
            "column": "city",
            "action": "skip",
            "status": "skipped",
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
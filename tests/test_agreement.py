from app.detection.agreement import (
    compare_detector_results,
)

from app.anomaly.models import (
    AnomalyRecord,
)


def make_anomaly(
    row_index,
    column,
    detector,
    severity="medium",
):

    return AnomalyRecord(
        row_index=row_index,

        column=column,

        value=999,

        anomaly_type="outlier",

        detector=detector,

        score=2.0,

        method=detector,

        severity=severity,

        reason="test anomaly",
    )


def test_agreement_full():

    results = {

        "iqr": [
            make_anomaly(
                5,
                "age",
                "iqr",
            )
        ],

        "zscore": [
            make_anomaly(
                5,
                "age",
                "zscore",
            )
        ],

        "lof": [
            make_anomaly(
                5,
                "age",
                "lof",
            )
        ],
    }

    agreements = compare_detector_results(
        results
    )

    assert len(agreements) == 1

    result = agreements[0]

    assert result.row_index == 5

    assert result.column == "age"

    assert result.agreement_count == 3

    assert result.agreement_ratio == 1.0

    assert result.confidence == 1.0

    assert set(
        result.detectors
    ) == {
        "iqr",
        "zscore",
        "lof",
    }


def test_agreement_partial():

    results = {

        "iqr": [
            make_anomaly(
                5,
                "age",
                "iqr",
            )
        ],

        "zscore": [
            make_anomaly(
                5,
                "age",
                "zscore",
            )
        ],

        "lof": [],

        "dbscan": [],
    }

    agreements = compare_detector_results(
        results
    )

    assert len(agreements) == 1

    result = agreements[0]

    assert result.agreement_count == 2

    assert result.agreement_ratio == 0.5

    assert result.confidence == 0.5


def test_agreement_multiple_anomalies():

    results = {

        "iqr": [
            make_anomaly(
                5,
                "age",
                "iqr",
            ),
            make_anomaly(
                10,
                "salary",
                "iqr",
            ),
        ],

        "zscore": [
            make_anomaly(
                5,
                "age",
                "zscore",
            )
        ],
    }

    agreements = compare_detector_results(
        results
    )

    assert len(agreements) == 2

    age_result = next(
        item
        for item in agreements
        if item.column == "age"
    )

    salary_result = next(
        item
        for item in agreements
        if item.column == "salary"
    )

    assert (
        age_result.agreement_count
        == 2
    )

    assert (
        age_result.agreement_ratio
        == 1.0
    )

    assert (
        salary_result.agreement_count
        == 1
    )

    assert (
        salary_result.agreement_ratio
        == 0.5
    )


def test_agreement_empty():

    results = {}

    agreements = compare_detector_results(
        results
    )

    assert agreements == []
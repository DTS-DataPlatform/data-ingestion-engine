from types import SimpleNamespace

from app.detection.hybrid.aggregator import (
    aggregate_anomalies,
)


def make_anomaly(
    row_index,
    column,
    detector,
    value,
):

    return SimpleNamespace(
        row_index=row_index,
        column=column,
        detector=detector,
        value=value,
        anomaly_type="outlier",
    )


def test_multivariate_detectors_are_grouped_by_row():

    anomalies = [
        make_anomaly(
            row_index=49,
            column="__row__",
            detector="isolation_forest",
            value={
                "age": 999,
                "salary": 200000000,
            },
        ),

        make_anomaly(
            row_index=49,
            column="__multivariate__",
            detector="lof",
            value={
                "age": 999,
                "salary": 200000000,
            },
        ),

        make_anomaly(
            row_index=49,
            column="__row__",
            detector="dbscan",
            value={
                "age": 999,
                "salary": 200000000,
            },
        ),
    ]

    selected_detectors = [
        "iqr",
        "zscore",
        "isolation_forest",
        "lof",
        "dbscan",
    ]

    results = aggregate_anomalies(
        anomalies,
        selected_detectors,
    )

    assert len(results) == 1

    result = results[0]

    assert result.row_index == 49

    assert result.column == "__multivariate__"

    assert result.detector_count == 3

    assert result.total_detectors == 5

    assert result.agreement_ratio == 0.6

    assert result.confidence == 0.6

    assert set(result.detectors) == {
        "isolation_forest",
        "lof",
        "dbscan",
    }
    
def test_univariate_and_multivariate_anomalies_are_separate():

    anomalies = [

        # AGE anomaly
        make_anomaly(
            row_index=49,
            column="age",
            detector="iqr",
            value=999,
        ),

        make_anomaly(
            row_index=49,
            column="age",
            detector="zscore",
            value=999,
        ),

        # Row-level anomalies
        make_anomaly(
            row_index=49,
            column="__row__",
            detector="isolation_forest",
            value={
                "age": 999,
                "salary": 200000000,
            },
        ),

        make_anomaly(
            row_index=49,
            column="__multivariate__",
            detector="lof",
            value={
                "age": 999,
                "salary": 200000000,
            },
        ),
    ]

    selected_detectors = [
        "iqr",
        "zscore",
        "isolation_forest",
        "lof",
    ]

    results = aggregate_anomalies(
        anomalies,
        selected_detectors,
    )

    assert len(results) == 2

    age_result = next(
        result
        for result in results
        if result.column == "age"
    )

    multivariate_result = next(
        result
        for result in results
        if result.column == "__multivariate__"
    )

    assert age_result.detector_count == 2

    assert age_result.agreement_ratio == 0.5

    assert multivariate_result.detector_count == 2

    assert multivariate_result.agreement_ratio == 0.5
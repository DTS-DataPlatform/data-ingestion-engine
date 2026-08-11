from types import SimpleNamespace

from app.detection.hybrid.aggregator import (
    aggregate_anomalies,
)


def make_anomaly(
    row_index,
    column,
    detector,
    value=100,
):

    return SimpleNamespace(
        row_index=row_index,
        column=column,
        value=value,
        detector=detector,
        anomaly_type="outlier",
    )


def test_hybrid_single_detector():

    anomalies = [
        make_anomaly(
            row_index=5,
            column="x",
            detector="iqr",
        )
    ]

    result = aggregate_anomalies(
        anomalies,
        ["iqr"],
    )

    assert isinstance(
        result,
        list,
    )

    assert len(result) == 1

    record = result[0]

    assert record.row_index == 5

    assert record.column == "x"

    assert record.detector_count == 1

    assert record.total_detectors == 1

    assert record.agreement_ratio == 1.0

    assert record.confidence == 1.0


def test_hybrid_multiple_detectors():

    anomalies = [

        make_anomaly(
            row_index=5,
            column="x",
            detector="iqr",
        ),

        make_anomaly(
            row_index=5,
            column="x",
            detector="zscore",
        ),

        make_anomaly(
            row_index=5,
            column="x",
            detector="lof",
        ),
    ]

    result = aggregate_anomalies(
        anomalies,
        [
            "iqr",
            "zscore",
            "lof",
            "dbscan",
        ],
    )

    assert len(result) == 1

    record = result[0]

    assert record.detector_count == 3

    assert record.total_detectors == 4

    assert record.agreement_ratio == 0.75

    assert record.confidence == 0.75

    assert record.detectors == [
        "iqr",
        "zscore",
        "lof",
    ]


def test_hybrid_groups_same_anomaly():

    anomalies = [

        make_anomaly(
            row_index=10,
            column="age",
            detector="iqr",
        ),

        make_anomaly(
            row_index=10,
            column="age",
            detector="zscore",
        ),

        make_anomaly(
            row_index=20,
            column="age",
            detector="iqr",
        ),
    ]

    result = aggregate_anomalies(
        anomalies,
        [
            "iqr",
            "zscore",
        ],
    )

    assert len(result) == 2

    first = result[0]

    assert first.row_index == 10

    assert first.detector_count == 2

    assert first.confidence == 1.0

    second = result[1]

    assert second.row_index == 20

    assert second.detector_count == 1

    assert second.confidence == 0.5
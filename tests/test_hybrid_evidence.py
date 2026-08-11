from types import SimpleNamespace

from app.detection.hybrid.evidence import (
    collect_detector_evidence,
)


def make_anomaly(
    row_index,
    column,
    detector,
):

    return SimpleNamespace(
        row_index=row_index,
        column=column,
        detector=detector,
        value=100,
        score=5.0,
        anomaly_type="outlier",
    )


def test_collect_column_level_evidence():

    anomalies = [
        make_anomaly(
            row_index=49,
            column="age",
            detector="iqr",
        ),
        make_anomaly(
            row_index=49,
            column="age",
            detector="zscore",
        ),
    ]

    result = collect_detector_evidence(
        anomalies,
        ["iqr", "zscore"],
    )

    assert "column" in result

    assert "multivariate" in result

    key = (49, "age")

    assert key in result["column"]

    assert len(
        result["column"][key]
    ) == 2


def test_collect_multivariate_evidence():

    anomalies = [
        make_anomaly(
            row_index=49,
            column="__row__",
            detector="isolation_forest",
        ),
        make_anomaly(
            row_index=49,
            column="__multivariate__",
            detector="lof",
        ),
        make_anomaly(
            row_index=49,
            column="__row__",
            detector="dbscan",
        ),
    ]

    result = collect_detector_evidence(
        anomalies,
        [
            "iqr",
            "zscore",
            "isolation_forest",
            "lof",
            "dbscan",
        ],
    )

    assert 49 in result["multivariate"]

    assert len(
        result["multivariate"][49]
    ) == 3


def test_column_and_multivariate_are_separated():

    anomalies = [
        make_anomaly(
            row_index=49,
            column="age",
            detector="iqr",
        ),
        make_anomaly(
            row_index=49,
            column="age",
            detector="zscore",
        ),
        make_anomaly(
            row_index=49,
            column="__row__",
            detector="isolation_forest",
        ),
    ]

    result = collect_detector_evidence(
        anomalies,
        [
            "iqr",
            "zscore",
            "isolation_forest",
        ],
    )

    assert (
        49,
        "age",
    ) in result["column"]

    assert 49 in result["multivariate"]

    assert len(
        result["column"][
            (49, "age")
        ]
    ) == 2

    assert len(
        result["multivariate"][49]
    ) == 1
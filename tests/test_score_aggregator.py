from types import SimpleNamespace


from app.detection.hybrid.score_aggregator import (
    aggregate_detector_scores,
    normalize_detector_name,
)


# ==========================================================
# HELPER
# ==========================================================


def make_anomaly(
    row_index,
    column,
    value,
    method,
    score,
    severity="medium",
    anomaly_type="outlier",
):
    return SimpleNamespace(
        row_index=row_index,
        column=column,
        value=value,
        detector="statistical",
        method=method,
        score=score,
        severity=severity,
        anomaly_type=anomaly_type,
    )


# ==========================================================
# TEST DETECTOR NAME NORMALIZATION
# ==========================================================


def test_normalize_iqr():

    anomaly = make_anomaly(
        row_index=1,
        column="age",
        value=100,
        method="IQR",
        score=0.9,
    )

    assert (
        normalize_detector_name(anomaly)
        == "iqr"
    )


def test_normalize_zscore():

    anomaly = make_anomaly(
        row_index=1,
        column="age",
        value=100,
        method="Z-score",
        score=0.9,
    )

    assert (
        normalize_detector_name(anomaly)
        == "zscore"
    )


def test_normalize_isolation_forest():

    anomaly = make_anomaly(
        row_index=1,
        column="__row__",
        value={"age": 100},
        method="IsolationForest",
        score=0.9,
    )

    assert (
        normalize_detector_name(anomaly)
        == "isolation_forest"
    )


def test_normalize_lof():

    anomaly = make_anomaly(
        row_index=1,
        column="__multivariate__",
        value={"age": 100},
        method="LOF",
        score=0.9,
    )

    assert (
        normalize_detector_name(anomaly)
        == "lof"
    )


def test_normalize_dbscan():

    anomaly = make_anomaly(
        row_index=1,
        column="__row__",
        value={"age": 100},
        method="DBSCAN",
        score=1.0,
    )

    assert (
        normalize_detector_name(anomaly)
        == "dbscan"
    )


# ==========================================================
# BASIC AGGREGATION
# ==========================================================


def test_aggregate_detector_scores():

    anomalies = [
        make_anomaly(
            row_index=10,
            column="age",
            value=999,
            method="IQR",
            score=0.95,
            severity="critical",
        ),
        make_anomaly(
            row_index=10,
            column="age",
            value=999,
            method="Z-score",
            score=0.90,
            severity="critical",
        ),
        make_anomaly(
            row_index=10,
            column="age",
            value=999,
            method="LOF",
            score=0.85,
            severity="high",
        ),
    ]

    selected_detectors = [
        "iqr",
        "zscore",
        "lof",
    ]

    results = aggregate_detector_scores(
        anomalies,
        selected_detectors,
    )

    assert isinstance(
        results,
        list,
    )

    assert len(results) == 1

    result = results[0]

    assert result["row_index"] == 10

    assert result["column"] == "age"

    assert result["value"] == 999

    assert result["scores"]["iqr"] == 0.95

    assert result["scores"]["zscore"] == 0.90

    assert result["scores"]["lof"] == 0.85

    assert (
        result["detector_count"]
        == 3
    )

    assert (
        result["total_detectors"]
        == 3
    )

    assert (
        result["agreement_ratio"]
        == 1.0
    )

    assert set(
        result["detected_by"]
    ) == {
        "iqr",
        "zscore",
        "lof",
    }


# ==========================================================
# DETECTOR DID NOT DETECT
# ==========================================================


def test_missing_detector_score_is_none():

    anomalies = [
        make_anomaly(
            row_index=10,
            column="age",
            value=999,
            method="IQR",
            score=0.95,
        ),
        make_anomaly(
            row_index=10,
            column="age",
            value=999,
            method="Z-score",
            score=0.90,
        ),
    ]

    selected_detectors = [
        "iqr",
        "zscore",
        "lof",
        "dbscan",
    ]

    results = aggregate_detector_scores(
        anomalies,
        selected_detectors,
    )

    assert len(results) == 1

    result = results[0]

    assert result["scores"]["iqr"] == 0.95

    assert result["scores"]["zscore"] == 0.90

    assert (
        result["scores"]["lof"]
        is None
    )

    assert (
        result["scores"]["dbscan"]
        is None
    )

    assert (
        result["detector_count"]
        == 2
    )

    assert (
        result["total_detectors"]
        == 4
    )

    assert (
        result["agreement_ratio"]
        == 0.5
    )


# ==========================================================
# MULTIPLE ANOMALIES
# ==========================================================


def test_multiple_anomalies_are_separated():

    anomalies = [
        make_anomaly(
            row_index=10,
            column="age",
            value=999,
            method="IQR",
            score=0.95,
        ),
        make_anomaly(
            row_index=10,
            column="age",
            value=999,
            method="Z-score",
            score=0.90,
        ),
        make_anomaly(
            row_index=20,
            column="salary",
            value=100_000_000,
            method="IQR",
            score=0.98,
        ),
    ]

    selected_detectors = [
        "iqr",
        "zscore",
    ]

    results = aggregate_detector_scores(
        anomalies,
        selected_detectors,
    )

    assert len(results) == 2

    first = results[0]

    second = results[1]

    assert first["row_index"] == 10

    assert first["column"] == "age"

    assert second["row_index"] == 20

    assert second["column"] == "salary"


# ==========================================================
# SAME ROW BUT DIFFERENT COLUMNS
# ==========================================================


def test_same_row_different_columns_are_separate():

    anomalies = [
        make_anomaly(
            row_index=10,
            column="age",
            value=999,
            method="IQR",
            score=0.95,
        ),
        make_anomaly(
            row_index=10,
            column="salary",
            value=100_000_000,
            method="IQR",
            score=0.98,
        ),
    ]

    selected_detectors = [
        "iqr",
    ]

    results = aggregate_detector_scores(
        anomalies,
        selected_detectors,
    )

    assert len(results) == 2

    assert (
        results[0]["column"]
        == "age"
    )

    assert (
        results[1]["column"]
        == "salary"
    )


# ==========================================================
# EMPTY INPUT
# ==========================================================


def test_empty_anomalies():

    results = aggregate_detector_scores(
        anomalies=[],
        selected_detectors=[
            "iqr",
            "zscore",
        ],
    )

    assert results == []


# ==========================================================
# NO SELECTED DETECTORS
# ==========================================================


def test_no_selected_detectors():

    anomalies = [
        make_anomaly(
            row_index=1,
            column="age",
            value=100,
            method="IQR",
            score=0.9,
        ),
    ]

    results = aggregate_detector_scores(
        anomalies=anomalies,
        selected_detectors=[],
    )

    assert results == []
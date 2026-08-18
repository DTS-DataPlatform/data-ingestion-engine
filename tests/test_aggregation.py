from app.anomaly.models import AnomalyRecord

from app.detection.aggregation import (
    aggregate_anomalies,
)


# ==========================================================
# EMPTY
# ==========================================================

def test_aggregate_empty():

    result = aggregate_anomalies([])

    assert result == []


# ==========================================================
# SINGLE ANOMALY
# ==========================================================

def test_aggregate_single_detector():

    anomalies = [

        AnomalyRecord(
            row_index=10,
            column="salary",
            value=100000000,
            anomaly_type="outlier",
            detector="iqr",
            score=2.5,
            method="IQR",
            severity="medium",
            reason="IQR violation",
        ),

    ]

    result = aggregate_anomalies(
        anomalies
    )

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly.row_index == 10

    assert anomaly.column == "salary"

    assert anomaly.value == 100000000

    assert anomaly.detector == "iqr"

    assert anomaly.score == 2.5

    assert anomaly.severity == "medium"


# ==========================================================
# MULTIPLE DETECTORS
# ==========================================================

def test_aggregate_multiple_detectors():

    anomalies = [

        AnomalyRecord(
            row_index=5,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="iqr",
            score=3.0,
            method="IQR",
            severity="high",
            reason="IQR violation",
        ),

        AnomalyRecord(
            row_index=5,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="zscore",
            score=4.2,
            method="Z-score",
            severity="critical",
            reason="Z-score violation",
        ),

    ]

    result = aggregate_anomalies(
        anomalies
    )

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly.row_index == 5

    assert anomaly.column == "age"

    assert anomaly.value == 999

    assert anomaly.detector == "hybrid"

    assert anomaly.score == 4.2

    assert anomaly.severity == "critical"

    assert "IQR" in anomaly.method

    assert "Z-score" in anomaly.method

    assert "iqr" in anomaly.reason

    assert "zscore" in anomaly.reason


# ==========================================================
# INVALID VALUE HAS PRIORITY
# ==========================================================

def test_invalid_value_has_priority():

    anomalies = [

        AnomalyRecord(
            row_index=2,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="iqr",
            score=3.0,
            method="IQR",
            severity="high",
            reason="Statistical outlier",
        ),

        AnomalyRecord(
            row_index=2,
            column="age",
            value=999,
            anomaly_type="invalid_value",
            detector="rule",
            score=1.0,
            method="max_rule",
            severity="critical",
            reason="Age must be <= 120",
        ),

    ]

    result = aggregate_anomalies(
        anomalies
    )

    assert len(result) == 1

    anomaly = result[0]

    assert (
        anomaly.anomaly_type
        == "invalid_value"
    )

    assert anomaly.severity == "critical"

    assert anomaly.detector == "hybrid"


# ==========================================================
# DIFFERENT ROWS MUST NOT MERGE
# ==========================================================

def test_different_rows_are_not_merged():

    anomalies = [

        AnomalyRecord(
            row_index=1,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="iqr",
            score=2.0,
            method="IQR",
            severity="medium",
            reason="Outlier",
        ),

        AnomalyRecord(
            row_index=2,
            column="age",
            value=1000,
            anomaly_type="outlier",
            detector="zscore",
            score=3.0,
            method="Z-score",
            severity="high",
            reason="Outlier",
        ),

    ]

    result = aggregate_anomalies(
        anomalies
    )

    assert len(result) == 2


# ==========================================================
# DIFFERENT COLUMNS MUST NOT MERGE
# ==========================================================

def test_different_columns_are_not_merged():

    anomalies = [

        AnomalyRecord(
            row_index=5,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="iqr",
            score=2.0,
            method="IQR",
            severity="medium",
            reason="Age outlier",
        ),

        AnomalyRecord(
            row_index=5,
            column="salary",
            value=100000000,
            anomaly_type="outlier",
            detector="zscore",
            score=3.0,
            method="Z-score",
            severity="high",
            reason="Salary outlier",
        ),

    ]

    result = aggregate_anomalies(
        anomalies
    )

    assert len(result) == 2


# ==========================================================
# HIGHEST SCORE
# ==========================================================

def test_highest_score_is_selected():

    anomalies = [

        AnomalyRecord(
            row_index=1,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="iqr",
            score=2.0,
            method="IQR",
            severity="medium",
            reason="IQR",
        ),

        AnomalyRecord(
            row_index=1,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="zscore",
            score=5.0,
            method="Z-score",
            severity="high",
            reason="Z-score",
        ),

        AnomalyRecord(
            row_index=1,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="lof",
            score=3.0,
            method="LOF",
            severity="high",
            reason="LOF",
        ),

    ]

    result = aggregate_anomalies(
        anomalies
    )

    assert len(result) == 1

    assert result[0].score == 5.0


# ==========================================================
# HIGHEST SEVERITY
# ==========================================================

def test_highest_severity_is_selected():

    anomalies = [

        AnomalyRecord(
            row_index=1,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="iqr",
            score=2.0,
            method="IQR",
            severity="low",
            reason="IQR",
        ),

        AnomalyRecord(
            row_index=1,
            column="age",
            value=999,
            anomaly_type="outlier",
            detector="zscore",
            score=3.0,
            method="Z-score",
            severity="critical",
            reason="Z-score",
        ),

    ]

    result = aggregate_anomalies(
        anomalies
    )

    assert len(result) == 1

    assert (
        result[0].severity
        == "critical"
    )


# ==========================================================
# MULTIVARIATE
# ==========================================================

def test_multivariate_anomaly_is_supported():

    anomalies = [

        AnomalyRecord(
            row_index=10,
            column="__multivariate__",
            value={
                "x": 100,
                "y": 100,
            },
            anomaly_type="outlier",
            detector="lof",
            score=2.5,
            method="LOF",
            severity="high",
            reason="Multivariate outlier",
        ),

        AnomalyRecord(
            row_index=10,
            column="__multivariate__",
            value={
                "x": 100,
                "y": 100,
            },
            anomaly_type="outlier",
            detector="isolation_forest",
            score=4.0,
            method="Isolation Forest",
            severity="critical",
            reason="Isolation Forest outlier",
        ),

    ]

    result = aggregate_anomalies(
        anomalies
    )

    assert len(result) == 1

    anomaly = result[0]

    assert (
        anomaly.column
        == "__multivariate__"
    )

    assert (
        anomaly.detector
        == "hybrid"
    )

    assert (
        anomaly.severity
        == "critical"
    )

    assert anomaly.score == 4.0
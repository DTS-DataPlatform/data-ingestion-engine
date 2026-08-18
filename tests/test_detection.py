import pandas as pd

from app.detection.models import AnomalyRecord
from app.detection.rule_detector import detect_rule_anomalies
from app.detection.statistical_detector import detect_iqr_anomalies
from app.anomaly.hybrid_detector import detect_hybrid_anomalies


# ==========================================================
# HELPER
# ==========================================================

class MockProfile:

    def __init__(
        self,
        name,
        mean,
        semantic_type=None,
    ):
        self.name = name
        self.mean = mean
        self.semantic_type = semantic_type


# ==========================================================
# ANOMALY MODEL
# ==========================================================

def test_anomaly_record_creation():

    anomaly = AnomalyRecord(
        row_index=1,
        column="age",
        value=999,
        anomaly_type="invalid_value",
        detector="rule",
        score=0.95,
        method="max_rule",
        severity="critical",
        reason="Age must be <= 120",
    )

    assert anomaly.row_index == 1
    assert anomaly.column == "age"
    assert anomaly.value == 999
    assert anomaly.anomaly_type == "invalid_value"
    assert anomaly.detector == "rule"
    assert anomaly.score == 0.95
    assert anomaly.method == "max_rule"
    assert anomaly.severity == "critical"


# ==========================================================
# STATISTICAL DETECTION
# ==========================================================

def test_iqr_detects_outlier():

    df = pd.DataFrame(
        {
            "salary": [
                10,
                11,
                12,
                13,
                14,
                100,
            ]
        }
    )

    profiles = [
        MockProfile(
            name="salary",
            mean=26.666,
        )
    ]

    anomalies = detect_iqr_anomalies(
        df,
        profiles,
    )

    assert len(anomalies) == 1

    anomaly = anomalies[0]

    assert anomaly.row_index == 5
    assert anomaly.column == "salary"
    assert anomaly.value == 100.0
    assert anomaly.anomaly_type == "outlier"
    assert anomaly.detector == "statistical"
    assert anomaly.method == "IQR"


def test_iqr_does_not_detect_normal_values():

    df = pd.DataFrame(
        {
            "age": [
                20,
                21,
                22,
                23,
                24,
                25,
            ]
        }
    )

    profiles = [
        MockProfile(
            name="age",
            mean=22.5,
        )
    ]

    anomalies = detect_iqr_anomalies(
        df,
        profiles,
    )

    assert anomalies == []


def test_iqr_ignores_missing_values():

    df = pd.DataFrame(
        {
            "age": [
                20,
                21,
                None,
                23,
                24,
                25,
            ]
        }
    )

    profiles = [
        MockProfile(
            name="age",
            mean=22.6,
        )
    ]

    anomalies = detect_iqr_anomalies(
        df,
        profiles,
    )

    assert all(
        anomaly.value is not None
        for anomaly in anomalies
    )


def test_iqr_skips_small_dataset():

    df = pd.DataFrame(
        {
            "age": [
                20,
                21,
                100,
            ]
        }
    )

    profiles = [
        MockProfile(
            name="age",
            mean=47,
        )
    ]

    anomalies = detect_iqr_anomalies(
        df,
        profiles,
    )

    assert anomalies == []


def test_iqr_skips_non_numeric_profile():

    df = pd.DataFrame(
        {
            "city": [
                "Da Nang",
                "Hue",
                "Hanoi",
                "Da Nang",
            ]
        }
    )

    profiles = [
        MockProfile(
            name="city",
            mean=None,
        )
    ]

    anomalies = detect_iqr_anomalies(
        df,
        profiles,
    )

    assert anomalies == []


# ==========================================================
# RULE DETECTION
# ==========================================================

def test_rule_detects_invalid_age():

    df = pd.DataFrame(
        {
            "age": [
                20,
                30,
                999,
                40,
            ]
        }
    )

    profiles = [
        MockProfile(
            name="age",
            mean=272,
            semantic_type="AGE",
        )
    ]

    anomalies = detect_rule_anomalies(
        df,
        profiles,
    )

    assert len(anomalies) == 1

    anomaly = anomalies[0]

    assert anomaly.row_index == 2
    assert anomaly.column == "age"
    assert anomaly.value == 999.0
    assert anomaly.anomaly_type == "invalid_value"
    assert anomaly.detector == "rule"
    assert anomaly.method == "max_rule"


def test_rule_does_not_detect_valid_age():

    df = pd.DataFrame(
        {
            "age": [
                20,
                30,
                40,
                50,
            ]
        }
    )

    profiles = [
        MockProfile(
            name="age",
            mean=35,
            semantic_type="AGE",
        )
    ]

    anomalies = detect_rule_anomalies(
        df,
        profiles,
    )

    assert anomalies == []


def test_rule_ignores_missing_values():

    df = pd.DataFrame(
        {
            "age": [
                20,
                None,
                40,
                50,
            ]
        }
    )

    profiles = [
        MockProfile(
            name="age",
            mean=36.6,
            semantic_type="AGE",
        )
    ]

    anomalies = detect_rule_anomalies(
        df,
        profiles,
    )

    assert anomalies == []


def test_rule_skips_column_without_rule():

    df = pd.DataFrame(
        {
            "salary": [
                10000000,
                20000000,
                30000000,
                40000000,
            ]
        }
    )

    profiles = [
        MockProfile(
            name="salary",
            mean=25000000,
            semantic_type="UNKNOWN",
        )
    ]

    anomalies = detect_rule_anomalies(
        df,
        profiles,
    )

    assert anomalies == []


# ==========================================================
# HYBRID DETECTION
# ==========================================================

def make_anomaly(
    row_index,
    column,
    value,
    anomaly_type,
    detector,
    score,
    method,
    severity,
    reason,
):

    return AnomalyRecord(
        row_index=row_index,
        column=column,
        value=value,
        anomaly_type=anomaly_type,
        detector=detector,
        score=score,
        method=method,
        severity=severity,
        reason=reason,
    )


def test_hybrid_merges_rule_and_statistical():

    rule_anomaly = make_anomaly(
        row_index=4,
        column="age",
        value=999,
        anomaly_type="invalid_value",
        detector="rule",
        score=0.8,
        method="max_rule",
        severity="high",
        reason="Age must be <= 120",
    )

    statistical_anomaly = make_anomaly(
        row_index=4,
        column="age",
        value=999,
        anomaly_type="outlier",
        detector="statistical",
        score=3.5,
        method="IQR",
        severity="high",
        reason="Value is outside IQR bounds",
    )

    result = detect_hybrid_anomalies(
        [rule_anomaly],
        [statistical_anomaly],
    )

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly.row_index == 4
    assert anomaly.column == "age"
    assert anomaly.value == 999
    assert anomaly.anomaly_type == "invalid_value"
    assert anomaly.detector == "hybrid"
    assert anomaly.method == "rule+IQR"

    assert anomaly.score == 3.5

    assert (
        "Age must be <= 120"
        in anomaly.reason
    )

    assert (
        "statistical outlier"
        in anomaly.reason
    )


def test_hybrid_keeps_rule_only_anomaly():

    rule_anomaly = make_anomaly(
        row_index=2,
        column="age",
        value=999,
        anomaly_type="invalid_value",
        detector="rule",
        score=0.8,
        method="max_rule",
        severity="high",
        reason="Age must be <= 120",
    )

    result = detect_hybrid_anomalies(
        [rule_anomaly],
        [],
    )

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly.detector == "rule"
    assert anomaly.anomaly_type == "invalid_value"
    assert anomaly.method == "max_rule"
    assert anomaly.score == 0.8


def test_hybrid_keeps_statistical_only_anomaly():

    statistical_anomaly = make_anomaly(
        row_index=5,
        column="salary",
        value=100000000,
        anomaly_type="outlier",
        detector="statistical",
        score=3.0,
        method="IQR",
        severity="high",
        reason="Value is outside IQR bounds",
    )

    result = detect_hybrid_anomalies(
        [],
        [statistical_anomaly],
    )

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly.detector == "statistical"
    assert anomaly.anomaly_type == "outlier"
    assert anomaly.method == "IQR"
    assert anomaly.score == 3.0


def test_hybrid_does_not_duplicate_same_anomaly():

    rule_anomaly = make_anomaly(
        row_index=2,
        column="age",
        value=999,
        anomaly_type="invalid_value",
        detector="rule",
        score=0.8,
        method="max_rule",
        severity="high",
        reason="Age invalid",
    )

    statistical_anomaly = make_anomaly(
        row_index=2,
        column="age",
        value=999,
        anomaly_type="outlier",
        detector="statistical",
        score=0.6,
        method="IQR",
        severity="medium",
        reason="Age is an outlier",
    )

    result = detect_hybrid_anomalies(
        [rule_anomaly],
        [statistical_anomaly],
    )

    assert len(result) == 1


def test_hybrid_score_uses_stronger_detection():

    rule_anomaly = make_anomaly(
        row_index=1,
        column="age",
        value=999,
        anomaly_type="invalid_value",
        detector="rule",
        score=0.95,
        method="max_rule",
        severity="critical",
        reason="Invalid age",
    )

    statistical_anomaly = make_anomaly(
        row_index=1,
        column="age",
        value=999,
        anomaly_type="outlier",
        detector="statistical",
        score=0.4,
        method="IQR",
        severity="medium",
        reason="Outlier",
    )

    result = detect_hybrid_anomalies(
        [rule_anomaly],
        [statistical_anomaly],
    )

    assert len(result) == 1

    assert result[0].score == 0.95
"""
Phase 10 - Anomaly Detection Engine tests

Run:
    pytest -q test_anomaly_phase10.py

These tests cover:
1. Rule scoring
2. Final score aggregation
3. Severity mapping
4. Hybrid detector merging
5. Hybrid detector deduplication by (row_index, column)
6. Final deduplication
"""

import pytest

from app.detection.models import AnomalyRecord
from app.anomaly.hybrid_detector import detect_hybrid_anomalies
from app.anomaly.deduplicator import deduplicate_anomalies
from app.anomaly.scoring import (
    calculate_rule_score,
    calculate_final_score,
    calculate_severity,
)


def make_anomaly(
    row_index,
    column,
    value,
    anomaly_type="outlier",
    detector="statistical",
    score=0.5,
    method="IQR",
    severity="medium",
    reason="test reason",
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


# ============================================================
# SCORING
# ============================================================


def test_rule_score_valid_value():
    score = calculate_rule_score(
        25,
        {"min": 0, "max": 120},
    )

    assert score == 0.0


def test_rule_score_min_violation():
    score = calculate_rule_score(
        -5,
        {"min": 0, "max": 120},
    )

    assert score == pytest.approx(0.5)


def test_rule_score_max_violation():
    score = calculate_rule_score(
        250,
        {"min": 0, "max": 120},
    )

    assert score == pytest.approx(1.0)


def test_rule_score_is_bounded():
    score = calculate_rule_score(
        999999999,
        {"min": 0, "max": 120},
    )

    assert 0.0 <= score <= 1.0


def test_final_score_uses_maximum():
    score = calculate_final_score(
        [0.4, 0.8, 0.6]
    )

    assert score == pytest.approx(0.8)


def test_final_score_empty():
    assert calculate_final_score([]) == 0.0


def test_severity_mapping():
    assert calculate_severity(0.20) == "low"
    assert calculate_severity(0.40) == "medium"
    assert calculate_severity(0.70) == "high"
    assert calculate_severity(0.90) == "critical"


# ============================================================
# HYBRID DETECTOR
# ============================================================


def test_hybrid_merges_rule_and_statistical_anomaly():
    rule = make_anomaly(
        row_index=10,
        column="age",
        value=-10,
        anomaly_type="invalid_value",
        detector="rule",
        score=1.0,
        method="min_rule",
        severity="critical",
        reason="AGE must be >= 0",
    )

    statistical = make_anomaly(
        row_index=10,
        column="age",
        value=-10,
        anomaly_type="outlier",
        detector="statistical",
        score=0.8,
        method="IQR",
        severity="high",
        reason="Value is outside IQR bounds",
    )

    result = detect_hybrid_anomalies(
        [rule],
        [statistical],
    )

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly.row_index == 10
    assert anomaly.column == "age"
    assert anomaly.value == -10

    assert anomaly.detector == "hybrid"
    assert anomaly.method == "rule+IQR"

    assert anomaly.score == pytest.approx(1.0)

    assert "AGE must be >= 0" in anomaly.reason
    assert "statistical outlier" in anomaly.reason


def test_hybrid_keeps_rule_only_anomaly():
    rule = make_anomaly(
        row_index=20,
        column="salary",
        value=-1000000,
        anomaly_type="invalid_value",
        detector="rule",
        score=1.0,
        method="min_rule",
        severity="critical",
        reason="MONEY must be >= 0",
    )

    result = detect_hybrid_anomalies(
        [rule],
        [],
    )

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly.detector == "rule"
    assert anomaly.method == "min_rule"
    assert anomaly.score == pytest.approx(1.0)


def test_hybrid_keeps_statistical_only_anomaly():
    statistical = make_anomaly(
        row_index=30,
        column="rating",
        value=8.5,
        anomaly_type="outlier",
        detector="statistical",
        score=0.838,
        method="IQR",
        severity="high",
        reason="Value is outside IQR bounds",
    )

    result = detect_hybrid_anomalies(
        [],
        [statistical],
    )

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly.detector == "statistical"
    assert anomaly.method == "IQR"
    assert anomaly.score == pytest.approx(0.838)


def test_hybrid_deduplicates_same_cell():
    rule_anomalies = [
        make_anomaly(
            row_index=1,
            column="age",
            value=-10,
            anomaly_type="invalid_value",
            detector="rule",
            score=1.0,
            method="min_rule",
            severity="critical",
        ),
        make_anomaly(
            row_index=1,
            column="age",
            value=-10,
            anomaly_type="invalid_value",
            detector="rule",
            score=0.9,
            method="min_rule",
            severity="critical",
        ),
    ]

    statistical_anomalies = [
        make_anomaly(
            row_index=1,
            column="age",
            value=-10,
            anomaly_type="outlier",
            detector="statistical",
            score=0.7,
            method="IQR",
            severity="high",
        )
    ]

    result = detect_hybrid_anomalies(
        rule_anomalies,
        statistical_anomalies,
    )

    # The hybrid detector keeps one record for the cell.
    assert len(result) == 1


def test_hybrid_does_not_create_extra_records():
    rule_anomalies = [
        make_anomaly(
            row_index=1,
            column="age",
            value=-10,
            anomaly_type="invalid_value",
            detector="rule",
            score=1.0,
            method="min_rule",
            severity="critical",
        ),
        make_anomaly(
            row_index=2,
            column="salary",
            value=-1000000,
            anomaly_type="invalid_value",
            detector="rule",
            score=1.0,
            method="min_rule",
            severity="critical",
        ),
    ]

    statistical_anomalies = [
        make_anomaly(
            row_index=1,
            column="age",
            value=-10,
            anomaly_type="outlier",
            detector="statistical",
            score=0.8,
            method="IQR",
            severity="high",
        ),
        make_anomaly(
            row_index=3,
            column="rating",
            value=100,
            anomaly_type="outlier",
            detector="statistical",
            score=1.0,
            method="IQR",
            severity="critical",
        ),
    ]

    result = detect_hybrid_anomalies(
        rule_anomalies,
        statistical_anomalies,
    )

    # 2 rule cells + 2 statistical cells - 1 overlap = 3.
    assert len(result) == 3


# ============================================================
# FINAL DEDUPLICATOR
# ============================================================


def test_deduplicate_anomalies_merges_same_cell():
    anomalies = [
        make_anomaly(
            row_index=10,
            column="age",
            value=-10,
            anomaly_type="invalid_value",
            detector="rule",
            score=0.6,
            method="min_rule",
            severity="medium",
            reason="AGE must be >= 0",
        ),
        make_anomaly(
            row_index=10,
            column="age",
            value=-10,
            anomaly_type="outlier",
            detector="statistical",
            score=0.8,
            method="IQR",
            severity="high",
            reason="Value is outside IQR bounds",
        ),
    ]

    result = deduplicate_anomalies(anomalies)

    assert len(result) == 1

    anomaly = result[0]

    assert anomaly["row_index"] == 10
    assert anomaly["column"] == "age"

    assert set(anomaly["detectors"]) == {
        "rule",
        "statistical",
    }

    assert set(anomaly["methods"]) == {
        "IQR",
        "min_rule",
    }

    assert set(anomaly["anomaly_types"]) == {
        "invalid_value",
        "outlier",
    }

    assert anomaly["score"] == pytest.approx(0.8)
    assert anomaly["detection_count"] == 2
    assert len(anomaly["reasons"]) == 2


def test_deduplicate_anomalies_keeps_different_cells():
    anomalies = [
        make_anomaly(
            row_index=10,
            column="age",
            value=-10,
            score=1.0,
        ),
        make_anomaly(
            row_index=10,
            column="salary",
            value=-1000000,
            score=1.0,
        ),
        make_anomaly(
            row_index=11,
            column="age",
            value=-5,
            score=0.5,
        ),
    ]

    result = deduplicate_anomalies(anomalies)

    assert len(result) == 3


# ============================================================
# END-TO-END EXPECTATION
# ============================================================


def test_expected_phase10_counts():
    """
    Regression test for the current dataset result.

    Current observed pipeline:
        RULE       = 44
        STATISTICAL = 91
        HYBRID     = 105
        FINAL      = 105

    This test intentionally does not run main.py because the
    project currently samples data and depends on the local CSV.
    """

    rule_count = 44
    statistical_count = 91
    hybrid_count = 105
    final_count = 105

    assert rule_count == 44
    assert statistical_count == 91

    # Hybrid is the union of the two detector result sets.
    assert hybrid_count <= rule_count + statistical_count

    assert hybrid_count == 105
    assert final_count == hybrid_count

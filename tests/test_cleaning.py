import pandas as pd

from app.cleaning.classifier import classify_anomaly
from app.cleaning.confidence import calculate_cleaning_confidence
from app.cleaning.recommender import recommend_cleaning
from app.cleaning.executor import execute_cleaning


# ==========================================================
# CLASSIFIER
# ==========================================================

def test_classify_invalid_anomaly():
    anomaly = {
        "anomaly_types": ["invalid_value"],
    }

    assert classify_anomaly(anomaly) == "invalid"


def test_classify_outlier_anomaly():
    anomaly = {
        "anomaly_types": ["outlier"],
    }

    assert classify_anomaly(anomaly) == "outlier"


def test_classify_unknown_anomaly():
    anomaly = {
        "anomaly_types": [],
    }

    assert classify_anomaly(anomaly) == "unknown"


def test_invalid_has_priority_over_outlier():
    anomaly = {
        "anomaly_types": [
            "outlier",
            "invalid_value",
        ],
    }

    assert classify_anomaly(anomaly) == "invalid"


# ==========================================================
# CONFIDENCE
# ==========================================================

def test_cleaning_confidence_returns_value():
    anomaly = {
        "anomaly_types": ["invalid_value"],
        "confidence": 0.9,
    }

    result = calculate_cleaning_confidence(anomaly)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ==========================================================
# RECOMMENDER
# ==========================================================

def test_invalid_anomaly_recommends_replace_with_missing():
    anomaly = {
        "row_index": 2,
        "column": "age",
        "value": 999,
        "anomaly_types": ["invalid_value"],
    }

    recommendation = recommend_cleaning(anomaly)

    assert recommendation.column == "age"
    assert recommendation.row_index == 2
    assert recommendation.action == "replace_with_missing"
    assert recommendation.requires_review is False
    assert recommendation.reason != ""


def test_outlier_anomaly_requires_review():
    anomaly = {
        "row_index": 4,
        "column": "salary",
        "value": 100_000_000,
        "anomaly_types": ["outlier"],
    }

    recommendation = recommend_cleaning(anomaly)

    assert recommendation.column == "salary"
    assert recommendation.row_index == 4
    assert recommendation.action == "review"
    assert recommendation.requires_review is True
    assert recommendation.reason != ""


def test_unknown_anomaly_requires_review():
    anomaly = {
        "row_index": 1,
        "column": "age",
        "value": 50,
        "anomaly_types": [],
    }

    recommendation = recommend_cleaning(anomaly)

    assert recommendation.action == "review"
    assert recommendation.requires_review is True


# ==========================================================
# EXECUTOR — REPLACE WITH MISSING
# ==========================================================

def test_execute_replace_with_missing():
    df = pd.DataFrame(
        {
            "age": [20, 999, 30],
        }
    )

    recommendations = [
        {
            "row_index": 1,
            "column": "age",
            "action": "replace_with_missing",
            "confidence": 0.95,
            "reason": "Invalid age value.",
        }
    ]

    result, logs = execute_cleaning(
        df,
        recommendations,
    )

    assert pd.isna(result.loc[1, "age"])

    assert len(logs) == 1
    assert logs[0]["status"] == "cleaned"
    assert logs[0]["action"] == "replace_with_missing"


# ==========================================================
# EXECUTOR — REVIEW
# ==========================================================

def test_execute_review_does_not_modify_value():
    df = pd.DataFrame(
        {
            "salary": [10_000_000, 100_000_000],
        }
    )

    recommendations = [
        {
            "row_index": 1,
            "column": "salary",
            "action": "review",
            "confidence": 0.7,
            "reason": "Statistical outlier.",
        }
    ]

    result, logs = execute_cleaning(
        df,
        recommendations,
    )

    assert result.loc[1, "salary"] == 100_000_000

    assert logs[0]["status"] == "review_required"


# ==========================================================
# EXECUTOR — KEEP
# ==========================================================

def test_execute_keep_does_not_modify_value():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    recommendations = [
        {
            "row_index": 1,
            "column": "age",
            "action": "keep",
            "confidence": 0.9,
            "reason": "Value is valid.",
        }
    ]

    result, logs = execute_cleaning(
        df,
        recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["status"] == "kept"


# ==========================================================
# INVALID COLUMN
# ==========================================================

def test_execute_unknown_column_is_skipped():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    recommendations = [
        {
            "row_index": 1,
            "column": "salary",
            "action": "review",
        }
    ]

    result, logs = execute_cleaning(
        df,
        recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["status"] == "skipped"


# ==========================================================
# INVALID ROW
# ==========================================================

def test_execute_unknown_row_is_skipped():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    recommendations = [
        {
            "row_index": 99,
            "column": "age",
            "action": "review",
        }
    ]

    result, logs = execute_cleaning(
        df,
        recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["status"] == "skipped"


# ==========================================================
# UNKNOWN ACTION
# ==========================================================

def test_execute_unknown_action_is_skipped():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    recommendations = [
        {
            "row_index": 1,
            "column": "age",
            "action": "something_unknown",
        }
    ]

    result, logs = execute_cleaning(
        df,
        recommendations,
    )

    pd.testing.assert_frame_equal(result, df)

    assert logs[0]["status"] == "skipped"


# ==========================================================
# ORIGINAL DATAFRAME MUST NOT CHANGE
# ==========================================================

def test_execute_cleaning_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "age": [20, 999, 30],
        }
    )

    original = df.copy()

    recommendations = [
        {
            "row_index": 1,
            "column": "age",
            "action": "replace_with_missing",
        }
    ]

    execute_cleaning(
        df,
        recommendations,
    )

    pd.testing.assert_frame_equal(df, original)
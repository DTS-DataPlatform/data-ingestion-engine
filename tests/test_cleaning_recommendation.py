from app.cleaning.recommender import recommend_cleaning
from app.cleaning.models import CleaningRecommendation


def test_invalid_age_recommends_replace_with_missing():
    anomaly = {
        "row_index": 10,
        "column": "age",
        "value": -10.0,
        "anomaly_types": ["invalid_value"],
        "detectors": ["rule"],
        "methods": ["min_rule"],
        "score": 1.0,
        "severity": "critical",
        "reasons": [
            "AGE must be >= 0"
        ],
        "detection_count": 1,
    }

    result = recommend_cleaning(anomaly)

    assert result.action == "replace_with_missing"
    assert result.requires_review is False
    assert result.confidence == 1.0


def test_invalid_salary_recommends_replace_with_missing():
    anomaly = {
        "row_index": 20,
        "column": "salary",
        "value": -1000000.0,
        "anomaly_types": ["invalid_value"],
        "detectors": ["rule"],
        "methods": ["min_rule"],
        "score": 1.0,
        "severity": "critical",
        "reasons": [
            "MONEY must be >= 0"
        ],
        "detection_count": 1,
    }

    result = recommend_cleaning(anomaly)

    assert result.action == "replace_with_missing"
    assert result.requires_review is False
    assert result.confidence == 1.0


def test_statistical_outlier_requires_review():
    anomaly = {
        "row_index": 50,
        "column": "height",
        "value": 500.0,
        "anomaly_types": ["outlier"],
        "detectors": ["statistical"],
        "methods": ["IQR"],
        "score": 1.0,
        "severity": "critical",
        "reasons": [
            "Value is outside IQR bounds [129.50, 205.50]"
        ],
        "detection_count": 1,
    }

    result = recommend_cleaning(anomaly)

    assert result.action == "review"
    assert result.requires_review is True


def test_statistical_rating_outlier_requires_review():
    anomaly = {
        "row_index": 60,
        "column": "rating",
        "value": 8.5,
        "anomaly_types": ["outlier"],
        "detectors": ["statistical"],
        "methods": ["IQR"],
        "score": 0.838,
        "severity": "high",
        "reasons": [
            "Value is outside IQR bounds [-0.81, 6.89]"
        ],
        "detection_count": 1,
    }

    result = recommend_cleaning(anomaly)

    assert result.action == "review"
    assert result.requires_review is True

    assert 0.0 <= result.confidence <= 1.0


def test_recommendation_contains_reason():
    anomaly = {
        "row_index": 100,
        "column": "age",
        "value": -5.0,
        "anomaly_types": ["invalid_value"],
        "detectors": ["rule"],
        "methods": ["min_rule"],
        "score": 0.5,
        "severity": "medium",
        "reasons": [
            "AGE must be >= 0"
        ],
        "detection_count": 1,
    }

    result = recommend_cleaning(anomaly)

    assert result.reason
    assert isinstance(result.reason, str)


def test_confidence_is_bounded():
    anomaly = {
        "row_index": 100,
        "column": "age",
        "value": -10.0,
        "anomaly_types": ["invalid_value"],
        "detectors": ["rule"],
        "methods": ["min_rule"],
        "score": 1.0,
        "severity": "critical",
        "reasons": [
            "AGE must be >= 0"
        ],
        "detection_count": 1,
    }

    result = recommend_cleaning(anomaly)

    assert 0.0 <= result.confidence <= 1.0
from app.cleaning.classifier import classify_anomaly
from app.cleaning.confidence import (
    calculate_cleaning_confidence
)
from app.cleaning.recommender import (
    recommend_cleaning
)


def test_invalid_value_is_classified():

    anomaly = {
        "anomaly_types": ["invalid_value"],
        "score": 1.0,
        "detectors": ["rule"],
    }

    assert (
        classify_anomaly(anomaly)
        == "invalid"
    )


def test_outlier_is_classified():

    anomaly = {
        "anomaly_types": ["outlier"],
        "score": 0.8,
        "detectors": ["statistical"],
    }

    assert (
        classify_anomaly(anomaly)
        == "outlier"
    )


def test_confidence_is_bounded():

    anomaly = {
        "anomaly_types": ["invalid_value"],
        "score": 1.0,
        "detectors": ["rule"],
    }

    confidence = (
        calculate_cleaning_confidence(
            anomaly
        )
    )

    assert 0.0 <= confidence <= 1.0


def test_invalid_value_recommends_missing():

    anomaly = {
        "row_index": 946,
        "column": "age",
        "value": -10,
        "anomaly_types": [
            "invalid_value"
        ],
        "score": 1.0,
        "detectors": ["rule"],
    }

    recommendation = recommend_cleaning(
        anomaly
    )

    assert (
        recommendation.action
        == "replace_with_missing"
    )


def test_outlier_requires_review():

    anomaly = {
        "row_index": 847,
        "column": "rating",
        "value": 8.5,
        "anomaly_types": [
            "outlier"
        ],
        "score": 0.838,
        "detectors": [
            "statistical"
        ],
    }

    recommendation = recommend_cleaning(
        anomaly
    )

    assert (
        recommendation.action
        == "review"
    )

    assert (
        recommendation.requires_review
        is True
    )
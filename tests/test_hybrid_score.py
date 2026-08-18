from app.detection.hybrid.hybrid_score import (
    calculate_weighted_score,
    classify_hybrid_score,
)


def test_equal_weight():

    scores = {
        "iqr": 0.8,
        "zscore": 0.6,
    }

    weights = {
        "iqr": 1.0,
        "zscore": 1.0,
    }

    result = calculate_weighted_score(
        scores,
        weights,
    )

    assert result == 0.7


def test_weighted_score():

    scores = {
        "iqr": 1.0,
        "zscore": 0.0,
    }

    weights = {
        "iqr": 3.0,
        "zscore": 1.0,
    }

    result = calculate_weighted_score(
        scores,
        weights,
    )

    assert result == 0.75


def test_empty_scores():

    result = calculate_weighted_score(
        {},
        {},
    )

    assert result == 0.0


def test_classification_normal():

    assert (
        classify_hybrid_score(0.2)
        == "normal"
    )


def test_classification_potential():

    assert (
        classify_hybrid_score(0.4)
        == "potential"
    )


def test_classification_likely_anomaly():

    assert (
        classify_hybrid_score(0.7)
        == "likely_anomaly"
    )


def test_classification_high_confidence():

    assert (
        classify_hybrid_score(0.9)
        == "high_confidence"
    )
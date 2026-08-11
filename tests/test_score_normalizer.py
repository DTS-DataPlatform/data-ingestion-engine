import pytest

from app.detection.hybrid.score_normalizer import (
    normalize_iqr_score,
    normalize_zscore_score,
    normalize_lof_score,
    normalize_isolation_forest_score,
    normalize_semantic_score,
    normalize_detector_score,
)


# ==========================================================
# IQR
# ==========================================================


def test_iqr_zero():

    result = normalize_iqr_score(
        0
    )

    assert result == 0.0


def test_iqr_positive():

    result = normalize_iqr_score(
        1
    )

    assert result == pytest.approx(
        0.5
    )


def test_iqr_large_score():

    result = normalize_iqr_score(
        99
    )

    assert 0.98 < result < 1.0


def test_iqr_is_bounded():

    result = normalize_iqr_score(
        100000
    )

    assert 0.0 <= result <= 1.0


# ==========================================================
# Z-SCORE
# ==========================================================


def test_zscore_zero():

    result = normalize_zscore_score(
        0
    )

    assert result == 0.0


def test_zscore_positive():

    result = normalize_zscore_score(
        3
    )

    assert result == pytest.approx(
        0.75
    )


def test_zscore_large_score():

    result = normalize_zscore_score(
        99
    )

    assert 0.98 < result < 1.0


# ==========================================================
# LOF
# ==========================================================


def test_lof_normal():

    result = normalize_lof_score(
        1.0
    )

    assert result == 0.0


def test_lof_below_one():

    result = normalize_lof_score(
        0.8
    )

    assert result == 0.0


def test_lof_anomaly():

    result = normalize_lof_score(
        2.0
    )

    assert result == pytest.approx(
        0.5
    )


def test_lof_large_score():

    result = normalize_lof_score(
        5.0
    )

    assert result == pytest.approx(
        0.8
    )


# ==========================================================
# ISOLATION FOREST
# ==========================================================


def test_isolation_forest_score():

    result = normalize_isolation_forest_score(
        0.5
    )

    assert result == 0.5


def test_isolation_forest_clamp_high():

    result = normalize_isolation_forest_score(
        2.0
    )

    assert result == 1.0


def test_isolation_forest_clamp_low():

    result = normalize_isolation_forest_score(
        -1.0
    )

    assert result == 0.0


# ==========================================================
# SEMANTIC
# ==========================================================


def test_semantic_normal():

    result = normalize_semantic_score(
        0.0
    )

    assert result == 0.0


def test_semantic_anomaly():

    result = normalize_semantic_score(
        1.0
    )

    assert result == 1.0


def test_semantic_clamp():

    result = normalize_semantic_score(
        2.0
    )

    assert result == 1.0


# ==========================================================
# GENERIC NORMALIZER
# ==========================================================


def test_generic_iqr():

    result = normalize_detector_score(
        "iqr",
        1.0,
    )

    assert result == pytest.approx(
        0.5
    )


def test_generic_zscore():

    result = normalize_detector_score(
        "zscore",
        3.0,
    )

    assert result == pytest.approx(
        0.75
    )


def test_generic_lof():

    result = normalize_detector_score(
        "lof",
        2.0,
    )

    assert result == pytest.approx(
        0.5
    )


def test_generic_semantic():

    result = normalize_detector_score(
        "semantic",
        0.8,
    )

    assert result == 0.8


def test_unknown_detector():

    with pytest.raises(
        ValueError
    ):

        normalize_detector_score(
            "unknown",
            1.0,
        )
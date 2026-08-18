from types import SimpleNamespace

from app.detection.detector_selector import (
    select_detectors,
)


def test_selector_small_numeric_dataset():

    characteristics = SimpleNamespace(
        rows=10,
        numeric_columns=2,
        skewed=False,
    )

    detectors = select_detectors(
        characteristics
    )

    assert "iqr" in detectors

    assert "zscore" in detectors

    assert "isolation_forest" not in detectors

    assert "lof" not in detectors

    assert "dbscan" not in detectors


def test_selector_large_dataset():

    characteristics = SimpleNamespace(
        rows=100,
        numeric_columns=3,
        skewed=False,
    )

    detectors = select_detectors(
        characteristics
    )

    assert "iqr" in detectors

    assert "zscore" in detectors

    assert "isolation_forest" in detectors

    assert "lof" in detectors

    assert "dbscan" in detectors


def test_selector_skewed_dataset():

    characteristics = SimpleNamespace(
        rows=100,
        numeric_columns=3,
        skewed=True,
    )

    detectors = select_detectors(
        characteristics
    )

    assert "iqr" in detectors

    assert "zscore" not in detectors

    assert "isolation_forest" in detectors

    assert "lof" in detectors

    assert "dbscan" in detectors


def test_selector_no_numeric_columns():

    characteristics = SimpleNamespace(
        rows=100,
        numeric_columns=0,
        skewed=False,
    )

    detectors = select_detectors(
        characteristics
    )

    assert "iqr" not in detectors
    assert "zscore" not in detectors
    assert "isolation_forest" not in detectors
    assert "lof" not in detectors
    assert "dbscan" not in detectors   
    
import pandas as pd
import pytest
from app.quality.evaluator import (
    calculate_missing_ratio,
    calculate_quality_score,
    evaluate_quality,
    compare_quality,
)


def test_missing_ratio_without_missing_values():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "salary": [1000, 2000, 3000],
        }
    )

    ratio = calculate_missing_ratio(df)

    assert ratio == 0.0


def test_missing_ratio_with_missing_values():

    df = pd.DataFrame(
        {
            "age": [20, None, 40],
            "salary": [1000, 2000, 3000],
        }
    )

    ratio = calculate_missing_ratio(df)

    assert ratio == 1 / 6


def test_quality_score_without_missing_values():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    score = calculate_quality_score(df)

    assert score == 100.0


def test_quality_score_with_missing_values():

    df = pd.DataFrame(
        {
            "age": [20, None, 40],
        }
    )

    score = calculate_quality_score(df)

    assert score == pytest.approx((2 / 3) * 100)


def test_evaluate_quality():

    df = pd.DataFrame(
        {
            "age": [20, None, 40],
            "salary": [1000, 2000, None],
        }
    )

    result = evaluate_quality(df)

    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["total_cells"] == 6
    assert result["missing_cells"] == 2

    assert result["missing_ratio"] == 2 / 6

    assert result["quality_score"] == pytest.approx(
    (4 / 6) * 100
)


def test_compare_quality():

    before = {
        "rows": 3,
        "columns": 2,
        "total_cells": 6,
        "missing_cells": 0,
        "missing_ratio": 0.0,
        "quality_score": 100.0,
    }

    after = {
        "rows": 3,
        "columns": 2,
        "total_cells": 6,
        "missing_cells": 2,
        "missing_ratio": 2 / 6,
        "quality_score": (4 / 6) * 100,
    }

    result = compare_quality(
        before,
        after,
    )

    assert result["missing_cells_change"] == 2

    assert result["missing_ratio_change"] == 2 / 6

    assert result["quality_score_change"] == (
        (4 / 6) * 100
        - 100
    )

    assert result["rows_change"] == 0
    assert result["columns_change"] == 0

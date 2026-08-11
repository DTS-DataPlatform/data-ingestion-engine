import pandas as pd
import pytest

from app.quality.comparison import (
    create_quality_snapshot,
    compare_quality,
)


def test_quality_snapshot():

    df = pd.DataFrame(
        {
            "age": [20, None, 30],
            "salary": [100, 200, None],
        }
    )

    snapshot = create_quality_snapshot(
        dataframe=df,
        anomaly_count=2,
    )

    assert snapshot.rows == 3

    assert snapshot.columns == 2

    assert snapshot.total_cells == 6

    assert snapshot.missing_cells == 2

    assert snapshot.missing_ratio == pytest.approx(
        2 / 6
    )

    assert snapshot.anomaly_count == 2

    assert snapshot.quality_score == pytest.approx(
        (4 / 6) * 100
    )
def test_quality_comparison():

    before_df = pd.DataFrame(
        {
            "age": [20, None, 30],
            "salary": [100, 200, None],
        }
    )

    after_df = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "salary": [100, 200, 150],
        }
    )

    before = create_quality_snapshot(
        before_df,
        anomaly_count=2,
    )

    after = create_quality_snapshot(
        after_df,
        anomaly_count=0,
    )

    comparison = compare_quality(
        before,
        after,
    )

    assert comparison.missing_reduction == 2

    assert comparison.anomaly_reduction == 2

    assert (
        comparison.missing_ratio_reduction
        == pytest.approx(2 / 6)
    )

    assert (
        comparison.quality_score_improvement
        == pytest.approx(
            100 - ((4 / 6) * 100)
        )
    )

    assert comparison.quality_improved is True
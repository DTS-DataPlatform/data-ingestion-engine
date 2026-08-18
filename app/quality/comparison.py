import pandas as pd

from .models import (
    QualitySnapshot,
    QualityComparison,
)


def create_quality_snapshot(
    dataframe: pd.DataFrame,
    anomaly_count: int = 0,
) -> QualitySnapshot:

    rows = len(dataframe)

    columns = len(dataframe.columns)

    total_cells = rows * columns

    missing_cells = int(
        dataframe.isna().sum().sum()
    )

    if total_cells > 0:

        missing_ratio = (
            missing_cells / total_cells
        )

        quality_score = (
            1.0 - missing_ratio
        ) * 100

    else:

        missing_ratio = 0.0

        quality_score = 100.0

    return QualitySnapshot(
        rows=rows,
        columns=columns,
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_ratio=missing_ratio,
        anomaly_count=anomaly_count,
        quality_score=quality_score,
    )
    
def compare_quality(
    before: QualitySnapshot,
    after: QualitySnapshot,
) -> QualityComparison:

    missing_reduction = (
        before.missing_cells
        - after.missing_cells
    )

    missing_ratio_reduction = (
        before.missing_ratio
        - after.missing_ratio
    )

    anomaly_reduction = (
        before.anomaly_count
        - after.anomaly_count
    )

    quality_score_improvement = (
        after.quality_score
        - before.quality_score
    )

    quality_improved = (
        quality_score_improvement > 0
        or missing_reduction > 0
        or anomaly_reduction > 0
    )

    return QualityComparison(
        before=before,
        after=after,
        missing_reduction=missing_reduction,
        missing_ratio_reduction=(
            missing_ratio_reduction
        ),
        anomaly_reduction=anomaly_reduction,
        quality_score_improvement=(
            quality_score_improvement
        ),
        quality_improved=quality_improved,
    )